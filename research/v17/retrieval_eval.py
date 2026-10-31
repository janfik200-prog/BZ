"""Задачи 5 и 6 ТЗ v16: простой территориальный фильтр против текущего механизма и вклад графа.

Вопросы — оба проверенных автором набора (research/v15: 89 и 84 вопроса), без изменений;
место и понятия вопроса — из сохранённых прогонов (новых вызовов модели нет), поэтому
варианты различаются только способом поиска. Релевантность — та же: статья и фрагмент,
по которым составлен вопрос.

Территория (задача 5):
  B0   только совпадение названия: статьи, где упомянуто место (те же, что у B3), по числу
       упоминаний; фрагменты — с наибольшим числом упоминаний; без векторов;
  B1   BGE-M3 без ограничения (vector);
  B2   BGE-M3 + простой фильтр метаданных: место, как оно названо в вопросе, — подстрока
       названия статьи; нет таких статей — весь корпус;
  B2t  BGE-M3 + написание из вопроса в тексте статьи, без сопоставления написаний
       (транслитерация, векторы и звучание названий — concepts.resolve — не используются);
  B3   текущий механизм: concepts.resolve + place_docs, затем BGE-M3 (vector+place);
  B4   гибридный поиск (RRF вектор + слова) + территория (hybrid+place).

Граф (задача 6), поверх B1 и B3 — третий список кандидатов, слияние RRF:
  +gf     факты графа, ближайшие к вопросу по вектору факта (связь + цитата, fact_vectors);
  +gn     обход графа: концы связей, совпавшие с понятиями вопроса (после нормализации
          имён и словаря синонимов), → фрагменты этих связей; +gn1 — и соседи на шаг;
  абляции: gn_raw — без нормализации имён (сравнение строк целиком); gf_untyped — и связи,
  исключённые типизацией (3 585); gf_cand — и неподтверждённые кандидаты (6 336).

Метрики: статья среди пяти, статья первой, фрагмент среди пяти, MRR@10 по статьям, время.
Неопределённость — парный бутстрэп, по объединённому набору — с группировкой по статье.

    retrieval_eval.py → results/retrieval_results.csv (по вопросам), results/retrieval_summary.json,
                        results/graph_ablation.csv
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V15 = HERE.parent / "v15"
sys.path.insert(0, str(V15))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

import search_eval as E  # noqa: E402

from georag.graph import api, store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.graph.vectors import fact_text  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect, vector_literal  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.text import name_key  # noqa: E402

RES = HERE / "results"
CACHE = HERE / ".cache"
K, PER_DOC, CAND, TOP = E.K, E.PER_DOC, E.CAND, 10
TERRITORY = ("B0", "B1", "B2", "B2t", "B3", "B4")
GRAPH = ("B1+gf", "B1+gn", "B1+gn1", "B3+gf", "B3+gn", "B3+gn1",
         "B3+gn_raw", "B3+gf_untyped", "B3+gf_cand")


def questions() -> list[dict]:
    from openpyxl import load_workbook

    out = []
    for tag, runs, sheet in (("set89", "article_qa_runs.json", "Разметка_вопросы_по_статьям_проверено.xlsx"),
                             ("set84", "article_qa2_runs.json", "Разметка_вопросы_по_статьям_2_проверено.xlsx")):
        items = json.loads((V15 / "results" / runs).read_text("utf-8"))
        ws = load_workbook(str(V15 / sheet), data_only=True)["Вопросы"]
        for i, x in enumerate(items, 1):
            if str(ws.cell(i + 1, 6).value).strip() == "1":
                out.append({**x, "set": tag, "qid": f"{tag}-{i}"})
    return out


class Facts:
    """Связи с векторами и концами — в памяти: граф (3 818), исключённые типизацией, кандидаты."""

    def __init__(self, conn, emb, syn):
        acc = store.accepted()
        graph = {f["id"] for f in store.all_facts(conn)}
        with conn.cursor() as cur:
            cur.execute(f"SELECT f.id, f.chunk_id, f.src, f.relation, f.dst, f.quote, {acc} FROM facts f ORDER BY f.id")
            rows = cur.fetchall()
            cur.execute("SELECT fact_id, embedding::text FROM fact_vectors")
            vecs = {fid: np.array(json.loads(e), dtype=np.float32) for fid, e in cur.fetchall()}
        self.chunk = {r[0]: r[1] for r in rows}
        self.status = {r[0]: ("graph" if r[0] in graph else "untyped" if r[6] else "candidate") for r in rows}
        need = [r for r in rows if r[0] not in vecs]
        CACHE.mkdir(exist_ok=True)
        path = CACHE / f"fact_vectors_extra_{len(need)}_{need[-1][0] if need else 0}.npy"
        if path.exists():
            extra = np.load(path)
        else:  # вектор кандидата — так же, как у подтверждённых (vectors.fact_text); в базу не пишется
            extra = np.array(emb.encode([fact_text(r[2], r[3], r[4], r[5] or "") for r in need], progress=True),
                             dtype=np.float32)
            np.save(path, extra)
        for r, v in zip(need, extra):
            vecs[r[0]] = v
        self.ids = np.array([r[0] for r in rows])
        m = np.stack([vecs[i] for i in self.ids])
        self.mat = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.kind = np.array([self.status[i] for i in self.ids])

        def node(name: str) -> str:
            k = name_key(name)
            return name_key(syn.names[k]) if k in syn.names else k

        self.node = node
        self.ends: dict[str, list[int]] = defaultdict(list)  # объект → связи графа
        self.raw: dict[str, list[int]] = defaultdict(list)  # написание целиком → связи графа
        self.nbr: dict[str, set[str]] = defaultdict(set)
        for fid, _c, src, _rel, dst, _q, _ok in rows:
            if fid not in graph:
                continue
            a, b = node(src), node(dst)
            self.ends[a].append(fid)
            self.ends[b].append(fid)
            self.nbr[a].add(b)
            self.nbr[b].add(a)
            for s in (src, dst):
                self.raw[s.strip().lower()].append(fid)

    def by_vector(self, vec, kinds: set[str], allowed_chunks: set[int] | None) -> list[int]:
        """Фрагменты ближайших связей (близость ≥ FACT_MIN_SIMILARITY), по фрагменту — раз."""
        sims = self.mat @ (np.asarray(vec, dtype=np.float32) / np.linalg.norm(vec))
        mask = np.isin(self.kind, list(kinds)) & (sims >= X.FACT_MIN_SIMILARITY)
        out: list[int] = []
        for i in np.argsort(-np.where(mask, sims, -2.0)):
            if not mask[i] or len(out) >= CAND:
                break
            c = self.chunk[int(self.ids[i])]
            if c not in out and (allowed_chunks is None or c in allowed_chunks):
                out.append(c)
        return out

    def by_nodes(self, concepts: list[list[str]], hop: bool, raw: bool) -> dict[int, float]:
        """Фрагменты связей, концы которых — понятия вопроса: оценка — число понятий;
        соседи на шаг — с половинным весом."""
        score: dict[int, float] = defaultdict(float)
        for spellings in concepts:
            if raw:
                facts = {f for s in spellings for f in self.raw.get(s.strip().lower(), [])}
                near: set[int] = set()
            else:
                nodes = {self.node(s) for s in spellings}
                facts = {f for n in nodes for f in self.ends.get(n, [])}
                near = ({f for n in nodes for m in self.nbr.get(n, ()) for f in self.ends.get(m, [])} - facts
                        if hop else set())
            for c in {self.chunk[f] for f in facts}:
                score[c] += 1.0
            for c in {self.chunk[f] for f in near}:
                score[c] += 0.5
        return score


def run_question(conn, emb, syn, F: Facts, chunk_doc, x) -> tuple[dict, dict, dict]:
    q, info = x["question"], x["info"]
    names, raw_place, concepts = info["names"], info["place"] or [], info["concepts"]
    out: dict[str, list[int]] = {}
    sec: dict[str, float] = {}
    why: dict[str, dict[int, str]] = {}

    t0 = time.perf_counter()
    vec = emb.encode_one(q)
    t_vec = time.perf_counter() - t0

    def vector(where):
        rows = X.vector_search(conn, vec, CAND, doc_id=where or None)
        return [r[0] for r in rows], {r[0]: float(r[11]) for r in rows}

    def top(ids):
        hits = [X.Hit(chunk_id=c, doc_id=chunk_doc[c], ord=0, text="") for c in ids]
        return [h.chunk_id for h in X.cap_per_doc(hits, PER_DOC)[:TOP]]

    t0 = time.perf_counter()
    docs3 = C.place_docs(conn, emb, names) if names else []
    t_place = time.perf_counter() - t0

    # B0 — только названия
    t0 = time.perf_counter()
    if docs3:
        keys = [C._distinct(nm).lower() for nm in names]
        with conn.cursor() as cur:
            cur.execute("SELECT id, doc_id, lower(text) FROM chunks WHERE doc_id = ANY(%s)", (docs3,))
            cnt = {cid: (doc, sum(t.count(k) for k in keys)) for cid, doc, t in cur.fetchall()}
        per_doc: dict[str, int] = defaultdict(int)
        for doc, c in cnt.values():
            per_doc[doc] += c
        order = sorted(cnt, key=lambda c: (-per_doc[cnt[c][0]], -cnt[c][1], c))
        out["B0"] = top(order)
    else:
        out["B0"] = []
    sec["B0"] = t_place + time.perf_counter() - t0

    t0 = time.perf_counter()
    v1, sim1 = vector(None)
    out["B1"] = top(v1)
    sec["B1"] = t_vec + time.perf_counter() - t0

    t0 = time.perf_counter()
    low = [C._distinct(p).lower() for p in raw_place if p.strip()]
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, lower(coalesce(title, '')) FROM documents")
        docs2 = sorted(d for d, title in cur.fetchall() if any(s and s in title for s in low))
    out["B2"] = top(vector(docs2)[0]) if docs2 else out["B1"]
    sec["B2"] = t_vec + time.perf_counter() - t0

    t0 = time.perf_counter()
    docs2t = C.place_docs(conn, emb, raw_place) if raw_place else []
    out["B2t"] = top(vector(docs2t)[0]) if docs2t else out["B1"]
    sec["B2t"] = t_vec + time.perf_counter() - t0

    t0 = time.perf_counter()
    v3, sim3 = vector(docs3) if docs3 else (v1, sim1)
    out["B3"] = top(v3)
    sec["B3"] = t_vec + t_place + time.perf_counter() - t0

    t0 = time.perf_counter()
    alts = S.synonym_variants(q, syn)
    hy = X.hybrid_search(conn, emb, q, limit=100, candidates=CAND, max_per_doc=0, alternatives=alts,
                         vector=vec, doc_id=docs3 or None)
    out["B4"] = top([h.chunk_id for h in hy])
    sec["B4"] = t_vec + t_place + time.perf_counter() - t0

    allowed3 = {c for c, d in chunk_doc.items() if d in set(docs3)} if docs3 else None

    def fuse(base_ids, graph_ids, label, t_base, t_graph):
        scores = X.rrf_merge([base_ids, graph_ids])
        order = sorted(scores, key=lambda c: -scores[c])
        out[label] = top(order)
        bset, gset = set(base_ids), set(graph_ids)
        why[label] = {c: ("оба" if c in bset and c in gset else "вектор" if c in bset else "граф")
                      for c in out[label]}
        sec[label] = t_base + t_graph

    def node_list(base_ids, sims, allowed, hop, raw):
        score = F.by_nodes(concepts, hop, raw)
        cand = [c for c in score if allowed is None or c in allowed]
        missing = [c for c in cand if c not in sims]
        if missing:
            for r in X._chunks_by_id(conn, missing, vec):
                sims[r[0]] = float(r[11])
        return sorted(cand, key=lambda c: (-score[c], -sims.get(c, 0.0)))[:CAND]

    for base, ids, sims, allowed, tb in (("B1", v1, dict(sim1), None, sec["B1"]),
                                         ("B3", v3, dict(sim3), allowed3, sec["B3"])):
        t0 = time.perf_counter()
        fuse(ids, F.by_vector(vec, {"graph"}, allowed), f"{base}+gf", tb, time.perf_counter() - t0)
        for hop, tag in ((False, "gn"), (True, "gn1")):
            t0 = time.perf_counter()
            fuse(ids, node_list(ids, sims, allowed, hop, False), f"{base}+{tag}", tb, time.perf_counter() - t0)
        if base == "B3":
            t0 = time.perf_counter()
            fuse(ids, node_list(ids, sims, allowed, False, True), "B3+gn_raw", tb, time.perf_counter() - t0)
            t0 = time.perf_counter()
            fuse(ids, F.by_vector(vec, {"graph", "untyped"}, allowed), "B3+gf_untyped", tb,
                 time.perf_counter() - t0)
            t0 = time.perf_counter()
            fuse(ids, F.by_vector(vec, {"graph", "untyped", "candidate"}, allowed), "B3+gf_cand", tb,
                 time.perf_counter() - t0)
    return out, sec, why


def metrics(x, ids, chunk_doc) -> dict[str, float]:
    docs = [chunk_doc[c] for c in ids]
    first5 = docs[:K]
    order = list(dict.fromkeys(docs))
    rr = next((1.0 / (i + 1) for i, d in enumerate(order[:TOP]) if d == x["doc_id"]), 0.0)
    return {"top5": int(x["doc_id"] in first5), "first": int(bool(first5) and first5[0] == x["doc_id"]),
            "frag": int(x["chunk_id"] in ids[:K]), "mrr10": rr}


def boot(diffs: list[float], groups: list[str] | None = None) -> list[float]:
    """Среднее и 95-процентный интервал парной разности; groups — бутстрэп по статьям."""
    rng = random.Random(1)
    if groups is None:
        res = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    else:
        by: dict[str, list[float]] = defaultdict(list)
        for d, g in zip(diffs, groups):
            by[g].append(d)
        keys = list(by)
        res = []
        for _ in range(10000):
            pick = [v for _ in keys for v in by[rng.choice(keys)]]
            res.append(statistics.mean(pick))
        res.sort()
    return [round(statistics.mean(diffs), 3), round(res[250], 3), round(res[9750], 3)]


COMPARE = [("B3", "B2"), ("B3", "B2t"), ("B3", "B1"), ("B2", "B1"), ("B2t", "B1"), ("B3", "B0"), ("B4", "B3"),
           ("B1+gf", "B1"), ("B1+gn", "B1"), ("B3+gf", "B3"), ("B3+gn", "B3"), ("B3+gn1", "B3"),
           ("B3+gn_raw", "B3"), ("B3+gf_untyped", "B3"), ("B3+gf_cand", "B3")]


def main() -> None:
    RES.mkdir(exist_ok=True)
    qs = questions()
    emb, syn = Embedder(), S.load()
    with connect(E.DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, doc_id FROM chunks")
            chunk_doc = dict(cur.fetchall())
        F = Facts(conn, emb, api._with_auto(conn, syn))
        for i, x in enumerate(qs, 1):
            x["res"], x["sec"], x["why"] = run_question(conn, emb, syn, F, chunk_doc, x)
            print(i, x["qid"], x["question"][:50], flush=True)
    for i, x in enumerate(qs):
        x["_i"] = i
    variants = list(TERRITORY + GRAPH)
    # совпадение с сохранёнными прогонами v15 (ablation.py: vector, vector+place)
    v15 = json.loads((V15 / "results" / "ablation.json").read_text("utf-8"))
    rows, per = [], defaultdict(lambda: defaultdict(list))
    for x in qs:
        for v in variants:
            m = metrics(x, x["res"][v], chunk_doc)
            for k, val in m.items():
                per[v][k].append(val)
            src = x["why"].get(v, {})
            rows.append([x["qid"], x["set"], x["doc_id"], x["chunk_id"], int(bool(x["info"]["names"])), v,
                         m["top5"], m["first"], m["frag"], round(m["mrr10"], 3), round(x["sec"][v], 3),
                         " ".join(map(str, x["res"][v][:K])),
                         " ".join(src.get(c, "") for c in x["res"][v][:K])])
    with (RES / "retrieval_results.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "query_set", "expected_doc", "expected_chunk", "has_place", "variant",
                    "article_top5", "article_top1", "passage_top5", "mrr10", "seconds", "top5_chunks",
                    "found_by"])
        w.writerows(rows)

    summary: dict = {"questions": len(qs), "by_set": {}, "v15_reference": {
        "set89": {"vector": v15["all"]["vector"], "vector+place": v15["all"]["vector+place"]}}}
    groups = {"all": qs, "set89": [x for x in qs if x["set"] == "set89"],
              "set84": [x for x in qs if x["set"] == "set84"]}
    for name, xs in list(groups.items()):
        groups[f"{name}_place"] = [x for x in xs if x["info"]["names"]]
    for name, xs in groups.items():
        idx = [x["_i"] for x in xs]
        part = {"n": len(xs)}
        for v in variants:
            part[v] = {k: round(statistics.mean(per[v][k][i] for i in idx), 3) for k in per[v]}
            part[v]["seconds_median"] = round(statistics.median(x["sec"][v] for x in xs), 3)
        g = [x["doc_id"] for x in xs] if name.startswith("all") else None
        for a, b in COMPARE:
            part[f"{a} - {b}"] = {k: boot([per[a][k][i] - per[b][k][i] for i in idx], g)
                                  for k in ("top5", "first", "frag", "mrr10")}
        summary["by_set"][name] = part
    graph_share = defaultdict(lambda: [0, 0, 0])
    for x in qs:
        for v, src in x["why"].items():
            for c, s in src.items():
                if c in x["res"][v][:K] and s == "граф":
                    graph_share[v][0] += 1
                    graph_share[v][1] += int(chunk_doc[c] == x["doc_id"])
                    graph_share[v][2] += int(c == x["chunk_id"])
    summary["graph_only_hits_in_top5"] = {v: {"hits": a, "from_source_article": b, "source_chunk": c}
                                          for v, (a, b, c) in graph_share.items()}
    (RES / "retrieval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    al = summary["by_set"]["all"]
    with (RES / "graph_ablation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "n", "article_top5", "article_top1", "passage_top5", "mrr10", "seconds_median",
                    "d_top1_vs_base", "ci_low", "ci_high", "d_passage_vs_base", "ci_low", "ci_high"])
        for v in ("B1", "B3") + GRAPH:
            d = al.get(f"{v} - {'B1' if v.startswith('B1') else 'B3'}")
            r = al[v]
            w.writerow([v, al["n"], r["top5"], r["first"], r["frag"], round(r["mrr10"], 3), r["seconds_median"],
                        *(d["first"] if d else ["", "", ""]), *(d["frag"] if d else ["", "", ""])])
    print(json.dumps({k: {v: al[v] for v in variants} for k in ["all"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
