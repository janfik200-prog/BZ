"""Переранжировщик bge-reranker-v2-m3 и метки фрагментов из графа — на уже сделанной разметке.

Часть A — пары. Все размеченные пары «вопрос — фрагмент» (три листа: 25 вопросов, 20
вопросов, доразметка). Насколько оценка отделяет полезные фрагменты от бесполезных
(AUC: вероятность, что случайный полезный оценён выше случайного бесполезного), в сравнении
с близостью векторов и долей ключевых понятий.

Часть B — выдача. Кандидаты — как у способа «place» (search_eval3.py: место → статьи о нём,
ключевые понятия выше), первые TOP_POOL; затем:
  place         — как есть (прежний результат);
  rerank        — порядок по оценке переранжировщика;
  place+rerank  — сначала доля понятий, при равной — оценка переранжировщика;
  place+tags    — сначала доля понятий в тексте, затем доля понятий среди объектов графа,
                  извлечённых из этого фрагмента (вид «метод», «минерал» и т. д.).
Пятёрки, где есть неразмеченные фрагменты, считаются по размеченным; их число — в отчёте.

    rerank_eval.py <xlsx set1> <xlsx set2> <xlsx доразметки>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import search_eval as E  # noqa: E402
import search_eval3 as S3  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

RES = HERE / "results"
MODEL = "BAAI/bge-reranker-v2-m3"
TOP_POOL = 20
MAX_CHARS = 2000


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main(x1: str, x2: str, x3: str) -> None:
    from openpyxl import load_workbook
    from sentence_transformers import CrossEncoder

    sets = S3._sets()
    key3 = json.loads((RES / "search3_key.json").read_text("utf-8"))
    labels = {"set1": S3._labels(x1, sets["set1"]["rows"]), "set2": S3._labels(x2, sets["set2"]["rows"])}
    marks = E._marks(load_workbook(x3, data_only=True)["Поиск"], 5)
    for r in key3["rows"]:
        if r["row"] in marks:
            labels[r["set"]][(r["query"], r["chunk_id"])] = marks[r["row"]]

    t0 = time.monotonic()
    ce = CrossEncoder(MODEL, device="cpu", max_length=512)
    emb, syn = Embedder(), S.load()
    res: dict = {"model": MODEL, "device": "cpu"}
    with connect(E.DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text, embedding::text FROM chunks")
        rows = cur.fetchall()
        text = {i: t or "" for i, _, t, _ in rows}
        cur.execute("SELECT chunk_id, src, dst FROM facts")
        tags: dict[int, list[str]] = {}
        for cid, src, dst in cur.fetchall():
            tags.setdefault(cid, []).extend([src, dst])

        # --- часть A: все размеченные пары ---
        pairs = [(name, q, c, v) for name, lab in labels.items() for (q, c), v in lab.items()]
        qtext = {name: data["queries"] for name, data in sets.items()}
        t1 = time.monotonic()
        scores = ce.predict([(qtext[n][q - 1], text[c][:MAX_CHARS]) for n, q, c, _ in pairs], batch_size=16)
        per_pair = time.monotonic() - t1
        qvec = {(n, q): emb.encode_one(qtext[n][q - 1]) for n, q, _, _ in pairs}
        sim: dict[tuple[str, int, int], float] = {}
        for n, q, c, _ in pairs:
            cur.execute("SELECT 1 - (embedding <=> %s::vector) FROM chunks WHERE id = %s",
                        ("[" + ",".join(f"{x:.6f}" for x in qvec[(n, q)]) + "]", c))
            sim[(n, q, c)] = float(cur.fetchone()[0])
        concepts = {(n, int(q)): o["concepts"] for n in ("set1", "set2") for q, o in key3["sets"][n].items()}
        res["pairs"] = {"n": len(pairs), "useful": sum(v for *_, v in pairs),
                        "seconds_per_pair_cpu": round(per_pair / len(pairs), 3)}
        for part in ("set1", "set2", "all"):
            idx = [i for i, p in enumerate(pairs) if part == "all" or p[0] == part]
            pos = [i for i in idx if pairs[i][3]]
            neg = [i for i in idx if not pairs[i][3]]
            res["pairs"][part] = {
                "auc_reranker": auc([float(scores[i]) for i in pos], [float(scores[i]) for i in neg]),
                "auc_vector": auc([sim[pairs[i][:3]] for i in pos], [sim[pairs[i][:3]] for i in neg]),
                "auc_concepts": auc(
                    [C.share(concepts[(pairs[i][0], pairs[i][1])], text[pairs[i][2]]) for i in pos],
                    [C.share(concepts[(pairs[i][0], pairs[i][1])], text[pairs[i][2]]) for i in neg]),
            }

        # --- часть B: выдача ---
        modes = ("place", "rerank", "place+rerank", "place+tags")
        out = {n: {m: [] for m in modes} for n in ("set1", "set2")}
        rerank_s = []
        for n in ("set1", "set2"):
            for q, o in key3["sets"][n].items():
                qi, qs = int(q), sets[n]["queries"][int(q) - 1]
                docs = C.place_docs(conn, emb, o["names"]) if o["names"] else []
                pool = X.hybrid_search(conn, emb, qs, limit=100, candidates=50, max_per_doc=0,
                                       alternatives=S.synonym_variants(qs, syn), doc_id=docs or None)
                cs = o["concepts"]
                base = X.cap_per_doc(C.rerank(pool, cs), E.PER_DOC)
                top = base[:TOP_POOL]
                t1 = time.monotonic()
                sc = ce.predict([(qs, h.text[:MAX_CHARS]) for h in top], batch_size=16) if top else []
                rerank_s.append(time.monotonic() - t1)
                r = {h.chunk_id: float(s) for h, s in zip(top, sc)}
                share = {h.chunk_id: C.share(cs, h.text) for h in top}
                tag = {h.chunk_id: C.share(cs, " ; ".join(tags.get(h.chunk_id, []))) for h in top}
                order = {
                    "place": [h.chunk_id for h in top],
                    "rerank": sorted(r, key=lambda c: -r[c]),
                    "place+rerank": sorted(r, key=lambda c: (-share[c], -r[c])),
                    "place+tags": sorted(r, key=lambda c: (-share[c], -tag[c])),
                }
                for m, ids in order.items():
                    out[n][m].append([labels[n].get((qi, c)) for c in ids[: E.K]])
        res["ranking"] = {}
        for n, by in out.items():
            res["ranking"][n] = {}
            for m, runs in by.items():
                vals = [v for run in runs for v in run]
                got = [v for v in vals if v is not None]
                res["ranking"][n][m] = {"judged": len(got), "useful": sum(got),
                                        "share_useful_judged": round(sum(got) / len(got), 3),
                                        "unjudged": vals.count(None)}
        res["rerank_seconds_per_question_cpu"] = round(sum(rerank_s) / len(rerank_s), 1)
        res["fragments_with_graph_tags"] = len(tags)
    res["total_minutes"] = round((time.monotonic() - t0) / 60, 1)
    (RES / "rerank_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
