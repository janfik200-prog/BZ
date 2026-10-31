"""Задачи 5 и 6 ТЗ v17: разбор по вопросам — B3 против B2t и потери от графовых кандидатов.

Те же 173 вопроса и те же функции поиска, что в retrieval_eval.py (параметры не меняются,
ничего не подбирается по вопросам). Для каждого вопроса дополнительно сохраняются
промежуточные списки: какие статьи разрешает территория в B2t (написание из вопроса) и в
B3 (concepts.resolve + place_docs), какие фрагменты дал граф и как изменилась выдача.

Задача 5: парные исходы B3/B2t по «статья первой» (исправил / испортил / оба верно / оба
неверно) и признаки случая: место не названо, написание в другой письменности или
транслитерация, несколько написаний, место только в тексте статьи (не в названии), в
вопросе несколько мест. Значимость: точный биномиальный тест по несовпавшим парам и
бутстрэп с группировкой по статье (из retrieval_summary.json).

Задача 6: вопросы, где +gf или +gn изменили «статья первой» или «фрагмент среди пяти»;
для каждого — исходная выдача, графовые кандидаты (со связями и узлами), новая выдача и
механизм изменения.

    pairwise_cases.py → results/territory_pairwise_cases.csv, results/graph_failure_cases.csv,
                        results/pairwise_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import retrieval_eval as R  # noqa: E402

from georag.graph import api  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

RES = HERE / "results"
CYR = re.compile(r"[а-яё]", re.I)
LAT = re.compile(r"[a-z]", re.I)


def script(s: str) -> str:
    return "кир" if CYR.search(s) and not LAT.search(s) else "лат" if LAT.search(s) and not CYR.search(s) else "смеш"


def binom_two_sided(k: int, n: int) -> float:
    """Точный двусторонний биномиальный тест (p = 0,5) для несовпавших пар."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    qs = R.questions()
    emb, syn = Embedder(), S.load()
    t_rows, g_rows = [], []
    with connect(R.E.DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, doc_id FROM chunks")
            chunk_doc = dict(cur.fetchall())
            cur.execute("SELECT doc_id, coalesce(title, '') FROM documents")
            titles = dict(cur.fetchall())
        F = R.Facts(conn, emb, api._with_auto(conn, syn))
        fact_of_chunk: dict[int, list[int]] = defaultdict(list)
        for fid, c in F.chunk.items():
            if F.status[fid] == "graph":
                fact_of_chunk[c].append(fid)
        degree = {n: len(v) for n, v in F.ends.items()}
        with conn.cursor() as cur:
            cur.execute("SELECT id, src, relation, dst FROM facts")
            fact_txt = {r[0]: f"{r[1]} — {r[2]} — {r[3]}" for r in cur.fetchall()}

        for i, x in enumerate(qs, 1):
            res, _sec, why = R.run_question(conn, emb, syn, F, chunk_doc, x)
            info = x["info"]
            raw, names, concepts = info["place"] or [], info["names"], info["concepts"]
            docs2t = set(C.place_docs(conn, emb, raw)) if raw else set()
            docs3 = set(C.place_docs(conn, emb, names)) if names else set()
            low = [C._distinct(p).lower() for p in raw if p.strip()]
            in_title = any(s and s in titles[x["doc_id"]].lower() for s in low)
            exp, exp_chunk = x["doc_id"], x["chunk_id"]

            def docs(v):
                return [chunk_doc[c] for c in res[v][:R.K]]

            first2t, first3 = int(bool(res["B2t"]) and docs("B2t")[0] == exp), int(bool(res["B3"]) and docs("B3")[0] == exp)
            outcome = {(0, 1): "B3 исправил", (1, 0): "B3 испортил", (1, 1): "оба верно", (0, 0): "оба неверно"}[
                (first2t, first3)]
            extra = sorted({n for n in names} - {p for p in raw})
            tags = []
            if not raw:
                tags.append("место не названо")
            if raw and names and {script(n) for n in names} - {script(p) for p in raw}:
                tags.append("другая письменность / транслитерация")
            if len(names) > len(raw) and raw:
                tags.append("несколько написаний")
            if raw and exp in docs2t and not in_title:
                tags.append("место только в тексте статьи")
            if len(raw) > 1:
                tags.append("несколько мест в вопросе")
            if raw and len(docs3) > 2 * max(1, len(docs2t)):
                tags.append("неоднозначный топоним: B3 расширил круг статей")
            t_rows.append({
                "query_id": x["qid"], "query": x["question"], "place_in_query": "; ".join(raw),
                "resolved_names_b3": "; ".join(names), "names_added_by_b3": "; ".join(extra),
                "expected_doc": exp, "expected_title": titles[exp],
                "docs_b2t": len(docs2t), "docs_b3": len(docs3),
                "expected_in_b2t": int(exp in docs2t), "expected_in_b3": int(exp in docs3),
                "b2t_first": first2t, "b3_first": first3, "outcome": outcome, "tags": "; ".join(tags),
                "b2t_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in res["B2t"][:R.K]),
                "b3_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in res["B3"][:R.K]),
            })

            base = res["B3"][:R.K]
            for v in ("B3+gf", "B3+gn"):
                new = res[v][:R.K]
                f0, f1 = int(bool(base) and chunk_doc[base[0]] == exp), int(bool(new) and chunk_doc[new[0]] == exp)
                p0, p1 = int(exp_chunk in base), int(exp_chunk in new)
                if (f0, p0) == (f1, p1):
                    continue
                graph_in = [c for c in new if why[v].get(c) == "граф"]  # новые кандидаты графа
                both_in = [c for c in new if why[v].get(c) == "оба"]  # нашли и вектор, и граф
                pushed = [c for c in base if c not in new]
                mech = []
                if p0 and not p1:
                    if any(chunk_doc[c] == exp for c in graph_in):
                        mech.append("нужный фрагмент вытеснен новым графовым фрагментом той же статьи (лимит 2 на статью)")
                    elif graph_in:
                        mech.append("нужный фрагмент вытеснен новыми графовыми фрагментами других статей")
                    else:
                        mech.append("перестановка RRF: фрагменты, найденные и вектором, и графом, поднялись выше нужного")
                if f0 and not f1:
                    mech.append("первое место занял новый графовый фрагмент другой статьи" if why[v].get(new[0]) == "граф"
                                else "перестановка RRF: первое место занял фрагмент другой статьи, найденный и вектором, "
                                     "и графом")
                if (not p0 and p1) or (not f0 and f1):
                    mech.append("граф помог: " + ("нужный фрагмент вошёл в пятёрку" if not p0 and p1 else
                                                  "нужная статья стала первой"))
                g_facts = [fid for c in graph_in + both_in for fid in fact_of_chunk.get(c, [])][:6]
                nodes = sorted({F.node(s) for sp in concepts for s in sp})
                hub = max((degree.get(n, 0) for n in nodes), default=0)
                g_rows.append({
                    "query_id": x["qid"], "variant": v, "query": x["question"],
                    "change": ("хуже" if (f1 - f0) + (p1 - p0) < 0 else "лучше" if (f1 - f0) + (p1 - p0) > 0
                               else "смешанно"),
                    "first_before": f0, "first_after": f1, "passage_before": p0, "passage_after": p1,
                    "base_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in base),
                    "graph_candidates_in_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in graph_in),
                    "graph_candidates_from_expected_doc": sum(chunk_doc[c] == exp for c in graph_in),
                    "both_lists_in_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in both_in),
                    "new_first_found_by": why[v].get(new[0], "") if new else "",
                    "graph_relations": " | ".join(fact_txt[f] for f in g_facts),
                    "new_top5": " ".join(f"{c}:{chunk_doc[c]}" for c in new),
                    "pushed_out": " ".join(map(str, pushed)),
                    "query_concepts": "; ".join(nodes), "max_node_degree": hub,
                    "mechanism": "; ".join(mech),
                })
            print(i, x["qid"], flush=True)

    with (RES / "territory_pairwise_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, list(t_rows[0]))
        w.writeheader()
        w.writerows(t_rows)
    with (RES / "graph_failure_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, list(g_rows[0]))
        w.writeheader()
        w.writerows(g_rows)

    oc = Counter(r["outcome"] for r in t_rows)
    fixed, broken = oc["B3 исправил"], oc["B3 испортил"]
    by_tag = defaultdict(Counter)
    for r in t_rows:
        for t in (r["tags"].split("; ") if r["tags"] else ["без особенностей"]):
            by_tag[t][r["outcome"]] += 1
    summary = {
        "questions": len(t_rows), "outcomes": dict(oc),
        "discordant": fixed + broken, "exact_binomial_p": round(binom_two_sided(fixed, fixed + broken), 4),
        "by_tag": {t: dict(c) for t, c in by_tag.items()},
        "expected_doc_in_territory": {"b2t": sum(r["expected_in_b2t"] for r in t_rows),
                                      "b3": sum(r["expected_in_b3"] for r in t_rows),
                                      "with_place": sum(1 for r in t_rows if r["place_in_query"])},
        "graph_changes": {v: dict(Counter(r["change"] for r in g_rows if r["variant"] == v))
                          for v in ("B3+gf", "B3+gn")},
        "graph_mechanisms": dict(Counter(m for r in g_rows if r["change"] != "лучше"
                                         for m in r["mechanism"].split("; ") if m)),
    }
    (RES / "pairwise_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
