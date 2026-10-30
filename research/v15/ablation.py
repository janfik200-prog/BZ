"""Вклад частей поиска на тех же 89 вопросах по статьям базы (article_qa.py), без новых
вызовов модели: понятия и место вопроса берутся из results/article_qa_runs.json.

Варианты: основа — объединение (hybrid) или поиск по смыслу (vector); к ней по отдельности и
вместе — два слоя по месту вопроса (place) и упорядочивание по ключевым понятиям (concepts).
hybrid+place+concepts — способ «система» из статьи (должен совпасть с сохранённым прогоном).

    ablation.py  → results/ablation.json
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import search_eval as E  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

RES = HERE / "results"
CHECKED = HERE / "Разметка_вопросы_по_статьям_проверено.xlsx"


def _selected() -> list[dict]:
    from openpyxl import load_workbook

    runs = json.loads((RES / "article_qa_runs.json").read_text("utf-8"))
    ws = load_workbook(str(CHECKED), data_only=True)["Вопросы"]
    return [x for i, x in enumerate(runs, 1) if str(ws.cell(i + 1, 6).value).strip() == "1"]


def _top(hits, concepts=None) -> list[int]:
    if concepts:
        hits = C.rerank(hits, concepts)
    return [h.chunk_id for h in X.cap_per_doc(hits, E.PER_DOC)[: E.K]]


def run_variants(conn, emb, syn, x) -> dict[str, list[int]]:
    q, concepts = x["question"], x["info"]["concepts"]
    names = x["info"]["names"]
    docs = C.place_docs(conn, emb, names) if names else []
    vec = emb.encode_one(q)
    alts = S.synonym_variants(q, syn)

    def hybrid(where):
        return X.hybrid_search(conn, emb, q, limit=100, candidates=50, max_per_doc=0, alternatives=alts,
                               vector=vec, doc_id=where or None)

    def vector(where):
        return [X._to_hit(r) for r in X.vector_search(conn, vec, 50, doc_id=where or None)]

    out = {}
    for base, fn in (("hybrid", hybrid), ("vector", vector)):
        pool, ppool = fn(None), (fn(docs) if docs else None)
        out[base] = _top(pool)
        out[f"{base}+place"] = _top(ppool if docs else pool)
        out[f"{base}+concepts"] = _top(pool, concepts)
        out[f"{base}+place+concepts"] = _top(ppool if docs else pool, concepts)
    return out


def _paired(d: list[int]) -> list[float]:
    rng = random.Random(1)
    boot = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(10000))
    return [round(statistics.mean(d), 3), round(boot[250], 3), round(boot[9750], 3)]


def main() -> None:
    sel = _selected()
    emb, syn = Embedder(), S.load()
    with connect(E.DSN) as conn:
        cur = conn.cursor()
        for x in sel:
            x["abl"] = run_variants(conn, emb, syn, x)
        ids = sorted({c for x in sel for ids in x["abl"].values() for c in ids})
        cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (ids,))
        doc = dict(cur.fetchall())
    same = sum(x["abl"]["hybrid+place+concepts"] == x["runs"]["system"] for x in sel)
    res: dict = {"questions": len(sel), "system_reproduced": same}
    variants = list(sel[0]["abl"])
    for part, xs in (("all", sel), ("place", [x for x in sel if x["info"]["names"]]),
                     ("noplace", [x for x in sel if not x["info"]["names"]])):
        res[part] = {"n": len(xs)}
        for v in variants:
            top5 = [int(x["doc_id"] in [doc[c] for c in x["abl"][v]]) for x in xs]
            first = [int(bool(x["abl"][v]) and doc[x["abl"][v][0]] == x["doc_id"]) for x in xs]
            frag = [int(x["chunk_id"] in x["abl"][v]) for x in xs]
            res[part][v] = {"top5": round(statistics.mean(top5), 3), "first": round(statistics.mean(first), 3),
                            "frag": round(statistics.mean(frag), 3)}
            res[part][v]["_first"], res[part][v]["_frag"], res[part][v]["_top5"] = first, frag, top5
        for a, b in (("vector+place+concepts", "vector"), ("vector+place", "vector"),
                     ("vector+concepts", "vector"), ("hybrid+place+concepts", "vector")):
            res[part][f"{a} - {b}"] = {
                m: _paired([p - q for p, q in zip(res[part][a][f"_{m}"], res[part][b][f"_{m}"])])
                for m in ("top5", "first", "frag")}
        for v in variants:
            for m in ("_first", "_frag", "_top5"):
                del res[part][v][m]
    (RES / "ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
