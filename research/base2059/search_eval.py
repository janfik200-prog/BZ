"""Поиск нужной статьи на этой базе: «по смыслу» (B1) против «по смыслу + территория» (B3).

Пилотный research/v17/retrieval_eval.py: те же run_question, metrics и boot, те же K, TOP,
PER_DOC. Вопросы — набор 1 этой базы (qa.py), только отмеченные автором «Годится = 1»;
место и понятия вопроса — из сохранённого прогона (qa/article_qa_runs.json), модель не
нужна. Варианты с графом (+gf, +gn) в отчёт не входят, поэтому вместо связей графа —
пустая заглушка: на B1 и B3 она не влияет. База только читается.

    python research/base2059/search_eval.py  →  results/search_summary.json, results/search_by_question.csv
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "v17"))
sys.path.insert(0, str(ROOT / "research" / "v15"))
sys.path.insert(0, str(ROOT))

import retrieval_eval as R  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
QA = HERE / "qa"
RES = HERE / "results"


class NoGraph:
    def by_vector(self, *a, **k):
        return []

    def by_nodes(self, *a, **k):
        return {}


def questions() -> list[dict]:
    from openpyxl import load_workbook

    items = json.loads((QA / "article_qa_runs.json").read_text("utf-8"))
    ws = load_workbook(str(QA / "Вопросы_набор1.xlsx"), data_only=True)["Вопросы"]
    out = []
    for i, x in enumerate(items, 1):
        if str(ws.cell(i + 1, 6).value).strip() == "1":
            out.append({**x, "qid": f"set1-{i}"})
    return out, len(items)


def main() -> None:
    qs, total = questions()
    emb, syn = Embedder(), S.load()
    rows = []
    with connect(DSN + "?options=-c%20default_transaction_read_only%3Don") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, doc_id FROM chunks")
            chunk_doc = dict(cur.fetchall())
        for x in qs:
            out, _sec, _why = R.run_question(conn, emb, syn, NoGraph(), chunk_doc, x)
            m = {v: R.metrics(x, out[v], chunk_doc) for v in ("B1", "B3")}
            rows.append({"qid": x["qid"], "doc_id": x["doc_id"], "place": bool(x["info"]["names"]),
                         **{f"{v}_{k}": m[v][k] for v in m for k in ("first", "top5")}})
            print(x["qid"], m["B1"]["first"], m["B3"]["first"], flush=True)
    RES.mkdir(exist_ok=True)
    with (RES / "search_by_question.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def share(part, key):
        return round(sum(r[key] for r in part) / len(part), 3) if part else None

    summary = {"questions_total": total, "questions_valid": len(rows)}
    for name, part in (("all", rows), ("with_place", [r for r in rows if r["place"]]),
                       ("without_place", [r for r in rows if not r["place"]])):
        summary[name] = {"n": len(part),
                         "B1_first": share(part, "B1_first"), "B3_first": share(part, "B3_first"),
                         "B1_top5": share(part, "B1_top5"), "B3_top5": share(part, "B3_top5"),
                         "d_first_B3_B1": R.boot([r["B3_first"] - r["B1_first"] for r in part]) if part else None}
    (RES / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
