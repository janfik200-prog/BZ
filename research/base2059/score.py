"""Итог ручной разметки базы из 2 059 статей (Разметка_база2059.xlsx, labeling.py).

Фильтр, типизация, отрицания, модальность — пилотный research/v16/labeling_score.py как
есть (ключ — key.json этой выборки, результаты — research/base2059/results). Лист «Связи»
— как лист «Факты» пилота (research/v15): доля верных, виды ошибок; для потерянных оговорок
— находит ли их действующее правило (extraction_errors.hedge_now) и расширенное (hedge_ext).
Считаются только строки, где стоит 1 или 0. База только читается.

    python research/base2059/score.py РАЗМЕТКА.xlsx  →  results/labeling_scores.json, results/links_scores.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "v16"))
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

import extraction_errors as XE  # noqa: E402
import labeling_score as LS  # noqa: E402

RES = HERE / "results"
DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")


def links(path: str) -> dict:
    from openpyxl import load_workbook

    key = json.loads((HERE / "key.json").read_text("utf-8"))
    ws = load_workbook(path, data_only=True)["Связи"]
    marks = []
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, 8).value
        if str(v).strip() in ("0", "1"):
            kind = (ws.cell(r, 9).value or "").strip().lower() or None
            marks.append((key["facts"][r - 2], int(str(v).strip()), kind))
    n = len(marks)
    ok = sum(v for _, v, _ in marks)
    kinds = Counter(k or "не указан" for _, v, k in marks if not v)
    with psycopg.connect(DSN + "?options=-c%20default_transaction_read_only%3Don") as c:
        rel = dict((i, (r, q)) for i, r, q in c.execute(
            "SELECT id, relation, quote FROM facts WHERE id = ANY(%s)", ([i for i, _, _ in marks],)).fetchall())
    hedge = [i for i, v, k in marks if not v and k == "оговорка"]
    out = {"labelled": n, "correct": ok, "share": round(ok / n, 3) if n else None, "ci": LS.wilson(ok, n),
           "errors_by_kind": dict(kinds),
           "hedge_lost": len(hedge),
           "hedge_found_now": sum(XE.hedge_now(*rel[i][::1]) for i in hedge),
           "hedge_found_ext": sum(XE.hedge_ext(*rel[i][::1]) for i in hedge)}
    (RES / "links_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out


def main(path: str) -> None:
    RES.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "key.json", RES / "labeling_key.json")
    LS.RES = RES
    print(json.dumps(links(path), ensure_ascii=False, indent=1))
    LS.main(path)


if __name__ == "__main__":
    main(sys.argv[1])
