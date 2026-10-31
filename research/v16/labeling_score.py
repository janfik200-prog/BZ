"""Оценка по размеченному Разметка_v16.xlsx (labeling_sheet.py): фильтр, типизация, отрицания,
модальность. Считается только по строкам, где автор поставил 1 или 0.

Фильтр: для порога t отсеиваются фрагменты с числом терминов < t. Доля фрагментов со связями
в каждой группе терминов — по разметке; по группам она переносится на весь корпус (выборка
стратифицирована: по 25 из группы), отсюда — доля фрагментов со связями, которые порог
сохранил бы (relation_recall), и их оценённое число среди отсеянных (missed_relations).
Экономия токенов — доля токенов отсеянных фрагментов среди не-дубликатов и не-подписей
(оценка по длине фрагментов; прямой замер на 100 фрагментах для порога 3 — 21,3 %).

    labeling_score.py [xlsx] → results/prefilter_evaluation.csv, results/labeling_scores.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BUCKETS = ("0", "1", "2", "3", "4", "5+")


def _marks(ws, col: int) -> dict[int, int]:
    out = {}
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, col).value
        if str(v).strip() in ("0", "1"):
            out[r - 1] = int(str(v).strip())
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def main(path: str) -> None:
    from openpyxl import load_workbook

    key = json.loads((RES / "labeling_key.json").read_text("utf-8"))
    wb = load_workbook(path, data_only=True)
    out: dict = {}

    fm = _marks(wb["Фильтр"], 4)
    by = defaultdict(lambda: [0, 0])
    examples = []
    for i, v in fm.items():
        item = key["filter"][i - 1]
        b = "5+" if item["terms"] >= 5 else str(item["terms"])
        by[b][0] += v
        by[b][1] += 1
        if v and item["terms"] < 3:
            examples.append({"row": i, **item, "relations": wb["Фильтр"].cell(i + 1, 5).value})
    pop, tok = key["filter_population"], key["filter_tokens"]
    share = {b: (by[b][0] / by[b][1] if by[b][1] else None) for b in BUCKETS}
    out["filter_by_terms"] = {b: {"labelled": by[b][1], "with_relations": by[b][0], "share": share[b],
                                  "ci": wilson(by[b][0], by[b][1]), "population": pop.get(b, 0)} for b in BUCKETS}
    rel = {b: pop.get(b, 0) * (share[b] or 0.0) for b in BUCKETS}
    total_rel, total_tok, total_chunks = sum(rel.values()), sum(tok.values()), sum(pop.values())
    rows = []
    for t in (1, 2, 3, 4):
        drop = [b for b in BUCKETS if b != "5+" and int(b) < t]
        missed = sum(rel[b] for b in drop)
        rows.append([t, sum(pop.get(b, 0) for b in drop), round(sum(pop.get(b, 0) for b in drop) / total_chunks, 3),
                     round(sum(tok.get(b, 0) for b in drop) / total_tok, 3),
                     round(1 - missed / total_rel, 3) if total_rel else "", round(missed)])
    with (RES / "prefilter_evaluation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "filtered_chunks", "filtered_share", "token_saving", "relation_recall",
                    "missed_relations"])
        w.writerows(rows)
    out["filter_thresholds"] = rows
    out["filter_missed_examples"] = examples

    for sheet, name, col in (("Типизация", "typing_wrongly_excluded", 7), ("Отрицания", "negation_correct", 7),
                             ("Модальность", "modality_hedge", 7)):
        m = _marks(wb[sheet], col)
        k, n = sum(m.values()), len(m)
        out[name] = {"labelled": n, "ones": k, "share": round(k / n, 3) if n else None, "ci": wilson(k, n)}
    if out["typing_wrongly_excluded"]["share"] is not None:
        out["typing_wrongly_excluded"]["population"] = key["typing_population"]
    (RES / "labeling_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "Разметка_v16.xlsx"))
