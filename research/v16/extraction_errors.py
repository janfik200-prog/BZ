"""Задача 2 ТЗ v16: ошибки извлечения по ручной разметке автора и сохранение модальности/отрицаний.

Разметка — research/v15/Разметка_v15_автор.xlsx, лист «Факты» (120 случайных связей графа,
1/0 и комментарий автора). Категория ошибки — по комментарию автора; если в комментарии
названы две причины, указаны обе (основная — первой).

Модальность: правило кода facts.hedged (пометка «предположительно», если в цитате есть
предположение) — как работает сейчас и с расширенным списком оборотов (HEDGE_EXTRA). Список
расширен по тем же 120 связям, поэтому его полнота на них — не оценка: нужна новая выборка
(лист «Модальность» в Разметка_v16.xlsx, labeling_sheet.py).

Отрицания: связь, в цитате которой есть отрицание, а в самой связи — нет. Это отбор на
проверку, а не мера ошибок: «не» может относиться к другой части предложения.

    extraction_errors.py → results/extraction_errors.csv, results/extraction_modality.json
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import psycopg  # noqa: E402

from georag.graph import facts as FX  # noqa: E402
from georag.graph import store  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
RES = HERE / "results"
SHEET = HERE.parent / "v15" / "Разметка_v15_автор.xlsx"

# Категории 16 ошибок — по комментариям автора (столбец «Комментарий» листа «Факты»).
CATEGORY = {
    6: ["связь не геологического содержания"],
    12: ["связь не геологического содержания"],
    20: ["обратное направление"],
    26: ["потеряна оговорка о предположительности"],
    51: ["потеряна оговорка о предположительности", "неверный тип связи"],
    61: ["неверный тип связи", "предположение (пометка в связи сохранена)"],
    68: ["неверный тип связи"],
    70: ["потеряна оговорка о предположительности"],
    77: ["потеряна оговорка о предположительности"],
    80: ["потеряна оговорка о предположительности"],
    81: ["неверный тип связи"],
    82: ["неверный тип связи"],
    88: ["неверный тип связи"],
    92: ["потеряна оговорка о предположительности"],
    115: ["неверный тип связи"],
    120: ["неверный тип связи"],
}
# Обороты предположения, которых нет в facts._HEDGE_QUOTE: найдены в ошибках этой выборки.
HEDGE_EXTRA = re.compile(
    r"предпо-?\s*лага|interpret|propos|\bcan\s+(?:reduce|be|indicate|affect)|\bcould\b|"
    r"предполага|интерпретир",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:not|no|neither|nor|absent|absence|lack|lacks|without)\b|(?<!\w)(?:не|нет|ни)(?!\w)|"
                      r"отсутств", re.IGNORECASE)
NEG_REL = re.compile(r"(?<!\w)(?:не|нет)(?!\w)|отсутств", re.IGNORECASE)


def hedge_now(relation: str, quote: str) -> bool:
    return FX.HEDGE in relation or bool(FX._HEDGE_RELATION.search(relation)) or FX.hedged(relation, quote) != relation


def hedge_ext(relation: str, quote: str) -> bool:
    return hedge_now(relation, quote) or bool(HEDGE_EXTRA.search(quote))


def main() -> None:
    from openpyxl import load_workbook

    ws = load_workbook(str(SHEET), data_only=True)["Факты"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if str(r[6]).strip() not in ("0", "1"):
            continue
        no, src, rel, dst, quote, title, ok, note = r[:8]
        no, ok, quote = int(no), int(ok), str(quote or "")
        cats = CATEGORY.get(no, []) if not ok else []
        assert bool(cats) == (not ok), no
        rows.append({"no": no, "src": src, "relation": rel, "dst": dst, "quote": quote, "article": title,
                     "correct": ok, "category": "; ".join(cats), "author_comment": note or "",
                     "hedge_in_quote": int(bool(FX._HEDGE_QUOTE.search(quote) or HEDGE_EXTRA.search(quote))),
                     "hedge_marked_now": int(hedge_now(rel, quote)),
                     "hedge_marked_extended": int(hedge_ext(rel, quote)),
                     "negation_in_quote": int(bool(NEGATION.search(quote))),
                     "negation_in_relation": int(bool(NEG_REL.search(rel)))})
    RES.mkdir(exist_ok=True)
    with (RES / "extraction_errors.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lost = [x for x in rows if "потеряна оговорка" in x["category"]]
    correct = [x for x in rows if x["correct"]]
    res = {
        "sample": len(rows), "correct": len(correct),
        "errors_by_primary_category": {},
        "lost_hedge_errors": len(lost),
        "lost_hedge_fixed_by_current_rule": sum(x["hedge_marked_now"] for x in lost),
        "lost_hedge_fixed_by_extended_rule": sum(x["hedge_marked_extended"] for x in lost),
        "correct_newly_marked_by_extended_rule": sum(x["hedge_marked_extended"] - x["hedge_marked_now"]
                                                     for x in correct),
        "negation_in_quote": sum(x["negation_in_quote"] for x in rows),
        "negation_in_quote_correct": sum(x["negation_in_quote"] and x["correct"] for x in rows),
        "negation_kept_in_relation": sum(x["negation_in_quote"] and x["negation_in_relation"] for x in rows),
    }
    for x in rows:
        if x["category"]:
            k = x["category"].split("; ")[0]
            res["errors_by_primary_category"][k] = res["errors_by_primary_category"].get(k, 0) + 1

    conn = psycopg.connect(DSN, autocommit=True)
    graph = store.all_facts(conn)
    res["graph"] = {
        "facts": len(graph),
        "marked_hedged": sum(FX.HEDGE in f["relation"] for f in graph),
        "hedge_in_quote_current_rule": sum(bool(FX._HEDGE_QUOTE.search(f["quote"] or "")) for f in graph),
        "would_be_marked_extended": sum(hedge_ext(f["relation"], f["quote"] or "") and not
                                        hedge_now(f["relation"], f["quote"] or "") for f in graph),
        "negation_in_quote": sum(bool(NEGATION.search(f["quote"] or "")) for f in graph),
        "negation_in_quote_not_in_relation": sum(bool(NEGATION.search(f["quote"] or "")) and not
                                                 NEG_REL.search(f["relation"]) for f in graph),
    }
    (RES / "extraction_modality.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
