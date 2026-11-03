"""Отчёт по набору вопросов: ответы чата → таблица Excel, колонка оценки — пустая.

Для встреч с геологами: свой список вопросов (config/questions-*.yaml), ответ бота, его
источники и время; колонку «Оценка» заполняет человек. --compare-dict — каждый вопрос
дважды, со справочным словарём и без (GEORAG_DICT): видно, что словарь меняет.

У вопросов-ловушек (вид «ложный» — заведомо неверная посылка: «почему в баженовской свите
много золота?») есть колонка «Реакция»: опроверг посылку, сказал «нет данных» или
согласился. Ставится по словам ответа — это подсказка, решает человек.

    python georag.py report config/questions-oil.yaml
    python georag.py report config/questions-oil.yaml --compare-dict --out .local/reports/x.xlsx

Таблица пишется в .local/reports (внутри папки проекта): ответы базы наружу не уходят.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import LOCAL

REPORTS = LOCAL / "reports"

# Опровержение посылки и «нет данных» — по словам ответа (подсказка для оценки).
_DENY = re.compile(
    r"\b(?:не\s+(?:подтвержда|подтвержд|соответству|являет|содерж|связан|указыва)|"
    r"неверн|ошибочн|опроверга|противоречит|напротив|не\s+так)",
    re.IGNORECASE,
)
_NO_DATA = re.compile(
    r"(?:в базе знаний нет|нет (?:ответа|информации|сведений|данных)|не (?:найден|упомина|"
    r"описан|приводится)|отсутству)",
    re.IGNORECASE,
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Вопросы из YAML: список строк или словарей {вопрос, вид, заметка}."""
    import yaml

    data = yaml.safe_load(path.read_text("utf-8")) or []
    items = data.get("вопросы", data) if isinstance(data, dict) else data
    out = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"вопрос": item}
        question = " ".join(str(item.get("вопрос") or item.get("question") or "").split())
        if question:
            out.append({
                "id": str(item.get("id") or i),
                "question": question,
                "kind": str(item.get("вид") or item.get("kind") or "обычный"),
                "note": str(item.get("заметка") or item.get("note") or ""),
            })
    return out


def reaction(answer: str) -> str:
    """Как бот встретил ложную посылку — по словам ответа."""
    if _DENY.search(answer):
        return "опроверг"
    if _NO_DATA.search(answer):
        return "нет данных"
    return "согласился? проверить"


def ask(conn: Any, embedder: Any, question: str, settings: Any) -> dict[str, Any]:
    """Один ответ чата: текст (после всех правок), источники, время, режим."""
    from .answer import answer

    text, sources, done = "", [], {}
    started = time.monotonic()
    for ev in answer(conn, embedder, question, [], settings):
        kind = ev.get("type")
        if kind == "token":
            text += ev["text"]
        elif kind in ("revised", "general"):
            text = ev["text"]
        elif kind == "sources":
            sources = ev["sources"]
        elif kind == "done":
            done = ev
        elif kind == "error":
            text = f"[ошибка] {ev.get('error')}"
    used = set(done.get("used") or [])
    shown = [s for s in sources if s["n"] in used] or sources
    return {
        "answer": text.strip(),
        "sources": "\n".join(
            f"[{s['n']}] {s.get('title') or s.get('doc_id')}"
            + (f", с. {', '.join(map(str, s['pages'][:3]))}" if s.get("pages") else "")
            + (" — словарь" if s.get("kind") == "словарь" else "")
            for s in shown
        ),
        "dictionary": ", ".join(done.get("dictionary") or []),
        "mode": done.get("mode", ""),
        "seconds": round(time.monotonic() - started, 1),
    }


def write(rows: list[dict[str, Any]], out: Path, title: str) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Ответы"
    head = ["№", "Вид", "Вопрос", "Словарь", "Ответ бота", "Источники", "Термины словаря",
            "Реакция на ложную посылку", "Время, с", "Оценка (верно / частично / неверно)",
            "Комментарий геолога"]
    ws.append(head)
    for r in rows:
        ws.append([r["id"], r["kind"], r["question"], r["dict"], r["answer"], r["sources"],
                   r["dictionary"], r.get("reaction", ""), r["seconds"], None, None])
    widths = (6, 10, 40, 9, 90, 50, 22, 18, 9, 18, 30)
    for col, w in zip("ABCDEFGHIJK", widths, strict=True):
        ws.column_dimensions[col].width = w
    fill = PatternFill("solid", fgColor="FFF2CC")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in range(1, ws.max_row + 1):
        ws.cell(r, 10).fill = fill
        ws.cell(r, 11).fill = fill
    ws.freeze_panes = "D2"
    info = wb.create_sheet("О прогоне")
    for line in (title, f"Вопросов: {len({r['id'] for r in rows})}, ответов: {len(rows)}",
                 f"Дата: {datetime.now():%d.%m.%Y %H:%M}",
                 "Колонку «Оценка» и «Комментарий» заполняет человек. «Реакция» — подсказка по "
                 "словам ответа, не оценка."):
        info.append([line])
    info.column_dimensions["A"].width = 110
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    from ..index.db import connect
    from ..index.embed import build_embedder
    from .settings import Settings

    p = argparse.ArgumentParser(prog="python georag.py report")
    p.add_argument("questions", help="YAML со списком вопросов")
    p.add_argument("--compare-dict", action="store_true", help="каждый вопрос со словарём и без")
    p.add_argument("--out", default="", help="куда записать .xlsx (по умолчанию .local/reports)")
    args = p.parse_args(argv)
    path = Path(args.questions)
    questions = load_questions(path)
    if not questions:
        print(f"В {path} нет вопросов.", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else REPORTS / f"{path.stem}-{datetime.now():%Y%m%d-%H%M}.xlsx"
    modes = [True, False] if args.compare_dict else [True]
    embedder = build_embedder()
    base = Settings()
    rows = []
    with connect() as conn:
        for q in questions:
            for with_dict in modes:
                settings = replace(base, dictionary=with_dict)
                got = ask(conn, embedder, q["question"], settings)
                row = {**q, **got, "dict": "да" if with_dict else "нет"}
                if q["kind"].startswith("лож"):
                    row["reaction"] = reaction(got["answer"])
                rows.append(row)
                print(f"{q['id']:>3} {'словарь' if with_dict else 'без словаря':<12} "
                      f"{got['seconds']:>6} с", flush=True)
    print(f"Таблица: {write(rows, out, f'Набор: {path.name}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
