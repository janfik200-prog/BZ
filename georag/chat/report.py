"""Отчёт по набору вопросов: ответы чата → таблица Excel, колонка оценки — пустая.

Для встреч с геологами: свой список вопросов (config/questions-*.yaml), ответ бота, его
источники и время; колонку «Оценка» заполняет человек. Лист «Фрагменты» — каждый
источник ответа отдельной строкой: название статьи или отчёта, имя файла, страницы,
раздел и сам текст фрагмента, — чтобы проверить ответ, не открывая базу. --compare-dict — каждый вопрос
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
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import LOCAL

REPORTS = LOCAL / "reports"
CELL_CHARS = 30000  # в ячейке Excel не больше 32 767 знаков

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
            out.append(
                {
                    "id": str(item.get("id") or i),
                    "question": question,
                    "kind": str(item.get("вид") or item.get("kind") or "обычный"),
                    "note": str(item.get("заметка") or item.get("note") or ""),
                }
            )
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
    files = file_names(conn, [s.get("doc_id") or "" for s in sources])
    return {
        "answer": text.strip(),
        "sources": "\n".join(source_line(s, files) for s in shown),
        "fragments": [
            {
                "n": s["n"],
                "used": s["n"] in used,
                "kind": s.get("kind") or "",
                "title": s.get("title") or s.get("doc_id") or "",
                "file": files.get(s.get("doc_id") or "", ""),
                "year": s.get("year") or "",
                "pages": ", ".join(map(str, s.get("pages") or [])),
                "headings": " / ".join(s.get("headings") or []),
                "text": s.get("text") or s.get("note") or "",
            }
            for s in sources
        ],
        "dictionary": ", ".join(done.get("dictionary") or []),
        "mode": done.get("mode", ""),
        "seconds": round(time.monotonic() - started, 1),
    }


def file_names(conn: Any, doc_ids: list[str]) -> dict[str, str]:
    """Имя файла каждой статьи или отчёта (documents.file_name): по нему документ
    находится в папке output. У словаря и сводок файла нет."""
    ids = [d for d in dict.fromkeys(doc_ids) if d and ":" not in d]
    if not ids:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, coalesce(file_name, '') FROM documents WHERE doc_id = ANY(%s)",
                (ids,),
            )
            return {str(d): str(f) for d, f in cur.fetchall()}
    except Exception:  # noqa: BLE001 — без имён файлов отчёт всё равно нужен
        conn.rollback()
        return {}


def source_line(s: dict[str, Any], files: dict[str, str]) -> str:
    """«[2] Название — файл x.docx, с. 12» для колонки «Источники»."""
    line = f"[{s['n']}] {s.get('title') or s.get('doc_id')}"
    if s.get("kind") == "словарь":
        line += " — словарь"
    elif files.get(s.get("doc_id") or ""):
        line += f" — файл {files[s['doc_id']]}"
    if s.get("pages"):
        line += f", с. {', '.join(map(str, s['pages'][:3]))}"
    return line


def _cell(value: Any) -> Any:
    """Строка без знаков, которых Excel не принимает, и не длиннее ячейки."""
    if not isinstance(value, str):
        return value
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    value = ILLEGAL_CHARACTERS_RE.sub("", value)
    return value if len(value) <= CELL_CHARS else value[:CELL_CHARS] + " …"


def write(rows: list[dict[str, Any]], out: Path, title: str) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Ответы"
    head = [
        "№",
        "Вид",
        "Вопрос",
        "Что считать правильным",
        "Словарь",
        "Ответ бота",
        "Источники",
        "Термины словаря",
        "Реакция на ложную посылку",
        "Время, с",
        "Оценка (верно / частично / неверно)",
        "Комментарий геолога",
    ]
    ws.append(head)
    for r in rows:
        ws.append(
            [
                _cell(v)
                for v in [
                r["id"],
                r["kind"],
                r["question"],
                r.get("note", ""),
                r["dict"],
                r["answer"],
                r["sources"],
                r["dictionary"],
                r.get("reaction", ""),
                r["seconds"],
                None,
                None,
            ]]
        )
    widths = (6, 10, 40, 30, 9, 90, 55, 22, 18, 9, 18, 30)
    for col, w in zip("ABCDEFGHIJKL", widths, strict=True):
        ws.column_dimensions[col].width = w
    fill = PatternFill("solid", fgColor="FFF2CC")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for row_no in range(1, ws.max_row + 1):
        ws.cell(row_no, 11).fill = fill
        ws.cell(row_no, 12).fill = fill
    ws.freeze_panes = "D2"

    # Каждый источник ответа — строкой: найти документ по имени файла и прочитать сам
    # фрагмент. «Использован» — бот сослался на него в ответе.
    frag = wb.create_sheet("Фрагменты")
    frag.append(
        [
            "№ вопроса",
            "Словарь",
            "[n]",
            "Использован в ответе",
            "Название статьи / отчёта",
            "Файл",
            "Год",
            "Страницы",
            "Раздел",
            "Текст фрагмента",
        ]
    )
    for r in rows:
        for f in r.get("fragments") or []:
            frag.append(
                [
                    _cell(v)
                    for v in (
                        r["id"],
                        r["dict"],
                        f"[{f['n']}]",
                        "да" if f["used"] else "нет",
                        f["title"] + (" (словарь)" if f["kind"] == "словарь" else ""),
                        f["file"],
                        f["year"],
                        f["pages"],
                        f["headings"],
                        f["text"],
                    )
                ]
            )
    for col, w in zip("ABCDEFGHIJ", (9, 8, 6, 11, 45, 30, 7, 10, 30, 110), strict=True):
        frag.column_dimensions[col].width = w
    for c in frag[1]:
        c.font = Font(bold=True)
    for row in frag.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    frag.freeze_panes = "E2"
    frag.auto_filter.ref = frag.dimensions
    info = wb.create_sheet("О прогоне")
    for line in (
        title,
        f"Вопросов: {len({r['id'] for r in rows})}, ответов: {len(rows)}",
        f"Дата: {datetime.now():%d.%m.%Y %H:%M}",
        "Колонку «Оценка» и «Комментарий» заполняет человек. «Реакция» — подсказка по "
        "словам ответа, не оценка.",
        "Лист «Фрагменты» — все источники каждого ответа: название, файл (в папке output), "
        "страницы, раздел и текст фрагмента. Фильтр по «№ вопроса» показывает источники "
        "одного ответа.",
    ):
        info.append([line])
    info.column_dimensions["A"].width = 110
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


@contextmanager
def _dictionary_env(on: bool) -> Iterator[None]:
    """GEORAG_DICT на время одного ответа: 1 — со словарём, 0 — без; прежнее — обратно."""
    before = os.environ.get("GEORAG_DICT")
    os.environ["GEORAG_DICT"] = "1" if on else "0"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("GEORAG_DICT", None)
        else:
            os.environ["GEORAG_DICT"] = before


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
                # Словарь выключается целиком: и определения (settings.dictionary), и
                # его написания в поиске по словам — их берёт search.dictionary_variants
                # по GEORAG_DICT, а не по настройкам ответа.
                with _dictionary_env(with_dict):
                    got = ask(conn, embedder, q["question"], settings)
                row = {**q, **got, "dict": "да" if with_dict else "нет"}
                if q["kind"].startswith("лож"):
                    row["reaction"] = reaction(got["answer"])
                rows.append(row)
                print(
                    f"{q['id']:>3} {'словарь' if with_dict else 'без словаря':<12} "
                    f"{got['seconds']:>6} с",
                    flush=True,
                )
    print(f"Таблица: {write(rows, out, f'Набор: {path.name}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
