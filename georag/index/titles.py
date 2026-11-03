"""Названия, авторы, год и территория документов — из таблицы отчётов (Excel).

Документы закрытой базы называются кодами файлов («06a6f50a-ae30-…»): модель писала «все
фрагменты взяты из статьи «06a6f50a…»» и не видела, о какой площади отчёт (встреча
30.10.2026). Таблица отчётов (.xlsx) знает названия. Строка таблицы сопоставляется с
документом по любому совпавшему коду: doc_id, имя файла без расширения, код отчёта в поле
«журнал» или источнике. Колонки узнаются по заголовкам (название, автор, год, территория).

    python georag.py titles таблица.xlsx --dry-run   заголовки и сколько совпало, без записи
    python georag.py titles таблица.xlsx             записать в базу

Печатаются только заголовки колонок и числа — не содержимое таблицы и документов.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Заголовок колонки → поле. Проверяется по началу слов заголовка, без регистра.
COLUMNS = {
    "title": ("назв", "заголов", "тема", "title", "наименов"),
    "authors": ("автор", "исполнит", "author", "состав"),
    "year": ("год", "year", "дата"),
    "place": (
        "территор",
        "площад",
        "район",
        "регион",
        "объект",
        "месторожд",
        "участ",
        "area",
        "region",
        "лицензи",
    ),
}
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_YEAR = re.compile(r"(19|20)\d\d")


@dataclass
class Row:
    keys: set[str]
    values: dict[str, str] = field(default_factory=dict)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split())


def columns(header: list[Any]) -> dict[str, int]:
    """Номера колонок полей по заголовкам; первая подходящая колонка — на поле."""
    out: dict[str, int] = {}
    for i, cell in enumerate(header):
        name = _norm(cell).lower()
        if not name:
            continue
        for key, starts in COLUMNS.items():
            if key not in out and any(w.startswith(starts) for w in re.findall(r"\w+", name)):
                out[key] = i
                break
    return out


def row_keys(cells: list[Any]) -> set[str]:
    """Коды строки: все UUID в ячейках и имена файлов без расширения."""
    keys: set[str] = set()
    for cell in cells:
        text = _norm(cell)
        keys.update(m.group().lower() for m in _ID.finditer(text))
        for name in re.findall(r"[\w.-]+\.(?:docx?|pdf|xlsx?|rtf|txt)", text, re.IGNORECASE):
            keys.add(Path(name).stem.lower())
    return keys


def read_table(path: Path) -> tuple[list[str], dict[str, int], list[Row]]:
    """Заголовки, колонки полей и строки таблицы (первый лист со строкой заголовков)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = [list(ws.iter_rows(values_only=True)) for ws in wb.worksheets]
    finally:
        wb.close()  # в режиме read_only файл иначе остаётся открытым (Windows не даст удалить)
    for rows in sheets:
        # Строка заголовков — первая, где узнано название или хотя бы два поля.
        for h, header in enumerate(rows[:20]):
            cols = columns(list(header))
            if "title" in cols or len(cols) >= 2:
                out = []
                for cells in rows[h + 1 :]:
                    cells = list(cells)
                    keys = row_keys(cells)
                    if not keys:
                        continue
                    values = {k: _norm(cells[i]) for k, i in cols.items() if i < len(cells)}
                    out.append(Row(keys, {k: v for k, v in values.items() if v}))
                return [_norm(c) for c in header], cols, out
    return [], {}, []


def document_keys(conn: Any) -> dict[str, set[str]]:
    """Коды каждого документа: doc_id, имя файла (если есть такая колонка), UUID в журнале,
    источнике и адресе."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
        )
        have = {c for (c,) in cur.fetchall()}
        extra = [c for c in ("file_name", "report", "journal", "source", "url") if c in have]
        cur.execute(f"SELECT doc_id, {', '.join(extra) or 'NULL'} FROM documents")
        rows = cur.fetchall()
    out: dict[str, set[str]] = {}
    for doc_id, *rest in rows:
        keys = {str(doc_id).lower(), Path(str(doc_id)).stem.lower()}
        for value in rest:
            text = _norm(value)
            keys.update(m.group().lower() for m in _ID.finditer(text))
            if text and "." in text and "/" not in text:
                keys.add(Path(text).stem.lower())
        out[str(doc_id)] = keys
    return out


def match(docs: dict[str, set[str]], rows: list[Row]) -> dict[str, Row]:
    """Документ → строка таблицы: по совпавшему коду; самый частный код — первым (код
    документа и файла раньше кода отчёта, общего для нескольких документов)."""
    by_key: dict[str, list[Row]] = {}
    for row in rows:
        for key in row.keys:
            by_key.setdefault(key, []).append(row)
    out: dict[str, Row] = {}
    for doc_id, keys in docs.items():
        candidates = [(len(by_key[k]), k) for k in keys if k in by_key]
        if candidates:
            out[doc_id] = by_key[min(candidates)[1]][0]
    return out


def title_of(values: dict[str, str]) -> str:
    """Название с территорией в скобках, если её нет в самом названии."""
    title = values.get("title", "")
    place = values.get("place", "")
    if place and place.lower() not in title.lower():
        title = f"{title} ({place})" if title else place
    return title[:500]


def apply(conn: Any, found: dict[str, Row]) -> int:
    changed = 0
    with conn.cursor() as cur:
        for doc_id, row in found.items():
            title = title_of(row.values)
            authors = [
                a.strip() for a in re.split(r"[;,]\s*", row.values.get("authors", "")) if a.strip()
            ]
            year_m = _YEAR.search(row.values.get("year", ""))
            sets, params = [], []
            if title:
                sets.append("title = %s")
                params.append(title)
            if authors:
                sets.append("authors = %s")
                params.append(authors[:12])
            if year_m:
                sets.append("year = %s")
                params.append(int(year_m.group()))
            if sets:
                cur.execute(
                    f"UPDATE documents SET {', '.join(sets)} WHERE doc_id = %s", (*params, doc_id)
                )
                changed += cur.rowcount
    conn.commit()
    return changed


def main(argv: list[str] | None = None) -> int:
    from .db import connect

    p = argparse.ArgumentParser(prog="python georag.py titles")
    p.add_argument("table", help="таблица отчётов .xlsx")
    p.add_argument("--dry-run", action="store_true", help="только показать, что совпало")
    args = p.parse_args(argv)
    path = Path(args.table)
    if not path.exists():
        print(f"Нет файла: {path}", file=sys.stderr)
        return 1
    header, cols, rows = read_table(path)
    if not rows:
        print(
            "В таблице не нашлось строки заголовков с названием и строк с кодами "
            "(UUID или именем файла).",
            file=sys.stderr,
        )
        return 1
    print("Заголовки таблицы: " + " | ".join(h or "—" for h in header))
    print("Узнаны колонки: " + ", ".join(f"{k} ← «{header[i]}»" for k, i in cols.items()))
    with connect() as conn:
        docs = document_keys(conn)
        found = match(docs, rows)
        print(f"Строк с кодами: {len(rows)}; документов в базе: {len(docs)}; совпало: {len(found)}")
        missing = len(docs) - len(found)
        if missing:
            print(f"Без строки в таблице: {missing} документов — их названия не меняются.")
        if args.dry_run:
            return 0
        print(f"Записано: {apply(conn, found)} документов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
