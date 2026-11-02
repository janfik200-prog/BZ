"""Словари: справочные (Геологический словарь ВСЕГЕИ и др.) и рабочие словари проекта.

Справочный словарь разбирается из PDF в статьи «термин — английский — определение» и
лежит в двух местах:

* папка dictionaries/<имя>/ — entries.jsonl (статья в строке) и meta.json: из неё словарь
  переносится на другую машину и загружается в базу заново;
* база (контейнер georag-db) — таблицы dictionaries и dict_entries: поиск по термину и
  по тексту определений, вкладка «Словари» на странице.

Рабочие словари проекта (config/geo-terms.txt, synonyms.yaml, term-translations.yaml)
показываются на той же вкладке как есть, из файлов.

    python georag.py dict import geol_dict2017_v1.pdf --id geol_dict2017_v1 --title "…"
    python georag.py dict load            папка dictionaries → база (все словари)
    python georag.py dict list            какие словари есть
    python georag.py dict find "адуляр"   статьи словарей по термину
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ROOT
from .text import name_key

FOLDER = ROOT / "dictionaries"
CONFIG = ROOT / "config"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dictionaries (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    entries INT NOT NULL DEFAULT 0,
    loaded_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS dict_entries (
    id BIGSERIAL PRIMARY KEY,
    dict_id TEXT NOT NULL REFERENCES dictionaries(id) ON DELETE CASCADE,
    ord INT NOT NULL,
    term TEXT NOT NULL,
    term_key TEXT NOT NULL,
    en TEXT,
    etym TEXT,
    definition TEXT NOT NULL,
    page INT,
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', coalesce(term, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(en, '')), 'A') ||
        setweight(to_tsvector('russian', coalesce(definition, '')), 'B')
    ) STORED
);
CREATE INDEX IF NOT EXISTS dict_entries_dict ON dict_entries (dict_id, ord);
CREATE INDEX IF NOT EXISTS dict_entries_key ON dict_entries (term_key);
CREATE INDEX IF NOT EXISTS dict_entries_tsv ON dict_entries USING gin (tsv);
"""


@dataclass
class Entry:
    term: str
    en: str
    etym: str
    definition: str
    page: int


# --- разбор PDF ------------------------------------------------------------------------ #

# Колонтитулы страницы: сверху — первые буквы статей («АЗУ»), снизу — номер страницы.
HEAD_Y = 62
FOOT_Y = 755
# Новая статья начинается жирным словом у левого края колонки; продолжение статьи — с
# отступом (≈7 pt), продолжение английского названия в скобках — тоже жирное, но с отступом.
INDENT = 4.0
_DASH = re.compile(r"\s+[–—-]\s+")


def _bold(span: dict[str, Any]) -> bool:
    return bool(span["flags"] & 16) or "Bold" in span["font"]


def _lines(page: Any) -> list[tuple[float, float, list[dict[str, Any]]]]:
    """Строки страницы по колонкам: (левый край колонки, x строки, спаны), сверху вниз."""
    # Правая колонка на нечётных страницах начинается у x≈289 — чуть левее середины (290):
    # граница колонок — на 45 % ширины, между левыми краями колонок (45–62 и 289–313).
    mid = page.rect.width * 0.45
    rows = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0 = spans[0]["bbox"][0], line["bbox"][1]
            if y0 < HEAD_Y or y0 > FOOT_Y:
                continue
            rows.append((int(x0 >= mid), y0, x0, spans))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    left = {col: min(r[2] for r in rows if r[0] == col) for col in {r[0] for r in rows}}
    return [(left[c], x, spans) for c, _, x, spans in rows]


def _text(spans: list[dict[str, Any]]) -> str:
    return "".join(s["text"] for s in spans)


def _join(parts: list[str]) -> str:
    """Строки статьи — в одну: перенос «гидрогеоло-|гическая» склеивается без дефиса."""
    out = ""
    for part in parts:
        part = part.strip().replace(" ", " ").replace(" ", " ")
        if out.endswith("-") and part[:1].islower() and not out.endswith(" -"):
            out = out[:-1] + part
        elif out.endswith("‑"):  # неразрывный дефис — часть слова («м‑лов»)
            out += part
        else:
            out = f"{out} {part}" if out else part
    # Мягкий перенос внутри строки на части страниц — знаком \x10 или \xad («издаю\x10 щая»);
    # знаки формул, которых в шрифте PDF нет (U+FFFD), и прочие служебные — вон.
    out = re.sub(r"[\x10\xad]\s*", "", out)
    out = re.sub(r"[\x00-\x08\x0b-\x1f\ufffd]", "", out)
    return re.sub(r"\s+", " ", out).strip()


def _entry(raw: str, page: int) -> Entry | None:
    """«Адуляр [по горам Адула, Швейцария; adularia] – прозрач. …» → поля статьи."""
    # Тире внутри скобок («[от фр. décollement – отклеивание; …]») — не граница определения.
    b, e = raw.find("["), raw.find("]")
    m = _DASH.search(raw, e + 1 if 0 <= b < e and b < (_DASH.search(raw) or re.search("$", raw)).start() else 0)
    if not m:
        return None
    head, definition = raw[: m.start()].strip(), raw[m.end() :].strip()
    en = etym = ""
    b = head.find("[")
    if b >= 0:
        inside = head[b + 1 :].rstrip("]").strip()
        head = head[:b].strip()
        if ";" in inside:
            etym, en = (x.strip() for x in inside.rsplit(";", 1))
        elif re.search("[а-яё]", inside, re.I):
            etym = inside
        else:
            en = inside
    if not head or not definition:
        return None
    return Entry(head, en.replace("ﬁ ", "fi").replace("ﬁ", "fi"), etym, definition, page)


def parse_pdf(path: Path, first: int = 1, last: int | None = None) -> list[Entry]:
    """Статьи словаря из PDF (страницы first…last, с 1)."""
    import pymupdf

    doc = pymupdf.open(path)
    entries: list[Entry] = []
    current: list[str] = []
    start_page = first

    def close() -> None:
        if current:
            e = _entry(_join(current), start_page)
            if e is not None:
                entries.append(e)

    for number in range(first - 1, last or len(doc)):
        for left, x, spans in _lines(doc[number]):
            new = _bold(spans[0]) and x - left < INDENT and spans[0]["text"].strip()[:1].isalnum()
            if new:
                close()
                current, start_page = [], number + 1
            if new or current:
                current.append(_text(spans))
    close()
    return entries


def detect_range(path: Path) -> tuple[int, int]:
    """Страницы со статьями: от первой страницы, где строка «Термин – …» начата жирным, до
    последней такой. Вступление, сокращения и выходные данные отсекаются."""
    import pymupdf

    doc = pymupdf.open(path)
    pages = []
    for i in range(len(doc)):
        heads = dots = 0
        for left, x, spans in _lines(doc[i]):
            m = _DASH.search(_text(spans))
            if _bold(spans[0]) and x - left < INDENT and m:
                heads += 1
                dots += _text(spans)[: m.start()].rstrip().endswith(".")
        # Список сокращений («агр. – агрегат») — не статьи: там термины кончаются точкой.
        if heads >= 2 and dots < heads / 2:
            pages.append(i + 1)
    if not pages:
        return 1, len(doc)
    return pages[0], pages[-1]


# --- папка ----------------------------------------------------------------------------- #


def save(dict_id: str, title: str, source: str, entries: list[Entry]) -> Path:
    folder = FOLDER / dict_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "entries.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    meta = {"id": dict_id, "title": title, "source": source, "entries": len(entries),
            "parsed_at": datetime.now().isoformat(timespec="seconds")}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return folder


def folders() -> list[dict[str, Any]]:
    """Словари в папке dictionaries: meta.json каждого."""
    out = []
    for meta in sorted(FOLDER.glob("*/meta.json")):
        try:
            out.append(json.loads(meta.read_text("utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def read(dict_id: str) -> list[Entry]:
    path = FOLDER / dict_id / "entries.jsonl"
    with path.open(encoding="utf-8") as f:
        return [Entry(**json.loads(line)) for line in f if line.strip()]


# --- база ------------------------------------------------------------------------------ #


def init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def load(conn: Any, dict_id: str) -> int:
    """Словарь из папки → база (прежние статьи этого словаря заменяются)."""
    meta = json.loads((FOLDER / dict_id / "meta.json").read_text("utf-8"))
    entries = read(dict_id)
    init(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dictionaries (id, title, source, entries, loaded_at) "
            "VALUES (%s, %s, %s, %s, now()) ON CONFLICT (id) DO UPDATE SET title = "
            "EXCLUDED.title, source = EXCLUDED.source, entries = EXCLUDED.entries, "
            "loaded_at = now()",
            (dict_id, meta["title"], meta.get("source"), len(entries)),
        )
        cur.execute("DELETE FROM dict_entries WHERE dict_id = %s", (dict_id,))
        with cur.copy(
            "COPY dict_entries (dict_id, ord, term, term_key, en, etym, definition, page) "
            "FROM STDIN"
        ) as copy:
            for i, e in enumerate(entries):
                copy.write_row((dict_id, i, e.term, name_key(e.term), e.en, e.etym,
                                e.definition, e.page))
    conn.commit()
    return len(entries)


def listing(conn: Any) -> list[dict[str, Any]]:
    """Справочные словари из базы."""
    init(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, source, entries FROM dictionaries ORDER BY title")
        return [{"id": i, "title": t, "source": s, "entries": n, "kind": "reference"}
                for i, t, s, n in cur.fetchall()]


def find(conn: Any, dict_id: str | None, query: str, letter: str = "", limit: int = 50,
         offset: int = 0) -> dict[str, Any]:
    """Статьи словаря: по началу термина, затем по словам в термине и определении.
    Пустой запрос — по алфавиту (с буквы letter)."""
    init(conn)
    where = "dict_id = %(d)s" if dict_id else "TRUE"
    params: dict[str, Any] = {"d": dict_id, "lim": limit, "off": offset}
    fields = "e.dict_id, e.term, e.en, e.etym, e.definition, e.page"
    q = query.strip()
    with conn.cursor() as cur:
        if not q:
            params["l"] = f"{letter}%"
            sql_where = f"{where} AND term ILIKE %(l)s"
            cur.execute(f"SELECT count(*) FROM dict_entries WHERE {sql_where}", params)
            total = cur.fetchone()[0]
            cur.execute(f"SELECT {fields} FROM dict_entries e WHERE {sql_where} "
                        "ORDER BY ord LIMIT %(lim)s OFFSET %(off)s", params)
        else:
            params.update({"p": f"{q}%", "q": q})
            match = ("(term ILIKE %(p)s OR en ILIKE %(p)s OR tsv @@ "
                     "websearch_to_tsquery('russian', %(q)s) OR tsv @@ "
                     "websearch_to_tsquery('english', %(q)s))")
            cur.execute(f"SELECT count(*) FROM dict_entries WHERE {where} AND {match}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT {fields} FROM dict_entries e WHERE {where} AND {match} "
                "ORDER BY (term ILIKE %(q)s) DESC, (term ILIKE %(p)s OR en ILIKE %(p)s) DESC, "
                "ts_rank(tsv, websearch_to_tsquery('russian', %(q)s)) DESC, ord "
                "LIMIT %(lim)s OFFSET %(off)s",
                params,
            )
        rows = [dict(zip(("dict_id", "term", "en", "etym", "definition", "page"), r))
                for r in cur.fetchall()]
    return {"total": total, "entries": rows}


# --- рабочие словари проекта (из файлов config) -------------------------------------------- #

WORKING = {
    "geo-terms": ("Термины первого этапа графа", "geo-terms.txt"),
    "synonyms": ("Синонимы названий", "synonyms.yaml"),
    "term-translations": ("Переводы терминов для поиска", "term-translations.yaml"),
}


def working_listing() -> list[dict[str, Any]]:
    out = []
    for key, (title, file) in WORKING.items():
        path = CONFIG / file
        if path.exists():
            out.append({"id": key, "title": title, "source": f"config/{file}",
                        "entries": len(working_entries(key)), "kind": "working"})
    return out


def working_entries(key: str) -> list[dict[str, Any]]:
    """Строки рабочего словаря в том же виде, что статьи справочного."""
    import yaml

    path = CONFIG / WORKING[key][1]
    if not path.exists():
        return []
    if key == "geo-terms":
        terms = [line.strip() for line in path.read_text("utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        return [{"term": t, "en": "", "etym": "", "definition": "", "page": None} for t in terms]
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if key == "synonyms":
        rows = []
        for section in ("имена", "связи"):
            for name, spellings in (data.get(section) or {}).items():
                rows.append({"term": str(name), "en": "", "etym": section,
                             "definition": ", ".join(str(s) for s in spellings or []),
                             "page": None})
        return rows
    return [{"term": str(k), "en": ", ".join(str(x) for x in v or []), "etym": "",
             "definition": "", "page": None} for k, v in data.items()]


def working_find(key: str, query: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    rows = working_entries(key)
    q = query.strip().lower()
    if q:
        rows = [r for r in rows if q in " ".join(str(v or "") for v in r.values()).lower()]
    return {"total": len(rows), "entries": rows[offset : offset + limit]}


# --- команда --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    from .index import db

    p = argparse.ArgumentParser(prog="python georag.py dict")
    sub = p.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="разобрать PDF словаря в папку и загрузить в базу")
    imp.add_argument("pdf")
    imp.add_argument("--id", default="")
    imp.add_argument("--title", default="")
    imp.add_argument("--pages", default="", help="страницы статей, например 22-430")
    imp.add_argument("--no-db", action="store_true", help="только в папку")
    ld = sub.add_parser("load", help="словари из папки dictionaries → база")
    ld.add_argument("id", nargs="?", default="")
    sub.add_parser("list")
    fd = sub.add_parser("find")
    fd.add_argument("query")
    fd.add_argument("--id", default=None)
    args = p.parse_args(argv)

    if args.cmd == "import":
        pdf = Path(args.pdf)
        dict_id = args.id or re.sub(r"[^\w-]+", "_", pdf.stem).strip("_").lower()
        if args.pages:
            first, last = (int(x) for x in args.pages.split("-"))
        else:
            first, last = detect_range(pdf)
        print(f"Разбираю {pdf.name}, страницы {first}–{last}…")
        entries = parse_pdf(pdf, first, last)
        folder = save(dict_id, args.title or pdf.stem, pdf.name, entries)
        print(f"Статей: {len(entries)} → {folder}")
        if not args.no_db:
            with db.connect() as conn:
                print(f"В базе: {load(conn, dict_id)} статей словаря {dict_id}")
        return 0
    if args.cmd == "load":
        ids = [args.id] if args.id else [m["id"] for m in folders()]
        with db.connect() as conn:
            for dict_id in ids:
                print(f"{dict_id}: {load(conn, dict_id)} статей")
        return 0
    if args.cmd == "list":
        with db.connect() as conn:
            for d in listing(conn) + working_listing():
                print(f"{d['id']:<24} {d['entries']:>7}  {d['title']}  ({d['source']})")
        return 0
    with db.connect() as conn:
        for e in find(conn, args.id, args.query, limit=10)["entries"]:
            en = f" [{e['en']}]" if e["en"] else ""
            print(f"{e['term']}{en} — {e['definition'][:300]}  (с. {e['page']})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
