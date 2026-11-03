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
ALTER TABLE dictionaries ADD COLUMN IF NOT EXISTS abbreviations JSONB;
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
    m = _DASH.search(
        raw, e + 1 if 0 <= b < e and b < (_DASH.search(raw) or re.search("$", raw)).start() else 0
    )
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


def abbreviation_pages(path: Path, first_entry_page: int) -> tuple[int, int]:
    """Страницы списка сокращений: перед статьями, где у большинства строк «X – Y» левая
    часть кончается точкой («агр. – агрегат»). Нет таких — пустой диапазон."""
    import pymupdf

    doc = pymupdf.open(path)
    pages = []
    for i in range(first_entry_page - 1):
        heads = dots = 0
        for left, x, spans in _lines(doc[i]):
            m = _DASH.search(_text(spans))
            if _bold(spans[0]) and x - left < INDENT and m:
                heads += 1
                dots += _text(spans)[: m.start()].rstrip().endswith(".")
        if heads >= 5 and dots >= heads / 2:
            pages.append(i + 1)
    return (pages[0], pages[-1]) if pages else (1, 0)


# --- папка ----------------------------------------------------------------------------- #


def save(
    dict_id: str,
    title: str,
    source: str,
    entries: list[Entry],
    pairs: list[tuple[str, str]] | None = None,
) -> Path:
    folder = FOLDER / dict_id
    folder.mkdir(parents=True, exist_ok=True)
    if pairs is not None:
        (folder / "abbreviations.json").write_text(
            json.dumps(pairs, ensure_ascii=False, indent=0), "utf-8"
        )
    with (folder / "entries.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    meta = {
        "id": dict_id,
        "title": title,
        "source": source,
        "entries": len(entries),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }
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
    abbr = FOLDER / dict_id / "abbreviations.json"
    pairs = json.loads(abbr.read_text("utf-8")) if abbr.exists() else []
    init(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dictionaries (id, title, source, entries, loaded_at, abbreviations) "
            "VALUES (%s, %s, %s, %s, now(), %s) ON CONFLICT (id) DO UPDATE SET title = "
            "EXCLUDED.title, source = EXCLUDED.source, entries = EXCLUDED.entries, "
            "loaded_at = now(), abbreviations = EXCLUDED.abbreviations",
            (
                dict_id,
                meta["title"],
                meta.get("source"),
                len(entries),
                json.dumps(pairs, ensure_ascii=False),
            ),
        )
        cur.execute("DELETE FROM dict_entries WHERE dict_id = %s", (dict_id,))
        with cur.copy(
            "COPY dict_entries (dict_id, ord, term, term_key, en, etym, definition, page) "
            "FROM STDIN"
        ) as copy:
            for i, e in enumerate(entries):
                copy.write_row(
                    (dict_id, i, e.term, name_key(e.term), e.en, e.etym, e.definition, e.page)
                )
    conn.commit()
    return len(entries)


def listing(conn: Any) -> list[dict[str, Any]]:
    """Справочные словари из базы."""
    init(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, source, entries FROM dictionaries ORDER BY title")
        return [
            {"id": i, "title": t, "source": s, "entries": n, "kind": "reference"}
            for i, t, s, n in cur.fetchall()
        ]


def find(
    conn: Any, dict_id: str | None, query: str, letter: str = "", limit: int = 50, offset: int = 0
) -> dict[str, Any]:
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
            cur.execute(
                f"SELECT {fields} FROM dict_entries e WHERE {sql_where} "
                "ORDER BY ord LIMIT %(lim)s OFFSET %(off)s",
                params,
            )
        else:
            params.update({"p": f"{q}%", "q": q})
            match = (
                "(term ILIKE %(p)s OR en ILIKE %(p)s OR tsv @@ "
                "websearch_to_tsquery('russian', %(q)s) OR tsv @@ "
                "websearch_to_tsquery('english', %(q)s))"
            )
            cur.execute(f"SELECT count(*) FROM dict_entries WHERE {where} AND {match}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT {fields} FROM dict_entries e WHERE {where} AND {match} "
                "ORDER BY (term ILIKE %(q)s) DESC, (term ILIKE %(p)s OR en ILIKE %(p)s) DESC, "
                "ts_rank(tsv, websearch_to_tsquery('russian', %(q)s)) DESC, ord "
                "LIMIT %(lim)s OFFSET %(off)s",
                params,
            )
        rows = [
            dict(zip(("dict_id", "term", "en", "etym", "definition", "page"), r))
            for r in cur.fetchall()
        ]
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
            out.append(
                {
                    "id": key,
                    "title": title,
                    "source": f"config/{file}",
                    "entries": len(working_entries(key)),
                    "kind": "working",
                }
            )
    return out


def working_entries(key: str) -> list[dict[str, Any]]:
    """Строки рабочего словаря в том же виде, что статьи справочного."""
    import yaml

    path = CONFIG / WORKING[key][1]
    if not path.exists():
        return []
    if key == "geo-terms":
        terms = [
            line.strip()
            for line in path.read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return [{"term": t, "en": "", "etym": "", "definition": "", "page": None} for t in terms]
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if key == "synonyms":
        rows = []
        for section in ("имена", "связи"):
            for name, spellings in (data.get(section) or {}).items():
                rows.append(
                    {
                        "term": str(name),
                        "en": "",
                        "etym": section,
                        "definition": ", ".join(str(s) for s in spellings or []),
                        "page": None,
                    }
                )
        return rows
    return [
        {
            "term": str(k),
            "en": ", ".join(str(x) for x in v or []),
            "etym": "",
            "definition": "",
            "page": None,
        }
        for k, v in data.items()
    ]


def working_find(key: str, query: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    rows = working_entries(key)
    q = query.strip().lower()
    if q:
        rows = [r for r in rows if q in " ".join(str(v or "") for v in r.values()).lower()]
    return {"total": len(rows), "entries": rows[offset : offset + limit]}


# --- сокращения ------------------------------------------------------------------------ #

# Сокращения словаря («м-л», «Тв.», «к-лы») — из его списка сокращений (у Геологического
# словаря — с. 20–21 PDF). Модель читала бы «Тв. 6» и «м-ниях» как придётся, поэтому перед
# тем как отдать определение модели, сокращения раскрываются (expand).
_NOTE = re.compile(r"\s*\([^)]*\)")


def abbreviation_pairs(entries: list[Entry]) -> list[tuple[str, str]]:
    """Строки списка сокращений → пары (сокращение, полное слово).

    «к-л, к-лы – кристалл, кристаллы» → две пары; «тв. – твердость (при цифре)» → одна, без
    пояснения в скобках. Однобуквенные без точки и дефиса («T, t») пропускаются: совпали бы
    с обозначениями в формулах."""
    pairs: list[tuple[str, str]] = []
    for e in entries:
        full = _NOTE.sub("", e.definition.split(" Примечания")[0]).strip()
        shorts = [s.strip() for s in e.term.split(",") if s.strip()]
        fulls = [f.strip() for f in full.split(",") if f.strip()]
        for i, short in enumerate(shorts):
            if len(short) <= 2 and not re.search(r"[.\-]", short):
                continue
            if re.fullmatch(r"\w\.", short):  # «г.», «п.» — как инициалы; только в сочетаниях
                continue
            target = fulls[i] if len(fulls) == len(shorts) else fulls[0] if fulls else ""
            if target:
                pairs.append((short, target))
    return pairs


def _abbr_pattern(short: str, full: str) -> tuple[re.Pattern[str], Any]:
    """Шаблон сокращения в тексте и замена. Сокращение с дефисом («м-л» → «минерал»)
    узнаётся и в других формах: «м-лов» → «минералов», «к-лах» → «кристаллах»."""
    if "-" in short and not short.endswith("."):
        head, tail = short.split("-", 1)
        if full.lower().endswith(tail.lower()) and tail:
            stem = full[: len(full) - len(tail)]
            pattern = re.compile(
                rf"(?<![\w-]){re.escape(head)}[-‑]({re.escape(tail[0])}\w*)", re.IGNORECASE
            )
            return pattern, lambda m: stem + m.group(1)
    pattern = re.compile(rf"(?<![\w-]){re.escape(short)}(?![\w-])", re.IGNORECASE)

    def same_case(m: re.Match[str]) -> str:
        word = full[:1].upper() + full[1:] if m.group(0)[:1].isupper() else full
        # Точка сокращения бывает и концом фразы: «Мон. Удлиненные к-лы» → «Моноклинный.
        # Удлиненные…» — точка остаётся, если дальше новая фраза или конец текста.
        after = m.string[m.end() :]
        if short.endswith(".") and (not after.strip() or re.match(r"\s+[A-ZА-ЯЁ]", after)):
            word += "."
        return word

    return pattern, same_case


def expand(text: str, pairs: list[tuple[str, str]]) -> str:
    """Сокращения словаря в тексте — полными словами (длинные — первыми)."""
    for short, full in sorted(pairs, key=lambda p: -len(p[0])):
        pattern, repl = _abbr_pattern(short, full)
        text = pattern.sub(repl, text)
    return text


def abbreviations(conn: Any, dict_id: str) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT abbreviations FROM dictionaries WHERE id = %s", (dict_id,))
        row = cur.fetchone()
    return [tuple(p) for p in (row[0] or [])] if row else []  # type: ignore[misc]


# --- словарь в чате: термины вопроса, отсылки, варианты запроса ------------------------- #

# Однословный термин, который встречается больше чем в COMMON доле фрагментов статей
# («золото», «разлом»), в справку не идёт: его смысл модели и так ясен, а место в окне нужно
# фрагментам. Составные термины («зона сдвига») идут всегда.
COMMON = 0.02
MAX_NGRAM = 5
_REFERENCE = re.compile(r"^(?:см\.|син\. термина|то же, что)\s+(.+?)\.?$", re.IGNORECASE)
_QWORD = re.compile(r"[\w‑-]+", re.U)


@dataclass
class Index:
    """Ключи терминов справочных словарей (name_key: падеж и число не важны) → статьи."""

    stamp: tuple[Any, ...]
    by_key: dict[str, list[dict[str, Any]]]
    titles: dict[str, str]


_INDEX: dict[str, Index] = {}


def index(conn: Any) -> Index | None:
    """Индекс терминов из базы; пересчёт — если словари в базе поменялись. Словарей нет —
    None (чат и поиск работают как раньше)."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('dict_entries') IS NOT NULL")
        if not cur.fetchone()[0]:
            return None
        cur.execute("SELECT count(*), coalesce(max(loaded_at)::text, '') FROM dictionaries")
        stamp = tuple(cur.fetchone())
        if not stamp[0]:
            return None
        known = _INDEX.get("x")
        if known is not None and known.stamp == stamp:
            return known
        cur.execute("SELECT id, title FROM dictionaries")
        titles = dict(cur.fetchall())
        cur.execute(
            "SELECT dict_id, ord, term, term_key, en, etym, definition, page FROM dict_entries"
        )
        rows = cur.fetchall()
    by_key: dict[str, list[dict[str, Any]]] = {}
    for dict_id, ord_, term, key, en, etym, definition, page in rows:
        entry = {
            "dict_id": dict_id,
            "ord": ord_,
            "term": term,
            "en": en or "",
            "etym": etym or "",
            "definition": definition,
            "page": page,
        }
        by_key.setdefault(key, []).append(entry)
        # «Гастроподы (Gastropoda; от греч. …)» — и по слову без латинского названия.
        bare = term.split(" (")[0].strip()
        if bare != term:
            by_key.setdefault(name_key(bare), []).append(entry)
        for name in (en or "").split(","):
            if name.strip():
                by_key.setdefault(name_key(name), []).append(entry)
    _INDEX.clear()
    _INDEX["x"] = Index(stamp, by_key, titles)
    return _INDEX["x"]


def rollback(conn: Any) -> None:
    """Откатить сбойный запрос к словарю, чтобы соединение годилось дальше; у заглушки
    соединения в проверках отката нет — и не надо."""
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        pass


def _follow(idx: Index, entry: dict[str, Any], hops: int = 2) -> dict[str, Any]:
    """Отсылка («см. Структурная ось β.», «син. термина гастроподы.») — к статье, куда она
    ведёт; запоминается, откуда пришли (via)."""
    current = entry
    for _ in range(hops):
        m = _REFERENCE.match(current["definition"].strip())
        if not m:
            break
        target = idx.by_key.get(name_key(m.group(1).split(" (")[0]))
        if not target:
            break
        current = {**target[0], "via": entry["term"]}
    return current


def _common(conn: Any, term: str, total: int) -> bool:
    if " " in term.strip() or not total:
        return False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('russian', %s)", (term,)
        )
        return cur.fetchone()[0] > COMMON * total


def terms_in(conn: Any, text: str, limit: int = 3) -> list[dict[str, Any]]:
    """Статьи словарей о терминах, названных в тексте вопроса: самые длинные совпадения
    первыми, без перекрытий; отсылки разворачиваются, общеупотребительные однословные
    термины пропускаются. Словарей в базе нет — пусто."""
    idx = index(conn)
    if idx is None:
        return []
    words = _QWORD.findall(text or "")
    spans: list[tuple[int, int, dict[str, Any]]] = []
    for n in range(min(MAX_NGRAM, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            if any(a < i + n and i < b for a, b, _ in spans):
                continue
            phrase = " ".join(words[i : i + n])
            if n == 1 and len(phrase) < 4:
                continue
            found = idx.by_key.get(name_key(phrase))
            if found:
                spans.append((i, i + n, found[0]))
    if not spans:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        total = cur.fetchone()[0]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for _, _, entry in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if _common(conn, entry["term"], total):
            continue
        entry = _follow(idx, entry)
        key = (entry["dict_id"], entry["ord"])
        if key not in seen:
            seen.add(key)
            out.append({**entry, "title": idx.titles.get(entry["dict_id"], entry["dict_id"])})
        if len(out) >= limit:
            break
    return out


def query_variants(conn: Any, query: str, limit: int = 4) -> list[str]:
    """Запрос с другими написаниями терминов из словаря: английским названием и термином,
    куда ведёт отсылка («Брюхоногие» → «гастроподы»). Для полнотекстового поиска: он
    ищет слова буквально, и «лиственитизация» не находит английскую статью."""
    idx = index(conn)
    if idx is None:
        return []
    out: list[str] = []
    words = _QWORD.findall(query or "")
    for n in range(min(MAX_NGRAM, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if n == 1 and len(phrase) < 4:
                continue
            found = idx.by_key.get(name_key(phrase))
            if not found:
                continue
            entry = found[0]
            target = _follow(idx, entry)
            names = [x.strip() for x in entry["en"].split(",") if x.strip()]
            if target is not entry:
                names.append(target["term"])
                names += [x.strip() for x in target["en"].split(",") if x.strip()]
            for name in names:
                if name_key(name) != name_key(phrase):
                    variant = " ".join([*words[:i], name, *words[i + n :]])
                    if variant not in out:
                        out.append(variant)
    return out[:limit]


def expand_headword(text: str, term: str) -> str:
    """Термин внутри своей статьи сокращён до начальных букв: «Термин Д. часто применяют»
    (Декольман), «А. с. используется» (Адсорбция селективная) — полным термином. Инициалы
    авторов («Д.С. Коржинский», «Д. Smith») не трогаются: после них — заглавная буква."""
    words = [w for w in re.split(r"[\s\-]+", term.split(" (")[0]) if w]
    if not words:
        return text
    initials = r"\s?".join(re.escape(w[0]) + r"\." for w in words)
    initials = initials[:1].upper() + initials[1:] if len(words) == 1 else initials
    pattern = re.compile(
        rf"(?<![\w.]){initials}(?=(?-i:\s+[а-яёa-z0-9(])|[,;:)]|\s*$)", re.IGNORECASE
    )
    return pattern.sub(term.split(" (")[0], text)


def as_sources(conn: Any, entries: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    """Статьи словаря — источниками ответа, как фрагменты статей: свой номер [n], вид
    «словарь». Сокращения раскрыты, отсылка («см. …») названа."""
    out = []
    pairs_by: dict[str, list[tuple[str, str]]] = {}
    for i, e in enumerate(entries):
        if e["dict_id"] not in pairs_by:
            pairs_by[e["dict_id"]] = abbreviations(conn, e["dict_id"])
        pairs = pairs_by[e["dict_id"]]
        head = e["term"] + (f" [{e['en']}]" if e["en"] else "")
        via = f"(по отсылке из статьи «{e['via']}») " if e.get("via") else ""
        text = f"{head} — {via}{expand_headword(expand(e['definition'], pairs), e['term'])}"
        out.append(
            {
                "n": start + i,
                "doc_id": f"словарь:{e['dict_id']}",
                "ord": e["ord"],
                "title": f"{e['title']}: «{e['term']}»",
                "year": None,
                "journal": None,
                "authors": [],
                "pages": [e["page"]] if e.get("page") else [],
                "headings": [],
                "url": "",
                "text": text,
                "similarity": None,
                "found_by": "словарь",
                "part": None,
                "kind": "словарь",
                "term": e["term"],
            }
        )
    return out


# «Что такое адуляр», «что значит декольман», «дай определение…» — вопрос об определении:
# ответ начинается со статьи словаря, статьи базы его дополняют.
DEFINITION = re.compile(
    r"^\s*(?:что\s+(?:такое|значит|означает|называют|понимают\s+под)|"
    r"(?:дай|дайте|приведи|приведите)\s+определение|определение\s+(?:термина|понятия)?|"
    r"значение\s+(?:термина|слова|понятия)|what\s+is|what\s+does)\b",
    re.IGNORECASE,
)


def is_definition(question: str) -> bool:
    return bool(DEFINITION.search(question or ""))


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
        pairs = abbreviation_pairs(parse_pdf(pdf, *abbreviation_pages(pdf, first)))
        print(f"Сокращений: {len(pairs)}")
        folder = save(dict_id, args.title or pdf.stem, pdf.name, entries, pairs)
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
