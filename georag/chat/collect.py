"""Сбор по предмету: все статьи, где он упомянут, — выписка по каждой и сводка.

Ответ на «с чем связан хлорит» поиск строил из 8 фрагментов, а хлорит упомянут в 214
фрагментах 46 статей: ответ видел 7 статей из 46 (замер 14.10.2026). Окно модели —
12 тысяч токенов, все 46 статей целиком в него не войдут. Поэтому — как глобальный
поиск GraphRAG и map-reduce в Chain-of-Agents, но по самим статьям, а не по сводкам
сообществ (наш граф для них слишком редкий):

1. Псевдонимы предмета (aliases): словарь синонимов, имена графа, близкие по вектору
   и без общих слов («хлорит» — «chlorite»), и английское название от модели — один
   раз, в таблицу collect_aliases.
2. Все фрагменты, где упомянут предмет (полнотекст по псевдонимам), — по статьям;
   статьи — по числу упоминаний, из каждой до CHUNKS_PER_ARTICLE лучших фрагментов
   (заранее — до DEEP_CHUNKS: полнее, но дольше).
3. Выписка по каждой статье — отдельным вызовом модели, несколько одновременно: что
   фрагменты прямо говорят о предмете, с дословной цитатой. Пункт, чьей цитаты во
   фрагменте нет, отбрасывается. Выписка не зависит от вопроса и хранится
   (collect_notes): второй вопрос о том же предмете статьи не перечитывает.
4. Сводка — код, без модели: пункт → в скольких статьях и в каких. Модель ответа
   видит сводку и выписку каждой статьи (заметкой, со своим номером) и ссылается на
   статьи.

Первый вопрос о новом предмете ждёт, пока модель прочтёт до 30 статей (около минуты).
Поэтому выписки готовятся и заранее (warm, ночной прогон): о чём уже спрашивали — дочитать
новые статьи, и самые упоминаемые предметы словаря и графа — прочитать впрок.

    python -m georag.chat.collect                  заготовить выписки (нужна модель)
    python -m georag.chat.collect --top 0          только то, о чём уже спрашивали
    python -m georag.chat.collect --dry-run        что будет прочитано, без модели
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from ..text import locate_quote, mentions, name_key

MAX_ARTICLES = 30  # статей с упоминанием, которые читает модель
CHUNKS_PER_ARTICLE = 2
# Заготовка заранее (warm) читает из статьи больше фрагментов. 4 вместо 2 на тех же 30
# статьях: доля эталона (2+ статей) в выписке на 5 предметах — в среднем 0,54 → 0,71
# (хлорит 0,57 → 0,79, самородное золото 0,26 → 0,62, мигматиты 0,71 → 0,71); чтение
# в 1,9 раза дольше (15.10.2026) — для живого вопроса долго, для ночи годится.
# Глубокая выписка годится и живому вопросу: её фрагменты включают его 2.
DEEP_CHUNKS = 4
MAP_TOKENS = 750  # потолок ответа выписки: столько на каждый фрагмент статьи и ещё раз
CHUNK_CHARS = 1800
WORKERS = 8  # выписок одновременно (vLLM считает их вместе)
ITEMS_PER_ARTICLE = 12  # пунктов заметки статьи, которые видит модель ответа
SUMMARY_CHARS = 6000
ALIAS_SIMILARITY = 0.83  # имя графа — то же, что предмет, но иначе написано
NOTE_ITEMS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS collect_notes (
    subject  text NOT NULL,
    doc_id   text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunks   bigint[] NOT NULL,
    items    jsonb NOT NULL,
    made_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (subject, doc_id)
);
CREATE TABLE IF NOT EXISTS collect_aliases (
    subject  text PRIMARY KEY,
    aliases  text[] NOT NULL,
    made_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS collect_asked (
    subject  text NOT NULL,
    facet    text NOT NULL,
    name     text NOT NULL,
    times    integer NOT NULL DEFAULT 1,
    asked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (subject, facet)
);
"""

SYSTEM = "Ты внимательно читаешь геологические статьи. Отвечай только JSON."
ALIAS_PROMPT = """Как называют «{subject}» в геологических статьях по-английски и какие у него
другие написания по-русски? Только сам предмет, не родственные понятия: для «хлорит» —
«chlorite», но не «chloritization». До 4 вариантов, каждый 1–4 слова.
Верни JSON: {{"варианты": ["..."]}}"""

MAP_PROMPT = """Предмет: «{subject}» (он же: {aliases}).
Фрагменты одной статьи «{title}»{year}:

{fragments}

Выпиши всё, что эти фрагменты прямо говорят о «{subject}»: с чем он связан — минералы,
породы, процессы и изменения, структуры, месторождения и районы, условия образования,
методы выявления. Каждый пункт:
- «с_чем» — С ЧЕМ связан «{subject}» (не сам «{subject}»): название минерала, породы,
  процесса, структуры, месторождения или метода, 1–4 слова НА ЯЗЫКЕ ИСХОДНОЙ ЦИТАТЫ.
  Не переводи название: epidote оставь epidote. Название должно быть в «цитата»;
- «вид» — что это, одним словом: минерал, порода, процесс, структура, месторождение,
  район, метод, признак, другое;
- «как» — вид связи, 1–4 слова: «ассоциирует с», «замещает», «вмещает», «сопровождает»,
  «образуется при», «выявлен по», «локализован в»…;
- «где» — месторождение, район или стадия, если названы; иначе пусто;
- «цитата» — дословный отрывок фрагмента до 300 знаков с названием и связью;
- «фрагмент» — номер фрагмента, например c123.
Каждый пункт — отдельно: перечень «хлорит, эпидот и кальцит» — это три пункта; выпиши
всё, а не главное. Не пиши того, чего во фрагментах нет. О предмете ничего — пустой список.

Например, «Хлорит ассоциирует с эпидотом в пропилитах Кокпатаса [c12]» → {{"с_чем": "эпидот",
"вид": "минерал", "как": "ассоциирует с", "где": "Кокпатас, пропилиты", "цитата": "Хлорит
ассоциирует с эпидотом", "фрагмент": "c12"}}.

Верни JSON: {{"пункты": [{{"с_чем": "...", "вид": "...", "как": "...", "где": "...", "цитата": "...", "фрагмент": "c123"}}]}}"""

KINDS = (
    "минерал", "порода", "процесс", "структура", "месторождение", "район", "метод", "признак",
    "другое",
)  # fmt: skip
# О чём спрашивают — по словам вопроса: «с какими минералами» → только минералы в сводке.
# Не узнали — сводка по всем видам.
_FACETS = (
    (re.compile(r"минерал|mineral", re.IGNORECASE), {"минерал"}),
    (re.compile(r"пород|rock|litholog", re.IGNORECASE), {"порода"}),
    (re.compile(r"процесс|изменени|метасомат|process|alteration", re.IGNORECASE), {"процесс"}),
    (re.compile(r"структур|разлом|structur|fault", re.IGNORECASE), {"структура"}),
    (re.compile(r"метод|данны|съёмк|съемк|method", re.IGNORECASE), {"метод"}),
    (
        re.compile(r"месторожден|район|где |deposit|district", re.IGNORECASE),
        {"месторождение", "район"},
    ),
)
FACET_MIN = 3  # меньше пунктов нужного вида — сводка по всем видам

# Прицельная выписка: вопрос о минералах (породах, процессах…) — модель выписывает все
# такие и только такие. Общая («что статья говорит о предмете») давала около 5 пунктов на
# статью: во фрагментах опора была у 71–100 % названий эталона, в выписке — 29–61 %.
# Прицельная на тех же статьях: хлорит 0,26 → 0,57, самородное золото 0,28 → 0,51,
# кварцевые жилы 0,73 → 0,69, и быстрее (около 56 с на предмет против 92) — 15.10.2026.
KIND_WORDS = {
    "минерал": "минералы и группы минералов",
    "порода": "породы и толщи",
    "процесс": "процессы и изменения пород",
    "структура": "структуры — разломы, зоны сдвига, складки, линеаменты",
    "метод": "методы и данные",
    "месторождение": "месторождения и рудопроявления",
    "район": "районы и рудные поля",
}
TARGET_PROMPT = """Предмет: «{subject}» (он же: {aliases}).
Фрагменты одной статьи «{title}»{year}:

{fragments}

Выпиши ВСЕ {kinds}, которые эти фрагменты прямо связывают с «{subject}»: встречаются вместе
с ним, сопровождают, замещают, вмещают, образуются при нём… Каждый — отдельным пунктом:
перечень «эпидот, кальцит и серицит» — это три пункта. Не пиши сам «{subject}» и то, что в
статье упомянуто, но с «{subject}» не связано. Только {kinds} — не {others}.
Каждый пункт:
- «с_чем» — название НА ЯЗЫКЕ ИСХОДНОЙ ЦИТАТЫ. Не переводи: epidote оставь epidote.
  Название должно присутствовать в «цитата»;
- «вид» — что это на самом деле, одним словом: минерал, порода, процесс, структура,
  месторождение, район, метод, признак, другое;
- «как» — вид связи, 1–4 слова;
- «где» — месторождение, район или стадия, если названы; иначе пусто;
- «цитата» — дословный отрывок фрагмента до 300 знаков с названием и связью;
- «фрагмент» — номер фрагмента, например c123.
Ничего такого нет — пустой список.

Верни JSON: {{"пункты": [{{"с_чем": "...", "вид": "...", "как": "...", "где": "...", "цитата": "...", "фрагмент": "c123"}}]}}"""
# Связь — только если рядом с цитатой (в пределах NEAR знаков) упомянут сам предмет:
# иначе модель выписывала соседей по статье («кварц можно картировать по тепловым ИК-
# данным» для хлорита). Проверка вслепую 30 пунктов: 12 неверных, из них 5 — такие.
NEAR = 300
# Версия заданий на выписку и проверки пунктов — в ключе кэша: поменялись — выписки
# читаются заново (5 — название на языке цитаты и проверка второго участника).
MAP_VERSION = 5


def facet(question: str, subject: str = "") -> set[str]:
    """Какие виды пунктов нужны вопросу; пусто — все. Слова самого предмета не в счёт:
    «методами выделяют зоны гидротермальных изменений» — о методах, не о процессах."""
    own = {w.lower()[:5] for w in subject.split() if len(w) > 3}
    asked = " ".join(w for w in (question or "").split() if w.lower()[:5] not in own)
    wanted: set[str] = set()
    for pattern, kinds in _FACETS:
        if pattern.search(asked):
            wanted |= kinds
    return wanted


@dataclass
class Collected:
    """Что собрал сбор по предмету: источники для ответа и счёт."""

    subject: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    articles: int = 0  # статей с упоминанием
    read: int = 0  # из них прочитано моделью (или взято из выписок прошлых вопросов)
    cached: int = 0
    items: int = 0  # пунктов с проверенной цитатой
    dropped: int = 0  # пунктов без цитаты во фрагменте
    seconds: float = 0.0
    facet: list[str] = field(default_factory=list)  # какие виды пунктов взяты (facet)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.subject,
            "collected": {
                "articles": self.articles,
                "read": self.read,
                "cached": self.cached,
                "items": self.items,
                "dropped": self.dropped,
                "seconds": round(self.seconds, 1),
                "facet": self.facet,
            },
        }


def init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def _clean(value: Any, limit: int = 120) -> str:
    return " ".join(str(value or "").split()).strip(" .,;:«»\"'")[:limit]


def aliases(conn: Any, embedder: Any, llm: Any, subject: str) -> list[str]:
    """Предмет и другие его названия: словарь, близкие имена графа, ответ модели (в кэше)."""
    subject = _clean(subject)
    key = name_key(subject)
    names: list[str] = [subject]
    with conn.cursor() as cur:
        cur.execute("SELECT aliases FROM collect_aliases WHERE subject = %s", (key,))
        row = cur.fetchone()
    if row:
        return list(dict.fromkeys([subject, *row[0]]))
    names += _dictionary_names(subject)
    names += _graph_neighbours(conn, embedder, subject)
    english = _english_names(llm, subject)
    names += english or []
    out = [n for n in dict.fromkeys(names) if 3 <= len(n) <= 60 and len(n.split()) <= 5]
    if english is None:  # без английского названия статьи на английском не найдутся — не храним
        return out
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO collect_aliases (subject, aliases) VALUES (%s, %s) "
            "ON CONFLICT (subject) DO UPDATE SET aliases = EXCLUDED.aliases, made_at = now()",
            (key, out),
        )
    conn.commit()
    return out


def _dictionary_names(subject: str) -> list[str]:
    """Название предмета в словаре синонимов и его написания; словаря нет — пусто."""
    try:
        from ..graph import synonyms

        syn = synonyms.load()
        canon = syn.name(subject) or subject
        return [canon, *syn.groups.get(canon, [])]
    except Exception:  # noqa: BLE001 — словаря нет: обойдёмся без него
        return []


def _english_names(llm: Any, subject: str) -> list[str] | None:
    """Английские названия предмета от модели; None — модель не ответила."""
    try:
        answer = llm.chat_json(
            SYSTEM, ALIAS_PROMPT.format(subject=subject), temperature=0, max_tokens=200
        )
        # От модели — только английские названия: русские «варианты» она сочиняет
        # («хлоритовая порода» для «хлорит»), русские написания даёт словарь.
        return [
            v
            for v in (_clean(x, 60) for x in (answer or {}).get("варианты") or [])
            if v and not _CYRILLIC.search(v)
        ]
    except Exception:  # noqa: BLE001 — модель молчит: без английского названия
        return None


def _graph_neighbours(conn: Any, embedder: Any, subject: str) -> list[str]:
    """Имена графа, близкие по вектору и без общих слов с предметом: «хлорит» — «chlorite»
    (0,84), но не «зона хлорита» (общее слово) и не «chloritization» (0,71)."""
    from ..index.db import vector_literal

    try:
        literal = vector_literal(embedder.encode_one(subject))
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, 1 - (embedding <=> %s::vector) FROM entity_names "
                "ORDER BY embedding <=> %s::vector LIMIT 8",
                (literal, literal),
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — таблицы автосинонимов нет
        conn.rollback()
        return []
    own = set(name_key(subject).split())
    width = len(subject.split()) + 1
    return [
        name
        for name, similarity in rows
        if similarity >= ALIAS_SIMILARITY
        and not own & set(name_key(name).split())
        and len(name.split()) <= width
    ]


def _names_tsquery(names: list[str]) -> str:
    """Запрос полнотекста «любое из названий»; пусто — ни одно название в запрос не годится."""
    from ..index.search import build_tsquery

    parts = [p for p in (build_tsquery(n) for n in names) if p.strip()]
    return " | ".join(f"({p})" for p in parts)


def mentioning(conn: Any, names: list[str]) -> list[dict[str, Any]]:
    """Фрагменты, где упомянут предмет (любое из названий), со статьёй и рангом."""
    query = _names_tsquery(names)
    if not query:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.ord, c.text, c.headings, c.pages, d.title, d.year,
                   d.journal, d.url, d.authors, ts_rank_cd(c.tsv, to_tsquery('russian', %s))
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ to_tsquery('russian', %s)
            """,
            (query, query),
        )
        rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "doc_id": r[1],
            "ord": r[2],
            "text": r[3] or "",
            "headings": list(r[4] or []),
            "pages": list(r[5] or []),
            "title": r[6] or r[1],
            "year": r[7],
            "journal": r[8],
            "url": r[9] or "",
            "authors": list(r[10] or [])[:3],
            "rank": float(r[11] or 0),
        }
        for r in rows
    ]


def by_article(
    rows: list[dict[str, Any]], per: int = CHUNKS_PER_ARTICLE
) -> list[list[dict[str, Any]]]:
    """Статьи с упоминанием — сначала где их больше; из каждой — per лучших фрагментов
    (порядок однозначный: лучшие 2 — всегда начало лучших 4)."""
    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        docs[r["doc_id"]].append(r)
    ordered = sorted(
        docs.values(), key=lambda rs: (-len(rs), -max(r["rank"] for r in rs), rs[0]["doc_id"])
    )
    return [sorted(rs, key=lambda r: (-r["rank"], r["chunk_id"]))[:per] for rs in ordered]


def map_prompt(
    subject: str, names: list[str], chunks: list[dict[str, Any]], wanted: set[str] | None = None
) -> str:
    """Задание на выписку по статье: общее или прицельное (wanted — какие виды нужны)."""
    first = chunks[0]
    fragments = "\n\n".join(
        f"[c{c['chunk_id']}] {' '.join(c['text'].split())[:CHUNK_CHARS]}" for c in chunks
    )
    others = ", ".join(n for n in names if n != subject) or "—"
    year = f" ({first['year']})" if first.get("year") else ""
    if wanted:
        kinds = " и ".join(KIND_WORDS[k] for k in sorted(wanted) if k in KIND_WORDS)
        rest = ", не ".join(
            KIND_WORDS[k].split(" — ")[0] for k in sorted(KIND_WORDS) if k not in wanted
        )
        return TARGET_PROMPT.format(
            subject=subject,
            aliases=others,
            title=first["title"][:150],
            year=year,
            fragments=fragments,
            kinds=kinds,
            others=rest,
        )
    return MAP_PROMPT.format(
        subject=subject, aliases=others, title=first["title"][:150], year=year, fragments=fragments
    )


def check_items(
    answer: Any, chunks: list[dict[str, Any]], names: list[str] | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Пункты выписки с цитатой, которая есть во фрагменте, и не о самом предмете (names —
    его названия: модель пишет «хлорит — ассоциирует с», забыв, с чем); сколько отброшено."""
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    own = {name_key(n) for n in names or []}
    items = answer.get("пункты") if isinstance(answer, dict) else None
    kept, dropped = [], 0
    for item in items if isinstance(items, list) else []:
        checked = _checked_item(item, texts, own, names)
        if checked is None:
            dropped += 1
        else:
            kept.append(checked)
    return kept, dropped


def _checked_item(
    item: Any, texts: dict[int, str], own: set[str], names: list[str] | None
) -> dict[str, Any] | None:
    """Пункт выписки в нашем виде — или None, если он не прошёл проверку: нет названия,
    цитаты нет во фрагменте, пункт о самом предмете (own — ключи его названий), предмет
    не упомянут рядом с цитатой, название не подтверждено цитатой."""
    if not isinstance(item, dict):
        return None
    what = _clean(item.get("с_чем") or item.get("что"), 60)
    quote = _clean(item.get("цитата"), 300)
    found = _quote_chunk(quote, item.get("фрагмент"), texts)
    if not what or found is None or name_key(what) in own:
        return None
    if names and not _near_subject(quote, texts[found], names):
        return None
    if not _quoted_entity(what, quote):
        return None
    kind = _clean(item.get("вид"), 20).lower()
    return {
        "что": what,
        "вид": kind if kind in KINDS else "другое",
        "как": _clean(item.get("как"), 40),
        "где": _clean(item.get("где"), 80),
        "цитата": quote,
        "фрагмент": found,
    }


def _quote_chunk(quote: str, fragment: Any, texts: dict[int, str]) -> int | None:
    """Фрагмент, где есть цитата: названный моделью (если он из этой статьи), иначе любой
    из фрагментов статьи; None — цитаты нет ни в одном."""
    number = "".join(ch for ch in str(fragment or "") if ch.isdigit())
    chunk_id = int(number) if number and int(number) in texts else None
    where = [chunk_id] if chunk_id else list(texts)
    return next((cid for cid in where if quote and locate_quote(quote, texts[cid])), None)


def _quoted_entity(what: str, quote: str) -> bool:
    """Название подтверждено цитатой, напрямую или через словарь синонимов.

    Неизвестные переводы не угадываем: модель должна оставить термин оригинала.
    Это проверка названия, а не доказательство типа и направления связи.
    """
    if mentions(what, quote):
        return True
    from ..graph import synonyms

    syn = synonyms.load()
    canon = syn.name(what)
    return bool(canon and any(mentions(n, quote) for n in [canon, *syn.groups.get(canon, [])]))


# Слова, без которых предмет узнаётся и так: «argillic alteration» в тексте — «argillic
# zone», «argillized rocks». По имени целиком проверка отбрасывала у процессов 2/3 верных
# пунктов («Propylitic, Mineral = Calcite»): аргиллизация 143 → 46, по основам — 122.
_GENERIC_STEMS = {"alter", "zone", "zones", "proce", "измен", "зона", "зоны", "проце"}
_STEM_WORD = re.compile(r"[^\W\d_]+")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _STEM_WORD.findall(text.lower().replace("ё", "е")) if len(w) >= 4}


_ITEM = re.compile(r"\{[^{}]*\}")


def salvage(text: str) -> dict[str, Any] | None:
    """Пункты из оборванного ответа: целые объекты до места обрыва. Иначе статья с
    длинной выпиской терялась целиком и перечитывалась при каждом вопросе."""
    items = []
    for match in _ITEM.finditer(text or ""):
        try:
            item = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and ("с_чем" in item or "что" in item):
            items.append(item)
    return {"пункты": items} if items else None


def _near_subject(quote: str, text: str, names: list[str]) -> bool:
    """Упомянут ли предмет в цитате или рядом с ней (NEAR знаков до и после): именем
    целиком или основами его значимых слов («argillized» для «argillic alteration»)."""
    plain = " ".join(text.split())
    span = locate_quote(quote, text) or quote
    at = plain.find(span)
    window = plain[max(0, at - NEAR) : at + len(span) + NEAR] if at >= 0 else plain
    if any(mentions(n, window) for n in names):
        return True
    have = _stems(window)
    return any((need := _stems(n) - _GENERIC_STEMS) and need <= have for n in names)


# Ключ написания: выписка оставляет название на языке цитаты (epidote в английской статье,
# эпидот — в русской), и сводка считала их разными пунктами, а хвост «Ещё в статьях базы»
# повторял по-английски то, что ответ уже назвал по-русски. Ключ — латиницей, с правилами
# названий минералов (-ite = -ит, c перед e/i = ц, h = г = х), без двойных букв и конечных
# гласных. На парах «calcite — кальцит»: 29 из 35 (не сходятся galena — галенит, pyrrhotite
# — пирротин, gypsum — гипс…); среди 1269 названий из выписок склеено 49 групп, все —
# написания одного и того же (гранат и гранит — разные), 21.10.2026.
_RU_LATIN = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "h", "д": "d", "е": "e", "ж": "zh", "з": "z",
        "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
        "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
        "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "u", "я": "a",
    }
)  # fmt: skip


def spelling_key(name: str) -> str:
    """Один ключ у английского и русского написания: «epidote», «эпидотом» → epidot."""
    words = []
    for word in name_key(name).split():
        word = word.replace("ё", "е")
        if _CYRILLIC.search(word):
            word = word.translate(_RU_LATIN)
        else:
            for a, b in (("ph", "f"), ("th", "t"), ("ch", "h"), ("qu", "kv"), ("oe", "e")):
                word = word.replace(a, b)
            word = word.replace("tz", "ts").replace("x", "ks").replace("w", "v")
            word = re.sub(r"c(?=[eiy])", "ts", word).replace("c", "k")
        word = re.sub(r"(.)\1+", r"\1", word.replace("g", "h").replace("y", "i"))
        words.append(re.sub(r"(?<=.)[aeiou]+$", "", word))
    return " ".join(words)


def _said_keys(text: str) -> set[str]:
    """Ключи написания слов текста и пар, троек слов подряд — что ответ уже назвал."""
    words = _STEM_WORD.findall(text.lower())
    return {
        spelling_key(" ".join(words[i : i + n]))
        for n in (1, 2, 3)
        for i in range(len(words) - n + 1)
    }


def summary_text(subject: str, groups: list[dict[str, Any]], articles: int) -> str:
    """Сводка — указатель: что упомянуто в скольких статьях и в каких (номера источников).

    Строкой на пункт с подробностями модель отвечала разделом на пункт — 40 разделов,
    без порядка (хлорит, 15.10.2026). Подробности — в выписках статей; здесь — частота."""
    head = (
        f"Что статьи базы говорят о «{subject}» — выписано моделью из {articles} статей, где "
        "он упомянут, каждая выписка проверена по цитате. После названия — в скольких "
        "статьях и номера статей-источников (подробности — в их выписках ниже)."
    )
    tiers = (
        ("Чаще всего (в 3 и более статьях)", 3, 10**6),
        ("В двух статьях", 2, 2),
        ("В одной статье", 1, 1),
    )
    lines, size = [head], len(head)
    for title, low, high in tiers:
        parts = []
        for g in groups:
            n = len(g["sources"])
            if not low <= n <= high:
                continue
            cites = "".join(f"[{k}]" for k in g["sources"][:4])
            part = f"{g['name']} — {n} {cites}" if n > 1 else f"{g['name']} {cites}"
            if size + len(part) + 2 > SUMMARY_CHARS:
                break
            parts.append(part)
            size += len(part) + 2
        if parts:
            lines.append(f"{title}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


# Хвост ответа: чего из сводки модель не назвала. Модель, даже со сводкой из 30 статей,
# перечисляет 10–20 пунктов: сводка покрывала 0,46–0,61 эталона, ответ — 0,20–0,46
# (8 широких вопросов, 15.10.2026). Остальное дописывает код — пункты проверены цитатами.
REST_ITEMS = 60
REST_SINGLE = 25  # пунктов из одной статьи — не больше (их длинный хвост — шум)


def rest_of_summary(answer: str, sources: list[dict[str, Any]]) -> str:
    """Строка «Ещё в статьях базы: …» — пункты сводки, которых нет в ответе, по убыванию
    числа статей, с номерами статей-источников; пусто — сводки нет или всё названо."""
    summary = next((s for s in sources if s.get("kind") == "сводка"), None)
    if summary is None:
        return ""
    said = answer.lower().replace("ё", "е")
    said_keys = _said_keys(said)
    available = {s["n"] for s in sources if s.get("kind") != "сводка"}
    rest, single = [], 0
    for g in summary.get("groups") or []:
        refs = [n for n in g["sources"] if n in available]
        if not refs:
            continue
        spellings = g.get("names") or [g["name"]]
        if any(mentions(n, said) or spelling_key(n) in said_keys for n in spellings):
            continue
        if len(refs) == 1:
            if single >= REST_SINGLE:
                continue
            single += 1
        cites = "".join(f"[{n}]" for n in refs[:3])
        rest.append(f"{g['name']} {cites}")
        if len(rest) >= REST_ITEMS:
            break
    if not rest:
        return ""
    return "**Ещё в статьях базы** (дополнительные пункты из выписок): " + "; ".join(rest) + "."


def collect(
    conn: Any,
    embedder: Any,
    llm: Any,
    subject: str,
    start: int = 1,
    question: str = "",
    wanted: set[str] | None = None,
    per: int = CHUNKS_PER_ARTICLE,
) -> Generator[dict[str, Any], None, Collected | None]:
    """Сбор по предмету — генератор событий status; в конце — Collected (или None:
    предмет нигде не упомянут). wanted — какие виды пунктов нужны, если не по вопросу,
    per — фрагментов из статьи (заготовка выписок, warm)."""
    started = time.monotonic()
    init(conn)
    names = aliases(conn, embedder, llm, subject)
    articles = by_article(mentioning(conn, names), per)
    if not articles:
        return None
    result = Collected(subject, articles=len(articles))
    articles = articles[:MAX_ARTICLES]
    # О чём спрашивают (минералы, породы…) — от этого задание на выписку и ключ кэша.
    if wanted is None:
        wanted = facet(question, subject)
    key = _notes_key(subject, wanted)
    if question:
        _remember(conn, subject, wanted)
    notes, todo = _split_cached(articles, _cached_notes(conn, key))
    result.cached = len(notes)
    if todo:
        yield _status(f"читаю статьи о «{subject}»: {len(todo)} из {len(articles)}…")
    read, result.dropped = yield from _read_articles(conn, llm, key, subject, names, wanted, todo)
    notes.update(read)
    result.read = len(notes)
    # Спросили о минералах — в сводке и выписках только минералы (если их хватает):
    # иначе модель отвечала списком вперемешку с процессами, методами и районами.
    fitting = _of_kinds(notes, wanted) if wanted else None
    if fitting is not None:
        notes, result.facet = fitting, sorted(wanted)
    sources = _note_sources(articles, notes, start)
    if not sources:
        return None
    result.items = sum(map(len, notes.values()))
    groups = _summary_groups(sources, notes)
    result.sources = [_summary_source(subject, groups, len(sources), start), *sources]
    result.seconds = time.monotonic() - started
    return result


def _status(text: str) -> dict[str, Any]:
    return {"type": "status", "text": text}


def _notes_key(subject: str, wanted: set[str]) -> str:
    """Ключ выписок в кэше (collect_notes): предмет, виды пунктов прицельной выписки и
    версия заданий (MAP_VERSION)."""
    kinds = "|" + ",".join(sorted(wanted)) if wanted else ""
    return f"{name_key(subject)}{kinds}@{MAP_VERSION}"


def _cached_notes(conn: Any, key: str) -> dict[str, tuple[list[int], Any]]:
    """Выписки прошлых вопросов: статья → (прочитанные фрагменты, пункты)."""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, chunks, items FROM collect_notes WHERE subject = %s", (key,))
        return {r[0]: (list(r[1]), r[2]) for r in cur.fetchall()}


def _split_cached(
    articles: list[list[dict[str, Any]]], cached: dict[str, tuple[list[int], Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Готовые выписки статей (статья → пункты) и статьи, которые надо прочитать."""
    notes: dict[str, list[dict[str, Any]]] = {}
    todo = []
    for chunks in articles:
        doc = chunks[0]["doc_id"]
        read_chunks, items = cached.get(doc, ([], None))
        # Готовая выписка годится, если прочитаны все нужные фрагменты (или больше —
        # заготовленная заранее глубокая).
        if _is_fresh(items) and {c["chunk_id"] for c in chunks} <= set(read_chunks):
            notes[doc] = items
        else:
            todo.append(chunks)
    return notes, todo


def _is_fresh(items: Any) -> bool:
    """Выписка нынешнего вида: список пунктов, у каждого есть «вид»."""
    return isinstance(items, list) and all("вид" in i for i in items)


def _read_articles(
    conn: Any,
    llm: Any,
    key: str,
    subject: str,
    names: list[str],
    wanted: set[str],
    todo: list[list[dict[str, Any]]],
) -> Generator[dict[str, Any], None, tuple[dict[str, list[dict[str, Any]]], int]]:
    """Выписки по статьям todo — моделью, WORKERS одновременно. Каждая проверяется по
    цитатам (check_items) и сразу пишется в кэш; каждые 8 статей — событие status.
    Итог — выписки (статья → пункты, в порядке готовности) и сколько пунктов отброшено."""
    notes: dict[str, list[dict[str, Any]]] = {}
    dropped = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {
            pool.submit(_read_article, llm, subject, names, chunks, wanted): chunks
            for chunks in todo
        }
        for done, future in enumerate(as_completed(futures), start=1):
            answer = future.result()
            if answer is None:
                continue
            chunks = futures[future]
            items, lost = check_items(answer, chunks, names)
            dropped += lost
            doc = chunks[0]["doc_id"]
            notes[doc] = items
            _save_note(conn, key, doc, chunks, items)
            if done % 8 == 0 and done < len(todo):
                yield _status(f"прочитано статей о «{subject}»: {done} из {len(todo)}…")
    return notes, dropped


def _read_article(
    llm: Any, subject: str, names: list[str], chunks: list[dict[str, Any]], wanted: set[str]
) -> Any:
    """Ответ модели — выписка по статье; оборванный ответ — что из него спасено (salvage);
    None — модель не ответила."""
    prompt = map_prompt(subject, names, chunks, wanted)
    try:
        return llm.chat_json(
            SYSTEM, prompt, temperature=0, max_tokens=MAP_TOKENS * (len(chunks) + 1)
        )
    except json.JSONDecodeError as exc:  # ответ оборвался на потолке длины
        return salvage(exc.doc)
    except Exception:  # noqa: BLE001 — модель не ответила по статье
        return None


def _save_note(
    conn: Any, key: str, doc: str, chunks: list[dict[str, Any]], items: list[dict[str, Any]]
) -> None:
    """Выписка статьи — в кэш, с номерами прочитанных фрагментов."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO collect_notes (subject, doc_id, chunks, items) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (subject, doc_id) DO UPDATE SET chunks = EXCLUDED.chunks, "
            "items = EXCLUDED.items, made_at = now()",
            (key, doc, [c["chunk_id"] for c in chunks], json.dumps(items, ensure_ascii=False)),
        )
    conn.commit()


def _of_kinds(
    notes: dict[str, list[dict[str, Any]]], wanted: set[str]
) -> dict[str, list[dict[str, Any]]] | None:
    """Выписки только с пунктами нужных видов; None — таких меньше FACET_MIN."""
    fitting = {d: [i for i in its if i.get("вид") in wanted] for d, its in notes.items()}
    return fitting if sum(map(len, fitting.values())) >= FACET_MIN else None


def _note_sources(
    articles: list[list[dict[str, Any]]], notes: dict[str, list[dict[str, Any]]], start: int
) -> list[dict[str, Any]]:
    """Источники: статья с выпиской — заметкой; номера — по порядку статей, после сводки
    (у неё номер start)."""
    sources: list[dict[str, Any]] = []
    for chunks in articles:
        items = notes.get(chunks[0]["doc_id"]) or []
        if items:
            sources.append(_note_source(chunks, items, start + 1 + len(sources)))
    return sources


def _note_source(
    chunks: list[dict[str, Any]], items: list[dict[str, Any]], n: int
) -> dict[str, Any]:
    """Статья источником номер n: её лучший фрагмент и заметка — пункты выписки строкой."""
    first = chunks[0]
    note = "; ".join(
        f"{i['что']} — {i['как']}" + (f" ({i['где']})" if i["где"] else "")
        for i in items[:ITEMS_PER_ARTICLE]
    )
    return {
        "n": n,
        "doc_id": first["doc_id"],
        "chunk_id": first["chunk_id"],
        "ord": first["ord"],
        "title": first["title"],
        "year": first["year"],
        "journal": first["journal"],
        "authors": first["authors"],
        "pages": first["pages"],
        "headings": first["headings"][-2:],
        "url": first["url"],
        "text": " ".join(first["text"].split()),
        "similarity": None,
        "found_by": "сбор",
        "part": None,
        "kind": "заметка",
        "note": note,
    }


def _summary_groups(
    sources: list[dict[str, Any]], notes: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Пункты выписок, сведённые по написанию (spelling_key): названия, связи, где и номера
    статей-источников; чем в большем числе статей, тем раньше."""
    groups: dict[str, dict[str, Any]] = {}
    for source in sources:
        n = source["n"]
        for item in notes[source["doc_id"]]:
            g = groups.setdefault(
                spelling_key(item["что"]) or item["что"].lower(),
                {"names": Counter(), "как": Counter(), "где": set(), "sources": []},
            )
            g["names"][item["что"]] += 1
            g["как"][item["как"]] += 1
            if item["где"]:
                g["где"].add(item["где"])
            if n not in g["sources"]:
                g["sources"].append(n)
    ordered = sorted(groups.values(), key=lambda g: -len(g["sources"]))
    for g in ordered:
        # Есть русское написание из статей базы — им (не переводом модели), иначе как в цитате.
        russian = [n for n, _ in g["names"].most_common() if _CYRILLIC.search(n)]
        g["name"] = russian[0] if russian else g["names"].most_common(1)[0][0]
    return ordered


def _summary_source(
    subject: str, groups: list[dict[str, Any]], articles: int, start: int
) -> dict[str, Any]:
    """Сводка источником номер start: текст для модели (summary_text) и группы для кода."""
    return {
        "n": start,
        "subject": subject,
        "doc_id": f"сводка:{subject}",
        "ord": -1,
        "title": f"Сводка по статьям: {subject}",
        "year": None,
        "journal": None,
        "authors": [],
        "pages": [],
        "headings": [],
        "url": "",
        "text": summary_text(subject, groups, articles),
        # Для кода, не для модели: что упомянуто в скольких статьях (rest_of_summary).
        "groups": [
            {"name": g["name"], "names": list(g["names"]), "sources": list(g["sources"])}
            for g in groups
        ],
        "similarity": None,
        "found_by": "сбор",
        "part": None,
        "kind": "сводка",
    }


def _remember(conn: Any, subject: str, wanted: set[str]) -> None:
    """О чём спросили (предмет и виды) — чтобы ночью дочитать к нему новые статьи."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO collect_asked (subject, facet, name) VALUES (%s, %s, %s) "
            "ON CONFLICT (subject, facet) DO UPDATE SET times = collect_asked.times + 1, "
            "name = EXCLUDED.name, asked_at = now()",
            (name_key(subject), ",".join(sorted(wanted)), _clean(subject)),
        )
    conn.commit()


# Заготовка выписок (warm). Предмет с готовыми выписками отвечает за 25–50 с вместо
# 85–130 с (хлорит, 15.10.2026); прочитать один предмет заново — около минуты, поэтому
# число предметов и время ограничены.
WARM_TOP = 20  # самых упоминаемых предметов словаря и графа
WARM_FACETS = ("", "минерал")  # «что известно о…» и «с какими минералами…»
WARM_MINUTES = 60
WARM_GRAPH = 150  # имён графа (по числу статей с фактами) — в кандидаты
# Предмет-метод («магниторазведка», «машинное обучение») — только общая выписка: «с какими
# минералами связана магниторазведка» не спросят.
_METHOD = re.compile(
    r"метод|анализ|съёмк|съемк|разведк|зондирован|моделирован|обучени|картирован", re.IGNORECASE
)


def _documents(conn: Any, names: list[str]) -> int:
    """Во скольких статьях упомянут предмет (любое из названий) — по полнотексту."""
    query = _names_tsquery(names)
    if not query:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT doc_id) FROM chunks WHERE tsv @@ to_tsquery('russian', %s)",
            (query,),
        )
        return int(cur.fetchone()[0])


def frequent(conn: Any, top: int = WARM_TOP) -> list[tuple[str, int, bool]]:
    """Предметы, о которых спросят скорее всего: названия словаря синонимов и русские
    имена графа — по числу статей, где они упомянуты; третье — метод ли это. Английские
    имена графа не берутся: спрашивают по-русски, а «chlorite» и «хлорит» — разные ключи
    выписок."""
    from ..graph import api, synonyms

    syn = synonyms.load()
    names: dict[str, list[str]] = {c: [c, *v] for c, v in syn.groups.items()}
    rows = api.datasets_response(conn, syn, limit=WARM_GRAPH)["datasets"]
    for r in rows:
        if _CYRILLIC.search(r["name"]) and len(r["name"].split()) <= 3:
            names.setdefault(r["name"], [r["name"]])
    g = api.graph(conn, syn)
    ranked = sorted(
        (
            (n, _documents(conn, v), g.kind(n) == "метод" or bool(_METHOD.search(n)))
            for n, v in names.items()
        ),
        key=lambda x: -x[1],
    )
    return [x for x in ranked if x[1] >= 2][:top]


def warm(
    conn: Any,
    embedder: Any,
    llm: Any,
    top: int = WARM_TOP,
    facets: tuple[str, ...] = WARM_FACETS,
    minutes: float = WARM_MINUTES,
    log: Any = print,
    dry_run: bool = False,
) -> dict[str, int]:
    """Выписки заранее: сперва то, о чём уже спрашивали (чаще — раньше; читаются только
    новые статьи), затем top самых упоминаемых предметов с каждым видом из facets.
    Время вышло — остальное в следующий раз."""
    init(conn)
    todo = _warm_todo(conn, top, facets)
    stats = Counter(subjects=0, read=0, cached=0, items=0, skipped=0, empty=0)
    deadline = time.monotonic() + minutes * 60
    for name, wanted in _once(todo):
        label = name + (f" — {', '.join(sorted(wanted))}" if wanted else "")
        if dry_run:
            log(f"  {label}")
            stats["subjects"] += 1
            continue
        if time.monotonic() > deadline:
            stats["skipped"] += 1
            continue
        result = _outcome(collect(conn, embedder, llm, name, wanted=wanted, per=DEEP_CHUNKS))
        stats["subjects"] += 1
        if result is None:
            stats["empty"] += 1
            log(f"  {label}: не упомянут")
            continue
        fresh = result.read - result.cached
        stats["read"] += fresh
        stats["cached"] += result.cached
        stats["items"] += result.items
        log(
            f"  {label}: статей {result.articles}, прочитано {fresh}, готовых {result.cached}, "
            f"пунктов {result.items}, {result.seconds:.0f} с"
        )
    return dict(stats)


def _warm_todo(conn: Any, top: int, facets: tuple[str, ...]) -> list[tuple[str, set[str]]]:
    """Что заготовить — предмет и виды пунктов: сперва о чём спрашивали (чаще — раньше),
    затем top самых упоминаемых предметов с каждым видом из facets (метод — только общая)."""
    with conn.cursor() as cur:
        cur.execute("SELECT name, facet FROM collect_asked ORDER BY times DESC, asked_at DESC")
        todo = [(name, _facet_kinds(f)) for name, f in cur.fetchall()]
    if top:
        todo += [
            (name, _facet_kinds(f))
            for name, _, method in frequent(conn, top)
            for f in (("",) if method else facets)
        ]
    return todo


def _facet_kinds(text: str) -> set[str]:
    """Виды пунктов из строки через запятую (как в collect_asked); пустая — все виды."""
    return set(text.split(",")) - {""}


def _once(todo: list[tuple[str, set[str]]]) -> Iterator[tuple[str, set[str]]]:
    """Предметы без повторов: тот же предмет (по name_key) с теми же видами — один раз."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for name, wanted in todo:
        mark = (name_key(name), tuple(sorted(wanted)))
        if mark not in seen:
            seen.add(mark)
            yield name, wanted


def _outcome(events: Generator[dict[str, Any], None, Collected | None]) -> Collected | None:
    """Итог сбора по предмету без его событий: генератор — до конца."""
    while True:
        try:
            next(events)
        except StopIteration as stop:
            result: Collected | None = stop.value
            return result


def main(argv: list[str] | None = None) -> int:
    from ..common import add_db_args, add_embedder_args, add_llm_args
    from ..index import db
    from ..index.embed import build_embedder
    from ..llm import LLM

    parser = argparse.ArgumentParser(description="Выписки по предметам заранее (сбор по предмету)")
    parser.add_argument("--top", type=int, default=WARM_TOP, help="самых упоминаемых предметов")
    parser.add_argument(
        "--facets",
        default=",".join(f or "все" for f in WARM_FACETS),
        help="виды пунктов через запятую: все, минерал, порода, процесс, структура, метод",
    )
    parser.add_argument("--minutes", type=float, default=WARM_MINUTES, help="сколько читать, мин")
    parser.add_argument("--dry-run", action="store_true", help="только список, без модели")
    parser.add_argument("--timeout", type=int, default=300, help="сколько ждать модель, с")
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    args = parser.parse_args(argv)
    facets = tuple("" if f.strip() in ("все", "") else f.strip() for f in args.facets.split(","))
    unknown = [f for f in facets if f and f not in KIND_WORDS]
    if unknown:
        print(f"Нет такого вида: {', '.join(unknown)}", file=sys.stderr)
        return 1
    llm = LLM(model=args.model, host=args.llm_host, timeout=args.timeout)
    status = {"ok": True} if args.dry_run else llm.status()
    if not status["ok"]:
        print(f"{status['error']} Выписки не готовлю.", file=sys.stderr)
        return 1
    embedder = None if args.dry_run else build_embedder(device=args.device)
    with db.connect(args.dsn) as conn:
        print(
            "Заготовка выписок: "
            + ("что будет прочитано" if args.dry_run else f"не дольше {args.minutes:.0f} мин")
        )
        stats = warm(
            conn,
            embedder,
            llm,
            args.top,
            facets,
            args.minutes,
            log=lambda line: print(line, flush=True),  # в лог ночного прогона — сразу
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        print(
            f"Предметов: {stats['subjects']}; статей прочитано {stats['read']}, уже было "
            f"{stats['cached']}; пунктов {stats['items']}"
            + (f"; не успел: {stats['skipped']}" if stats["skipped"] else "")
        )
    return 0


def mentioned(name: str, text: str) -> bool:  # для проверок: то же сравнение, что у графа
    return mentions(name, text)


if __name__ == "__main__":
    raise SystemExit(main())
