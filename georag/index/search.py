"""Гибридный поиск: вектор + полнотекст, объединение через RRF.

Зачем два поиска, а не один. Вектор понимает смысл и переводит между языками:
запрос «выделение рудных узлов» найдёт статью про ore cluster delineation. Но он
плохо держит точные строки — «Анабарский щит», «Куонамский», номер листа
Q-49-XXI. Полнотекстовый поиск наоборот: точные слова находит, смысл — нет.

RRF (Reciprocal Rank Fusion) складывает не оценки, а места в списках:

    score(d) = Σ 1 / (k + rank(d))

Оценки двух поисков несопоставимы (косинус против ts_rank), а места —
сопоставимы. k = 60 — значение из исходной статьи Cormack et al.; оно гасит
вклад хвоста, но не даёт первому месту одного списка задавить всё остальное.

Третий поиск (facts=True) — по векторам фактов графа (graph/vectors.py): факт
«от — связь — к» с цитатой — одно утверждение, и вопрос ровно о нём находит его
ближе, чем фрагмент в 2000 знаков, где оно растворено. Близкий факт приводит свой
фрагмент — сверх найденного, до FACT_SLOTS мест. С двумя другими поисками по местам
он не сливается: так он перемешивал их порядок, и выдача становилась хуже.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    ord: int
    text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    title: str = ""
    year: int | None = None
    journal: str | None = None
    url: str = ""
    authors: list[str] = field(default_factory=list)
    score: float = 0.0
    found_by: str = ""  # вектор | текст | оба, и «+ связь» — нашёлся и через факт
    vec_rank: int | None = None
    fts_rank: int | None = None
    fact_rank: int | None = None
    # Нашёлся ли по словам «строго» — все слова запроса есть в тексте. Мягкий
    # поиск (хотя бы часть слов) сам по себе ничего не доказывает.
    fts_strict: bool = False
    # Косинусная близость к запросу, от 0 до 1. У BGE-M3 текст про то же самое
    # даёт примерно 0.6 и выше, случайный сосед — около 0.3-0.45. По ней и
    # отсекается «ближайшее, но не по делу».
    similarity: float | None = None
    # Факт графа, который привёл фрагмент (лучший из его фактов), и его близость к
    # запросу.
    fact: str = ""
    fact_similarity: float | None = None

    def citation(self) -> str:
        pages = (
            f", с. {'–'.join(str(p) for p in (self.pages[:1] + self.pages[-1:]))}"
            if self.pages
            else ""
        )
        year = f", {self.year}" if self.year else ""
        return f"{self.title[:80]}{year}{pages} — {self.url}"


SELECT_FIELDS = """
    c.id, c.doc_id, c.ord, c.text, c.headings, c.pages,
    d.title, d.year, d.journal, d.url, d.authors
"""


def _filters(
    year_from: int | None, source: str | None, doc_id: str | None = None
) -> tuple[str, dict[str, Any]]:
    """doc_id — только фрагменты одной статьи (вопрос о названной статье, chat/article.py)."""
    where: list[str] = []
    params: dict[str, Any] = {}
    if year_from:
        where.append("d.year >= %(year_from)s")
        params["year_from"] = year_from
    if source:
        where.append("d.source = %(source)s")
        params["source"] = source
    if doc_id:
        where.append("c.doc_id = %(doc_id)s")
        params["doc_id"] = doc_id
    return (" AND " + " AND ".join(where) if where else ""), params


# Сколько соседей HNSW смотрит за один поиск. По умолчанию в pgvector 40 — и
# больше 40 строк он не вернёт, сколько ни проси в LIMIT: веб-поиск просил 50
# и молча получал 40. Берём с запасом от запрошенного.
EF_SEARCH_MIN = 100


def _tune_hnsw(cur: Any, limit: int, filtered: bool) -> None:
    """Настройки векторного индекса на время одного поиска (до конца транзакции)."""
    cur.execute(f"SET LOCAL hnsw.ef_search = {min(max(EF_SEARCH_MIN, 2 * int(limit)), 1000)}")
    if not filtered:
        return
    # С фильтром по году или источнику HNSW сначала берёт ближайших, потом
    # отсеивает — и может вернуть почти пусто. pgvector с 0.8 умеет добирать
    # (iterative scan). Старая версия этой настройки не знает — тогда без неё.
    cur.execute("SAVEPOINT georag_hnsw")
    try:
        cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    except Exception:  # noqa: BLE001 — pgvector старше 0.8
        cur.execute("ROLLBACK TO SAVEPOINT georag_hnsw")
    else:
        cur.execute("RELEASE SAVEPOINT georag_hnsw")


def _transaction(conn: Any) -> Any:
    """SET LOCAL действует только внутри транзакции. У соединения чат-бота
    autocommit (оно живёт, пока модель пишет ответ, и не должно держать
    транзакцию открытой минутами) — поэтому векторный поиск идёт в своей."""
    begin = getattr(conn, "transaction", None)
    return begin() if callable(begin) else nullcontext()


def vector_search(
    conn: Any,
    vector: list[float],
    limit: int,
    year_from: Any = None,
    source: Any = None,
    doc_id: Any = None,
) -> list[tuple[Any, ...]]:
    from .db import vector_literal

    clause, params = _filters(year_from, source, doc_id)
    params.update({"q": vector_literal(vector), "limit": limit})
    with _transaction(conn), conn.cursor() as cur:
        _tune_hnsw(cur, limit, bool(clause))
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS}, 1 - (c.embedding <=> %(q)s::vector) AS score
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.embedding IS NOT NULL {clause}
            ORDER BY c.embedding <=> %(q)s::vector
            LIMIT %(limit)s
            """,
            params,
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


# Русская конфигурация Postgres снимает окончания у кириллицы, а латиницу
# отдаёт английскому стеммеру — одна на оба языка. Но беглую гласную она не
# знает: «рудный узел» даёт основу «узел», а «рудного узла» — «узл», и одно по
# другому не находилось. Поэтому запрос собирается здесь: у слова с беглой
# гласной — оба варианта через «или».
TSQUERY = "to_tsquery('russian', %(q)s)"
# Кавычки и минус — синтаксис строки поиска («точная фраза», -исключить):
# такой запрос отдаётся Postgres как есть.
TSQUERY_WEB = "websearch_to_tsquery('russian', %(q)s)"

_WORD = re.compile(r"[\w\-]{3,}", re.UNICODE)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# Только окончания, где беглая гласная и бывает: узел, угол, песок, конец. Слова
# на -ом, -ем, -ов — это падежи, их не трогаем («золотом» — не «золотм»).
_FLEETING = re.compile(r"^(.*[бвгджзклмнпрстфхцчшщ])[еоё]([лкцнр])$")


# Существительное на -ом, -ов, -ем, -ев: стеммер Postgres срезает это как окончание.
# «разлом» → 'разл', а «разлома», «разломы» → 'разлом' — и запрос «Персияновский
# разлом» не находил ни одного фрагмента со словом «Персияновского разлома». Вариант
# с «-а» стеммится в ту же основу, что и косвенные падежи.
_NOUN_TAIL = re.compile(r"^[а-яё]{3,}(?:ом|ов|ем|ев)$")

# Обратно — из косвенной формы к начальной: «рудного узла» → «узел». Стеммер даёт
# «узла» основу «узл», а «узел» — «узел», и запрос в косвенной форме не находил текст
# в начальной. С окончания существительного снимается падеж; основа на согласную с
# л, к, ц, н, р — вариант со вставленной «е» и «о» (узел, песок, конец, угол).
_CASE_ENDINGS = (
    "ами",
    "ями",
    "ах",
    "ях",
    "ам",
    "ям",
    "ом",
    "ем",
    "ов",
    "ей",
    "а",
    "у",
    "ы",
    "е",
    "и",
    "ю",
)
_CLUSTER = re.compile(r"^(.+[бвгджзклмнпрстфхцчшщ])([лкцнр])$")


def _with_vowel(word: str) -> list[str]:
    """«узла» → [«узел», «узол»]: начальная форма слова с беглой гласной."""
    for ending in _CASE_ENDINGS:
        stem = word[: -len(ending)]
        if word.endswith(ending) and len(stem) >= 3:
            match = _CLUSTER.match(stem)
            if match:
                return [f"{match.group(1)}{vowel}{match.group(2)}" for vowel in "ео"]
            return []
    return []


def _variants(word: str) -> str:
    low = word.lower()
    options = [low]
    match = _FLEETING.match(low)
    if match:
        options.append(f"{match.group(1)}{match.group(2)}")
    options += [v for v in _with_vowel(low) if v not in options]
    if _NOUN_TAIL.match(low):
        options.append(low + "а")
    return f"({' | '.join(options)})" if len(options) > 1 else low


def build_tsquery(query: str, any_word: bool = False) -> str:
    """Текст запроса → выражение для to_tsquery: все слова (или хотя бы одно)."""
    words = [w for w in _TOKEN.findall(query) if w.lower() not in {"or", "and", "not"}]
    return (" | " if any_word else " & ").join(_variants(w) for w in words)


def _is_web_syntax(query: str) -> bool:
    return '"' in query or bool(re.search(r"(^|\s)-\w", query))


def text_search(
    conn: Any,
    query: str,
    limit: int,
    year_from: Any = None,
    source: Any = None,
    any_word: bool = False,
    alternatives: list[str] | None = None,
    doc_id: Any = None,
) -> list[tuple[Any, ...]]:
    """alternatives — тот же запрос с другими написаниями из словаря синонимов:
    находится фрагмент, где есть все слова хотя бы одного из вариантов."""
    clause, params = _filters(year_from, source, doc_id)
    if _is_web_syntax(query) and not any_word:
        tsq, q = TSQUERY_WEB, query
    else:
        parts = [
            p
            for p in (build_tsquery(x, any_word) for x in [query, *(alternatives or [])])
            if p.strip()
        ]
        tsq = TSQUERY
        q = " | ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else "")
    if not q.strip():
        return []
    params.update({"q": q, "limit": limit})
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS},
                   ts_rank_cd(c.tsv, {tsq}) AS score
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ {tsq} {clause}
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        found: list[tuple[Any, ...]] = cur.fetchall()
        return found


def _to_hit(row: tuple[Any, ...]) -> Hit:
    return Hit(
        chunk_id=row[0],
        doc_id=row[1],
        ord=row[2],
        text=row[3],
        headings=list(row[4] or []),
        pages=list(row[5] or []),
        title=row[6] or "",
        year=row[7],
        journal=row[8],
        url=row[9] or "",
        authors=list(row[10] or []),
    )


# Поиск по фактам (2026-10-14, .local/logs/facts-ab-20261014; поиск по тексту вопроса,
# первые 12 фрагментов). Слить его с двумя другими по местам (RRF) — хуже: на 29
# вопросах оценки ключевого 0,90 → 0,88, на 30 вопросах из графа нужных статей
# +1/−3. Отдать до двух мест фрагментам, которые привёл близкий факт, — ключевого
# 0,90 → 0,90 и 0,73 → 0,87, нужных статей +3/−0. Близость лучшего факта у вопросов
# не по теме — до 0,46, у вопросов по базе — в середине 0,61.
FACT_MIN_SIMILARITY = 0.6  # близость факта к запросу, с которой он приводит фрагмент
FACT_SLOTS = 2  # сколько мест сверх найденного у фрагментов, которые привели факты


def fact_search(
    conn: Any,
    vector: list[float],
    limit: int,
    year_from: Any = None,
    source: Any = None,
    doc_id: Any = None,
) -> list[tuple[int, float, str]]:
    """Ближайшие факты → их фрагменты: (chunk_id, близость лучшего факта, факт) в
    порядке близости, по фрагменту — один раз. Таблицы векторов фактов ещё нет
    (граф не строили) — пусто."""
    from .db import vector_literal

    clause, params = _filters(year_from, source, doc_id)
    join = (
        "JOIN chunks c ON c.id = v.chunk_id JOIN documents d ON d.doc_id = c.doc_id"
        if clause
        else ""
    )
    params.update({"q": vector_literal(vector), "limit": limit})
    try:
        with _transaction(conn), conn.cursor() as cur:
            _tune_hnsw(cur, limit, bool(clause))
            cur.execute(
                f"""
                SELECT v.chunk_id, 1 - (v.embedding <=> %(q)s::vector), v.text
                FROM fact_vectors v {join}
                WHERE true {clause}
                ORDER BY v.embedding <=> %(q)s::vector
                LIMIT %(limit)s
                """,
                params,
            )
            rows = cur.fetchall()
    except Exception as exc:
        if getattr(exc, "sqlstate", None) != "42P01":  # undefined_table — графа нет
            raise
        return []
    found: dict[int, tuple[int, float, str]] = {}
    for chunk_id, similarity, text in rows:
        found.setdefault(chunk_id, (chunk_id, float(similarity), text))
    return list(found.values())


def _chunks_by_id(conn: Any, ids: list[int], vector: list[float]) -> list[tuple[Any, ...]]:
    """Фрагменты, которые нашлись только через факты, — с их собственной близостью."""
    from .db import vector_literal

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS}, 1 - (c.embedding <=> %(q)s::vector)
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.id = ANY(%(ids)s)
            """,
            {"q": vector_literal(vector), "ids": list(ids)},
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def rrf_merge(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Места в списках → общая оценка. Вход: списки id в порядке релевантности."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


def hybrid_search(
    conn: Any,
    embedder: Any,
    query: str,
    limit: int = 10,
    candidates: int = 50,
    rrf_k: int = RRF_K,
    year_from: int | None = None,
    source: str | None = None,
    max_per_doc: int = 2,
    min_similarity: float = 0.0,
    require_words: bool = False,
    alternatives: list[str] | None = None,
    vector: list[float] | None = None,
    facts: bool = False,
    doc_id: str | None = None,
) -> list[Hit]:
    """alternatives — написания запроса из словаря синонимов (synonyms.synonym_variants):
    идут только в полнотекст, вектор и так понимает перевод. vector — уже посчитанный
    вектор запроса (search_many считает их пачкой). facts — ещё и поиск по векторам
    фактов графа: в конце выдачи, сверх limit, — до FACT_SLOTS фрагментов, которые
    привели близкие факты. doc_id — искать только во фрагментах этой статьи."""
    if vector is None:
        vector = embedder.encode_one(query)
    vec_rows = vector_search(conn, vector, candidates, year_from, source, doc_id)
    fts_rows = text_search(
        conn, query, candidates, year_from, source, alternatives=alternatives, doc_id=doc_id
    )
    strict = True
    # Полнотекст требует все слова сразу, и длинный вопрос словами целиком
    # почти никогда не находится. Тогда — мягкий заход: хотя бы часть слов.
    # Такое совпадение само по себе не доказательство, его проверяет порог близости.
    if not fts_rows and len(_WORD.findall(query)) >= 3:
        fts_rows = text_search(
            conn,
            query,
            candidates,
            year_from,
            source,
            any_word=True,
            alternatives=alternatives,
            doc_id=doc_id,
        )
        strict = False

    hits: dict[int, Hit] = {}
    for row in vec_rows + fts_rows:
        hits.setdefault(row[0], _to_hit(row))
    for row in vec_rows:
        hits[row[0]].similarity = round(float(row[11]), 4)

    vec_ids = [row[0] for row in vec_rows]
    fts_ids = [row[0] for row in fts_rows]
    scores = rrf_merge([vec_ids, fts_ids], k=rrf_k)

    for position, chunk_id in enumerate(vec_ids, start=1):
        hits[chunk_id].vec_rank = position
    for position, chunk_id in enumerate(fts_ids, start=1):
        hits[chunk_id].fts_rank = position
        hits[chunk_id].fts_strict = strict

    for chunk_id, score in scores.items():
        hit = hits[chunk_id]
        hit.score = score
        hit.found_by = _found_by(hit)

    kept = [h for h in hits.values() if _passes(h, min_similarity, require_words)]
    ordered = sorted(kept, key=lambda h: h.score, reverse=True)
    found = cap_per_doc(ordered, max_per_doc)[:limit]
    if facts and not require_words:
        found += _fact_extras(conn, vector, candidates, year_from, source, hits, found, doc_id)
    return found


def _fact_extras(
    conn: Any,
    vector: list[float],
    limit: int,
    year_from: Any,
    source: Any,
    hits: dict[int, Hit],
    found: list[Hit],
    doc_id: Any = None,
) -> list[Hit]:
    """Фрагменты, которые привели близкие факты, — до FACT_SLOTS сверх найденного.

    У найденного и так — пометка, какой близкий факт о нём (found_by «+ связь»)."""
    rows = fact_search(conn, vector, limit, year_from, source, doc_id)
    rows = [r for r in rows if r[1] >= FACT_MIN_SIMILARITY]
    taken = {h.chunk_id for h in found}
    wanted = [r for r in rows if r[0] not in taken][:FACT_SLOTS]
    missing = [chunk_id for chunk_id, _, _ in wanted if chunk_id not in hits]
    for row in _chunks_by_id(conn, missing, vector) if missing else []:
        hits[row[0]] = _to_hit(row)
        hits[row[0]].similarity = round(float(row[11]), 4)
    for position, (chunk_id, similarity, text) in enumerate(rows, start=1):
        hit = hits.get(chunk_id)
        if hit is not None:
            hit.fact_rank = position
            hit.fact_similarity = round(similarity, 4)
            hit.fact = text
            hit.found_by = _found_by(hit)
    return [hits[chunk_id] for chunk_id, _, _ in wanted if chunk_id in hits]


def _found_by(hit: Hit) -> str:
    """Кто нашёл: вектор, текст, оба — и «+ связь», если ещё и через факт графа."""
    if hit.vec_rank and hit.fts_rank:
        base = "оба"
    elif hit.vec_rank or hit.fts_rank:
        base = "вектор" if hit.vec_rank else "текст"
    else:
        return "связь"
    return f"{base} + связь" if hit.fact_rank else base


def _passes(hit: Hit, min_similarity: float, require_words: bool) -> bool:
    """Достаточно ли доказательств, что кусок относится к запросу.

    Совпадение по словам — само по себе доказательство: полнотекстовый поиск
    Postgres требует, чтобы в тексте нашлись все слова запроса. Поэтому такой
    кусок проходит порог близости без разговоров. А найденный только вектором
    обязан быть действительно близким, иначе это просто наименее непохожее.
    """
    if require_words:
        return hit.fts_rank is not None and hit.fts_strict
    if hit.fts_rank is not None and hit.fts_strict:
        return True
    return (hit.similarity or 0.0) >= min_similarity


def cap_per_doc(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    """Не больше N кусков одной статьи в выдаче.

    Без этого одна подробная статья занимает всю первую страницу: её куски
    похожи друг на друга и на запрос одинаково. Человеку нужен обзор того,
    что вообще есть в базе, а не десять абзацев одного текста.
    """
    if max_per_doc <= 0:
        return hits
    seen: dict[str, int] = {}
    kept, extra = [], []
    for hit in hits:
        count = seen.get(hit.doc_id, 0)
        if count < max_per_doc:
            kept.append(hit)
            seen[hit.doc_id] = count + 1
        else:
            extra.append(hit)
    # Лишние куски не выбрасываем совсем: если статей в базе мало, пусть
    # выдача добирается ими, но уже после всех остальных документов.
    return kept + extra
