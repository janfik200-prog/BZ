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
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    found_by: str = ""           # вектор | текст | оба
    vec_rank: int | None = None
    fts_rank: int | None = None
    # Косинусная близость к запросу, от 0 до 1. У BGE-M3 текст про то же самое
    # даёт примерно 0.6 и выше, случайный сосед — около 0.3-0.45. По ней и
    # отсекается «ближайшее, но не по делу».
    similarity: float | None = None

    def citation(self) -> str:
        pages = f", с. {'–'.join(str(p) for p in (self.pages[:1] + self.pages[-1:]))}" if self.pages else ""
        year = f", {self.year}" if self.year else ""
        return f"{self.title[:80]}{year}{pages} — {self.url}"


SELECT_FIELDS = """
    c.id, c.doc_id, c.ord, c.text, c.headings, c.pages,
    d.title, d.year, d.journal, d.url, d.authors
"""


def _filters(year_from: int | None, source: str | None) -> tuple[str, dict]:
    where, params = [], {}
    if year_from:
        where.append("d.year >= %(year_from)s")
        params["year_from"] = year_from
    if source:
        where.append("d.source = %(source)s")
        params["source"] = source
    return (" AND " + " AND ".join(where) if where else ""), params


def vector_search(conn, vector: list[float], limit: int, year_from=None, source=None) -> list[tuple]:
    from .db import vector_literal

    clause, params = _filters(year_from, source)
    params.update({"q": vector_literal(vector), "limit": limit})
    with conn.cursor() as cur:
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
        return cur.fetchall()


def text_search(conn, query: str, limit: int, year_from=None, source=None) -> list[tuple]:
    clause, params = _filters(year_from, source)
    params.update({"q": query, "limit": limit})
    with conn.cursor() as cur:
        # websearch_to_tsquery понимает кавычки и минус, как строка поиска в браузере,
        # и не падает на произвольном тексте — в отличие от to_tsquery.
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS},
                   ts_rank_cd(c.tsv, websearch_to_tsquery('russian', %(q)s)) AS score
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ websearch_to_tsquery('russian', %(q)s) {clause}
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return cur.fetchall()


def _to_hit(row: tuple) -> Hit:
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


def rrf_merge(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Места в списках → общая оценка. Вход: списки id в порядке релевантности."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


def hybrid_search(
    conn,
    embedder,
    query: str,
    limit: int = 10,
    candidates: int = 50,
    rrf_k: int = RRF_K,
    year_from: int | None = None,
    source: str | None = None,
    max_per_doc: int = 2,
    min_similarity: float = 0.0,
    require_words: bool = False,
) -> list[Hit]:
    vector = embedder.encode_one(query)
    vec_rows = vector_search(conn, vector, candidates, year_from, source)
    fts_rows = text_search(conn, query, candidates, year_from, source)

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

    for chunk_id, score in scores.items():
        hit = hits[chunk_id]
        hit.score = score
        hit.found_by = (
            "оба" if hit.vec_rank and hit.fts_rank else ("вектор" if hit.vec_rank else "текст")
        )

    kept = [h for h in hits.values() if _passes(h, min_similarity, require_words)]
    ordered = sorted(kept, key=lambda h: h.score, reverse=True)
    return cap_per_doc(ordered, max_per_doc)[:limit]


def _passes(hit: Hit, min_similarity: float, require_words: bool) -> bool:
    """Достаточно ли доказательств, что кусок относится к запросу.

    Совпадение по словам — само по себе доказательство: полнотекстовый поиск
    Postgres требует, чтобы в тексте нашлись все слова запроса. Поэтому такой
    кусок проходит порог близости без разговоров. А найденный только вектором
    обязан быть действительно близким, иначе это просто наименее непохожее.
    """
    if require_words:
        return hit.fts_rank is not None
    if hit.fts_rank is not None:
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
