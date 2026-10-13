"""Поиск фрагментов по порогу близости: запросы разбора → гибридный поиск → слияние.

Так отбирается, когда модель молчит; merged_search даёт и кандидатов для модели
(retrieval). К короткому фрагменту добавляется продолжение из того же раздела."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ..index import db
from ..index.search import RRF_K, hybrid_search
from .plan import Part, Plan
from .settings import Settings

CANDIDATES = 12  # сколько брать из каждого запроса до слияния
PART_MIN = 2  # гарантированных мест у каждой части
# Сколько поисков одного вопроса идут одновременно (каждый в своём соединении к базе).
SEARCH_WORKERS = 4

# Порог близости — как «отсечь непохожее» во вкладке поиска. Совпадение по
# словам запроса проходит всегда: оно само по себе доказательство.
MIN_SIMILARITY = 0.45

# Короткий фрагмент (заголовок, подпись, обрывок абзаца) без соседей почти
# ничего не говорит. К такому добавляется следующий кусок той же статьи,
# если он из того же раздела. Номер у них общий.
SHORT_FRAGMENT = 900
NEIGHBOR_CHARS = 1200


def source_dict(n: int, hit: Any, part: Any = None) -> dict[str, Any]:
    return {
        "n": n,
        "doc_id": hit.doc_id,
        "chunk_id": hit.chunk_id,  # по нему под ответом — связи графа из этого фрагмента
        "ord": hit.ord,
        "title": hit.title or hit.doc_id,
        "year": hit.year,
        "journal": hit.journal,
        "authors": hit.authors[:3],
        "pages": hit.pages,
        "headings": hit.headings[-2:],
        "url": hit.url,
        "text": " ".join(hit.text.split()),
        "similarity": hit.similarity,
        "found_by": hit.found_by,
        "part": part,
    }


def search_all(conn: Any, embedder: Any, queries: list[str]) -> list[list[Any]]:
    """Несколько запросов разом. Векторы запросов — одной пачкой (12 запросов: 36 мс
    против 172 по одному), поиски в базе — одновременно в нескольких соединениях
    (70 мс против 240): на вопрос из нескольких частей это около 0,35 с.
    Заглушки в проверках — по одному, как раньше."""
    if len(queries) < 2 or not db.is_real(conn) or not hasattr(embedder, "encode"):
        return [hybrid_search(conn, embedder, q, limit=CANDIDATES, max_per_doc=0) for q in queries]
    vectors = embedder.encode(queries)

    def one(pair: tuple[str, Any]) -> list[Any]:
        query, vector = pair
        with db.sibling(conn) as other:
            return hybrid_search(
                other, embedder, query, limit=CANDIDATES, max_per_doc=0, vector=vector
            )

    with ThreadPoolExecutor(min(SEARCH_WORKERS, len(queries))) as pool:
        return list(pool.map(one, zip(queries, vectors, strict=True)))


def merged_search(
    conn: Any, embedder: Any, queries: list[str], cache: dict[str, Any] | None = None
) -> list[Any]:
    """Все запросы → одна выдача. Сливается по местам в выдачах (RRF), как
    вектор и слова внутри одного поиска: фрагмент, найденный несколькими
    запросами, поднимается выше. Порог здесь не ставится — его ставит
    find_sources. cache — чтобы один запрос не искать дважды (общая выдача
    и выдачи частей вопроса строятся из одних и тех же запросов)."""
    hits: dict[int, Any] = {}
    scores: dict[int, float] = {}
    todo = [q for q in dict.fromkeys(queries) if cache is None or q not in cache]
    fresh = dict(zip(todo, search_all(conn, embedder, todo), strict=True))
    if cache is not None:
        cache.update(fresh)
    for query in queries:
        found = cache[query] if cache is not None and query in cache else fresh[query]
        for rank, hit in enumerate(found, start=1):
            key = hit.chunk_id
            if key not in hits:
                # Копия: попадания из кэша общие для разных слияний, их не правим.
                hits[key] = replace(hit)
            known = hits[key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if hit.similarity is not None and (known.similarity or 0) < hit.similarity:
                known.similarity = hit.similarity
            if hit.fts_rank is not None and known.fts_rank is None:
                known.fts_rank = hit.fts_rank
            known.fts_strict = known.fts_strict or hit.fts_strict
    ordered = sorted(hits.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in ordered:
        hit.score = scores[hit.chunk_id]
    return ordered


def is_close(hit: Any) -> bool:
    """Близок ли фрагмент: все слова запроса в тексте или похожесть от порога."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return strict or (hit.similarity or 0.0) >= MIN_SIMILARITY


def fragment_key(item: Any) -> tuple[Any, ...]:
    """Место фрагмента в базе — (статья, номер куска): у найденного и у источника."""
    if isinstance(item, dict):
        return (item.get("doc_id"), item.get("ord"))
    return (item.doc_id, item.ord)


def find_sources(
    conn: Any,
    embedder: Any,
    queries: Any,
    settings: Settings,
    start: int = 1,
    taken: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Близкие фрагменты для ответа; дальние отбрасываются совсем.

    queries — список запросов или разбор вопроса (Plan). В разборе из
    нескольких частей у каждой части свои гарантированные места: сначала
    лучшие фрагменты каждой части, затем остаток — лучшим из общей выдачи.
    taken — фрагменты, что уже есть в ответе (из датасета): не повторяются
    и считаются в ограничение «не больше N из статьи»."""
    plan = queries if isinstance(queries, Plan) else Plan([Part("", list(queries))])
    top_k = settings.top_k if limit is None else limit
    taken = taken or []
    cache: dict[str, Any] = {}
    seen = {fragment_key(s) for s in taken}
    per_doc: dict[str, int] = {}
    for s in taken:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    chosen: list[tuple[Any, ...]] = []

    def take(hit: Any, part: Any, strict: bool = True) -> bool:
        if fragment_key(hit) in seen:
            return False
        if strict and settings.per_doc and per_doc.get(hit.doc_id, 0) >= settings.per_doc:
            return False
        seen.add(fragment_key(hit))
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        chosen.append((hit, part))
        return True

    parts = plan.parts if len(plan.parts) > 1 else []
    if parts:
        quota = max(PART_MIN, (top_k - 2) // len(parts))
        for i, part in enumerate(parts, start=1):
            got = 0
            for hit in merged_search(conn, embedder, part.queries or [part.question], cache):
                if got >= quota or len(chosen) >= top_k:
                    break
                if is_close(hit) and take(hit, i):
                    got += 1
            part.found = got
    ranked = [h for h in merged_search(conn, embedder, plan.queries, cache) if is_close(h)]
    for hit in ranked:
        if len(chosen) >= top_k:
            break
        take(hit, None)
    # Статей в базе мало — добирается лишними кусками тех же статей, но после
    # всех остальных (как в cap_per_doc). Во втором круге — нет: там нужно новое.
    if limit is None:
        for hit in ranked:
            if len(chosen) >= top_k:
                break
            take(hit, None, strict=False)
    if not parts and plan.parts:
        plan.parts[0].found = len(chosen)
    sources = [source_dict(n, h, part) for n, (h, part) in enumerate(chosen, start=start)]
    if conn is not None:
        add_neighbors(conn, sources)
    return sources


def add_neighbors(conn: Any, sources: list[dict[str, Any]]) -> None:
    """Короткому фрагменту — продолжение из той же статьи и того же раздела.

    Поиск находит самый похожий кусок, а он бывает обрывком: заголовок и две
    строки, дальше «см. ниже». Для развёрнутого ответа модели нужно то, что
    идёт следом. Добавленный текст идёт под тем же номером — ссылка [n]
    по-прежнему ведёт на ту же статью.
    """
    short = [s for s in sources if len(s["text"]) < SHORT_FRAGMENT]
    if not short:
        return
    wanted = {(s["doc_id"], s["ord"] + 1) for s in short}
    taken = {(s["doc_id"], s["ord"]) for s in sources}
    wanted -= taken
    if not wanted:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, ord, text, headings, pages FROM chunks
                WHERE (doc_id, ord) IN (SELECT * FROM unnest(%s::text[], %s::int[]))
                """,
                ([d for d, _ in wanted], [o for _, o in wanted]),
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — без соседей ответ всё равно будет
        return
    after = {(r[0], r[1]): r for r in rows}
    for s in short:
        row = after.get((s["doc_id"], s["ord"] + 1))
        if not row:
            continue
        same_section = list(row[3] or [])[-1:] == list(s.get("headings") or [])[-1:]
        if not same_section:
            continue
        extra = " ".join((row[2] or "").split())[:NEIGHBOR_CHARS]
        s["text"] = f"{s['text']} {extra}"
        s["pages"] = sorted(set(s.get("pages") or []) | set(row[4] or []))
        s["with_next"] = True
