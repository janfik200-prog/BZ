"""Поиск фрагментов по порогу близости: запросы разбора → гибридный поиск → слияние.

Так отбирается, когда модель молчит; merged_search даёт и кандидатов для модели
(retrieval). К фрагменту, оборванному на полуслове или начатому с середины мысли,
добавляется соседний кусок того же раздела (add_neighbors)."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from ..index import db
from ..index.search import FACT_MIN_SIMILARITY, RRF_K, hybrid_search
from .plan import Part, Plan
from .settings import Settings

CANDIDATES = 12  # сколько брать из каждого запроса до слияния
PART_MIN = 2  # гарантированных мест у каждой части
# Сколько поисков одного вопроса идут одновременно (каждый в своём соединении к базе).
SEARCH_WORKERS = 4

# Порог близости — как «отсечь непохожее» во вкладке поиска. Совпадение по
# словам запроса проходит всегда: оно само по себе доказательство.
MIN_SIMILARITY = 0.45

# Фрагмент — кусок раздела, и мысль бывает начата в предыдущем куске («Однако в
# другом районе…», «Таким образом…»), а закончена в следующем. По базе 13.10.2026
# (4272 фрагмента): 8 % начинаются посреди фразы, 4 % — со ссылки назад, 21 %
# кончаются не на конце фразы, 24 % короче 900 знаков. К такому фрагменту
# добавляется соседний кусок того же раздела — целыми предложениями, под тем же
# номером: ссылка [n] по-прежнему ведёт на ту же статью.
SHORT_FRAGMENT = 900  # короткий (заголовок, подпись, обрывок) — продолжение до NEIGHBOR_CHARS
NEIGHBOR_CHARS = 1200
PREV_CHARS = 600  # начало мысли — из конца предыдущего куска
TAIL_CHARS = 500  # конец оборванной фразы — из начала следующего
_BACK_REF = re.compile(
    r"(?:Однако|При этом|Таким образом|Поэтому|Следовательно|Кроме того|Вместе с тем|"
    r"Тем не менее|В то же время|Это|Эти|Этот|Эта|Такие|Такой|Такая|Здесь|However|Thus|"
    r"Therefore|Hence|Moreover|In addition|This|These|Such)\b"
)
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def starts_mid_thought(text: str) -> bool:
    """Начат с середины фразы (с маленькой буквы) или со ссылки на сказанное раньше."""
    head = text.lstrip()
    return bool(head) and (head[0].islower() or bool(_BACK_REF.match(head)))


def ends_mid_thought(text: str) -> bool:
    """Оборван: не на конце фразы (двоеточие перед списком — тоже обрыв). Таблица — нет."""
    tail = text.rstrip().rstrip("»\"')]")
    return bool(tail) and not text.rstrip().endswith("|") and tail[-1] not in ".!?…"


def _head(text: str, limit: int, sentences: int = 99) -> str:
    """Начало текста целыми предложениями, не длиннее limit (первое — как влезет)."""
    out = ""
    for part in _SENTENCE.split(" ".join(text.split()))[:sentences]:
        if out and len(out) + 1 + len(part) > limit:
            break
        out = f"{out} {part}" if out else part
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + " …"


def _tail(text: str, limit: int) -> str:
    """Конец текста целыми предложениями, не длиннее limit (последнее — как влезет)."""
    out = ""
    for part in reversed(_SENTENCE.split(" ".join(text.split()))):
        if out and len(out) + 1 + len(part) > limit:
            break
        out = f"{part} {out}" if out else part
    return out if len(out) <= limit else "… " + out[-limit:].split(" ", 1)[-1]


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


def search_all(
    conn: Any, embedder: Any, queries: list[str], facts: bool = False, doc: str | None = None
) -> list[list[Any]]:
    """Несколько запросов разом. Векторы запросов — одной пачкой (12 запросов: 36 мс
    против 172 по одному), поиски в базе — одновременно в нескольких соединениях
    (70 мс против 240): на вопрос из нескольких частей это около 0,35 с.
    Заглушки в проверках — по одному, как раньше. facts — ещё и поиск по фактам
    графа (index.search: фрагменты, которые привели близкие факты); только в
    настоящей базе. doc — только во фрагментах этой статьи (Plan.article)."""
    more: dict[str, Any] = {"facts": True} if facts and db.is_real(conn) else {}
    if doc:
        more["doc_id"] = doc
    if len(queries) < 2 or not db.is_real(conn) or not hasattr(embedder, "encode"):
        return [
            hybrid_search(conn, embedder, q, limit=CANDIDATES, max_per_doc=0, **more)
            for q in queries
        ]
    vectors = embedder.encode(queries)

    def one(pair: tuple[str, Any]) -> list[Any]:
        query, vector = pair
        with db.sibling(conn) as other:
            return hybrid_search(
                other, embedder, query, limit=CANDIDATES, max_per_doc=0, vector=vector, **more
            )

    with ThreadPoolExecutor(min(SEARCH_WORKERS, len(queries))) as pool:
        return list(pool.map(one, zip(queries, vectors, strict=True)))


# Статья найдена, а фрагмент с ответом — нет: из 17 проваленных вопросов об одном факте
# (вопросы из графа, 15.10.2026) у 8 статья была среди найденного, а нужного куска не было.
# Тогда из найденных статей берутся ещё по SIBLINGS кусков, ближайших к вопросу.
SIBLINGS = 2
SIBLING_DOCS = 6

# Внутри статьи название объекта часто есть лишь в начале раздела. Здесь нужны
# слова самого факта: «замещает хлорит», а не повтор названия месторождения.
_ARTICLE_STOP = {
    "какой", "какая", "какие", "каких", "каким", "каког", "какую", "каком", "каков",
    "ответ", "кратк", "подро", "рудоп", "место", "район", "участ", "минер", "стать",
    "сведе", "извес", "найди", "покаж", "переч", "собер",
}  # fmt: skip


def article_query(question: str) -> str:
    """Термины факта без имён территории и общих слов вопроса."""
    from .dataset import proper_stems

    names = proper_stems(question)
    words = re.findall(r"[^\W\d_][\w-]*", question.lower())
    terms = [w for w in words if len(w) >= 4 and w[:5] not in _ARTICLE_STOP | names]
    return " ".join(dict.fromkeys(terms)) or question


def same_article(
    conn: Any, embedder: Any, question: str, docs: list[str], known: set[int]
) -> list[Any]:
    """Дополнительные куски найденных статей: по словам факта и по вектору.

    По два результата каждого способа из статьи; известные куски не повторяются.
    Предыдущий кусок даёт местный контекст, который не обязан повторяться в факте.
    """
    if not docs or not db.is_real(conn):
        return []
    from ..index.db import vector_literal
    from ..index.search import SELECT_FIELDS, _to_hit, build_tsquery

    literal = vector_literal(embedder.encode_one(question))
    params = {
        "q": literal,
        "docs": docs,
        "known": list(known),
        "n": SIBLINGS,
        "terms": build_tsquery(article_query(question), any_word=True),
    }
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT {SELECT_FIELDS},
                       ts_rank_cd(c.tsv, to_tsquery('russian', %(terms)s)) AS score,
                       row_number() OVER (
                           PARTITION BY c.doc_id
                           ORDER BY ts_rank_cd(c.tsv, to_tsquery('russian', %(terms)s)) DESC, c.id
                       ) AS place
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = ANY(%(docs)s) AND NOT (c.id = ANY(%(known)s))
                  AND c.tsv @@ to_tsquery('russian', %(terms)s)
            ) ranked WHERE place <= %(n)s ORDER BY place, score DESC
            """,
            params,
        )
        lexical = cur.fetchall()
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT {SELECT_FIELDS}, 1 - (c.embedding <=> %(q)s::vector) AS score,
                       row_number() OVER (
                           PARTITION BY c.doc_id ORDER BY c.embedding <=> %(q)s::vector
                       ) AS place
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = ANY(%(docs)s) AND NOT (c.id = ANY(%(known)s))
            ) ranked WHERE place <= %(n)s
            """,
            params,
        )
        rows = cur.fetchall()
    hits = []
    seen = set(known)
    for row in lexical + rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        hit = _to_hit(row)
        if row in rows:
            hit.similarity = round(float(row[11]), 4)
        hit.found_by = "та же статья"
        with conn.cursor() as cur:
            cur.execute(
                "SELECT text,pages FROM chunks WHERE doc_id=%s AND ord=%s",
                (hit.doc_id, hit.ord - 1),
            )
            previous = cur.fetchone()
        if previous:
            hit.text = " ".join(previous[0].split())[-600:] + "\n" + hit.text
            hit.pages = sorted(set(list(previous[1] or []) + hit.pages))
        hits.append(hit)
    return hits


def merged_search(
    conn: Any,
    embedder: Any,
    queries: list[str],
    cache: dict[str, Any] | None = None,
    facts: bool = False,
    doc: str | None = None,
) -> list[Any]:
    """Все запросы → одна выдача. Сливается по местам в выдачах (RRF), как
    вектор и слова внутри одного поиска: фрагмент, найденный несколькими
    запросами, поднимается выше. Порог здесь не ставится — его ставит
    find_sources. cache — чтобы один запрос не искать дважды (общая выдача
    и выдачи частей вопроса строятся из одних и тех же запросов)."""
    hits: dict[int, Any] = {}
    scores: dict[int, float] = {}
    todo = [q for q in dict.fromkeys(queries) if cache is None or q not in cache]
    fresh = dict(zip(todo, search_all(conn, embedder, todo, facts, doc), strict=True))
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
            if (hit.fact_similarity or 0) > (known.fact_similarity or 0):
                known.fact_similarity, known.fact = hit.fact_similarity, hit.fact
                if "связь" not in known.found_by:
                    known.found_by = f"{known.found_by} + связь" if known.found_by else "связь"
    ordered = sorted(hits.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in ordered:
        hit.score = scores[hit.chunk_id]
    return ordered


def is_close(hit: Any) -> bool:
    """Близок ли фрагмент: все слова запроса в тексте, похожесть от порога или его
    привёл близкий факт графа (выписан из этого фрагмента, подтверждён вопросами)."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return (
        strict
        or (hit.similarity or 0.0) >= MIN_SIMILARITY
        or (getattr(hit, "fact_similarity", None) or 0.0) >= FACT_MIN_SIMILARITY
    )


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
            asked = part.queries or [part.question]
            for hit in merged_search(
                conn, embedder, asked, cache, settings.fact_search, plan.article
            ):
                if got >= quota or len(chosen) >= top_k:
                    break
                if is_close(hit) and take(hit, i):
                    got += 1
            part.found = got
    merged = merged_search(conn, embedder, plan.queries, cache, settings.fact_search, plan.article)
    ranked = [h for h in merged if is_close(h)]
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
    """Соседний кусок той же статьи и того же раздела — где без него мысль неполна.

    Начат с середины мысли — спереди конец предыдущего куска (до PREV_CHARS):
    там начало фразы, оговорка или то, на что ссылается «Таким образом».
    Короткий — сзади продолжение (до NEIGHBOR_CHARS); оборван на полуслове —
    сзади конец фразы из следующего (до TAIL_CHARS). Кусок, который и так
    среди фрагментов ответа, не повторяется; таблица в соседи не берётся — её строки
    вместо начала фразы только занимают окно модели. Номер у добавленного — тот же."""
    taken = {(s["doc_id"], s["ord"]) for s in sources}
    Key = tuple[str, int] | None
    plan: list[tuple[dict[str, Any], Key, Key, bool]] = []
    for s in sources:
        if s.get("kind") in ("датасет", "заметка"):
            continue
        text = s["text"]
        prev = (s["doc_id"], s["ord"] - 1) if starts_mid_thought(text) else None
        short = len(text) < SHORT_FRAGMENT
        nxt = (s["doc_id"], s["ord"] + 1) if short or ends_mid_thought(text) else None
        prev = None if prev in taken else prev
        nxt = None if nxt in taken else nxt
        if prev or nxt:
            plan.append((s, prev, nxt, short))
    wanted = {key for _, p, n, _ in plan for key in (p, n) if key}
    if not wanted:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, ord, text, headings, pages, has_table FROM chunks
                WHERE (doc_id, ord) IN (SELECT * FROM unnest(%s::text[], %s::int[]))
                """,
                ([d for d, _ in wanted], [o for _, o in wanted]),
            )
            rows = {(r[0], r[1]): r for r in cur.fetchall()}
    except Exception:  # noqa: BLE001 — без соседей ответ всё равно будет
        return

    def fits(s: dict[str, Any], row: Any) -> bool:
        """Сосед годится: тот же раздел и не таблица."""
        return not row[5] and list(row[3] or [])[-1:] == list(s.get("headings") or [])[-1:]

    for s, prev, nxt, short in plan:
        pages = set(s.get("pages") or [])
        before = rows.get(prev) if prev else None
        if before and fits(s, before) and (piece := _tail(before[2] or "", PREV_CHARS)):
            s["text"] = f"{piece} {s['text']}"
            s["with_prev"] = True
            pages |= set(before[4] or [])
        after = rows.get(nxt) if nxt else None
        if after and fits(s, after):
            piece = _head(
                after[2] or "", NEIGHBOR_CHARS if short else TAIL_CHARS, 99 if short else 1
            )
            if piece:
                s["text"] = f"{s['text']} {piece}"
                s["with_next"] = True
                pages |= set(after[4] or [])
        s["pages"] = sorted(pages)
