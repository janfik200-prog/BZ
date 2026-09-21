"""Хранение графа: названия из статей, доказательства понятий, метки фрагментов.

Три слоя, от грубого к точному:

* **entities / mentions** — названия, которые правила вытащили из текста
  («…ский рудный узел», «…ское рудопроявление»). Это обзор: что вообще
  упоминается в корпусе и в каких статьях. Связей между ними здесь нет —
  «встретились в одной статье» для базы данных проекта ничего не значит.
* **tags** — какие территории и методы из словаря названы во фрагменте.
  По ним запрос отбирает фрагменты: «по Билляхской зоне», «по данным ASTER».
* **evidence** — главное: фрагмент описывает понятие из словаря. У каждой
  строки есть дословная цитата, похожесть на определение понятия, слова
  словаря, по которым фрагмент нашёлся, и решение модели, если она его
  проверяла. Это то, что уходит в базу данных обоснованием паспорта.

Связи «понятие проявлено на территории» и «понятие изучается методом» не
хранятся отдельно: они выводятся из evidence и tags одного и того же фрагмента.
Так у каждой из них по построению есть основание — фрагмент с цитатой.
"""

from __future__ import annotations

UNCHECKED = "не проверено"
CONFIRMED = "подтверждено моделью"
REJECTED = "отклонено моделью"
VERDICTS = (UNCHECKED, CONFIRMED, REJECTED)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS entities (
    key   text PRIMARY KEY,
    name  text NOT NULL,
    kind  text NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    entity_key text NOT NULL REFERENCES entities(key) ON DELETE CASCADE,
    doc_id     text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id   bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_key, chunk_id)
);

CREATE INDEX IF NOT EXISTS mentions_doc_idx ON mentions (doc_id);
CREATE INDEX IF NOT EXISTS mentions_entity_idx ON mentions (entity_key);

-- Свободные связи, которые раньше называла модель («находится в», «содержит»),
-- убраны: каждая статья давала новые названия связи, и по такому графу нельзя
-- было сделать ни одного запроса. Теперь типы связей задаёт словарь.
DROP TABLE IF EXISTS relations;

CREATE TABLE IF NOT EXISTS tags (
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    kind      text   NOT NULL CHECK (kind IN ('территория', 'метод')),
    name      text   NOT NULL,
    PRIMARY KEY (chunk_id, kind, name)
);
CREATE INDEX IF NOT EXISTS tags_name_idx ON tags (kind, name);

CREATE TABLE IF NOT EXISTS evidence (
    id          bigserial PRIMARY KEY,
    concept     text   NOT NULL,     -- код понятия, он же kb.concept.code в БД проекта
    chunk_id    bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id      text   NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    quote       text   NOT NULL,     -- дословно из фрагмента
    similarity  real,                -- косинус фрагмента и определения понятия
    terms       text[] NOT NULL DEFAULT '{{}}',  -- слова словаря, найденные во фрагменте
    found_by    text   NOT NULL CHECK (found_by IN ('вектор', 'словарь', 'оба')),
    verdict     text   NOT NULL DEFAULT '{UNCHECKED}'
                       CHECK (verdict IN ('{UNCHECKED}', '{CONFIRMED}', '{REJECTED}')),
    model       text,
    reason      text,
    checked_at  timestamptz,
    UNIQUE (concept, chunk_id)
);
CREATE INDEX IF NOT EXISTS evidence_concept_idx ON evidence (concept, verdict);

-- Когда и каким словарём собрана разметка: это «as_of» в ответах наружу.
CREATE TABLE IF NOT EXISTS graph_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);
"""


def init(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


# --------------------------------------------------------------------------- #
#  Названия из статей (правила)
# --------------------------------------------------------------------------- #
def clear_entities(conn) -> None:
    """Названия перестраиваются с чистого листа: правила могли измениться."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM mentions")
        cur.execute("DELETE FROM entities")


def chunks_for_graph(conn) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text FROM chunks ORDER BY doc_id, ord")
        return cur.fetchall()


def save(conn, entity, doc_id: str, chunk_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities (key, name, kind) VALUES (%s, %s, %s)
            ON CONFLICT (key) DO NOTHING
            """,
            (entity.key, entity.name, entity.kind),
        )
        cur.execute(
            """
            INSERT INTO mentions (entity_key, doc_id, chunk_id) VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (entity.key, doc_id, chunk_id),
        )


def top_entities(conn, kind: str | None = None, limit: int = 200) -> list[dict]:
    """Названия по числу статей, где они встречаются."""
    where = "WHERE e.kind = %(kind)s" if kind else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT e.key, e.name, e.kind,
                   count(DISTINCT m.doc_id) AS docs, count(*) AS mentions
            FROM entities e JOIN mentions m ON m.entity_key = e.key
            {where}
            GROUP BY e.key, e.name, e.kind
            ORDER BY docs DESC, mentions DESC, e.name
            LIMIT %(limit)s
            """,
            {"kind": kind, "limit": limit},
        )
        return [
            {"key": r[0], "name": r[1], "kind": r[2], "docs": int(r[3]), "mentions": int(r[4])}
            for r in cur.fetchall()
        ]


def entity_documents(conn, key: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_id, d.title, d.year, d.journal, d.url, count(*) AS hits
            FROM mentions m JOIN documents d ON d.doc_id = m.doc_id
            WHERE m.entity_key = %s
            GROUP BY d.doc_id, d.title, d.year, d.journal, d.url
            ORDER BY hits DESC, d.year DESC NULLS LAST
            """,
            (key,),
        )
        return [
            {"doc_id": r[0], "title": r[1], "year": r[2], "journal": r[3], "url": r[4],
             "hits": int(r[5])}
            for r in cur.fetchall()
        ]


# --------------------------------------------------------------------------- #
#  Метки фрагментов: территории и методы
# --------------------------------------------------------------------------- #
def replace_tags(conn, rows: list[tuple[int, str, str]]) -> int:
    """Метки пересчитываются целиком: словарь мог измениться."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tags")
        if rows:
            cur.executemany(
                "INSERT INTO tags (chunk_id, kind, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )
    return len(rows)


# --------------------------------------------------------------------------- #
#  Доказательства понятий
# --------------------------------------------------------------------------- #
def sync_evidence(conn, candidates: list, concepts: list[str]) -> dict:
    """Записать кандидатов шага 1 так, чтобы не потерять решения модели.

    Проверка моделью — секунды на фрагмент, и перестроение разметки не должно
    её обнулять. Поэтому:

    * строки понятий, которых больше нет в словаре, удаляются;
    * найденное снова обновляется (похожесть, слова, кто нашёл), а цитата —
      только если модель ещё не ставила свою;
    * непроверенное, что больше не находится, удаляется;
    * проверенное остаётся, даже если выпало из кандидатов: решение модели
      дороже, чем сдвинутый порог.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM evidence WHERE NOT (concept = ANY(%s))", (concepts,))
        dropped_concepts = cur.rowcount

        if candidates:
            cur.executemany(
                f"""
                INSERT INTO evidence (concept, chunk_id, doc_id, quote, similarity, terms, found_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (concept, chunk_id) DO UPDATE SET
                    similarity = EXCLUDED.similarity,
                    terms      = EXCLUDED.terms,
                    found_by   = EXCLUDED.found_by,
                    quote      = CASE WHEN evidence.verdict = '{UNCHECKED}'
                                      THEN EXCLUDED.quote ELSE evidence.quote END
                """,
                [(c.concept, c.chunk_id, c.doc_id, c.quote, c.similarity, list(c.terms),
                  c.found_by) for c in candidates],
            )

        cur.execute(
            f"""
            DELETE FROM evidence e
            WHERE e.verdict = '{UNCHECKED}'
              AND NOT EXISTS (
                  SELECT 1 FROM unnest(%s::text[], %s::bigint[]) AS k(concept, chunk_id)
                  WHERE k.concept = e.concept AND k.chunk_id = e.chunk_id)
            """,
            ([c.concept for c in candidates], [c.chunk_id for c in candidates]),
        )
        dropped_stale = cur.rowcount
    return {"dropped_concepts": dropped_concepts, "dropped_stale": dropped_stale}


def unchecked(conn, limit: int | None = None) -> list[tuple]:
    """Что ещё не видела модель: сначала то, что нашли оба способа."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT e.id, e.concept, c.text
            FROM evidence e JOIN chunks c ON c.id = e.chunk_id
            WHERE e.verdict = '{UNCHECKED}'
            ORDER BY (e.found_by = 'оба') DESC, e.similarity DESC NULLS LAST, e.id
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def set_verdict(conn, evidence_id: int, verdict: str, quote: str | None,
                model: str, reason: str | None) -> None:
    if verdict not in VERDICTS:
        raise ValueError(verdict)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE evidence
               SET verdict = %s, quote = coalesce(%s, quote), model = %s, reason = %s,
                   checked_at = now()
             WHERE id = %s
            """,
            (verdict, quote, model, reason, evidence_id),
        )


def set_meta(conn, **values) -> None:
    with conn.cursor() as cur:
        for key, value in values.items():
            cur.execute(
                """
                INSERT INTO graph_meta (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, str(value)),
            )


def meta(conn) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM graph_meta")
        return {k: v for k, v in cur.fetchall()}


def concept_summary(conn) -> dict[str, dict]:
    """Сколько доказательств у каждого понятия и в каком они состоянии."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT concept,
                   count(*) FILTER (WHERE verdict <> '{REJECTED}'),
                   count(*) FILTER (WHERE verdict = '{CONFIRMED}'),
                   count(*) FILTER (WHERE verdict = '{UNCHECKED}'),
                   count(*) FILTER (WHERE verdict = '{REJECTED}'),
                   count(DISTINCT doc_id) FILTER (WHERE verdict <> '{REJECTED}')
            FROM evidence GROUP BY concept
            """
        )
        return {
            r[0]: {"fragments": int(r[1]), "confirmed": int(r[2]), "unchecked": int(r[3]),
                   "rejected": int(r[4]), "documents": int(r[5])}
            for r in cur.fetchall()
        }


def concept_links(conn, concept: str) -> list[dict]:
    """Территории и методы, названные в тех же фрагментах, что описывают понятие."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT t.kind, t.name, count(DISTINCT e.chunk_id), count(DISTINCT e.doc_id)
            FROM evidence e JOIN tags t ON t.chunk_id = e.chunk_id
            WHERE e.concept = %s AND e.verdict <> '{REJECTED}'
            GROUP BY t.kind, t.name
            ORDER BY count(DISTINCT e.doc_id) DESC, t.name
            """,
            (concept,),
        )
        return [
            {"kind": r[0], "name": r[1], "fragments": int(r[2]), "documents": int(r[3])}
            for r in cur.fetchall()
        ]


def evidence_for(
    conn,
    concept: str,
    territory: str | None = None,
    method: str | None = None,
    checked_only: bool = False,
    limit: int = 10,
    per_doc: int = 2,
) -> tuple[list[dict], dict]:
    """Фрагменты, описывающие понятие, по шаблону запроса, и сколько их всего.

    Второе значение — {"fragments": …, "documents": …}: сколько фрагментов и статей
    подходит под запрос до всех ограничений показа. Считать «найдено» после
    ограничения «не больше N из статьи» нельзя: тогда число расходится с тем,
    что показано у понятия, и выглядит как ошибка.

    Порядок: подтверждённое моделью, затем найденное обоими способами, затем
    по похожести. Отклонённое моделью не отдаётся никогда. Не больше `per_doc`
    фрагментов одной статьи: базе данных нужны разные публикации, а не десять
    абзацев одной.
    """
    where = ["e.concept = %(concept)s", f"e.verdict <> '{REJECTED}'"]
    if checked_only:
        where.append(f"e.verdict = '{CONFIRMED}'")
    if territory:
        where.append("EXISTS (SELECT 1 FROM tags t WHERE t.chunk_id = e.chunk_id "
                     "AND t.kind = 'территория' AND t.name = %(territory)s)")
    if method:
        where.append("EXISTS (SELECT 1 FROM tags t WHERE t.chunk_id = e.chunk_id "
                     "AND t.kind = 'метод' AND t.name = %(method)s)")

    sql = f"""
        WITH ranked AS (
            SELECT e.*,
                   row_number() OVER (
                       PARTITION BY e.doc_id
                       ORDER BY (e.verdict = '{CONFIRMED}') DESC, (e.found_by = 'оба') DESC,
                                e.similarity DESC NULLS LAST, e.id) AS in_doc
            FROM evidence e
            WHERE {' AND '.join(where)}
        ),
        totals AS (
            SELECT count(*) AS fragments, count(DISTINCT doc_id) AS documents FROM ranked
        )
        SELECT r.id, r.chunk_id, r.doc_id, c.ord, c.text, c.pages, c.headings,
               r.quote, r.similarity, r.terms, r.found_by, r.verdict, r.model,
               d.title, d.year, d.doi, d.url, d.authors, d.journal,
               coalesce((SELECT array_agg(t.name ORDER BY t.name) FROM tags t
                         WHERE t.chunk_id = r.chunk_id AND t.kind = 'территория'), '{{}}'),
               coalesce((SELECT array_agg(t.name ORDER BY t.name) FROM tags t
                         WHERE t.chunk_id = r.chunk_id AND t.kind = 'метод'), '{{}}'),
               totals.fragments, totals.documents
        FROM ranked r
        JOIN chunks c ON c.id = r.chunk_id
        JOIN documents d ON d.doc_id = r.doc_id
        CROSS JOIN totals
        WHERE %(per_doc)s <= 0 OR r.in_doc <= %(per_doc)s
        ORDER BY (r.verdict = '{CONFIRMED}') DESC, (r.found_by = 'оба') DESC,
                 r.similarity DESC NULLS LAST, r.id
        LIMIT %(limit)s
    """
    params = {"concept": concept, "territory": territory, "method": method,
              "limit": limit, "per_doc": per_doc}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    out = []
    for r in rows:
        pages = list(r[5] or [])
        out.append({
            "evidence_id": r[0], "chunk_id": r[1], "doc_id": r[2], "ord": r[3],
            "text": " ".join((r[4] or "").split()),
            "pages": pages, "page": pages[0] if pages else None,
            "headings": list(r[6] or [])[-2:],
            "quote": r[7],
            "similarity": round(float(r[8]), 4) if r[8] is not None else None,
            "terms": list(r[9] or []), "found_by": r[10],
            "verdict": r[11], "model": r[12],
            "title": r[13], "year": r[14], "doi": r[15], "url": r[16],
            "authors": list(r[17] or [])[:3], "journal": r[18],
            "territories": list(r[19] or []), "methods": list(r[20] or []),
        })
    totals = ({"fragments": int(rows[0][21]), "documents": int(rows[0][22])}
              if rows else {"fragments": 0, "documents": 0})
    return out, totals


def counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT kind, count(*) FROM entities GROUP BY kind")
        by_kind = {row[0]: int(row[1]) for row in cur.fetchall()}
        cur.execute("SELECT count(*) FROM mentions")
        mentions = int(cur.fetchone()[0])
        cur.execute(f"SELECT count(*) FILTER (WHERE verdict <> '{REJECTED}') FROM evidence")
        evidence = int(cur.fetchone()[0])
        cur.execute("SELECT count(*) FROM tags")
        tags = int(cur.fetchone()[0])
    return {
        "by_kind": by_kind,
        "entities": sum(by_kind.values()),
        "mentions": mentions,
        "evidence": evidence,
        "tags": tags,
    }


def graph_data(conn, checked_only: bool = False) -> dict:
    """Всё для картинки графа: связи понятий с территориями и методами и вес узлов.

    Вес — число статей, а не фрагментов: десять абзацев одной статьи — это одно
    свидетельство, а не десять.
    """
    clause = (f"e.verdict = '{CONFIRMED}'" if checked_only else f"e.verdict <> '{REJECTED}'")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT e.concept, t.kind, t.name,
                   count(DISTINCT e.chunk_id), count(DISTINCT e.doc_id)
            FROM evidence e JOIN tags t ON t.chunk_id = e.chunk_id
            WHERE {clause}
            GROUP BY e.concept, t.kind, t.name
            """
        )
        edges = [{"concept": r[0], "kind": r[1], "name": r[2],
                  "fragments": int(r[3]), "documents": int(r[4])} for r in cur.fetchall()]
        cur.execute(
            f"""
            SELECT e.concept, count(DISTINCT e.chunk_id), count(DISTINCT e.doc_id)
            FROM evidence e WHERE {clause} GROUP BY e.concept
            """
        )
        concepts = {r[0]: {"fragments": int(r[1]), "documents": int(r[2])}
                    for r in cur.fetchall()}
        cur.execute(
            f"""
            SELECT t.kind, t.name, count(DISTINCT e.doc_id)
            FROM evidence e JOIN tags t ON t.chunk_id = e.chunk_id
            WHERE {clause} GROUP BY t.kind, t.name
            """
        )
        terms = {(r[0], r[1]): int(r[2]) for r in cur.fetchall()}
    return {"edges": edges, "concepts": concepts, "terms": terms}


def network_data(conn, checked_only: bool = False, max_entities: int = 200) -> dict:
    """Сеть знаний «как в Obsidian»: статьи в центре, от них — всё, о чём они.

    Рёбра только двух видов, и оба проверяемы:

    * статья **описывает** понятие — есть фрагмент-доказательство с цитатой;
    * статья **упоминает** территорию, метод, объект или полезное ископаемое —
      название найдено в её тексте.

    Связей «понятие — понятие» напрямую нет: они видны как общие статьи между
    ними, ровно так, как в Obsidian две темы связаны общими заметками.
    """
    clause = (f"verdict = '{CONFIRMED}'" if checked_only else f"verdict <> '{REJECTED}'")
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, title, year, url, doi FROM documents ORDER BY year DESC NULLS LAST")
        documents = [{"doc_id": r[0], "title": r[1], "year": r[2], "url": r[3], "doi": r[4]}
                     for r in cur.fetchall()]
        cur.execute(
            f"""
            SELECT doc_id, concept, count(*), bool_or(verdict = '{CONFIRMED}')
            FROM evidence WHERE {clause} GROUP BY doc_id, concept
            """
        )
        described = [{"doc_id": r[0], "concept": r[1], "fragments": int(r[2]),
                      "confirmed": bool(r[3])} for r in cur.fetchall()]
        cur.execute(
            """
            SELECT c.doc_id, t.kind, t.name, count(*)
            FROM tags t JOIN chunks c ON c.id = t.chunk_id
            GROUP BY c.doc_id, t.kind, t.name
            """
        )
        tagged = [{"doc_id": r[0], "kind": r[1], "name": r[2], "fragments": int(r[3])}
                  for r in cur.fetchall()]
        # Названия, найденные правилами: самые частые, иначе сеть тонет в единичных.
        cur.execute(
            """
            WITH top AS (
                SELECT e.key FROM entities e JOIN mentions m ON m.entity_key = e.key
                WHERE e.kind IN ('объект', 'ископаемое')
                GROUP BY e.key ORDER BY count(DISTINCT m.doc_id) DESC, e.key LIMIT %s
            )
            SELECT m.doc_id, e.key, e.name, e.kind, count(*)
            FROM mentions m JOIN entities e ON e.key = m.entity_key
            WHERE e.key IN (SELECT key FROM top)
            GROUP BY m.doc_id, e.key, e.name, e.kind
            """,
            (max_entities,),
        )
        mentioned = [{"doc_id": r[0], "key": r[1], "name": r[2], "kind": r[3],
                      "fragments": int(r[4])} for r in cur.fetchall()]
    return {"documents": documents, "described": described, "tagged": tagged,
            "mentioned": mentioned}
