"""Хранение графа: сущности, их упоминания и связи между ними.

Связь здесь не выдумана и не взвешена вручную: два объекта связаны, если о них
написано в одной статье. Это самое скромное определение из возможных, зато
проверяемое — за каждой связью стоит список статей, который можно открыть
и прочитать. Никаких «сила связи 0.73» без объяснения, откуда она взялась.
"""

from __future__ import annotations

SCHEMA = """
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

-- Названные связи находит модель: «находится в», «приурочено к», «применён к».
-- Правила такого не дают — у них связь только «упомянуты в одной статье».
CREATE TABLE IF NOT EXISTS relations (
    from_key  text NOT NULL REFERENCES entities(key) ON DELETE CASCADE,
    to_key    text NOT NULL REFERENCES entities(key) ON DELETE CASCADE,
    type      text NOT NULL,
    doc_id    text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (from_key, to_key, type, chunk_id)
);

CREATE INDEX IF NOT EXISTS relations_from_idx ON relations (from_key);
CREATE INDEX IF NOT EXISTS relations_to_idx ON relations (to_key);
"""


def init(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def clear(conn) -> None:
    """Перестроение графа начинается с чистого листа: правила могли измениться."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM relations")
        cur.execute("DELETE FROM mentions")
        cur.execute("DELETE FROM entities")
    conn.commit()


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


def save_relation(conn, relation, doc_id: str, chunk_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO relations (from_key, to_key, type, doc_id, chunk_id)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (relation.from_key, relation.to_key, relation.type, doc_id, chunk_id),
        )


def named_relations(conn, key: str, limit: int = 20) -> list[dict]:
    """Названные связи сущности в обе стороны, с числом статей-подтверждений."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.type, e.key, e.name, e.kind, direction, count(DISTINCT r.doc_id)
            FROM (
                SELECT from_key AS mine, to_key AS other, type, doc_id, 'из' AS direction
                FROM relations WHERE from_key = %(key)s
                UNION ALL
                SELECT to_key AS mine, from_key AS other, type, doc_id, 'в' AS direction
                FROM relations WHERE to_key = %(key)s
            ) r
            JOIN entities e ON e.key = r.other
            GROUP BY r.type, e.key, e.name, e.kind, direction
            ORDER BY count(DISTINCT r.doc_id) DESC, e.name
            LIMIT %(limit)s
            """,
            {"key": key, "limit": limit},
        )
        return [
            {"type": r[0], "key": r[1], "name": r[2], "kind": r[3],
             "direction": r[4], "docs": int(r[5])}
            for r in cur.fetchall()
        ]


def top_entities(conn, kind: str | None = None, limit: int = 200) -> list[dict]:
    """Сущности по числу статей, где они встречаются."""
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


def related(conn, key: str, limit: int = 15) -> list[dict]:
    """С кем эта сущность встречается в одних и тех же статьях."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.key, e.name, e.kind, count(DISTINCT other.doc_id) AS together
            FROM mentions mine
            JOIN mentions other
              ON other.doc_id = mine.doc_id AND other.entity_key <> mine.entity_key
            JOIN entities e ON e.key = other.entity_key
            WHERE mine.entity_key = %s
            GROUP BY e.key, e.name, e.kind
            ORDER BY together DESC, e.name
            LIMIT %s
            """,
            (key, limit),
        )
        return [
            {"key": r[0], "name": r[1], "kind": r[2], "together": int(r[3])}
            for r in cur.fetchall()
        ]


def counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT kind, count(*) FROM entities GROUP BY kind")
        by_kind = {row[0]: int(row[1]) for row in cur.fetchall()}
        cur.execute("SELECT count(*) FROM mentions")
        mentions = int(cur.fetchone()[0])
        cur.execute("SELECT count(*) FROM relations")
        relations = int(cur.fetchone()[0])
    return {
        "by_kind": by_kind,
        "entities": sum(by_kind.values()),
        "mentions": mentions,
        "relations": relations,
    }
