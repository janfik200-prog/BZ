"""База: схема, подключение, запись документов и чанков.

Одна таблица на документы, одна на чанки. У чанка три способа быть найденным:

* embedding — вектор BGE-M3, косинусная близость, индекс HNSW;
* tsv — полнотекстовый поиск Postgres, индекс GIN. Он ловит то, что вектор
  пропускает: точные названия площадей, номера ГОСТов, фамилии;
* поля документа — год, источник, язык: по ним отсекают лишнее до поиска.

Вектор передаётся строкой с приведением `%s::vector`. Так не нужен бинарный
адаптер, который на Windows любит не совпасть версией с psycopg.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

VECTOR_DIM = 1024  # BGE-M3

# Порт 5433 — тот, что публикует docker-compose.yml, чтобы не спорить
# с уже установленным на машине PostgreSQL на 5432.
DEFAULT_DSN = "postgresql://georag:georag@localhost:5433/georag"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      text PRIMARY KEY,
    title       text NOT NULL DEFAULT '',
    authors     text[] NOT NULL DEFAULT '{}',
    year        int,
    journal     text,
    doi         text,
    url         text NOT NULL DEFAULT '',
    source      text NOT NULL DEFAULT '',
    language    text,
    status      text NOT NULL DEFAULT '',
    sha256      text NOT NULL DEFAULT '',
    pages       int NOT NULL DEFAULT 0,
    parser      text NOT NULL DEFAULT '',
    accuracy    real NOT NULL DEFAULT 0,
    fetched_at  text NOT NULL DEFAULT '',
    added_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord         int NOT NULL,
    text        text NOT NULL,
    headings    text[] NOT NULL DEFAULT '{}',
    pages       int[] NOT NULL DEFAULT '{}',
    n_tokens    int NOT NULL DEFAULT 0,
    has_table   boolean NOT NULL DEFAULT false,
    embedding   vector(%(dim)s),
    -- Русский словарь снимает окончания: «рудного узла» находится по «рудный узел».
    -- Латиница через него проходит почти без изменений, поэтому конфигурация одна.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED,
    UNIQUE (doc_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
"""

# HNSW строится отдельно: на пустой таблице он бесполезен, а на больших данных
# создаётся долго. Поэтому — отдельным шагом и только если его ещё нет.
HNSW = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


def dsn_from_env(default: str = DEFAULT_DSN) -> str:
    return os.environ.get("GEORAG_DSN") or default


def vector_literal(values) -> str:
    """Вектор в том виде, в каком его понимает pgvector: [0.1,0.2,...]."""
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


@contextmanager
def connect(dsn: str | None = None):
    import psycopg

    conn = psycopg.connect(dsn or dsn_from_env())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn, dim: int = VECTOR_DIM, with_hnsw: bool = True) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA % {"dim": dim})
        if with_hnsw:
            cur.execute(HNSW)
    conn.commit()


def upsert_document(conn, record: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (doc_id, title, authors, year, journal, doi, url,
                                   source, language, status, sha256, pages, parser,
                                   accuracy, fetched_at)
            VALUES (%(doc_id)s, %(title)s, %(authors)s, %(year)s, %(journal)s, %(doi)s,
                    %(url)s, %(source)s, %(language)s, %(status)s, %(sha256)s, %(pages)s,
                    %(parser)s, %(accuracy)s, %(fetched_at)s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title, authors = EXCLUDED.authors, year = EXCLUDED.year,
                journal = EXCLUDED.journal, doi = EXCLUDED.doi, url = EXCLUDED.url,
                source = EXCLUDED.source, language = EXCLUDED.language,
                status = EXCLUDED.status, sha256 = EXCLUDED.sha256, pages = EXCLUDED.pages,
                parser = EXCLUDED.parser, accuracy = EXCLUDED.accuracy,
                fetched_at = EXCLUDED.fetched_at
            """,
            record,
        )


def replace_chunks(conn, doc_id: str, rows: list[dict]) -> int:
    """Чанки документа пишутся целиком: переиндексация не должна плодить дубли."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        for row in rows:
            cur.execute(
                """
                INSERT INTO chunks (doc_id, ord, text, headings, pages, n_tokens,
                                    has_table, embedding)
                VALUES (%(doc_id)s, %(ord)s, %(text)s, %(headings)s, %(pages)s,
                        %(n_tokens)s, %(has_table)s, %(embedding)s::vector)
                """,
                row,
            )
    return len(rows)


def indexed_docs(conn) -> dict[str, int]:
    """Что уже в базе: doc_id → сколько чанков. По этому решаем, что пропустить."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_id, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id
            """
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        docs = cur.fetchone()[0]
        cur.execute("SELECT count(*), count(embedding) FROM chunks")
        chunks, embedded = cur.fetchone()
        cur.execute("SELECT min(year), max(year) FROM documents WHERE year IS NOT NULL")
        lo, hi = cur.fetchone()
    return {
        "documents": docs,
        "chunks": chunks,
        "embedded": embedded,
        "years": (lo, hi),
    }


def documents(conn, limit: int = 500) -> list[dict]:
    """Список статей в базе с числом фрагментов — для обзора того, что разобрано."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_id, d.title, d.year, d.journal, d.url, d.authors,
                   d.source, d.parser, d.pages, count(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id
            ORDER BY d.year DESC NULLS LAST, d.title
            LIMIT %s
            """,
            (limit,),
        )
        return [
            {
                "doc_id": row[0], "title": row[1], "year": row[2], "journal": row[3],
                "url": row[4], "authors": list(row[5] or []), "source": row[6],
                "parser": row[7], "pages": row[8], "chunks": int(row[9]),
            }
            for row in cur.fetchall()
        ]


def document_chunks(conn, doc_id: str) -> list[dict]:
    """Разобранная статья по порядку — то, что получилось из PDF."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.ord, c.text, c.headings, c.pages, c.n_tokens, c.has_table
            FROM chunks c WHERE c.doc_id = %s ORDER BY c.ord
            """,
            (doc_id,),
        )
        return [
            {
                "ord": row[0], "text": row[1], "headings": list(row[2] or []),
                "pages": list(row[3] or []), "n_tokens": row[4], "has_table": row[5],
            }
            for row in cur.fetchall()
        ]
