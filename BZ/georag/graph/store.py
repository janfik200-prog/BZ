"""Хранилище графа: факты, которые Qwen3 выписала из фрагментов статей.

Факт — «от — связь — к» с дословной цитатой из фрагмента. Имена хранятся
как написала модель (плюс ключ для сравнения без падежей), а сводятся к
одному имени словарём синонимов при чтении: словарь можно править, не
прогоняя модель заново.

Таблицы:

* facts      — факты: фрагмент, статья, от, связь, к, цитата, модель;
* facts_pass — какие фрагменты модель уже прошла и что код отбросил.

Обе ссылаются на chunks с ON DELETE CASCADE: статью убрали из базы — её
факты ушли вместе с фрагментами. Переиндексация фрагменты не удаляет, а
обновляет на месте (index/db.py, replace_chunks): факты остаются, стираются
только у фрагментов с изменившимся текстом — их модель пройдёт заново.
"""

from __future__ import annotations

from collections import Counter

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id        bigserial PRIMARY KEY,
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id    text   NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    src       text   NOT NULL,
    relation  text   NOT NULL,
    dst       text   NOT NULL,
    src_key   text   NOT NULL,
    dst_key   text   NOT NULL,
    quote     text   NOT NULL,
    model     text,
    made_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (chunk_id, src_key, relation, dst_key)
);
CREATE INDEX IF NOT EXISTS facts_src_key ON facts (src_key);
CREATE INDEX IF NOT EXISTS facts_dst_key ON facts (dst_key);
CREATE INDEX IF NOT EXISTS facts_doc ON facts (doc_id);

CREATE TABLE IF NOT EXISTS facts_pass (
    chunk_id  bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model     text,
    facts     int NOT NULL DEFAULT 0,
    rejected  text[] NOT NULL DEFAULT '{}',
    done_at   timestamptz NOT NULL DEFAULT now()
);
"""


def init(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def todo(conn, limit: int | None = None) -> list[tuple]:
    """Фрагменты, которых модель ещё не видела: (chunk_id, doc_id, текст)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.text FROM chunks c
            WHERE NOT EXISTS (SELECT 1 FROM facts_pass p WHERE p.chunk_id = c.id)
            ORDER BY c.doc_id, c.ord
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def save(conn, chunk_id: int, doc_id: str, facts: list, rejected: list[str], model: str) -> None:
    """Результат модели по фрагменту — целиком вместо прошлого."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts WHERE chunk_id = %s", (chunk_id,))
        if facts:
            cur.executemany(
                """
                INSERT INTO facts (chunk_id, doc_id, src, relation, dst, src_key, dst_key,
                                   quote, model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                """,
                [(chunk_id, doc_id, f.src, f.relation, f.dst, f.src_key, f.dst_key, f.quote,
                  model) for f in facts],
            )
        cur.execute(
            """
            INSERT INTO facts_pass (chunk_id, model, facts, rejected) VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET model = EXCLUDED.model, facts = EXCLUDED.facts,
                rejected = EXCLUDED.rejected, done_at = now()
            """,
            (chunk_id, model, len(facts), list(rejected)[:40]),
        )


def clear(conn) -> None:
    """Забыть всё, что сделала модель, — перед проходом заново (правила поменялись)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts")
        cur.execute("DELETE FROM facts_pass")


def all_facts(conn) -> list[dict]:
    """Все факты со статьёй: из них в памяти собирается граф (фактов — тысячи)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote,
                   d.title, d.year, d.url, d.doi
            FROM facts f JOIN documents d ON d.doc_id = f.doc_id
            ORDER BY f.id
            """
        )
        return [{"id": r[0], "chunk_id": r[1], "doc_id": r[2], "src": r[3], "relation": r[4],
                 "dst": r[5], "quote": r[6], "title": r[7] or r[2], "year": r[8],
                 "url": r[9] or "", "doi": r[10]} for r in cur.fetchall()]


def summary(conn) -> dict:
    """Сколько фрагментов прошла модель, сколько фактов, когда последний раз."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), coalesce(sum(facts), 0), "
                    "coalesce(sum(cardinality(rejected)), 0), max(done_at) FROM facts_pass")
        passed, facts, rejected, last = cur.fetchone() or (0, 0, 0, None)
        cur.execute("SELECT count(*) FROM chunks")
        chunks = (cur.fetchone() or (0,))[0]
    return {"passed": int(passed or 0), "facts": int(facts or 0),
            "rejected": int(rejected or 0), "chunks": int(chunks or 0),
            "as_of": last.isoformat() if hasattr(last, "isoformat") else last}


def rejected_reasons(conn) -> Counter:
    """Почему код отбрасывал ответы модели: причина → сколько раз."""
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(rejected) FROM facts_pass")
        reasons: Counter = Counter()
        for (text,) in cur.fetchall():
            reasons[str(text).rsplit(": ", 1)[-1]] += 1
    return reasons


def fragments(conn, chunk_ids: list[int]) -> dict[int, dict]:
    """Текст фрагментов — для ответа чат-бота по фактам."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.ord, c.text, c.headings, c.pages, d.title, d.year, d.url,
                   d.authors, d.journal
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.id = ANY(%s)
            """,
            (list(chunk_ids),),
        )
        return {r[0]: {"chunk_id": r[0], "doc_id": r[1], "ord": r[2],
                       "text": " ".join((r[3] or "").split()), "headings": list(r[4] or [])[-2:],
                       "pages": list(r[5] or []), "title": r[6] or r[1], "year": r[7],
                       "url": r[8] or "", "authors": list(r[9] or [])[:3], "journal": r[10]}
                for r in cur.fetchall()}
