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
from typing import Any

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

-- Сколько из 4 проверок вопросами факт прошёл (confirm.py); NULL — ещё не проверяли.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS votes smallint;

CREATE TABLE IF NOT EXISTS facts_pass (
    chunk_id  bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model     text,
    facts     int NOT NULL DEFAULT 0,
    rejected  text[] NOT NULL DEFAULT '{}',
    done_at   timestamptz NOT NULL DEFAULT now()
);
"""


def init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def todo(conn: Any, limit: int | None = None) -> list[tuple[Any, ...]]:
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
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def save(
    conn: Any, chunk_id: int, doc_id: str, facts: list[Any], rejected: list[str], model: str
) -> None:
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
                [
                    (
                        chunk_id,
                        doc_id,
                        f.src,
                        f.relation,
                        f.dst,
                        f.src_key,
                        f.dst_key,
                        f.quote,
                        model,
                    )
                    for f in facts
                ],
            )
        cur.execute(
            """
            INSERT INTO facts_pass (chunk_id, model, facts, rejected) VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET model = EXCLUDED.model, facts = EXCLUDED.facts,
                rejected = EXCLUDED.rejected, done_at = now()
            """,
            (chunk_id, model, len(facts), list(rejected)[:40]),
        )


def stored_facts(conn: Any) -> list[tuple[Any, ...]]:
    """Все факты для перепроверки кодом: (id, chunk_id, от, связь, к, цитата)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_id, src, relation, dst, quote FROM facts ORDER BY id")
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def drop_fact(conn: Any, fact_id: int, chunk_id: int, reason: str) -> None:
    """Убрать факт; причина — в список отброшенного по его фрагменту (graph --new)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts WHERE id = %s", (fact_id,))
        cur.execute(
            "UPDATE facts_pass SET facts = greatest(facts - 1, 0), "
            "rejected = array_append(rejected, %s) WHERE chunk_id = %s",
            (reason, chunk_id),
        )


def set_relation(conn: Any, fact_id: int, relation: str) -> bool:
    """Новая связь факта, если такого же факта с ней во фрагменте ещё нет."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE facts f SET relation = %s WHERE f.id = %s AND NOT EXISTS (
                SELECT 1 FROM facts g WHERE g.chunk_id = f.chunk_id AND g.src_key = f.src_key
                AND g.dst_key = f.dst_key AND g.relation = %s)
            """,
            (relation, fact_id, relation),
        )
        return bool(cur.rowcount)


def unconfirmed(conn: Any, limit: int | None = None) -> list[tuple[Any, ...]]:
    """Факты, которых ещё не проверяли вопросами: (id, от, связь, к, цитата)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, src, relation, dst, quote FROM facts WHERE votes IS NULL ORDER BY id LIMIT %s",
            (limit,),
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def set_votes(conn: Any, fact_id: int, votes: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE facts SET votes = %s WHERE id = %s", (votes, fact_id))


def votes_summary(conn: Any) -> dict[int | None, int]:
    """Сколько фактов с каким числом подтверждений (None — не проверены)."""
    with conn.cursor() as cur:
        cur.execute("SELECT votes, count(*) FROM facts GROUP BY votes")
        return {r[0]: int(r[1]) for r in cur.fetchall()}


def clear(conn: Any) -> None:
    """Забыть всё, что сделала модель, — перед проходом заново (правила поменялись)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts")
        cur.execute("DELETE FROM facts_pass")


# Сколько из 4 проверок вопросами (confirm.py) должен пройти факт, чтобы попасть в граф.
MIN_VOTES = 3


def all_facts(conn: Any, min_votes: int = MIN_VOTES) -> list[dict[str, Any]]:
    """Факты со статьёй: из них в памяти собирается граф (фактов — тысячи).

    Факт, проверенный вопросами (confirm.py) и подтверждённый меньше чем min_votes
    проверками из 4, в граф не идёт, но в базе остаётся. Непроверенный — идёт."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote,
                   d.title, d.year, d.url, d.doi, f.votes
            FROM facts f JOIN documents d ON d.doc_id = f.doc_id
            WHERE f.votes IS NULL OR f.votes >= %s
            ORDER BY f.id
            """,
            (min_votes,),
        )
        return [
            {
                "id": r[0],
                "chunk_id": r[1],
                "doc_id": r[2],
                "src": r[3],
                "relation": r[4],
                "dst": r[5],
                "quote": r[6],
                "title": r[7] or r[2],
                "year": r[8],
                "url": r[9] or "",
                "doi": r[10],
                "votes": r[11] if len(r) > 11 else None,
            }
            for r in cur.fetchall()
        ]


def summary(conn: Any) -> dict[str, Any]:
    """Сколько фрагментов прошла модель, сколько фактов, когда последний раз."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), coalesce(sum(facts), 0), "
            "coalesce(sum(cardinality(rejected)), 0), max(done_at) FROM facts_pass"
        )
        passed, facts, rejected, last = cur.fetchone() or (0, 0, 0, None)
        cur.execute("SELECT count(*) FROM chunks")
        chunks = (cur.fetchone() or (0,))[0]
    return {
        "passed": int(passed or 0),
        "facts": int(facts or 0),
        "rejected": int(rejected or 0),
        "chunks": int(chunks or 0),
        "as_of": last.isoformat() if last is not None and hasattr(last, "isoformat") else last,
    }


def rejected_reasons(conn: Any) -> Counter[str]:
    """Почему код отбрасывал ответы модели: причина → сколько раз."""
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(rejected) FROM facts_pass")
        reasons: Counter[str] = Counter()
        for (text,) in cur.fetchall():
            reasons[str(text).rsplit(": ", 1)[-1]] += 1
    return reasons


def fragments(conn: Any, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
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
        return {
            r[0]: {
                "chunk_id": r[0],
                "doc_id": r[1],
                "ord": r[2],
                "text": " ".join((r[3] or "").split()),
                "headings": list(r[4] or [])[-2:],
                "pages": list(r[5] or []),
                "title": r[6] or r[1],
                "year": r[7],
                "url": r[8] or "",
                "authors": list(r[9] or [])[:3],
                "journal": r[10],
            }
            for r in cur.fetchall()
        }
