"""Векторы фактов: поиск по связям графа вместе с поиском по фрагментам.

Фрагмент — около 2000 знаков, и его вектор — смесь всего, о чём он: утверждение
«золото мобилизуется вместе с гематитом» в нём растворено. Вектор факта — одно
утверждение, и вопрос ровно о нём находит его ближе. Пространство то же (BGE-M3),
векторы фрагментов не меняются: найденный факт приводит свой фрагмент, и в поиске
(index/search.py, facts=True) это третья выдача рядом с векторной и полнотекстовой.

Векторы — только у подтверждённых фактов (votes ≥ store.MIN_VOTES): иначе факт
приводил бы фрагмент по утверждению, которого в нём, может быть, нет.

Таблица fact_vectors: факт, его фрагмент, текст, по которому считан вектор, и вектор.
Пересчитывается только новое: текст факта поменялся (правка связи) — вектор заново;
факт удалён (вместе с фрагментом — каскадом) или больше не подтверждён — вектор уходит.
"""

from __future__ import annotations

from typing import Any

from . import store

BATCH = 256  # сколько фактов считать за раз

SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_vectors (
    fact_id   bigint PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    text      text NOT NULL,
    embedding vector(%(dim)s) NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_vectors_hnsw ON fact_vectors
    USING hnsw (embedding vector_cosine_ops);
"""


def fact_text(src: str, relation: str, dst: str, quote: str) -> str:
    """Текст факта для вектора: связь и цитата, из которой она выписана."""
    return f"{src} — {relation} — {dst}. {' '.join(quote.split())}"


def init(conn: Any, dim: int = 1024) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA % {"dim": dim})
    conn.commit()


def update(conn: Any, embedder: Any, redo: bool = False, log: Any = print) -> dict[str, int]:
    """Векторы новых и изменившихся подтверждённых фактов; лишние — вон.

    redo — всё заново (поменялся текст для вектора или модель эмбеддингов)."""
    from ..index.db import vector_literal

    init(conn, len(embedder.encode_one("проверка")))
    with conn.cursor() as cur:
        if redo:
            cur.execute("DELETE FROM fact_vectors")
        cur.execute(
            "SELECT id, chunk_id, src, relation, dst, quote FROM facts WHERE votes >= %s",
            (store.MIN_VOTES,),
        )
        wanted = {r[0]: (r[1], fact_text(*r[2:])) for r in cur.fetchall()}
        cur.execute("SELECT fact_id, text FROM fact_vectors")
        have = dict(cur.fetchall())
        gone = [i for i in have if i not in wanted]
        if gone:
            cur.execute("DELETE FROM fact_vectors WHERE fact_id = ANY(%s)", (gone,))
    conn.commit()
    todo = [i for i, (_, text) in wanted.items() if have.get(i) != text]
    stats = {"facts": len(wanted), "new": len(todo), "removed": len(gone)}
    for start in range(0, len(todo), BATCH):
        ids = todo[start : start + BATCH]
        vectors = embedder.encode([wanted[i][1] for i in ids])
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO fact_vectors (fact_id, chunk_id, text, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT (fact_id) DO UPDATE SET text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (i, wanted[i][0], wanted[i][1], vector_literal(v))
                    for i, v in zip(ids, vectors, strict=True)
                ],
            )
        conn.commit()
        if log:
            log(f"  векторов фактов посчитано {min(start + BATCH, len(todo))} из {len(todo)}")
    return stats
