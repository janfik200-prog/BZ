"""Кластеры графа — темы по алгоритму Лейдена (план графа v2, А7).

Большой граф целиком на карте не показать: тысячи узлов — каша. Лейден разбивает
его на сообщества — предметы, связанные между собой теснее, чем с остальными; на
странице граф открывается по одному кластеру.

Узел — предмет после словаря синонимов и автосинонимов («Donetsk basin» и «Донецкий
бассейн» — один); вес ребра — в скольких статьях связаны. Берутся факты, что идут в
граф (store.accepted), без помеченных «не о геологии». Связи считаются в базе, а не
в памяти Python: на миллионах фактов это одна группировка SQL.

Кластер меньше MIN_SIZE предметов не сохраняется — это обрывки. Подпись кластера —
TITLE_NAMES самых связанных его предметов. Пересчёт — после graph (как names и
vectors) или командой graph --clusters; разбиение повторяемо (seed).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..text import name_key
from . import store

MIN_SIZE = 3
TITLE_NAMES = 3
SEED = 20261027

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_clusters (
    id        int PRIMARY KEY,
    title     text NOT NULL,
    size      int NOT NULL,
    made_at   timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS entity_clusters (
    key       text PRIMARY KEY,
    cluster   int NOT NULL REFERENCES graph_clusters(id) ON DELETE CASCADE
);
"""


def init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def _edges(conn: Any) -> list[tuple[str, str, int]]:
    """Пары ключей имён и в скольких статьях они связаны (в любую сторону)."""
    geo = (
        " AND NOT (f.typed_at IS NOT NULL AND f.rel_type IS NULL)" if store.has_types(conn) else ""
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT least(f.src_key, f.dst_key), greatest(f.src_key, f.dst_key),
                   count(DISTINCT f.doc_id)
            FROM facts f WHERE {store.accepted()}{geo} AND f.src_key <> f.dst_key
            GROUP BY 1, 2
            """)
        return [(str(a), str(b), int(n)) for a, b, n in cur.fetchall()]


def compute(conn: Any, syn: Any) -> dict[str, Any]:
    """Разбить граф на кластеры и записать их. syn — словарь с автосинонимами."""
    import igraph as ig
    import leidenalg as la

    init(conn)
    # Ключ → узел: каноническое имя словаря (его ключ) или сам ключ.
    canon: dict[str, str] = {}

    def node(key: str) -> str:
        if key not in canon:
            name = syn.names.get(key)
            canon[key] = name_key(name) if name else key
        return canon[key]

    weights: Counter[tuple[str, str]] = Counter()
    for a, b, docs in _edges(conn):
        u, v = node(a), node(b)
        if u != v:
            weights[(min(u, v), max(u, v))] += docs
    names = sorted({x for pair in weights for x in pair})
    if not names:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM graph_clusters")
        conn.commit()
        return {"entities": 0, "clusters": 0, "clustered": 0}
    index = {n: i for i, n in enumerate(names)}
    g = ig.Graph(n=len(names), edges=[(index[u], index[v]) for u, v in weights])
    g.es["weight"] = list(weights.values())
    part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", seed=SEED)

    # Подпись: самые связанные предметы кластера — их написание, как чаще пишут.
    strength = g.strength(weights="weight")
    # Подпись узла: имя словаря, если узел — из словаря; иначе самое частое написание
    # всех его ключей.
    spelled = store.spellings(conn, sorted(canon))
    written: dict[str, Counter[str]] = defaultdict(Counter)
    label: dict[str, str] = {}
    for key, n in canon.items():
        written[n].update(spelled.get(key) or Counter())
        if syn.names.get(key):
            label[n] = syn.names[key]
    for n, counts in written.items():
        if n not in label and counts:
            label[n] = counts.most_common(1)[0][0]

    members: dict[int, list[int]] = defaultdict(list)
    for vertex, cluster in enumerate(part.membership):
        members[cluster].append(vertex)
    big = sorted(
        (vs for vs in members.values() if len(vs) >= MIN_SIZE),
        key=lambda vs: -sum(strength[v] for v in vs),
    )
    number_of: dict[int, int] = {v: number for number, vs in enumerate(big, 1) for v in vs}
    with conn.cursor() as cur:
        cur.execute("DELETE FROM graph_clusters")  # entity_clusters — каскадом
        for number, vs in enumerate(big, start=1):
            top = sorted(vs, key=lambda v: -strength[v])[:TITLE_NAMES]
            title = ", ".join(label.get(names[v], names[v]) for v in top)
            cur.execute(
                "INSERT INTO graph_clusters (id, title, size) VALUES (%s, %s, %s)",
                (number, title, len(vs)),
            )
        rows = [
            (key, number_of[index[n]])
            for key, n in canon.items()
            if n in index and index[n] in number_of
        ]
        cur.executemany(
            "INSERT INTO entity_clusters (key, cluster) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET cluster = EXCLUDED.cluster",
            rows,
        )
    conn.commit()
    return {
        "entities": len(names),
        "clusters": len(big),
        "clustered": sum(len(vs) for vs in big),
        "modularity": round(part.modularity, 3),
    }


def load(conn: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Кластер по ключу имени и список кластеров (для страницы). Не считали — пусто."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, cluster FROM entity_clusters")
            by_key = {str(k): int(c) for k, c in cur.fetchall()}
            cur.execute("SELECT id, title, size FROM graph_clusters ORDER BY id")
            listed = [{"id": int(i), "title": str(t), "size": int(s)} for i, t, s in cur.fetchall()]
    except Exception:  # noqa: BLE001 — таблиц ещё нет: кластеров нет
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return {}, []
    return by_key, listed
