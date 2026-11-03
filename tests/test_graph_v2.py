"""План графа v2, А5–А7: буфер редких связей, подграф из базы, кластеры Лейдена.
Без базы: запросы — в подставную (fakedb), кластеры — на выдуманных связях."""

from __future__ import annotations

from typing import Any

import pytest
from fakedb import FakeConn

from georag.graph import api, clusters, store
from georag.graph.synonyms import Synonyms


# --------------------------------------------------------------------------- #
#  А5: что идёт в граф, что — в кандидаты
# --------------------------------------------------------------------------- #
def test_accepted_condition() -> None:
    sql = store.accepted()
    assert "coalesce(f.votes, 0) >= 3" in sql  # непроверенный — не NULL, а 0
    assert "count(DISTINCT o.doc_id) > 1" in sql  # повтор — только в другой статье
    # в любую сторону: пара концов без направления
    assert "(least(f.src_key, f.dst_key), greatest(f.src_key, f.dst_key)) IN" in sql
    assert "facts o WHERE" not in sql  # без fresh устаревших нет — фильтра тоже


def test_all_facts_candidates_and_keys() -> None:
    conn = FakeConn([])
    store.all_facts(conn, candidates=True, keys=["золот"], like=["персиянов разлом"])
    sql, params = conn.log[-1]
    assert "WHERE NOT (coalesce(f.votes, 0) >= 3" in sql
    assert "f.src_key = ANY(%s) OR f.src_key LIKE ANY(%s)" in sql
    assert ["%персиянов разлом%"] in params and ["золот"] in params


def test_all_facts_both_ends() -> None:
    conn = FakeConn([])
    store.all_facts(conn, keys=["a", "b"], both_ends=True)
    sql, _params = conn.log[-1]
    assert ") AND (f.dst_key = ANY(%s)" in sql


# --------------------------------------------------------------------------- #
#  А6: ключи всех написаний предмета
# --------------------------------------------------------------------------- #
def test_keys_cover_synonyms() -> None:
    syn = Synonyms(names={"donetsk basin": "Донецкий бассейн", "донбасс": "Донецкий бассейн"})
    keys = api._keys(syn, ["Донецкий бассейн"])
    assert {"donetsk basin", "донбасс"} <= keys
    assert api._keys(syn, ["donetsk basin"]) == keys  # с любого написания — те же ключи


def test_small_graph_stays_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(api, "large", lambda conn: False)
    monkeypatch.setattr(api, "graph", lambda conn, syn: calls.append("весь") or "граф")
    assert api.graph_for(None, Synonyms(), "золото") == "граф" and calls == ["весь"]


def test_large_graph_without_hint_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "large", lambda conn: True)
    monkeypatch.setattr(api, "_with_auto", lambda conn, syn: syn)
    g = api.graph_for(None, Synonyms(), None)
    assert not g.entities  # перебирать все имена большого графа не будем


# --------------------------------------------------------------------------- #
#  А7: кластеры Лейдена
# --------------------------------------------------------------------------- #
def _cluster_conn() -> FakeConn:
    # Две плотные тройки, связанные одним слабым ребром, и отдельная пара.
    edges = [
        ("золот", "пирит", 5),
        ("пирит", "арсенопирит", 4),
        ("золот", "арсенопирит", 4),
        ("гранит", "гнейс", 5),
        ("гнейс", "сланц", 4),
        ("гранит", "сланц", 4),
        ("пирит", "гранит", 1),
        ("уголь", "торф", 3),
    ]

    def answer(sql: str, params: Any) -> list[Any]:
        if sql.lstrip().startswith("SELECT least(f.src_key"):  # рёбра кластеров
            return edges
        if sql.startswith("SELECT key, raw"):
            return [(k, k, 1, 0) for k in {x for a, b, _ in edges for x in (a, b)}]
        return []

    return FakeConn(answer=answer)


def test_leiden_clusters() -> None:
    pytest.importorskip("leidenalg")
    conn = _cluster_conn()
    stats = clusters.compute(conn, Synonyms())
    assert stats["entities"] == 8 and stats["clusters"] == 2  # пара «уголь, торф» мельче 3
    written = dict(p for p in conn.queries("INSERT INTO entity_clusters"))
    assert written["золот"] == written["пирит"] == written["арсенопирит"]
    assert written["гранит"] == written["гнейс"] == written["сланц"]
    assert written["золот"] != written["гранит"] and "уголь" not in written
    titles = [p[1] for p in conn.queries("INSERT INTO graph_clusters")]
    assert any("пирит" in t for t in titles)


def test_synonyms_merge_cluster_nodes() -> None:
    pytest.importorskip("leidenalg")
    conn = _cluster_conn()
    syn = Synonyms(names={"золот": "золото", "пирит": "золото"})  # «пирит» сведён к «золоту»
    stats = clusters.compute(conn, syn)
    assert stats["entities"] == 7  # два ключа — один узел


# --------------------------------------------------------------------------- #
#  Подтверждение: у модели сразу workers фактов, без ожидания пачки
# --------------------------------------------------------------------------- #
def test_confirm_keeps_model_busy() -> None:
    import threading
    import time

    from georag.graph import confirm

    rows = [
        (i, "золото", "приурочено к", "кварцевым жилам", "Золото приурочено к кварцевым жилам.")
        for i in range(1, 201)
    ]
    conn = FakeConn(answer=lambda sql, params: rows if sql.startswith("SELECT id, src") else [])
    busy = peak = 0
    lock = threading.Lock()

    class Model:
        def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
            nonlocal busy, peak
            with lock:
                busy += 1
                peak = max(peak, busy)
            time.sleep(0.01 if "1." in user else 0)
            with lock:
                busy -= 1
            return {"1": "кварцевым жилам", "2": "золото", "3": "кварцевым жилам", "4": "золото"}

    stats = confirm.run(conn, Model(), workers=64, log=None)
    assert stats == {"todo": 200, "done": 200, "kept": 200, "errors": 0}
    assert len(conn.queries("UPDATE facts SET votes")) == 200
    assert peak > 16  # больше, чем фрагментов при выписке: факты идут к модели сразу
