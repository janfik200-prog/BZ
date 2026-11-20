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


# --------------------------------------------------------------------------- #
#  Темы Лейдена в датасете, связях под ответом, таблице и Телеграме
# --------------------------------------------------------------------------- #
def _topics_conn() -> FakeConn:
    from georag.text import name_key

    def answer(sql: str, params: Any) -> list[Any]:
        if sql.startswith("SELECT key, cluster"):
            return [(name_key("золото"), 1), (name_key("пирит"), 1), (name_key("гранит"), 2)]
        if sql.startswith("SELECT id, title, size"):
            return [(1, "золото, пирит", 5), (2, "гранит, гнейс", 3)]
        return []

    return FakeConn(answer=answer)


def test_topics_lookup() -> None:
    topics = clusters.Topics(_topics_conn())
    assert topics and topics.title("Золото") == "золото, пирит"  # по ключу имени
    assert topics.of("гранит") == {"id": 2, "title": "гранит, гнейс", "size": 3}
    assert topics.of("торф") is None and topics.title("торф") is None


def test_topics_without_clusters() -> None:
    topics = clusters.Topics(FakeConn([]))  # тем не считали — подписей нет, ошибок тоже
    assert not topics and topics.of("золото") is None


def test_telegram_caption_topic() -> None:
    from georag.tg import ui

    data = {"name": "золото", "facts": [], "documents": 1, "includes": []}
    assert "Тема графа" not in ui.graph_caption(data, 0)
    data["topic"] = {"id": 1, "title": "золото, пирит", "size": 5}
    assert "Тема графа: золото, пирит (5 предм.)" in ui.graph_caption(data, 0)


# --------------------------------------------------------------------------- #
#  Связи из закрытого списка и строгий режим модели-строителя (research/v19)
# --------------------------------------------------------------------------- #
def test_relation_forms_normalized() -> None:
    from georag.graph import facts

    assert facts.normalize_relation("содержат") == ("содержит", False)
    assert facts.normalize_relation("приурочивается к") == ("приурочено к", False)
    assert facts.normalize_relation("содержится в") == ("содержит", True)  # наоборот
    assert facts.normalize_relation("пересекает") == ("пересекает", False)
    assert facts.normalize_relation("залегает на") == ("залегает на", False)  # не из списка


def test_passive_relation_flips_ends() -> None:
    from georag.graph import facts

    text = "Золото содержится в кварцевых жилах."
    got = facts.check({"факты": [[1, "золото", "содержится в", "кварцевых жилах"]]}, text)
    assert [(f.src, f.relation, f.dst) for f in got.facts] == [
        ("кварцевых жилах", "содержит", "золото")
    ]


def test_strict_mode_rejects() -> None:
    from georag.graph import facts

    text = (
        "Оруденение не связано с гранитами. "
        "Дайки соответствуют зоне разлома. "
        "1 - пески; 2 - свита Зонг Чау - сланцы, известняки; 3 - свита Иен - угли. "
        "Аномалии расположены на глубине 18 km. "
        "Золото приурочено к кварцевым жилам."
    )
    answer = {"факты": [
        [1, "оруденение", "не связано с", "гранитами"],
        [2, "дайки", "соответствуют", "зоне разлома"],
        [3, "свита Зонг Чау", "содержит", "сланцы"],
        [4, "аномалии", "расположено в", "18 km"],
        [5, "золото", "приурочено к", "кварцевым жилам"],
    ]}  # fmt: skip
    loose = facts.check(answer, text)
    strict = facts.check(answer, text, strict=True)
    assert len(loose.facts) >= 4
    assert [(f.src, f.dst) for f in strict.facts] == [("золото", "кварцевым жилам")]
    reasons = " ".join(strict.rejected)
    assert "отрицание" in reasons and "не из закрытого списка" in reasons
    assert "легенды" in reasons


def test_strict_mode_negated_quote_and_language() -> None:
    from georag.graph import facts

    assert facts.strict_reason("Cu staining", "связано с", "pyrite",
                               "The Cu staining is probably unrelated to pyrite.") is not None
    assert facts.strict_reason("période", "сопровождается", "travaux",
                               "La période de production était avec des travaux dans les mines.")
    assert facts.strict_reason("золото", "приурочено к", "кварцевые жилы",
                               "Золото приурочено к кварцевым жилам.") is None
    assert facts.hedged("связано с", "Можно предположить, что X связано с Y.").startswith(
        "предположительно")


def test_strict_and_list_by_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from georag.graph import entities, facts

    monkeypatch.delenv("GEORAG_GRAPH_STRICT", raising=False)
    monkeypatch.delenv(entities.ENTITY_LIST_ENV, raising=False)
    assert facts.strict_for("qwen3-8b") and not facts.strict_for("qwen3-14b")
    assert not entities.entity_list_enabled("qwen3-8b") and entities.entity_list_enabled("qwen3-14b")
    monkeypatch.setenv("GEORAG_GRAPH_STRICT", "0")
    monkeypatch.setenv(entities.ENTITY_LIST_ENV, "1")
    assert not facts.strict_for("qwen3-8b") and entities.entity_list_enabled("qwen3-8b")


def test_strict_negation_only_for_same_relation() -> None:
    from georag.graph import facts

    quote = ("Новые данные показывают, что оруденение не связано с разломом, "
             "а приурочено к контакту гранитной интрузии.")
    assert facts.strict_reason("оруденение", "приурочено к", "контакту гранитной интрузии",
                               quote) is None
    assert facts.strict_reason("оруденение", "связано с", "разломом", quote) is not None
    assert facts.strict_reason("березиты", "содержит", "золото",
                               "Однако в березитах золото не обнаружено.") is not None


def test_strict_outdated_claim() -> None:
    from georag.graph import facts

    quote = "Ранее считалось, что золотое оруденение приурочено к зоне Персияновского разлома."
    assert "устаревшее" in facts.strict_reason("золотое оруденение", "приурочено к",
                                               "зона Персияновского разлома", quote)


def test_strict_direction_follows_text() -> None:
    from georag.graph import facts

    text = ("Некоторое количество даек и силлов магматических пород пространственно "
            "приурочено к зоне Персияновского разлома.")
    answer = {"факты": [[1, "зоне Персияновского разлома", "приурочено к",
                         "дайки и силлы магматических пород"]]}
    got = facts.check(answer, text, strict=True)
    assert [(f.src, f.dst) for f in got.facts] == [
        ("дайки и силлы магматических пород", "зоне Персияновского разлома")
    ]
    loose = facts.check(answer, text)  # без строгого режима — как ответила модель
    assert loose.facts[0].src == "зоне Персияновского разлома"


def test_strict_checks_own_sentence_only() -> None:
    from georag.graph import facts

    quote = ("Ранее считалось, что оруденение приурочено к зоне разлома. Новые данные "
             "показывают, что оруденение приурочено к контакту гранитной интрузии.")
    assert facts.strict_reason("оруденение", "приурочено к", "контакту гранитной интрузии",
                               quote) is None
    assert facts.strict_reason("оруденение", "приурочено к", "зоне разлома", quote) is not None
