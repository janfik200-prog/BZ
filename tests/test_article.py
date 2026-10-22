"""Вопрос о названной статье (chat/article.py): название узнаётся в вопросе, не идёт
отдельной частью, ищется только в этой статье; не нашлось — «в статье об этом нет».
Без базы и модели."""

from __future__ import annotations

from typing import Any

import pytest
from fakedb import FakeConn

from georag.chat import answer as chat
from georag.chat import retrieval
from georag.chat.article import ABOUT, NamedArticle, find_title, in_article
from georag.chat.plan import Part, Plan
from georag.chat.retrieval import Retrieval
from georag.chat.settings import Settings
from georag.index import search
from georag.index.search import Hit
from georag.tg import ui

TITLE = "Gold veins of the Massif Central: a Bayesian model of fault geometry"
TITLES = [
    ("short", "Gold veins"),  # короче MIN_WORDS — в вопросе не узнаётся
    ("part", "Gold veins of the Massif Central"),
    ("full", TITLE),
]


@pytest.mark.parametrize(
    ("question", "rest"),
    [
        (
            f"{TITLE}\nИспользуется ли в этой статье метод Монте-Карло?",
            "Используется ли в этой статье метод Монте-Карло?",
        ),  # noqa: E501
        (f"В статье «{TITLE.upper()}» есть метод Смита", "В статье есть метод Смита?"),
        ("gold veins of the massif central — a bayesian model of fault geometry.", ABOUT),
    ],
)
def test_title_found_and_cut(question: str, rest: str) -> None:
    found = find_title(question, TITLES)
    assert found is not None
    assert found.doc_id == "full"  # самое длинное из подходящих
    assert found.rest == rest


def test_short_title_and_plain_question_not_matched() -> None:
    assert find_title("Какие gold veins во Франции?", TITLES) is None
    assert find_title("Что известно о жилах золота во Франции?", TITLES) is None
    # Слово названия внутри другого слова — не название.
    assert find_title("Gold veins of the Massif Centrality", TITLES[1:2]) is None


def test_in_article_switches_off_whole_base_search() -> None:
    plan = Plan([Part("q", [])], subject="Франция", collect=True, on_topic=False, catalog=object())
    in_article(plan, NamedArticle("full", TITLE, "q"))
    assert plan.article == "full"
    assert (plan.subject, plan.collect, plan.catalog, plan.on_topic) == (None, False, None, True)


def test_doc_filter_in_search_sql() -> None:
    conn = FakeConn(rows=[])
    search.text_search(conn, "метод Смита", 10, doc_id="full")
    sql, params = conn.log[-1]
    assert "c.doc_id = %(doc_id)s" in sql and params["doc_id"] == "full"
    assert search._filters(None, None)[0] == ""


def yield_from(gen: Any) -> Any:
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_prepare_plans_rest_and_searches_article(monkeypatch: pytest.MonkeyPatch) -> None:
    planned: list[str] = []
    searched: list[Any] = []

    def planner(question: str, history: Any, settings: Settings) -> Plan:
        planned.append(question)
        return Plan([Part(question, ["Monte Carlo"])], subject="Монте-Карло", collect=True)

    def retrieve(conn: Any, emb: Any, question: str, plan: Plan, *a: Any) -> Any:
        searched.append((question, plan.article, plan.collect, len(plan.parts)))
        return Retrieval([])
        yield

    monkeypatch.setattr(chat, "retrieve", retrieve)
    question = f"{TITLE}\nИспользуется ли в этой статье метод Монте-Карло?"
    prepared = yield_from(
        chat.prepare(
            None,
            None,
            question,
            [],
            Settings(),
            planner,
            datasets=None,
            graph_hits=None,
            article=lambda conn, q, h: find_title(q, TITLES),
        )
    )
    assert planned == ["Используется ли в этой статье метод Монте-Карло?"]  # без названия
    # Судья и ответ видят вопрос целиком, ищется только в статье, одной частью.
    assert searched == [(question, "full", False, 1)]
    assert prepared.info()["article"] == TITLE

    events = list(chat.answer(None, None, question, [], Settings(), prepared=prepared))
    general = next(e for e in events if e["type"] == "general")
    assert TITLE in general["text"] and general["not_found"] and general["article"] == TITLE
    record = {"question": question, "info": general, "not_found": True}
    assert "другие статьи базы не смотрел" in ui.how_text(record)
    assert events[-1]["tokens"] is None  # модель из общих знаний не отвечала


def test_article_candidates_not_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Все кандидаты — из одной статьи: ограничение «3 из статьи» их не срезает."""
    asked: list[Any] = []
    hits = [Hit(i, "full", i, f"фрагмент {i}", title=TITLE, similarity=0.6) for i in range(10)]

    def candidates(
        conn: Any, emb: Any, queries: list[str], cache: Any, limit: int, facts: bool, doc: Any
    ) -> list[Any]:
        asked.append(doc)
        return hits

    seen: list[int] = []

    def judge(question: str, plan: Any, batch: list[dict[str, Any]], *a: Any) -> Any:
        seen.extend(c["id"] for c in batch)
        return {"подходят": [1], "не_хватает": [], "запросы": []}

    monkeypatch.setattr(retrieval, "candidates", candidates)
    plan = Plan([Part("q", ["Monte Carlo"])], main="q", article="full")
    found = retrieval.run(retrieval.retrieve(None, None, "q", plan, Settings(notes=False), judge))
    assert asked and set(asked) == {"full"}
    assert len(seen) == 10 and found.sources[0]["doc_id"] == "full"
