"""Место вопроса: поиск только в статьях о нём; статей нет — «по территории ничего не найдено»
(chat/answer.py, NOT_IN_PLACE). До 08.11.2026 искалось по всей базе, и ответ о районе Ванба
собирался в том числе из статьи о Среднем Урале."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat import answer as chat
from georag.chat.plan import Part, Plan
from georag.chat.retrieval import Retrieval
from georag.chat.settings import Settings


def _no_place() -> chat.Prepared:
    plan = Plan(parts=[Part("Какие структуры контролируют оруденение в районе Ванба?", [])])
    plan.place = ["Ванба", "Wamba"]
    return chat.Prepared(plan, [], Retrieval([], judged=False), no_place=True)


def test_no_articles_about_place_says_nothing_found() -> None:
    def llm(*_: Any, **__: Any) -> Any:
        raise AssertionError("модель не должна отвечать: статей о месте нет")

    events = list(
        chat.answer(None, None, "Какие структуры контролируют оруденение в районе Ванба?", [],
                    Settings(), llm=llm, prepared=_no_place())
    )
    general = [e for e in events if e["type"] == "general"]
    assert len(general) == 1
    assert general[0]["text"] == chat.NOT_IN_PLACE.format(place="Ванба")
    assert "Ванба" in general[0]["text"] and "ничего не найдено" in general[0]["text"]
    done = events[-1]
    assert done["type"] == "done" and done["no_place"] and done["not_found"]
    assert done["used"] == []


def test_place_search_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEORAG_CONCEPTS", raising=False)
    assert Settings().concepts
    monkeypatch.setenv("GEORAG_CONCEPTS", "0")
    assert not Settings().concepts

