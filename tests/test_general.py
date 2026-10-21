"""Вопрос о геологии, а в базе пусто (chat/answer.py): без GEORAG_GENERAL — отказ, без ответа
модели из общих знаний; с ним — ответ модели с пометкой. Без базы и модели."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat import answer as chat
from georag.chat.plan import Part, Plan
from georag.chat.retrieval import Retrieval
from georag.chat.settings import Settings
from georag.tg import ui

QUESTION = "Сколько статей про Францию?"


def llm(messages: list[dict[str, Any]], settings: Settings) -> Any:
    yield {"text": "Франция расположена в Западной Европе."}
    yield {"done": True, "tokens": 9}


def events(settings: Settings, on_topic: bool = True) -> list[dict[str, Any]]:
    plan = Plan([Part(QUESTION, ["Франция"])], on_topic=on_topic)
    prepared = chat.Prepared(plan, [], Retrieval([]))
    return list(chat.answer(None, None, QUESTION, [], settings, llm=llm, prepared=prepared))


@pytest.mark.parametrize("allowed", [False, True])
def test_general_knowledge_switch(allowed: bool) -> None:
    got = events(Settings(general_knowledge=allowed))
    general = next(e for e in got if e["type"] == "general")
    tokens = [e for e in got if e["type"] == "token"]
    done = got[-1]
    assert done["type"] == "done" and done["mode"] == "без базы"
    assert bool(tokens) is allowed
    assert bool(general.get("not_found")) is not allowed
    assert bool(done.get("not_found")) is not allowed
    if not allowed:
        assert general["text"] == chat.NOT_FOUND and done["tokens"] is None


def test_off_topic_refusal_untouched() -> None:
    got = events(Settings(general_knowledge=True), on_topic=False)
    assert got[-1].get("off_topic") and not [e for e in got if e["type"] == "token"]


@pytest.mark.parametrize(
    ("raw", "on"),
    [("1", True), ("да", True), ("TRUE", True), ("", False), ("0", False), ("abc", False)],
)
def test_env_switch(monkeypatch: pytest.MonkeyPatch, raw: str, on: bool) -> None:
    monkeypatch.setenv("GEORAG_GENERAL", raw)
    assert Settings().general_knowledge is on


def test_telegram_refusal_is_not_model_answer() -> None:
    done = {"mode": "без базы", "not_found": True, "model": "qwen3-14b", "seconds": 1.0}
    text = ui.answer_text("", chat.NOT_FOUND, [], done)
    assert "не из базы знаний" not in text and "(пусто)" not in text
    assert "выключен" in ui.how_text({"question": QUESTION, "mode": "без базы", "not_found": True})
    model = ui.answer_text("Ответ.", chat.NOT_IN_BASE, [], {"mode": "без базы"})
    assert "ответ модели, не из базы знаний" in model
