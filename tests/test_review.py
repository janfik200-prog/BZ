"""Проверки к исправлениям ревью перед релизом: каждая падает без своего исправления.

Без базы, модели и сети: база — подставная (tests/fakedb.py), Телеграм — заглушка."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
#  Очередь ответов Телеграм-бота (tg/bot.py, _wait_turn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("broken", ["send", "edit"])
def test_queue_survives_network_error(broken: str) -> None:
    """Сеть моргнула, пока вопрос ждал очереди: вопрос не теряется, а место у модели
    (общее со страницей) не пропадает навсегда. Ошибка requests — не TelegramError."""
    requests = pytest.importorskip("requests")
    from georag.tg.bot import Bot, Session
    from georag.turns import Turns

    class Tg:
        def send(self, chat_id: int, text: str, markup: Any = None) -> dict[str, Any]:
            if broken == "send":
                raise requests.ConnectionError("сеть моргнула")
            return {"message_id": 1}

        def edit(self, *args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("сеть моргнула")

    bot = Bot(Tg(), None, None, None, log=lambda *a: None)
    bot.slots = Turns(1)
    answered: list[str] = []
    bot.answer = lambda chat_id, session, text, history=None: answered.append(text)  # type: ignore[method-assign]
    bot.slots.acquire()  # место занято чужим ответом
    worker = threading.Thread(
        target=bot._answer_safe, args=(7, Session(), "вопрос", None), daemon=True
    )
    worker.start()
    deadline = time.monotonic() + 5
    while bot.slots.waiting == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)  # сообщение о месте в очереди уже отправлялось
    bot.slots.release()
    worker.join(5)
    assert answered == ["вопрос"]
    assert bot.slots.acquire(blocking=False)  # место вернулось в очередь
