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


# --------------------------------------------------------------------------- #
#  Раздача KaTeX (web/server.py, _static)
# --------------------------------------------------------------------------- #
def test_static_rejects_foreign_paths_before_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Путь с корнем (сетевая папка «//сервер/…») и «..» отсекаются до resolve(): на Windows
    resolve() такого пути — уже вход на чужой сервер, а ссылку даст любая страница."""
    from pathlib import Path

    from georag.web import server

    touched: list[str] = []
    real = Path.resolve

    def spy(self: Path, strict: bool = False) -> Path:
        touched.append(str(self))
        return self if "evil" in str(self) else real(self, strict)  # и без правки — не в сеть

    monkeypatch.setattr(Path, "resolve", spy)
    sent: list[tuple[int, str]] = []
    handler = server.Handler.__new__(server.Handler)
    handler._send = lambda code, body, kind: sent.append((code, kind))  # type: ignore[method-assign]
    handler._json = lambda payload, code=200: sent.append((code, "json"))  # type: ignore[method-assign]
    handler._static("//evil.example/share/x.js")
    handler._static("katex/../../server.py")
    assert sent == [(404, "json"), (404, "json")]
    assert not [p for p in touched if "evil" in p or "server.py" in p]
    handler._static("katex/katex.min.js")  # своё — как раньше
    assert sent[-1] == (200, "text/javascript; charset=utf-8")
