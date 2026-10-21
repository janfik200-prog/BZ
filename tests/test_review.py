"""Проверки к исправлениям ревью перед релизом: каждая падает без своего исправления.

Без базы, модели и сети: база — подставная (tests/fakedb.py), Телеграм — заглушка."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
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


# --------------------------------------------------------------------------- #
#  Формулы (index/formulas.py)
# --------------------------------------------------------------------------- #
RHO = r"\rho = \rho_0 (1 - \varphi) \quad (1)"
GRAV = r"\Delta g = \frac{4}{3} \pi G \Delta\rho R^{3} / h^{2} \quad (2)"


def test_formula_rows_without_nul(tmp_path: Path) -> None:
    """Нулевой символ из PDF в разметке docling не доходит до таблицы formulas: Postgres
    его не принимает, и загрузка всей статьи падала (у фрагментов его снимает _clean)."""
    from fakedb import FakeConn, returning_ids

    from georag.index import formulas

    doc = {
        "texts": [
            {"label": "text", "text": "Плотность\x00 пород выражается через пористость:"},
            {"label": "formula", "text": RHO + "\x00", "orig": "ρ = ρ0 (1 − φ)\x00 (1)"},
            {"label": "text", "text": "где ρ0\x00 — плотность скелета."},
        ]
    }
    (tmp_path / "d1.docling.json").write_text(json.dumps(doc), encoding="utf-8")
    items = formulas.prepare("d1", tmp_path)
    assert [f["number"] for f in items] == ["1"]
    conn = FakeConn(answer=returning_ids(chunks=[]))
    formulas.store(conn, "d1", items, {}, [[0.1]])
    [row] = conn.queries("INSERT INTO formulas")
    assert not [v for v in row if isinstance(v, str) and "\x00" in v]


# --------------------------------------------------------------------------- #
#  Формулы в проверке ответа (chat/checks.py)
# --------------------------------------------------------------------------- #
LOOSE = "А это — общие слова модели без всякой ссылки на статью."


def test_formula_opened_at_line_end() -> None:
    """«по формуле (2) [3]: $$» — формула открылась в конце строки текста (так в примере
    правила 8). Тело формулы — не утверждение без ссылки, а фраза без ссылки после
    формулы — по-прежнему убирается."""
    from georag.chat.checks import coverage, drop_uncited

    answer = (
        f"По формуле (2) из статьи [3]: $$\n{GRAV}\n$$\n"
        f"где Δg — аномалия силы тяжести над рудным телом [3].\n{LOOSE}"
    )
    kept, removed = drop_uncited(answer)
    assert GRAV in kept and removed == [LOOSE]
    assert coverage(answer) == (1, 2)  # «где Δg — …» со ссылкой и LOOSE без неё


def test_coverage_does_not_swallow_text_between_formulas() -> None:
    """Формула в начале строки с текстом после неё — не блок: строки до следующей
    формулы не пропадают из подсчёта утверждений."""
    from georag.chat.checks import coverage

    answer = (
        "$$x$$ — плотность пород, она растёт с глубиной залегания и давлением [1].\n"
        "Вторая длинная фраза без всякой ссылки, просто общие слова модели здесь.\n"
        "Третья длинная фраза без всякой ссылки, тоже общие слова модели и только.\n"
        "$$y = x$$\n"
    )
    assert coverage(answer) == (1, 3)


def test_plain_latex_arrows() -> None:
    r"""\rightarrow и \leftarrow — стрелки, а не «arrow»: \right и \left срезали начало."""
    from georag.index import formulas

    assert formulas.plain_latex(r"a \rightarrow b \leftarrow c") == "a → b ← c"
    assert formulas.plain_latex(r"\left( x \right)") == "( x )"


def test_describe_sentence_before_without_double_dot() -> None:
    """Фраза перед формулой уже кончается точкой — в описании нет «..»."""
    from georag.index import formulas

    item = {"number": "1", "latex": RHO, "before": "Модель простая. Так считают плотность."}
    assert formulas.describe(item) == "Формула (1): ρ = ρ_0 (1 − φ). Так считают плотность."


def test_named_reference_whole_word_and_number() -> None:
    """«сформулированы 3 гипотезы» — не ссылка на формулу (3), «уравнения 2015 года» — не
    на формулу (201); настоящие ссылки — как были."""
    from georag.index import formulas

    numbers = {"3", "20", "201"}
    assert formulas.references("Были сформулированы 3 гипотезы.", numbers) == set()
    assert formulas.references("Система уравнения 2015 года.", numbers) == set()
    assert formulas.references("Из формулы 3 и уравнения (20) следует", numbers) == {"3", "20"}
