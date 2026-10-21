"""Сверка ссылок ответа с фрагментами (chat/support.py) и спасение фраз без ссылки по
числам (checks.cite_by_numbers) — без модели: её ответ подставляется."""

from __future__ import annotations

import re
import time
from typing import Any

import pytest

from georag.chat import support
from georag.chat.answer import _answer_from
from georag.chat.checks import cite_by_numbers, drop_uncited, numbers
from georag.chat.plan import Plan
from georag.chat.settings import Settings

SOURCES = [
    {
        "n": 1,
        "title": "Massif Central",
        "doc_id": "a",
        "authors": ["Jean Martin"],
        "year": 2013,
        "text": "Gold veins of the Massif Central, France, are controlled by lineaments. "
        "Lineament density exceeds 2.5 km per km2 near the deposits.",
    },
    {
        "n": 2,
        "title": "Radon",
        "doc_id": "b",
        "year": 2022,
        "text": "Radon concentration at TTN-07 reached a maximum of 5343.33 Bq/m3, mean 3612.08.",
    },
    {
        "n": 3,
        "title": "Aeromagnetics",
        "doc_id": "c",
        "year": 1997,
        "text": "1:50 000 high-precision aeromagnetic survey; anomalies of 50-500 nT.",
    },
]


# --------------------------------------------------------------------------- #
#  Числа и ссылка по числам (без модели)
# --------------------------------------------------------------------------- #
def test_numbers_normalized() -> None:
    assert numbers("максимум 5343,33 Бк, масштаб 1:50 000, 50–500 нТл [3]") == {
        "5343.33",
        "50000",
        "50",
        "500",
    }
    assert numbers("в 3 точках [12]") == set()  # одиночные цифры и номера ссылок — не в счёт
    assert numbers("50 000 и 1 200") == {"50000", "1200"}  # неразрывные пробелы


def test_uncited_fact_with_numbers_gets_citation() -> None:
    answer = (
        "Корреляция радона с плотностью линеаментов изучена на рудном поле [2].\n"
        "Максимальная концентрация радона в точке TTN-07 составила 5343,33 Бк/м³.\n"
        "Линеаменты вообще часто бывают связаны с выходами газов из недр земли."
    )
    text, rescued = cite_by_numbers(answer, SOURCES)
    assert rescued == ["Максимальная концентрация радона в точке TTN-07 составила 5343,33 Бк/м³."]
    assert "5343,33 Бк/м³ [2]." in text
    text, removed = drop_uncited(text)
    assert "5343,33" in text and removed == [
        "Линеаменты вообще часто бывают связаны с выходами газов из недр земли."
    ]


def test_numbers_split_between_fragments_get_no_citation() -> None:
    answer = "Плотность линеаментов 2.5 км/км² и аномалии до 500 нТл отмечены на площади."
    assert cite_by_numbers(answer, SOURCES) == (answer, [])


RHO = r"$$\rho = \rho_0 (1 - \varphi) + \frac{a}{b} + \sum_{i=1}^{n} x_i + 50 000$$"


def test_cite_by_numbers_keeps_formulas_and_untouched_answer() -> None:
    """Формула ($$…$$) — не утверждение, хоть в ней и число из фрагмента; ответ, где
    ссылки ставить некуда, возвращается как был, без перестановки пробелов."""
    answer = (
        f"По формуле плотности пород [3]:\n{RHO}\nКонец  ответа без чисел, но достаточно длинный."
    )
    assert cite_by_numbers(answer, SOURCES) == (answer, [])


# --------------------------------------------------------------------------- #
#  Сверка смысла (модель подставляется)
# --------------------------------------------------------------------------- #
class Client:
    """Модель-проверщик: вердикт — по словам утверждения (judge), а не по его номеру:
    при перепроверке по одному у утверждения номер 1. second — вердикт перепроверки,
    если он другой."""

    def __init__(self, judge: Any, second: Any = None) -> None:
        self.judge, self.second = judge, second
        self.asked: list[str] = []

    def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        self.asked.append(user)
        if self.judge is None:
            raise RuntimeError("модель молчит")
        claims = re.findall(r"(?m)^(\d+)\) «(.+?)»", user)
        judge = (
            self.second if self.second and len(claims) == 1 and len(self.asked) > 1 else self.judge
        )
        return {n: judge(text) for n, text in claims}


def run(
    monkeypatch: pytest.MonkeyPatch, answer: str, judge: Any, second: Any = None
) -> tuple[tuple[str, list[str], int], Client]:
    client = Client(judge, second)
    monkeypatch.setattr(Settings, "client", lambda self: client)
    return support.check_support("?", answer, SOURCES, Settings()), client


ANSWER = (
    "**Золото**\n"
    "В Центральном массиве (Франция) золотые жилы контролируются линеаментами [1].\n"
    "В Пиренеях (Испания) анализ линеаментов дал сведения о золотых жилах [1]."
)

YES = {"ответ": "да", "чего_нет": ""}
NO = {"ответ": "нет", "чего_нет": "во фрагменте Франция, а не Испания"}


def spain(verdict: Any) -> Any:
    """Опровергает утверждение про Испанию, остальное подтверждает."""
    return lambda claim: verdict if "Испан" in claim else YES


def test_unsupported_claim_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    (text, removed, recited), client = run(monkeypatch, ANSWER, spain(NO))
    assert "Франция" in text and "Испан" not in text
    assert removed == [ANSWER.splitlines()[2]] and recited == 0
    assert "Massif Central" in client.asked[0]  # модель видит текст фрагмента
    assert "(Martin, 2013)" in client.asked[0]  # в том же виде, что при ответе
    assert '"1": {"ответ"' in client.asked[0]  # ключи JSON — с одинарными скобками
    assert len(client.asked) == 2
    assert "Пиренеях" in client.asked[1] and "Центральном" not in client.asked[1]  # по одному


def test_partial_with_foreign_name_is_refuted(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Частично», а нет во фрагменте — места из утверждения: приписано чужое, убрать.
    «Частично» с добавкой общими словами — оставить."""
    foreign = {"ответ": "частично", "чего_нет": "Пиренеи"}
    (text, removed, _), _ = run(monkeypatch, ANSWER, spain(foreign))
    assert "Испан" not in text and len(removed) == 1
    extra = {"ответ": "частично", "чего_нет": "сведения о золотых жилах"}
    (text, removed, _), _ = run(monkeypatch, ANSWER, spain(extra))
    assert text == ANSWER and removed == []
    # Имя, которое в процитированном фрагменте есть, — не чужое.
    answer = "В статье о радоне в точке TTN-07 изучена связь с плотностью линеаментов [2]."
    named = {"ответ": "частично", "чего_нет": "TTN-07 упомянута не во всех фрагментах"}
    (text, removed, _), _ = run(monkeypatch, answer, lambda c: named)
    assert text == answer and removed == []


def test_second_look_saves_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Общий проход опроверг, перепроверка по одному — нет: утверждение остаётся."""
    (text, removed, _), _ = run(monkeypatch, ANSWER, spain(NO), second=lambda c: YES)
    assert text == ANSWER and removed == []


def test_doubt_keeps_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Частично» без причины, «нет» без названной причины, строка вместо разбора,
    молчащая модель — ответ не трогается."""
    bare_no = {"ответ": "нет", "чего_нет": ""}
    for judge in (spain({"ответ": "частично"}), spain(bare_no), spain("нет"), None):
        (text, removed, _), _ = run(monkeypatch, ANSWER, judge)
        assert text == ANSWER and removed == []


def test_remarks_about_sources_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Во фрагментах нет точных значений точности выделения линеаментов в процентах [1]."
    (text, removed, _), client = run(monkeypatch, answer, lambda c: NO)
    assert text == answer and removed == [] and not client.asked


def test_number_moves_citation_to_its_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Аэромагнитная съёмка площади выполнена в масштабе 1:50 000 [1]."
    (text, removed, recited), client = run(monkeypatch, answer, lambda c: YES)
    assert text == "Аэромагнитная съёмка площади выполнена в масштабе 1:50 000 [3]."
    assert recited == 1 and removed == []
    assert "фрагменты 3" in client.asked[0]  # проверяется уже с правильной ссылкой


def test_connective_after_removed_claim_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Как у drop_uncited: связка «Например,» после убранной фразы снимается."""
    answer = (
        "В Пиренеях (Испания) анализ линеаментов дал сведения о золотых жилах [1]. "
        "Например, в Центральном массиве золотые жилы контролируются линеаментами [1]."
    )
    (text, removed, _), _ = run(monkeypatch, answer, spain(NO))
    assert text == "В Центральном массиве золотые жилы контролируются линеаментами [1]."


# --------------------------------------------------------------------------- #
#  Ответ целиком: события revised и done
# --------------------------------------------------------------------------- #
def llm(messages: list[dict[str, Any]], settings: Settings) -> Any:
    yield {"text": ANSWER}
    yield {"done": True, "tokens": 50}


def answer_events(check: Any, settings: Settings | None = None) -> list[dict[str, Any]]:
    return list(
        _answer_from(
            llm,
            "Где золотые жилы контролируются линеаментами?",
            SOURCES,
            [],
            Plan(parts=[]),
            [],
            settings or Settings(),
            {},
            time.monotonic(),
            support=check,
        )
    )


def test_answer_reports_unsupported() -> None:
    def check(q: str, a: str, s: Any, st: Any) -> tuple[str, list[str], int]:
        lines = a.splitlines()
        return "\n".join(lines[:2]), [lines[2]], 0

    got = answer_events(check)
    revised = next(e for e in got if e["type"] == "revised")
    assert revised["unsupported"] == [ANSWER.splitlines()[2]] and revised["removed"] == []
    assert revised["rescued"] == 0 and revised["recited"] == 0
    done = got[-1]
    assert done["type"] == "done" and done["unsupported"] == 1 and done["recited"] == 0
    assert done["support_seconds"] is not None


def test_answer_kept_when_check_is_off_or_removes_all() -> None:
    def everything(q: str, a: str, s: Any, st: Any) -> tuple[str, list[str], int]:
        return "", a.splitlines(), 0

    for check, settings in ((everything, None), (everything, Settings(check_support=False))):
        got = answer_events(check, settings)
        assert not [e for e in got if e["type"] == "revised"]
        assert got[-1]["unsupported"] == 0


def test_recite_needs_distinctive_number_and_skips_literature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Живая проверка 21.10.2026: двузначное «30 м» и номер «(лит. 10-12)» уводили ссылку
    с верного фрагмента на чужой."""
    sources = [
        {
            "n": 1,
            "title": "SRTM",
            "doc_id": "a",
            "text": "SRTM DEM has a relatively high resolution.",
        },
        {"n": 2, "title": "Landsat", "doc_id": "b", "text": "Images of 15, 30, 60 m were used."},
        {"n": 3, "title": "Maps", "doc_id": "c", "text": "Scale 1:100000 maps [10-12] were built."},
    ]
    client = Client(lambda c: YES)
    monkeypatch.setattr(Settings, "client", lambda self: client)
    for claim in (
        "SRTM имеет высокое пространственное разрешение (до 30 метров) [1].",
        "Используются снимки с разрешением 10 м (масштаб 1:100 000) [1].",
    ):
        text, removed, recited = support.check_support("?", claim, sources, Settings())
        assert text == claim and recited == 0


def test_numbers_in_several_fragments_give_no_citation() -> None:
    """Живая проверка 21.10.2026: «2005» стоял в двух фрагментах, фраза получала ссылку
    не на тот, где сказано."""
    sources = [
        {"n": 1, "text": "Subjectivity is unavoidable (Gomez and Kavzoglu 2005)."},
        {"n": 2, "text": "Assessment is based on visual examination (Jordan and Schott 2005)."},
    ]
    answer = "Визуальный осмотр результатов применяется для оценки точности (Jordan, Schott 2005)."
    assert cite_by_numbers(answer, sources) == (answer, [])
