"""Промт «выписки» (GEORAG_PROMPT=quote) и сверка выписок с фрагментами (chat/quotes.py)."""

from __future__ import annotations

import os
from unittest import mock

from georag.chat import messages as M
from georag.chat.checks import NO_ANSWER_QUOTE, is_refusal, says_not_in_base
from georag.chat.quotes import check_quotes, check_synthesis, is_foreign, split_synthesis
from georag.chat.settings import Settings, prompt_style

BERESITE = (
    "Березит — средне- и мелкозернистая метасоматическая горная порода, содержащая кварц, "
    "серицит, анкерит и пирит. Размещение отдельных тел березита контролируется разрывными "
    "нарушениями, зонами рассланцевания и катаклаза."
)
PYRITE = (
    "Пирит довольно широко распространён, встречается обычно в виде кубических "
    "метакристаллов. Содержание пирита в некоторых фрагментах пород может достигать 5 %."
)
SOURCES = [
    {"n": 1, "doc_id": "a", "title": "Золото Среднего Урала", "year": 2025, "text": PYRITE},
    {"n": 2, "doc_id": "d", "title": "Геологический словарь", "kind": "словарь", "text": BERESITE},
]


def test_style_from_environment() -> None:
    with mock.patch.dict(os.environ, {"GEORAG_PROMPT": "quote"}):
        assert prompt_style() == "quote"
        assert Settings().prompt == "quote"


def test_quote_messages_use_quote_prompt_and_notes() -> None:
    messages = M.build_messages("Что такое березит?", SOURCES, [], style="quote")
    assert messages[0]["content"] == M.QUOTE_SYSTEM
    user = messages[-1]["content"]
    assert "Отвечай выписками из этих фрагментов" in user
    assert "первой приведи выписку определения из словаря [2]" in user
    assert "по Геологическому словарю [N]" not in user  # пометка пересказа
    assert NO_ANSWER_QUOTE in M.QUOTE_SYSTEM  # по ней код узнаёт отказ


def test_quote_prompt_has_no_single_article_note() -> None:
    same = [{**SOURCES[0], "n": 1}, {**SOURCES[0], "n": 2, "text": BERESITE}]
    assert "из одного источника" in M.build_messages("Вопрос?", same, [])[-1]["content"]
    assert "из одного источника" not in M.build_messages("Вопрос?", same, [], style="quote")[-1][
        "content"
    ]


def test_refusal_phrase_of_quote_prompt() -> None:
    assert is_refusal("В предоставленных фрагментах нет ответа на этот вопрос.")
    assert says_not_in_base("В предоставленных фрагментах нет ответа на этот вопрос.")
    assert not is_refusal("Во фрагментах приведены сведения, противоречащие предпосылке вопроса:")


def test_verbatim_quote_stays() -> None:
    answer = f"[2] «Геологический словарь»\n{BERESITE}\n\n[1] «Золото Среднего Урала» (2025)\n{PYRITE}"
    text, removed, (kept, total) = check_quotes(answer, SOURCES)
    assert text == answer
    assert removed == []
    assert kept == total == 4


def test_words_of_the_model_are_removed() -> None:
    added = "Таким образом, березиты служат надёжным поисковым признаком золотого оруденения."
    answer = f"[2] «Геологический словарь»\n{BERESITE} {added}"
    text, removed, (kept, total) = check_quotes(answer, SOURCES)
    assert added not in text
    assert removed == [added]
    assert "катаклаза." in text
    assert (kept, total) == (2, 3)


def test_quote_without_fragment_number_is_removed() -> None:
    text, removed, _ = check_quotes(f"{BERESITE}\n\n[1]\n{PYRITE}", SOURCES)
    assert "Березит" not in text
    assert len(removed) == 2
    assert text.startswith("[1]")


def test_block_left_with_heading_only_is_dropped() -> None:
    invented = "Золото здесь связано исключительно с кварцевыми жилами северо-западного простирания."
    answer = f"[1] «Золото Среднего Урала» (2025)\n{invented}\n\n[2]\n{BERESITE}"
    text, removed, _ = check_quotes(answer, SOURCES)
    assert "Золото Среднего Урала" not in text
    assert removed == [invented]
    assert text.startswith("[2]")


def test_text_of_another_fragment_stays() -> None:
    answer = f"[2]\n{PYRITE}"
    text, removed, (kept, total) = check_quotes(answer, SOURCES)
    assert removed == []
    assert kept == total == 2
    assert text == answer


def test_service_phrases_of_the_prompt_stay() -> None:
    answer = (
        "Во фрагментах приведены сведения, противоречащие предпосылке вопроса:\n\n"
        f"[2]\n{BERESITE}\n\nНе раскрыта часть вопроса о том, из какой магмы они кристаллизуются."
    )
    text, removed, _ = check_quotes(answer, SOURCES)
    assert text == answer
    assert removed == []


def test_translation_of_foreign_fragment_is_not_checked() -> None:
    english = {
        "n": 1,
        "doc_id": "e",
        "title": "Gold of the Urals",
        "text": "Pyrite is widespread and usually occurs as cubic metacrystals in altered rocks.",
    }
    assert is_foreign(english["text"])
    answer = "Перевод фрагмента [1]\nПирит широко распространён и обычно встречается в виде кубических метакристаллов."
    text, removed, (kept, total) = check_quotes(answer, [english])
    assert text == answer
    assert removed == []
    assert total == 0


def test_unknown_fragment_number_is_removed() -> None:
    text, removed, _ = check_quotes(f"[7]\n{BERESITE}\n\n[1]\n{PYRITE}", SOURCES)
    assert "Березит" not in text
    assert text.startswith("[1]")


def test_translation_mark_on_russian_fragment_is_checked() -> None:
    """«Перевод фрагмента [N]» у русского фрагмента — не повод не сверять: модель так
    помечала и русские фрагменты (k02, 11.11.2026)."""
    invented = "Березиты образуются только при остывании гранитной магмы на большой глубине."
    text, removed, _ = check_quotes(f"Перевод фрагмента [2]: {invented}", SOURCES)
    assert removed == [f"Перевод фрагмента [2]: {invented}"]
    assert text == ""


SYNTHESIS = (
    "**Сопоставление по выпискам**\n"
    "Березит — метасоматическая порода с кварцем и серицитом [2], а пирит в нём достигает 5 % [1].\n"
    "Поэтому березиты всегда богаче золотом, чем граниты."
)


def test_comparison_question_gets_the_note() -> None:
    assert M.asks_comparison("Чем орогенные месторождения отличаются от эпитермальных?")
    assert M.asks_comparison("Почему узлы пересечения разломов благоприятны для оруденения?")
    assert not M.asks_comparison("Что такое березит?")
    user = M.build_messages("Чем березит отличается от лиственита?", SOURCES, [], style="quote")
    assert "Сопоставление по выпискам" in user[-1]["content"]
    plain = M.build_messages("Что такое березит?", SOURCES, [], style="quote")
    assert "правилу 11" not in plain[-1]["content"]
    assert "Сопоставление по выпискам" in M.QUOTE_SYSTEM


def test_synthesis_is_split_off_and_keeps_only_cited_sentences() -> None:
    answer = f"[2]\n{BERESITE}\n\n{SYNTHESIS}"
    quotes, section = split_synthesis(answer)
    assert quotes == f"[2]\n{BERESITE}"
    assert section.startswith("**Сопоставление по выпискам**")
    text, removed = check_synthesis(section, SOURCES)
    assert "пирит в нём достигает 5 % [1]" in text
    assert removed == ["Поэтому березиты всегда богаче золотом, чем граниты."]


def test_synthesis_with_unknown_number_or_no_sentences_is_dropped() -> None:
    text, removed = check_synthesis(
        "**Сопоставление по выпискам**\nБерезиты и листвениты образуются в одних зонах разломов [7].",
        SOURCES,
    )
    assert text == ""
    assert len(removed) == 1


def test_no_synthesis_section() -> None:
    assert split_synthesis(f"[2]\n{BERESITE}") == (f"[2]\n{BERESITE}", "")


def test_synthesis_drops_lines_of_bare_numbers() -> None:
    text, removed = check_synthesis(
        "**Сопоставление по выпискам**\nПирит в березитах достигает 5 % по данным разреза [1].\n[7][8]",
        SOURCES,
    )
    assert "[7][8]" not in text
    assert removed == []
