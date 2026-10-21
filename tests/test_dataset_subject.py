"""Предмет вопроса без графа и модели (georag/chat/dataset.py): предмет вопроса о связях
по тексту, основы имён собственных и проверка, что сущность графа их содержит."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat.dataset import named, proper_stems, subject_in


# --------------------------------------------------------------------------- #
#  Предмет вопроса о связях — по тексту
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "subject"),
    [
        ("С какими породами связаны мигматиты?", "мигматиты"),
        ("С чем связана аргиллизация?", "аргиллизация"),
        ("С чем связана аргиллизация", "аргиллизация"),
        ("С какими процессами и минералами связана пропилитизация?", "пропилитизация"),
        ("Что сопровождает золотое оруденение в черносланцевых толщах?", "золотое оруденение"),
        ("Какие минералы ассоциируют с самородным золотом?", "самородным золотом"),
        ("С чем ассоциирует пирит или арсенопирит?", "пирит"),
        ("С ЧЕМ СВЯЗАНА АРГИЛЛИЗАЦИЯ?", "АРГИЛЛИЗАЦИЯ"),
    ],
)
def test_subject_in_after_link_verb_without_tail(question: str, subject: str) -> None:
    assert subject_in(question) == subject


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("Как связаны разломы и оруденение?", id="link-between-two"),
        pytest.param("Где находится месторождение?", id="no-link-verb"),
        pytest.param("С чем связан Au?", id="too-short"),
        pytest.param("Хлорит связан с чем?", id="question-word"),
        pytest.param(
            "С чем связаны крупные золоторудные месторождения черносланцевых толщ Ленского "
            "района?",
            id="too-long",
        ),
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
    ],
)
def test_subject_in_unknown_gives_none(question: Any) -> None:
    assert subject_in(question) is None


# --------------------------------------------------------------------------- #
#  Основы имён собственных
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "stems"),
    [
        pytest.param("Что известно о Билляхской зоне разломов?", {"билля"}, id="adjective"),
        pytest.param("Что известно об Озерном?", {"озерн"}, id="capital-word"),
        pytest.param("Какая свита содержит месторождение Polaris?", {"polar"}, id="latin"),
        pytest.param("шахтаминский комплекс", {"шахта"}, id="lowercase-adjective"),
        pytest.param(
            "Что известно о Северо-Восточном Донбассе?", {"север", "донба"}, id="hyphenated"
        ),
        pytest.param(
            "Что известно о Кокпатасе?\nИ о Даугызтау?", {"кокпа", "даугы"}, id="two-sentences"
        ),
        pytest.param("Хлорит. Билляхская зона разломов.", {"билля"}, id="sentence-start"),
        pytest.param("С какими минералами ассоциирует хлорит?", set(), id="common-nouns"),
        pytest.param("геологическое строение", set(), id="adjective-not-from-name"),
        pytest.param("С чем связан хлорит? Собери сведения.", set(), id="instruction"),
        pytest.param("Что известно о Ба и Fe?", set(), id="short-words"),
        pytest.param("", set(), id="empty"),
        pytest.param(None, set(), id="none"),
    ],
)
def test_proper_stems(text: Any, stems: set[str]) -> None:
    assert proper_stems(text) == stems


# --------------------------------------------------------------------------- #
#  Сущность графа должна содержать имя собственное вопроса
# --------------------------------------------------------------------------- #
BILLYAKH = "Что известно о Билляхской зоне разломов?"


@pytest.mark.parametrize(
    ("name", "hint", "question", "expected"),
    [
        pytest.param(
            "зонам разломов", "Билляхская зона разломов", BILLYAKH, None, id="general-instead"
        ),
        pytest.param(
            "Билляхская зона", "Билляхская зона разломов", BILLYAKH, "Билляхская зона", id="same"
        ),
        pytest.param("зонам разломов", None, BILLYAKH, None, id="name-from-question"),
        pytest.param(
            "хлорит", None, "С какими минералами ассоциирует хлорит?", "хлорит", id="no-name"
        ),
        pytest.param("chlorite", "хлорит", "С чем связан хлорит?", "chlorite", id="translation"),
        pytest.param(None, "хлорит", "С чем связан хлорит?", None, id="no-entity"),
        pytest.param("", "Билляхская зона", BILLYAKH, "", id="empty-entity"),
    ],
)
def test_named(name: Any, hint: Any, question: str, expected: Any) -> None:
    assert named(name, hint, question) == expected


@pytest.mark.xfail(
    reason="подсказка модели — название, а не предложение, но proper_stems считает её первое "
    "слово началом фразы: имя собственное в начале подсказки («Озерное месторождение») не "
    "видно, и вместо Озерного годится любое «месторождения» графа",
    raises=AssertionError,
    strict=True,
)
def test_named_proper_name_at_start_of_hint() -> None:
    question = "Что известно об Озерном месторождении?"
    assert named("месторождения", "Озерное месторождение", question) is None
