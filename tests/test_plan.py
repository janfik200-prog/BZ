"""Разбор вопроса (georag/chat/plan.py): узкий вопрос, вопрос о связях, запросы модели
и её JSON-разбор. Ответ модели подаётся готовым словарём — сервер модели не нужен."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat.plan import Plan, about_links, clean_queries, is_narrow, parse_plan
from georag.chat.settings import Settings

NARROW = "Какая свита содержит месторождение Polaris?"
WIDE = "Что известно о хлорите?"


def parts_of(plan: Plan) -> list[tuple[str, list[str]]]:
    return [(p.question, p.queries) for p in plan.parts]


# --------------------------------------------------------------------------- #
#  Узкий вопрос — об одном факте
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "Какая свита содержит месторождение Polaris в районе Нунавута?",
        "Где изучали систему Кап Симпсон с помощью HyMap?",
        "С каким минералом ассоциирует каолинит в месторождении Cuprite?",
        "В каком районе находится месторождение Polaris?",
        "Когда сформировалось месторождение Сухой Лог?",
        "Сколько стадий оруденения выделяют на Кокпатасе?",
        "Кто описал Билляхскую зону?",
        "Какие минералы замещает хлорит?",
        "  Где\nизучали   систему Кап Симпсон?",
    ],
)
def test_is_narrow_single_fact_questions(question: str) -> None:
    assert is_narrow(question)


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("С какими минералами ассоциирует хлорит?", id="plural-relations"),
        pytest.param("Какие гидротермальные изменения видны на снимках ASTER?", id="plural"),
        pytest.param("Какую роль играют узлы пересечения разломов?", id="role"),
        pytest.param("Какой минерал образуется и как его выявляют?", id="and-how"),
        pytest.param("Чем отличается аргиллизация от пропилитизации?", id="difference"),
        pytest.param("Где находится Кокпатас? Когда его открыли?", id="two-questions"),
        pytest.param("Сравни Кокпатас и Даугызтау: какой богаче?", id="compare"),
        pytest.param("Что такое сдвиговые зоны?", id="definition"),
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
    ],
)
def test_is_narrow_false(question: Any) -> None:
    assert not is_narrow(question)


# --------------------------------------------------------------------------- #
#  Вопрос о связях предмета — его ведёт граф
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "С какими минералами связано золото?",
        "С чем связаны линеаменты?",
        "Что контролируют разломы и с чем они связаны?",
        "С какими минералами ассоциирует хлорит?",
        "Какие метасоматические изменения сопровождают золотое оруденение?",
        "Где встречается арсенопирит?",
        "Что влияет на размещение оруденения?",
        "С чем сопряжены дайки?",
        "С ЧЕМ СВЯЗАНО ЗОЛОТО?",
    ],
)
def test_about_links_true(question: str) -> None:
    assert about_links(question)


@pytest.mark.parametrize(
    "question",
    [
        "Как выделяют рудные узлы по плотности линеаментов?",
        "Что такое сдвиговые зоны?",
        "Как связаны разломы и оруденение?",
        "",
    ],
)
def test_about_links_false(question: str) -> None:
    assert not about_links(question)


# --------------------------------------------------------------------------- #
#  Запросы от модели
# --------------------------------------------------------------------------- #
def test_clean_queries_quotes_spaces_and_repeats() -> None:
    values = ['"золото" Донбасса', "  золото   Донбасса ", "Gold ore", "gold ORE"]
    assert clean_queries(values, 10) == ["золото Донбасса", "Gold ore"]


@pytest.mark.parametrize(
    ("query", "kept"),
    [
        pytest.param("a", False, id="1-char"),
        pytest.param("ab", True, id="2-chars"),
        pytest.param("x" * 120, True, id="120-chars"),
        pytest.param("x" * 121, False, id="121-chars"),
        pytest.param('""', False, id="only-quotes"),
        pytest.param("   ", False, id="only-spaces"),
    ],
)
def test_clean_queries_length_bounds(query: str, kept: bool) -> None:
    assert (clean_queries([query], 5) != []) is kept


@pytest.mark.parametrize(
    ("values", "limit", "expected"),
    [
        pytest.param(["q1", "q2", "q3", "q4"], 2, ["q1", "q2"], id="limit"),
        pytest.param(["q1", "Q1", "q2", "q3"], 2, ["q1", "q2"], id="limit-after-dedup"),
        pytest.param(["q1"], 0, [], id="zero-limit"),
        pytest.param(None, 3, [], id="none"),
        pytest.param([], 3, [], id="empty"),
        pytest.param("золото Донбасса", 3, [], id="string-not-list"),
    ],
)
def test_clean_queries_limits_and_empty(values: Any, limit: int, expected: list[str]) -> None:
    assert clean_queries(values, limit) == expected


@pytest.mark.xfail(
    reason="clean_queries превращает не-строки в запросы через str(): null → «None», "
    "объект → его repr; по parse_plan «непонятное — без запросов»",
    raises=AssertionError,
    strict=True,
)
def test_clean_queries_skips_non_strings() -> None:
    assert clean_queries([None, {"a": 1}, ["x", "y"], "золото"], 5) == ["золото"]


# --------------------------------------------------------------------------- #
#  Разбор вопроса из JSON модели
# --------------------------------------------------------------------------- #
def test_parse_plan_parts_queries_subject() -> None:
    """Части с запросами без повторов, не больше settings.parts; предмет и «всё»."""
    data = {
        "вопрос_целиком": "Какие признаки у рудных узлов и какими методами их выделяют?",
        "части": [
            {
                "вопрос": "какие признаки у  рудных узлов",
                "запросы": [
                    "признаки рудных узлов",
                    "ore cluster criteria",
                    "Ore cluster criteria",
                ],
            },
            {"вопрос": "какими методами их выделяют", "запросы": ["методы выделения узлов"]},
            {"вопрос": "третья", "запросы": ["q3"]},
            {"вопрос": "четвёртая", "запросы": ["q4"]},
        ],
        "предмет": "Анабарский щит",
        "всё": False,
        "по_теме": True,
    }
    plan = parse_plan(data, "вопрос", Settings())
    assert parts_of(plan) == [
        ("какие признаки у рудных узлов", ["признаки рудных узлов", "ore cluster criteria"]),
        ("какими методами их выделяют", ["методы выделения узлов"]),
        ("третья", ["q3"]),
    ]
    assert plan.subject == "Анабарский щит"
    assert plan.collect is False
    assert plan.on_topic is True
    assert plan.standalone == data["вопрос_целиком"]


def test_parse_plan_settings_limit_parts_and_queries() -> None:
    data = {"части": [{"вопрос": f"ч{i}", "запросы": ["a1", "a2", "a3"]} for i in range(5)]}
    plan = parse_plan(data, "вопрос", Settings(parts=2, queries=2))
    assert parts_of(plan) == [("ч0", ["a1", "a2"]), ("ч1", ["a1", "a2"])]
    assert len(parse_plan(data, "вопрос", Settings(parts=0)).parts) == 1  # хотя бы одна


def test_parse_plan_english_keys() -> None:
    data = {
        "parts": [{"question": "minerals", "queries": ["chlorite minerals"]}],
        "subject": "chlorite",
        "collect": "true",
        "on_topic": False,
        "standalone": "  What   minerals? ",
    }
    plan = parse_plan(data, "вопрос", Settings())
    assert parts_of(plan) == [("minerals", ["chlorite minerals"])]
    assert (plan.subject, plan.collect, plan.on_topic, plan.standalone) == (
        "chlorite",
        True,
        False,
        "What minerals?",
    )


def test_parse_plan_old_format_only_queries() -> None:
    plan = parse_plan({"queries": ["a1", "b2"]}, "вопрос", Settings())
    assert parts_of(plan) == [("вопрос", ["a1", "b2"])]


def test_parse_plan_part_without_text_takes_question() -> None:
    """Часть без вопроса, но с запросами — вопрос целиком; без того и другого — пропуск."""
    data = {"части": [{"вопрос": "", "запросы": ["q1"]}, {"вопрос": " ", "запросы": []}]}
    assert parts_of(parse_plan(data, "вопрос", Settings())) == [("вопрос", ["q1"])]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("ерунда", id="string"),
        pytest.param(None, id="none"),
        pytest.param([1, 2], id="list"),
        pytest.param(5, id="number"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"части": "строка"}, id="parts-string"),
        pytest.param({"части": None}, id="parts-null"),
        pytest.param({"части": [None, 3, "x", {}]}, id="parts-garbage-items"),
    ],
)
def test_parse_plan_garbage_gives_one_part_without_queries(data: Any) -> None:
    plan = parse_plan(data, WIDE, Settings())
    assert parts_of(plan) == [(WIDE, [])]
    assert plan.subject is None
    assert plan.collect is False
    assert plan.on_topic is True
    assert plan.standalone == ""


@pytest.mark.xfail(
    reason="parse_plan падает TypeError, если «части» или «запросы» — число или true: "
    "по строке документации непонятное — один вопрос без запросов, а исключение "
    "из plan_question никто не ловит",
    raises=TypeError,
    strict=True,
)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"части": 5}, id="parts-number"),
        pytest.param({"части": [{"вопрос": "x", "запросы": 5}]}, id="queries-number"),
        pytest.param({"части": [{"вопрос": "x", "запросы": True}]}, id="queries-true"),
        pytest.param({"queries": 5}, id="old-format-number"),
    ],
)
def test_parse_plan_non_iterable_values_do_not_crash(data: Any) -> None:
    plan = parse_plan(data, WIDE, Settings())
    assert [p.queries for p in plan.parts] == [[]]


@pytest.mark.parametrize(
    ("value", "subject"),
    [
        pytest.param("  Анабарский\n щит ", "Анабарский щит", id="spaces"),
        pytest.param("null", None, id="null-string"),
        pytest.param("Null", None, id="null-case"),
        pytest.param("None", None, id="none-string"),
        pytest.param("нет", None, id="net"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="blank"),
        pytest.param(None, None, id="null"),
        pytest.param(5, None, id="number"),
        pytest.param(["золото"], None, id="list"),
    ],
)
def test_parse_plan_subject_values(value: Any, subject: str | None) -> None:
    assert parse_plan({"предмет": value}, "вопрос", Settings()).subject == subject


def test_parse_plan_subject_from_territory_key() -> None:
    assert parse_plan({"территория": "Донбасс"}, "вопрос", Settings()).subject == "Донбасс"


@pytest.mark.parametrize(
    ("value", "collect"),
    [
        (True, True), ("true", True), ("True", True), ("да", True), ("Да", True), ("1", True),
        (1, True), (False, False), ("false", False), ("нет", False), (None, False), (0, False),
        ("может быть", False),
    ],
)  # fmt: skip
def test_parse_plan_collect_values(value: Any, collect: bool) -> None:
    assert parse_plan({"всё": value}, "вопрос", Settings()).collect is collect


def test_parse_plan_collect_key_without_yo() -> None:
    assert parse_plan({"все": True}, "вопрос", Settings()).collect is True


def test_parse_plan_links_question_collects_even_if_model_said_no() -> None:
    """Вопрос о связях идёт через граф, даже если модель не отметила «всё»."""
    data = {"части": [], "предмет": "золото", "всё": False}
    assert parse_plan(data, "С какими минералами связано золото?", Settings()).collect


@pytest.mark.parametrize(
    ("data", "on_topic"),
    [
        ({"по_теме": False}, False), ({"по_теме": "false"}, False), ({"по_теме": "нет"}, False),
        ({"по_теме": "0"}, False), ({"по_теме": 0}, False), ({"on_topic": False}, False),
        ({"по_теме": True}, True), ({"по_теме": None}, True), ({}, True),
    ],
)  # fmt: skip
def test_parse_plan_on_topic_values(data: dict[str, Any], on_topic: bool) -> None:
    assert parse_plan(data, "вопрос", Settings()).on_topic is on_topic


def test_parse_plan_standalone_collapsed_and_cut() -> None:
    plan = parse_plan({"вопрос_целиком": "слово\n " * 200}, "вопрос", Settings())
    assert len(plan.standalone) == 300
    assert "\n" not in plan.standalone and "  " not in plan.standalone


def test_parse_plan_narrow_question_merges_parts() -> None:
    """Узкий вопрос — одна часть, запросы частей вперемешку, не больше queries + 1."""
    data = {
        "части": [
            {"вопрос": "какая свита", "запросы": ["Polaris свита", "Polaris formation"]},
            {"вопрос": "где район", "запросы": ["Нунавут", "Nunavut district"]},
            {"вопрос": "что ещё", "запросы": ["Polaris Zn-Pb", "Polaris carbonate"]},
        ]
    }
    plan = parse_plan(data, NARROW, Settings())
    assert plan.narrow
    assert parts_of(plan) == [
        (NARROW, ["Polaris свита", "Нунавут", "Polaris Zn-Pb", "Polaris formation"])
    ]


def test_parse_plan_wide_question_keeps_parts() -> None:
    data = {"части": [{"вопрос": "a", "запросы": ["a1"]}, {"вопрос": "b", "запросы": ["b1"]}]}
    plan = parse_plan(data, "Какие свиты и районы связаны с Polaris?", Settings())
    assert not plan.narrow
    assert parts_of(plan) == [("a", ["a1"]), ("b", ["b1"])]


def test_parse_plan_narrow_but_collect_is_not_narrow() -> None:
    """«Где встречается X?» — и узкий, и о связях: сбор важнее, вопрос не узкий."""
    data = {"части": [{"вопрос": "a", "запросы": ["a1"]}]}
    plan = parse_plan(data, "Где встречается арсенопирит?", Settings())
    assert plan.collect and not plan.narrow


@pytest.mark.xfail(
    reason="запросы частей узкого вопроса сводятся через zip(..., strict=False), который "
    "обрывается на самой короткой части: лишние запросы длинных частей теряются",
    raises=AssertionError,
    strict=True,
)
def test_parse_plan_narrow_merge_keeps_queries_of_longer_part() -> None:
    data = {
        "части": [
            {"вопрос": "a", "запросы": ["a1", "a2", "a3"]},
            {"вопрос": "b", "запросы": ["b1"]},
        ]
    }
    assert parse_plan(data, NARROW, Settings()).parts[0].queries == ["a1", "b1", "a2", "a3"]


@pytest.mark.xfail(
    reason="часть узкого вопроса без запросов обнуляет все запросы: zip с пустым списком "
    "пуст, и поиск идёт только по самому вопросу",
    raises=AssertionError,
    strict=True,
)
def test_parse_plan_narrow_merge_part_without_queries() -> None:
    data = {
        "части": [
            {"вопрос": "a", "запросы": ["a1", "a2"]},
            {"вопрос": "b", "запросы": []},
        ]
    }
    assert parse_plan(data, NARROW, Settings()).parts[0].queries == ["a1", "a2"]


@pytest.mark.xfail(
    reason="одинаковые запросы разных частей при сведении остаются дважды: проверка "
    "«q not in merged» в генераторе списка видит merged до добавления всей группы, "
    "и повтор занимает место среди queries + 1",
    raises=AssertionError,
    strict=True,
)
def test_parse_plan_narrow_merge_has_no_repeats() -> None:
    data = {
        "части": [
            {"вопрос": "a", "запросы": ["Polaris", "a2"]},
            {"вопрос": "b", "запросы": ["Polaris", "b2"]},
        ]
    }
    assert parse_plan(data, NARROW, Settings()).parts[0].queries == ["Polaris", "a2", "b2"]
