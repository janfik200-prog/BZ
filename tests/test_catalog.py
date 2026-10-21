"""Каталожные вопросы (chat/catalog.py): разбор, запрос к базе, ответ — без модели и базы."""

from __future__ import annotations

from typing import Any

import pytest
from fakedb import FakeConn

from georag.chat import answer as chat
from georag.chat import catalog
from georag.chat.plan import Part, Plan, parse_plan
from georag.chat.settings import Settings
from georag.tg import ui

FOUND = [
    {
        "doc_id": "a",
        "title": "Gold veins of the Massif Central, France",
        "year": 2024,
        "authors": ["Jean Martin", "Paul Durand", "Anne Petit", "Luc Bernard"],
        "journal": None,
        "url": "",
        "doi": "10.1/a",
        "by_title": True,
        "hits": 9,
    },
    {
        "doc_id": "b",
        "title": "Variscan faults",
        "year": 2019,
        "authors": [],
        "journal": None,
        "url": "",
        "doi": "",
        "by_title": False,
        "hits": 3,
    },
]


# --------------------------------------------------------------------------- #
#  Разбор и чистка вывода модели
# --------------------------------------------------------------------------- #
def test_parse_catalog_from_plan() -> None:
    data = {
        "части": [{"вопрос": "Сколько статей про Францию?", "запросы": ["Франция"]}],
        "по_теме": False,  # модель сочла вопрос о статьях не геологией
        "каталог": {
            "тема": "Франция",
            "основы": ["France", "French", "Франц", "Fr", "France"],
            "год_от": "2015",
            "год_до": None,
            "автор": "null",
        },
    }
    plan = parse_plan(data, "?", Settings())
    q = plan.catalog
    assert q is not None and plan.on_topic  # каталожный вопрос — всегда по теме
    assert q.stems == ["France", "French", "Франц"]  # короткие и повторы — вон
    assert (q.topic, q.year_from, q.year_to, q.author) == ("Франция", 2015, None, "")
    plain = parse_plan({"части": [], "каталог": None, "по_теме": False}, "?", Settings())
    assert plain.catalog is None and not plain.on_topic


def test_parse_cleans_model_output() -> None:
    """Основы строкой, слова «статьи», годы и диапазоны, фамилия автора среди основ —
    убираются."""
    q = catalog.parse(
        {
            "тема": "статьи 2020–2023 годов про Sentinel-1",
            "основы": "Sentinel-1, радар 2021, 2020-2023, France articles; papers",
            "год_от": 2020,
            "год_до": "2023",
        }
    )
    assert q is not None
    assert q.stems == ["Sentinel-1", "радар", "France"]
    assert q.topic == "Sentinel-1" and (q.year_from, q.year_to) == (2020, 2023)
    q = catalog.parse(
        {
            "тема": "статьи Иваситы о золоте",
            "основы": ["ивасита", "Iwasita", "работы Иваситы", "Ивасита, gold"],
            "автор": "Iwasita",
        }
    )
    assert q is not None and q.stems == ["gold"] and q.author == "Iwasita"
    assert catalog._author_key("Iwasita") == "Iwas"  # сходится с Iwashita в базе
    assert catalog._author_key("Inoue") == "Inou"
    assert catalog._author_key("Smith, J.") == "Smit"


# --------------------------------------------------------------------------- #
#  Запрос к базе (подставная база проверяет подстановки, как psycopg)
# --------------------------------------------------------------------------- #
def test_find_builds_valid_query() -> None:
    conn = FakeConn(rows=[("a", "T", 2020, ["X"], None, "", "", True, 4)])
    query = catalog.CatalogQuery(
        stems=["France", "SAR"], year_from=2015, year_to=2023, author="Inoue"
    )
    found = catalog.find(conn, query)
    sql, params = conn.log[-1]
    assert found[0]["by_title"] and found[0]["hits"] == 4
    assert params["s0"] == "%France%" and params["s1"] == r"\mSAR\M"  # короткое — целым словом
    assert params["author"] == r"\mInou" and params["min_chunks"] == catalog.MIN_CHUNKS
    assert "d.year >= %(year_from)s" in sql and "d.year <= %(year_to)s" in sql
    assert sql.index("by_title DESC") < sql.index("hits DESC") < sql.index("d.year DESC")
    assert catalog.author_pattern("Ivasita") == r"\mI[vw]as"


def test_find_without_topic_lists_by_author() -> None:
    conn = FakeConn(rows=[])
    catalog.find(conn, catalog.CatalogQuery(author="Iwashita"))
    sql, params = conn.log[-1]
    assert "WHERE (true)" in sql and params["author"] == r"\mI[vw]ash"


# --------------------------------------------------------------------------- #
#  Ответ собирает код
# --------------------------------------------------------------------------- #
def test_describe_counts_and_lists() -> None:
    query = catalog.CatalogQuery(stems=["France"], topic="Франция")
    head, listing = catalog.describe(query, FOUND)
    assert head.startswith("**В базе знаний 2 статьи о теме «Франция».**")
    assert "в названии — у 1, только в тексте" in head.lower() and "Искал: France" in head
    first, second = listing.splitlines()
    assert first == (
        "1. Gold veins of the Massif Central, France (2024). Jean Martin, Paul Durand, "
        "Anne Petit и др. — в названии"
    )
    assert second == "2. Variscan faults (2019) — в тексте, кусков: 3"
    head, listing = catalog.describe(catalog.CatalogQuery(author="Inoue"), [])
    assert "0 статей с автором «Inoue»" in head and listing == ""


def test_describe_author_as_written_in_base_and_years() -> None:
    found = [dict(FOUND[1], authors=["Kenji Iwashita"])]
    query = catalog.CatalogQuery(author="Iwasita", year_from=2020, year_to=2023)
    head, _ = catalog.describe(query, found)
    assert "1 статья с автором «Kenji Iwashita» за 2020–2023 годы" in head


def test_describe_long_list_is_cut() -> None:
    many = [dict(FOUND[1], doc_id=str(i)) for i in range(catalog.LIST_MAX + 5)]
    head, listing = catalog.describe(catalog.CatalogQuery(stems=["fault"]), many)
    assert len(listing.splitlines()) == catalog.LIST_MAX and "остальные 5" in head


def test_catalog_question_skips_fragment_search(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = Plan(
        [Part("Сколько статей про Францию?", [])],
        catalog=catalog.CatalogQuery(stems=["France"], topic="Франция"),
    )
    asked: list[Any] = []

    def find(conn: Any, query: Any) -> list[dict[str, Any]]:
        asked.append(query.stems)
        return FOUND

    def no_search(*a: Any, **k: Any) -> Any:
        raise AssertionError("каталожный вопрос не ищет по фрагментам")

    monkeypatch.setattr(chat, "retrieve", no_search)
    prepared = yield_from(
        chat.prepare(
            None,
            None,
            "Сколько статей про Францию?",
            [],
            Settings(),
            lambda q, h, s: plan,
            catalog=find,
        )
    )
    events = list(
        chat.answer(None, None, "Сколько статей про Францию?", [], Settings(), prepared=prepared)
    )
    done = events[-1]
    assert asked == [["France"]]
    assert done["mode"] == "каталог" and done["catalog"]["count"] == 2
    assert done["tokens"] is None and not [e for e in events if e["type"] == "general"]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "2 статьи" in text and "Variscan faults" in text
    assert "без модели" in ui.answer_text(text, None, [], done)


def yield_from(gen: Any) -> Any:
    """Результат генератора prepare (return), события status — мимо."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


@pytest.mark.parametrize(
    ("author", "match", "miss"),
    [
        ("Хашим", "Mazlan Hashim", "Ivan Petrov"),  # модель написала автора кириллицей
        ("Ивасита", "Kenji Iwashita", "Mazlan Hashim"),
        ("Iwasita", "Kenji Iwashita", "Ivan Petrov"),
    ],
)
def test_author_pattern_any_script(author: str, match: str, miss: str) -> None:
    import re

    rx = re.compile(catalog.author_pattern(author).replace(r"\m", r"\b"), re.IGNORECASE)
    assert rx.search(match) and not rx.search(miss)


@pytest.mark.parametrize(
    "data",
    [
        # Живая проверка 21.10.2026: «научные работы» оставалось основой «научные».
        {
            "тема": "работы Иваситы",
            "основы": ["работы", "статьи", "научные работы"],
            "автор": "Ивасита",
        },
        {"тема": "статьи за 2020 год", "основы": ["2020", "статьи"], "год_от": 2020},
    ],
)
def test_topic_only_author_or_years_has_no_stems(data: dict[str, Any]) -> None:
    q = catalog.parse(data)
    assert q is not None and q.stems == [] and q.topic == ""
