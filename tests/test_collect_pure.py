"""Сбор по предмету (georag/chat/collect.py) — только чистые функции: о чём вопрос,
ключ написания, разбор выписки модели, порядок статей, сводка, хвост ответа и задание
на выписку. База, модель и закрытые («_») имена модуля не нужны."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat import collect as CO
from georag.graph import synonyms


@pytest.fixture(autouse=True)
def no_dictionary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка названия по словарю синонимов — с пустым словарём, а не с config/."""
    empty = synonyms.Synonyms()
    monkeypatch.setattr(synonyms, "load", lambda *a, **kw: empty)
    monkeypatch.setattr(synonyms, "cached", lambda *a, **kw: empty)


def row(cid: int, doc: str, rank: float, text: str = "") -> dict[str, Any]:
    """Фрагмент с упоминанием предмета, как его отдаёт полнотекст (mentioning)."""
    return {
        "chunk_id": cid,
        "doc_id": doc,
        "ord": cid,
        "text": text or "Хлорит ассоциирует с эпидотом в пропилитах.",
        "headings": [],
        "pages": [1],
        "title": f"Статья {doc}",
        "year": 2020,
        "journal": None,
        "url": "",
        "authors": [],
        "rank": rank,
    }


def source(n: int) -> dict[str, Any]:
    return {"n": n, "doc_id": f"d{n}", "title": "Статья", "text": "Текст фрагмента."}


# --------------------------------------------------------------------------- #
#  О чём спрашивают: виды пунктов
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "kinds"),
    [
        ("С какими минералами ассоциирует хлорит?", {"минерал"}),
        ("С какими породами связаны мигматиты?", {"порода"}),
        ("Какие процессы сопровождают окварцевание?", {"процесс"}),
        ("Какие метасоматические изменения связаны с золотом?", {"процесс"}),
        ("С какими разломами связано оруденение?", {"структура"}),
        ("Какими методами выделяют пропилиты?", {"метод"}),
        ("Где встречается арсенопирит?", {"месторождение", "район"}),
        ("Какие минералы и породы вмещают золото?", {"минерал", "порода"}),
        ("minerals associated with chlorite", {"минерал"}),
        ("Что известно о хлорите?", set()),
        ("", set()),
        (None, set()),
    ],
)
def test_facet_by_question_words(question: Any, kinds: set[str]) -> None:
    assert CO.facet(question) == kinds


def test_facet_ignores_words_of_subject() -> None:
    """«методами выделяют зоны гидротермальных изменений» — о методах, не о процессах."""
    question = "Какими методами выделяют зоны гидротермальных изменений?"
    assert CO.facet(question) == {"метод", "процесс"}
    assert CO.facet(question, "зоны гидротермальных изменений") == {"метод"}


def test_facet_kinds_have_target_wording() -> None:
    """Каждый вид, который узнаёт facet, есть в KIND_WORDS — иначе задание без видов."""
    asked = "минералы породы процессы структуры методы месторождения районы"
    assert CO.facet(asked) <= set(CO.KIND_WORDS)


@pytest.mark.xfail(
    reason="основа «данны» (для «какие данные») ловит оборот «по данным статей»: вопрос "
    "«что известно» о предмете становится вопросом о методах, и сводка — только методы",
    raises=AssertionError,
    strict=True,
)
def test_facet_by_data_of_articles_is_not_about_methods() -> None:
    assert CO.facet("Что известно о хлорите по данным статей?", "хлорит") == set()


# --------------------------------------------------------------------------- #
#  Ключ написания: по-английски и по-русски — один
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("english", "russian"),
    [
        ("epidote", "эпидотом"), ("calcite", "кальцит"), ("chalcopyrite", "халькопирит"),
        ("goethite", "гётит"), ("quartz", "кварц"), ("sphalerite", "сфалерит"),
        ("arsenopyrite", "арсенопирит"), ("pyrite", "пирит"), ("chlorite", "хлорит"),
        ("Epidote", "epidote"), ("кварцевые жилы", "кварцевых жил"),
    ],
)  # fmt: skip
def test_spelling_key_same_for_one_name(english: str, russian: str) -> None:
    assert CO.spelling_key(english) == CO.spelling_key(russian)


@pytest.mark.parametrize(
    ("first", "second"),
    [("гранат", "гранит"), ("пирит", "пирротин"), ("магнетит", "магнезит"), ("calcite", "calcium")],
)
def test_spelling_key_differs_for_different_names(first: str, second: str) -> None:
    assert CO.spelling_key(first) != CO.spelling_key(second)


@pytest.mark.parametrize("name", ["", "   ", "—"])
def test_spelling_key_empty(name: str) -> None:
    assert CO.spelling_key(name) == ""


# --------------------------------------------------------------------------- #
#  Оборванный ответ выписки
# --------------------------------------------------------------------------- #
def test_salvage_keeps_whole_items_before_cut() -> None:
    cut = '{"пункты": [{"с_чем": "эпидот", "цитата": "Хлорит ассоциирует"}, {"с_чем": "кал'
    assert CO.salvage(cut) == {"пункты": [{"с_чем": "эпидот", "цитата": "Хлорит ассоциирует"}]}


def test_salvage_accepts_old_key_and_skips_foreign_objects() -> None:
    text = '{"пункты": [{"что": "кварц"}, {"другое": 1}, {\'с_чем\': \'x\'}, {"с_чем": "b"}'
    assert CO.salvage(text) == {"пункты": [{"что": "кварц"}, {"с_чем": "b"}]}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"пункты": [{"с_чем": "эп', id="cut-in-first-item"),
        pytest.param('{"пункты": []}', id="no-items"),
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param("не JSON вовсе", id="not-json"),
    ],
)
def test_salvage_nothing_to_save(text: Any) -> None:
    assert CO.salvage(text) is None


# --------------------------------------------------------------------------- #
#  Пункты выписки: цитата во фрагменте, не о самом предмете, рядом с ним
# --------------------------------------------------------------------------- #
QUOTE = "Хлорит ассоциирует с эпидотом"


def test_check_items_keeps_checked_item_normalised() -> None:
    answer = {
        "пункты": [
            {
                "с_чем": " эпидот. ",
                "вид": "Минерал",
                "как": "ассоциирует  с",
                "где": "пропилиты",
                "цитата": f"«{QUOTE}»",
                "фрагмент": "c1",
            }
        ]
    }
    items, dropped = CO.check_items(answer, [row(1, "a", 0.1)], ["хлорит"])
    assert dropped == 0
    assert items == [
        {
            "что": "эпидот",
            "вид": "минерал",
            "как": "ассоциирует с",
            "где": "пропилиты",
            "цитата": QUOTE,
            "фрагмент": 1,
        }
    ]


@pytest.mark.parametrize(
    ("item", "kind"),
    [
        pytest.param({"вид": "минералы"}, "другое", id="unknown-kind"),
        pytest.param({}, "другое", id="no-kind"),
        pytest.param({"вид": "ПОРОДА"}, "порода", id="case"),
    ],
)
def test_check_items_kind_lowercased_unknown_is_other(item: dict[str, Any], kind: str) -> None:
    answer = {"пункты": [{"с_чем": "эпидот", "цитата": QUOTE, "фрагмент": "c1", **item}]}
    items, _ = CO.check_items(answer, [row(1, "a", 0.1)], ["хлорит"])
    assert [i["вид"] for i in items] == [kind]


def test_check_items_unknown_fragment_number_searches_all_chunks() -> None:
    chunks = [row(1, "a", 0.1, "Другой текст статьи без нужной цитаты."), row(2, "a", 0.2)]
    for number in ("c999", "", None):
        answer = {"пункты": [{"с_чем": "эпидот", "цитата": QUOTE, "фрагмент": number}]}
        items, dropped = CO.check_items(answer, chunks, ["хлорит"])
        assert ([i["фрагмент"] for i in items], dropped) == ([2], 0)


def test_check_items_old_key_and_field_limits() -> None:
    answer = {
        "пункты": [
            {"что": "эпидот", "как": "к" * 100, "где": "г" * 200, "цитата": QUOTE, "фрагмент": 1}
        ]
    }
    items, _ = CO.check_items(answer, [row(1, "a", 0.1)], ["хлорит"])
    assert [(i["что"], len(i["как"]), len(i["где"])) for i in items] == [("эпидот", 40, 80)]


FAR = (
    "Хлорит встречается в пропилитах."
    + " Далее описана другая часть разреза." * (CO.NEAR // 30 + 2)
    + " Кварц можно картировать по тепловым инфракрасным данным."
)


@pytest.mark.parametrize(
    ("item", "text", "names"),
    [
        pytest.param(
            {"с_чем": "золото", "цитата": "хлорит содержит золото", "фрагмент": "c1"},
            "",
            ["хлорит"],
            id="quote-not-in-fragment",
        ),
        pytest.param({"с_чем": "", "цитата": QUOTE}, "", ["хлорит"], id="empty-name"),
        pytest.param({"цитата": QUOTE}, "", ["хлорит"], id="no-name"),
        pytest.param({"с_чем": "эпидот", "цитата": ""}, "", ["хлорит"], id="empty-quote"),
        pytest.param({"с_чем": "эпидот"}, "", ["хлорит"], id="no-quote"),
        pytest.param(
            {"с_чем": "эпидот", "цитата": "с эпидотом"}, "", ["хлорит"], id="too-short-quote"
        ),
        pytest.param({"с_чем": "Хлорита", "цитата": QUOTE}, "", ["хлорит"], id="subject-itself"),
        pytest.param(
            {"с_чем": "chlorite", "цитата": "Chlorite is associated with epidote"},
            "Chlorite is associated with epidote in propylites.",
            ["хлорит", "chlorite"],
            id="subject-other-language",
        ),
        pytest.param({"с_чем": "алмаз", "цитата": QUOTE}, "", ["хлорит"], id="name-not-in-quote"),
        pytest.param(
            {"с_чем": "кварц", "цитата": "Кварц можно картировать по тепловым инфракрасным данным"},
            FAR,
            ["хлорит"],
            id="far-from-subject",
        ),
        pytest.param(
            {"с_чем": "эпидот", "цитата": QUOTE, "фрагмент": "c3"},
            "",
            ["хлорит"],
            id="quote-in-other-fragment",
        ),
        pytest.param("строка", "", ["хлорит"], id="not-object"),
        pytest.param(None, "", ["хлорит"], id="null"),
    ],
)
def test_check_items_drops_unchecked_item(item: Any, text: str, names: list[str]) -> None:
    chunks = [row(1, "a", 0.1, text), row(3, "a", 0.1, "Другой текст статьи, совсем о другом.")]
    assert CO.check_items({"пункты": [item]}, chunks, names) == ([], 1)


def test_check_items_without_names_skips_subject_checks() -> None:
    """Без названий предмета — ни «о самом предмете», ни «рядом с предметом»."""
    item = {"с_чем": "кварц", "цитата": "Кварц можно картировать по тепловым инфракрасным данным"}
    items, dropped = CO.check_items({"пункты": [item]}, [row(2, "a", 0.1, FAR)])
    assert ([i["что"] for i in items], dropped) == (["кварц"], 0)


def test_check_items_counts_kept_and_dropped_together() -> None:
    answer = {
        "пункты": [
            {"с_чем": "эпидот", "цитата": QUOTE, "фрагмент": "c1"},
            {"с_чем": "золото", "цитата": "хлорит содержит золото", "фрагмент": "c1"},
            {"с_чем": "", "цитата": QUOTE},
            {"с_чем": "Хлорита", "цитата": QUOTE},
        ]
    }
    items, dropped = CO.check_items(answer, [row(1, "a", 0.1)], ["хлорит", "chlorite"])
    assert [i["что"] for i in items] == ["эпидот"]
    assert dropped == 3


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param("пункты", id="string"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"пункты": None}, id="items-null"),
        pytest.param({"пункты": "эпидот"}, id="items-string"),
        pytest.param({"пункты": []}, id="items-empty"),
    ],
)
def test_check_items_garbage_answer(answer: Any) -> None:
    assert CO.check_items(answer, [row(1, "a", 0.1)], ["хлорит"]) == ([], 0)


# --------------------------------------------------------------------------- #
#  Статьи по числу упоминаний
# --------------------------------------------------------------------------- #
ROWS = [row(1, "a", 0.1), row(2, "b", 0.5), row(3, "b", 0.2), row(4, "b", 0.9), row(5, "c", 0.3)]


def ids(articles: list[list[dict[str, Any]]]) -> list[list[int]]:
    return [[c["chunk_id"] for c in article] for article in articles]


def test_by_article_more_mentions_first_best_chunks_first() -> None:
    assert ids(CO.by_article(ROWS)) == [[4, 2], [5], [1]]


def test_by_article_deep_read_starts_with_quick_read() -> None:
    """Глубокая выписка — больше фрагментов, и быстрые 2 — их начало."""
    deep = ids(CO.by_article(ROWS, 4))
    assert deep == [[4, 2, 3], [5], [1]]
    assert [d[: CO.CHUNKS_PER_ARTICLE] for d in deep] == ids(CO.by_article(ROWS))


def test_by_article_ties_by_rank_then_doc_then_chunk() -> None:
    rows = [row(1, "z", 0.5), row(2, "y", 0.5), row(3, "x", 0.9)]
    assert ids(CO.by_article(rows)) == [[3], [2], [1]]
    assert ids(CO.by_article([row(7, "z", 0.5), row(3, "z", 0.5)])) == [[3, 7]]


def test_by_article_empty() -> None:
    assert CO.by_article([]) == []


# --------------------------------------------------------------------------- #
#  Сводка — указатель по частоте
# --------------------------------------------------------------------------- #
def test_summary_text_tiers_by_frequency() -> None:
    groups = [
        {"name": "альбит", "sources": [2, 3, 4, 5, 6, 7]},
        {"name": "эпидот", "sources": [2, 3, 5]},
        {"name": "кальцит", "sources": [4, 6]},
        {"name": "кварц", "sources": [4]},
        {"name": "пусто", "sources": []},
    ]
    lines = CO.summary_text("хлорит", groups, 6).split("\n")
    assert "«хлорит»" in lines[0] and "из 6 статей" in lines[0]
    assert lines[1:] == [
        "Чаще всего (в 3 и более статьях): альбит — 6 [2][3][4][5]; эпидот — 3 [2][3][5].",
        "В двух статьях: кальцит — 2 [4][6].",
        "В одной статье: кварц [4].",
    ]


def test_summary_text_without_groups_is_only_head() -> None:
    text = CO.summary_text("хлорит", [], 0)
    assert "\n" not in text and "«хлорит»" in text


def test_summary_text_is_bounded_and_keeps_frequent() -> None:
    """Пунктов больше, чем входит: частые остаются, редкий за пределом — нет."""
    groups = [{"name": f"минерал{i:03d}", "sources": [1, 2, 3]} for i in range(400)]
    groups.append({"name": "редкий", "sources": [5]})
    text = CO.summary_text("хлорит", groups, 30)
    assert len(text) <= CO.SUMMARY_CHARS * 1.02  # заголовки ярусов в предел не входят
    assert "минерал000 — 3" in text
    assert "редкий" not in text


# --------------------------------------------------------------------------- #
#  Хвост ответа: чего из сводки модель не назвала
# --------------------------------------------------------------------------- #
SUMMARY = {
    "kind": "сводка",
    "groups": [
        {"name": "эпидот", "sources": [2, 3, 5]},
        {"name": "кальцит", "sources": [4, 6]},
        {"name": "биотит", "sources": [7]},
    ],
}


def test_rest_of_summary_lists_unnamed_items() -> None:
    rest = CO.rest_of_summary(
        "Хлорит ассоциирует с эпидотом [2][3].", [SUMMARY, *[source(n) for n in range(2, 8)]]
    )
    assert (
        rest
        == "**Ещё в статьях базы** (дополнительные пункты из выписок): кальцит [4][6]; биотит [7]."
    )


@pytest.mark.parametrize(
    ("answer", "sources"),
    [
        pytest.param(
            "эпидот, кальцит и биотит", [SUMMARY, source(2), source(4), source(7)], id="all-named"
        ),
        pytest.param("что угодно", [source(1)], id="no-summary"),
        pytest.param("что угодно", [], id="no-sources"),
        pytest.param("что угодно", [{"kind": "сводка"}], id="summary-without-groups"),
        pytest.param("что угодно", [SUMMARY], id="no-available-articles"),
    ],
)
def test_rest_of_summary_empty(answer: str, sources: list[dict[str, Any]]) -> None:
    assert CO.rest_of_summary(answer, sources) == ""


@pytest.mark.parametrize(
    ("answer", "group"),
    [
        pytest.param("Хлорит с **эпидотом** [2].", {"name": "epidote"}, id="english-in-summary"),
        pytest.param(
            "Хлорит с epidote [2].",
            {"name": "эпидот", "names": ["эпидот", "epidote"]},
            id="other-spelling",
        ),
        pytest.param("Хлорит с гетитом [2].", {"name": "гётит"}, id="yo"),
        pytest.param("Хлорит с гётитом [2].", {"name": "гетит"}, id="ye"),
    ],
)
def test_rest_of_summary_knows_other_spellings(answer: str, group: dict[str, Any]) -> None:
    summary = {"kind": "сводка", "groups": [{**group, "sources": [2]}]}
    assert CO.rest_of_summary(answer, [summary, source(2)]) == ""


def test_rest_of_summary_cites_only_available_and_first_three() -> None:
    summary = {"kind": "сводка", "groups": [{"name": "эпидот", "sources": [2, 25, 3, 4, 5]}]}
    rest = CO.rest_of_summary("", [summary, *[source(n) for n in range(2, 6)]])
    assert rest.endswith(": эпидот [2][3][4].")


def test_rest_of_summary_limits_single_article_items() -> None:
    summary = {
        "kind": "сводка",
        "groups": [{"name": f"один{i}", "sources": [2]} for i in range(40)],
    }
    rest = CO.rest_of_summary("", [summary, source(2)])
    assert rest.count("[2]") == CO.REST_SINGLE


def test_rest_of_summary_limits_all_items() -> None:
    groups = [{"name": f"два{i}", "sources": [2, 3]} for i in range(CO.REST_ITEMS + 20)]
    rest = CO.rest_of_summary("", [{"kind": "сводка", "groups": groups}, source(2), source(3)])
    assert rest.count("[2][3]") == CO.REST_ITEMS


# --------------------------------------------------------------------------- #
#  Задание на выписку по статье
# --------------------------------------------------------------------------- #
def test_map_prompt_general_task() -> None:
    chunks = [row(12, "a", 0.1, "Хлорит   ассоциирует\n с эпидотом."), row(13, "a", 0.1)]
    prompt = CO.map_prompt("хлорит", ["хлорит", "chlorite"], chunks)
    assert "Предмет: «хлорит» (он же: chlorite)." in prompt
    assert "Фрагменты одной статьи «Статья a» (2020):" in prompt
    assert "[c12] Хлорит ассоциирует с эпидотом." in prompt
    assert "[c13] " in prompt
    assert '{"пункты": [{"с_чем": "...", ' in prompt  # пример JSON без двойных скобок


def test_map_prompt_edge_values() -> None:
    """Без других названий — «—», без года — без скобок; длинное обрезано; фигурные
    скобки в тексте статьи не ломают шаблон."""
    chunk = {**row(1, "a", 0.1, "{subject} {0} " + "я" * 3000), "title": "Т" * 300, "year": None}
    prompt = CO.map_prompt("хлорит", ["хлорит"], [chunk])
    assert "(он же: —)" in prompt
    assert "«" + "Т" * 150 + "»:" in prompt
    assert "{subject} {0} " in prompt
    body = prompt.split("[c1] ", 1)[1].split("\n", 1)[0]
    assert len(body) == CO.CHUNK_CHARS


def test_map_prompt_targeted_task() -> None:
    prompt = CO.map_prompt("хлорит", ["хлорит"], [row(1, "a", 0.1)], {"минерал"})
    assert "Выпиши ВСЕ минералы и группы минералов, которые" in prompt
    assert "Только минералы и группы минералов — не " in prompt
    for other in ("породы и толщи", "методы и данные", "структуры"):
        assert f"не {other}" in prompt
    assert "не минералы" not in prompt


def test_map_prompt_two_kinds() -> None:
    prompt = CO.map_prompt("хлорит", ["хлорит"], [row(1, "a", 0.1)], {"месторождение", "район"})
    assert "Выпиши ВСЕ месторождения и рудопроявления и районы и рудные поля" in prompt


@pytest.mark.parametrize("wanted", [None, set()])
def test_map_prompt_without_kinds_is_general(wanted: Any) -> None:
    prompt = CO.map_prompt("хлорит", ["хлорит"], [row(1, "a", 0.1)], wanted)
    assert prompt == CO.map_prompt("хлорит", ["хлорит"], [row(1, "a", 0.1)])
    assert "Выпиши всё, что эти фрагменты прямо говорят о «хлорит»" in prompt


def test_map_prompt_needs_at_least_one_chunk() -> None:
    with pytest.raises(IndexError):
        CO.map_prompt("хлорит", ["хлорит"], [])
