"""Проверка ответа модели (georag/chat/checks.py): номера фрагментов, отказ, утверждения
без ссылки, повторы и зацикливание. Чистые функции — без базы и модели."""

from __future__ import annotations

import pytest

from georag.chat.checks import (
    citations,
    coverage,
    drop_repeats,
    drop_uncited,
    is_looping,
    is_refusal,
    says_not_in_base,
)

RHO = r"\rho = \rho_0 (1 - \varphi) + \frac{a}{b} + \sum_{i=1}^{n} x_i"  # длиннее 40 знаков
UNCITED = "Это общие слова модели без всякой ссылки на статью вообще."


# --------------------------------------------------------------------------- #
#  Отказ: «В базе знаний нет ответа» (правило 3) и «в базе знаний нет …»
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "start",
    [
        pytest.param("В базе знаний нет ответа на этот вопрос.", id="exact-phrase"),
        pytest.param("В базе знаний нет ответа", id="phrase-only"),
        pytest.param("**«В базе знаний нет ответа»** на этот вопрос", id="bold-and-quotes"),
        pytest.param("в БАЗЕ знаний нет ответа на вопрос", id="case"),
        pytest.param("  \nВ базе знаний нет ответа.", id="leading-space"),
    ],
)
def test_is_refusal_recognises_rule_phrase(start: str) -> None:
    assert is_refusal(start)


@pytest.mark.parametrize(
    "start",
    [
        pytest.param("В базе знаний описаны три метода [1].", id="normal-answer"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("В базе знаний нет", id="unfinished-phrase"),
        pytest.param("Ответ: в базе знаний нет ответа", id="not-at-start"),
    ],
)
def test_is_refusal_rejects_other_text(start: str) -> None:
    assert not is_refusal(start)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("В базе знаний нет информации о металлогении Анабарского щита.", id="info"),
        pytest.param("В базе знаний нет ответа на этот вопрос.", id="rule-phrase"),
        pytest.param("**В базе знаний** нет данных о золоте.", id="bold"),
        pytest.param("«В базе знаний» нет сведений.", id="quotes"),
    ],
)
def test_says_not_in_base_true(text: str) -> None:
    assert says_not_in_base(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("В базе знаний есть три статьи о золоте [1].", id="has"),
        pytest.param("Ответ: в базе знаний нет сведений.", id="not-at-start"),
        pytest.param("Золото связано с кварцем [1].", id="answer"),
    ],
)
def test_says_not_in_base_false(text: str) -> None:
    assert not says_not_in_base(text)


# --------------------------------------------------------------------------- #
#  Номера фрагментов в ответе
# --------------------------------------------------------------------------- #
def test_citations_collects_used_and_unknown() -> None:
    used, unknown = citations("Утверждение [1]. Ещё [2][3]. И [1, 4]. Выдумка [9].", 4)
    assert used == [1, 2, 3, 4]
    assert unknown == [9]


@pytest.mark.parametrize(
    ("answer", "count", "expected"),
    [
        pytest.param("", 3, ([], []), id="empty"),
        pytest.param("Просто текст без ссылок.", 3, ([], []), id="no-citations"),
        pytest.param("Пирит [1; 2], кварц [2 ,3].", 3, ([1, 2, 3], []), id="separators"),
        pytest.param("[3] [1] [2] [1]", 3, ([1, 2, 3], []), id="sorted-unique"),
        pytest.param("Номер ноль [0].", 2, ([], [0]), id="zero-is-unknown"),
        pytest.param("Фрагментов нет, а ссылка [1].", 0, ([], [1]), id="no-sources"),
        pytest.param("Граница [3] при трёх фрагментах.", 3, ([3], []), id="last-number"),
        pytest.param("В 2021 году (см. [2]).", 3, ([2], []), id="year-not-citation"),
        pytest.param("См. [статью](http://x/1) и [1].", 3, ([1], []), id="markdown-link"),
    ],
)
def test_citations_edge_cases(
    answer: str, count: int, expected: tuple[list[int], list[int]]
) -> None:
    assert citations(answer, count) == expected


# --------------------------------------------------------------------------- #
#  Сколько утверждений со ссылкой
# --------------------------------------------------------------------------- #
def test_coverage_counts_claims_not_intro() -> None:
    """Вводная фраза «По данным статей базы:» — не утверждение; пункт без ссылки — да."""
    text = (
        "По данным статей базы:\n"
        "- Оруденение приурочено к узлам пересечения разломов и зонам дробления [1].\n"
        "- Калиевый метасоматоз и окварцевание выделяют по снимкам ASTER.[2][3]\n"
        "- Россыпи в погребённых палеодолинах указывают на коренной источник выше.\n\n"
        "Прямых оценок золотоносности Анабарского щита во фрагментах нет [4]."
    )
    assert coverage(text) == (3, 4)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("", (0, 0), id="empty"),
        pytest.param(" \n\n ", (0, 0), id="blank"),
        pytest.param("Пирит [1]. Кварц.", (0, 0), id="short-phrases"),
        pytest.param(
            "Плотность линеаментов максимальна вблизи рудных узлов на севере щита.",
            (0, 1),
            id="fact-without-citation",
        ),
        pytest.param(
            "Калиевый метасоматоз сменяется окварцеванием по всей зоне.[2]",
            (1, 1),
            id="citation-after-period",
        ),
        pytest.param(
            "Калиевый метасоматоз сменяется окварцеванием по всей зоне [2]. "
            "Плотность линеаментов максимальна вблизи рудных узлов на севере щита.",
            (1, 2),
            id="cited-then-uncited",
        ),
        pytest.param(
            "Для Анабарского щита в найденных фрагментах прямых оценок золотоносности нет.",
            (0, 0),
            id="about-fragments",
        ),
        pytest.param(
            "Во фрагментах сказано, что оруденение связано с разломами северо-востока [1].",
            (1, 1),
            id="about-fragments-with-citation",
        ),
        pytest.param(
            "1. Узлы приурочены к пересечениям разломов северо-восточного простирания [1].\n"
            "2) Плотность линеаментов максимальна вблизи рудных узлов на севере щита.\n"
            "• Россыпи в погребённых палеодолинах указывают на коренной источник [2].",
            (2, 3),
            id="list-markers",
        ),
    ],
)
def test_coverage_cases(answer: str, expected: tuple[int, int]) -> None:
    assert coverage(answer) == expected


def test_coverage_headings_are_not_claims() -> None:
    """Заголовки разделов — не утверждения; пункт с жирным началом — утверждение."""
    headed = (
        "**1) Роль узлов пересечения разломов в геологии**\n"
        "Узлы пересечения разломов контролируют оруденение в Восточном Донбассе [1].\n"
        "### 2) Как используют гравитационные аномалии при прогнозе?\n"
        "- **Аномалии золота:** золото сопровождается аномалиями серебра в зоне."
    )
    assert coverage(headed) == (1, 2)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(f"$${RHO}$$", id="one-line"),
        pytest.param(f"  $${RHO}$$.  ", id="punctuation-after"),
        pytest.param(f"$$\n{RHO}\n$$", id="multi-line"),
    ],
)
def test_coverage_formula_line_is_not_claim(answer: str) -> None:
    """Формула отдельной строкой ($$…$$, правило 8) — не утверждение без ссылки."""
    assert coverage(answer) == (0, 0)
    lead = "Плотность пород считают по формуле из статьи о песчаниках района [3]:\n"
    assert coverage(lead + answer) == (1, 1)


@pytest.mark.xfail(
    reason="coverage, перенося ссылку «. [2] » перед точкой, съедает пробел после неё: "
    "следующая фраза без ссылки склеивается с подкреплённой и считается со ссылкой "
    "(drop_uncited ту же фразу убирает)",
    raises=AssertionError,
    strict=True,
)
@pytest.mark.parametrize("gap", [pytest.param(" ", id="space"), pytest.param("", id="no-space")])
def test_coverage_citation_after_period_does_not_cover_next_sentence(gap: str) -> None:
    text = (
        f"Калиевый метасоматоз сменяется окварцеванием по всей зоне.{gap}[2] "
        "Плотность линеаментов максимальна вблизи рудных узлов на севере щита."
    )
    assert coverage(text) == (1, 2)


@pytest.mark.xfail(
    reason="coverage режет предложение на «млн.» (делит по любой точке с пробелом), а "
    "drop_uncited по комментарию к _SENTENCE_END считает «120 млн. лет» одной фразой: "
    "полностью подкреплённый ответ получает покрытие 0 из 1",
    raises=AssertionError,
    strict=True,
)
def test_coverage_million_years_is_one_sentence() -> None:
    text = "Гранитоиды комплекса внедрились около 120 млн. лет назад в обстановке растяжения [2]."
    assert coverage(text) == (1, 1)


# --------------------------------------------------------------------------- #
#  Утверждения без ссылки уходят из ответа
# --------------------------------------------------------------------------- #
def test_drop_uncited_main_scenario() -> None:
    """Фразы без ссылки убраны; со ссылкой, подводка к списку, фраза о самих фрагментах
    и заголовок с текстом — остались; пустой пункт и заголовок без текста — ушли."""
    answer = "\n".join(
        [
            "**Признаки золотоносных зон**",
            "Золотоносные зоны всегда представляют большой интерес для геологов мира.",
            "Золото приурочено к зонам березитизации в гранитоидах массива [1].",
            "Поисковые признаки золотоносных зон в статьях называют следующие:",
            "- Аномалии мышьяка и сурьмы во вторичных ореолах рассеяния [2].",
            "- Метод широко применяется во многих регионах мира и даёт хорошие результаты.",
            "Во фрагментах не сказано, как выделяют такие зоны по радарным снимкам.",
            "",
            "**Вывод**",
            "В целом все перечисленные признаки очень важны для успешного прогноза.",
        ]
    )
    text, removed = drop_uncited(answer)
    assert text == "\n".join(
        [
            "**Признаки золотоносных зон**",
            "Золото приурочено к зонам березитизации в гранитоидах массива [1].",
            "Поисковые признаки золотоносных зон в статьях называют следующие:",
            "- Аномалии мышьяка и сурьмы во вторичных ореолах рассеяния [2].",
            "Во фрагментах не сказано, как выделяют такие зоны по радарным снимкам.",
        ]
    )
    assert removed == [
        "Золотоносные зоны всегда представляют большой интерес для геологов мира.",
        "Метод широко применяется во многих регионах мира и даёт хорошие результаты.",
        "В целом все перечисленные признаки очень важны для успешного прогноза.",
    ]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("Узлы связаны с разломами [1].", id="one-sentence"),
        pytest.param("Пирит [1]. Кварц.", id="short-phrases"),
        pytest.param(
            "**Методы**\n- Плотность линеаментов считают по снимкам Landsat [2].\n\n"
            "Золото приурочено к зонам березитизации в гранитоидах массива [1].",
            id="heading-list-paragraph",
        ),
    ],
)
def test_drop_uncited_keeps_cited_answer_unchanged(answer: str) -> None:
    assert drop_uncited(answer) == (answer, [])


@pytest.mark.parametrize("answer", ["", "   ", "\n\n\n"])
def test_drop_uncited_empty_answer(answer: str) -> None:
    assert drop_uncited(answer) == ("", [])


def test_drop_uncited_citation_after_period_belongs_to_sentence() -> None:
    text, removed = drop_uncited(
        "Узлы связаны с разломами. [1] Это доказано многократно во всех регионах мира."
    )
    assert text == "Узлы связаны с разломами. [1]"
    assert removed == ["Это доказано многократно во всех регионах мира."]


def test_drop_uncited_keeps_bullet_marker() -> None:
    """Из пункта уходит только фраза без ссылки; пункт целиком без ссылки — вся строка."""
    answer = (
        "- Узлы связаны с разломами северо-восточного простирания [1]. "
        "Общие слова модели без всякой ссылки на статью.\n"
        "1) Метод в целом широко применяется во многих регионах мира и даёт результаты."
    )
    text, removed = drop_uncited(answer)
    assert text == "- Узлы связаны с разломами северо-восточного простирания [1]."
    assert removed == [
        "Общие слова модели без всякой ссылки на статью.",
        "Метод в целом широко применяется во многих регионах мира и даёт результаты.",
    ]


def test_drop_uncited_empty_sections_lose_headings() -> None:
    """Заголовок, за которым сразу следующий заголовок или конец, — уходит."""
    answer = (
        "**Признаки**\n### Методы\n"
        "Узлы приурочены к пересечениям разломов северо-восточного простирания [1].\n"
        "__Итог__\n" + UNCITED
    )
    text, removed = drop_uncited(answer)
    assert text == (
        "### Методы\nУзлы приурочены к пересечениям разломов северо-восточного простирания [1]."
    )
    assert removed == [UNCITED]


def test_drop_uncited_collapses_blank_lines_and_citation_only_lines() -> None:
    text, removed = drop_uncited("Узлы связаны с разломами [1].\n\n\n\n[3]. [7].\n\nКварц [2].")
    assert text == "Узлы связаны с разломами [1].\n\nКварц [2]."
    assert removed == []


@pytest.mark.parametrize(
    "connective",
    ["Например, аргиллизация", "В частности, аргиллизация", "Кроме того, аргиллизация",
     "При этом, аргиллизация", "Также аргиллизация"],
)  # fmt: skip
def test_drop_uncited_strips_connective_after_removed_sentence(connective: str) -> None:
    """Связка у следующей фразы снимается, фраза начинается с большой буквы."""
    first = "Гидротермальные изменения широко распространены в изученных районах мира."
    text, removed = drop_uncited(f"{first} {connective} выявляется по полосе 2,17 мкм [7].")
    assert text == "Аргиллизация выявляется по полосе 2,17 мкм [7]."
    assert removed == [first]


def test_drop_uncited_keeps_connective_after_kept_sentence() -> None:
    answer = "Золото связано с кварцем [1]. Например, аргиллизация выявляется по снимкам [7]."
    assert drop_uncited(answer) == (answer, [])


@pytest.mark.parametrize(
    "follower",
    [
        "Эти длины волн включают VNIR, SWIR и TIR [3].",
        "Таким образом, съёмка охватывает VNIR, SWIR и TIR [3].",
        "Они охватывают VNIR, SWIR и TIR [3].",
        "Их используют для поиска изменённых пород [3].",
        "Данные каналы включают VNIR, SWIR и TIR [3].",
    ],
)
def test_drop_uncited_keeps_sentence_referred_back(follower: str) -> None:
    """Следующая фраза ссылается назад («Эти…», «Таким образом…») — предыдущая остаётся."""
    answer = f"Снимки ASTER охватывают четырнадцать каналов в разных диапазонах. {follower}"
    assert drop_uncited(answer) == (answer, [])


def test_drop_uncited_back_reference_without_citation_does_not_save() -> None:
    first = "Снимки ASTER охватывают четырнадцать каналов в разных диапазонах спектра."
    second = "Эти длины волн широко используются геологами во всех странах мира."
    assert drop_uncited(f"{first} {second}") == ("", [first, second])


def test_drop_uncited_million_years_cited_sentence_kept_whole() -> None:
    """«120 млн. лет» — одна фраза: ссылка в конце относится ко всему предложению."""
    answer = "Интрузии внедрились около 120 млн. лет назад [2]. " + UNCITED
    assert drop_uncited(answer) == (
        "Интрузии внедрились около 120 млн. лет назад [2].",
        [UNCITED],
    )


def test_drop_uncited_million_years_uncited_sentence_removed_without_tail() -> None:
    sentence = "Интрузии внедрились около 120 млн. лет назад, как и гранитоиды соседнего массива."
    assert drop_uncited(sentence) == ("", [sentence])


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param(f"$${RHO}$$", id="one-line"),
        pytest.param(f"$${RHO}$$.", id="punctuation-after"),
        pytest.param(f"$$\n{RHO}\n$$", id="multi-line"),
        pytest.param(f"$$\n{RHO}\n$$,", id="multi-line-comma"),
    ],
)
def test_drop_uncited_keeps_formula_line(formula: str) -> None:
    """Формула отдельной строкой — не утверждение; текст без ссылки после неё уходит."""
    answer = f"Плотность считают по формуле (1) из статьи [3]:\n{formula}\n{UNCITED}"
    text, removed = drop_uncited(answer)
    assert text == f"Плотность считают по формуле (1) из статьи [3]:\n{formula}"
    assert removed == [UNCITED]


@pytest.mark.xfail(
    reason="_ABOUT_SOURCES ищет подстроку «фрагмент» и принимает геологические «фрагменты "
    "пород» (брекчии) за фразу о фрагментах базы: утверждение без ссылки остаётся",
    raises=AssertionError,
    strict=True,
)
def test_drop_uncited_rock_fragments_are_geology_not_sources() -> None:
    sentence = "Брекчии содержат угловатые фрагменты вмещающих пород и обломки жильного кварца."
    assert drop_uncited(sentence) == ("", [sentence])


# --------------------------------------------------------------------------- #
#  Повторы
# --------------------------------------------------------------------------- #
def test_drop_repeats_removes_repeated_line_but_not_short_items() -> None:
    """Повтор строки (ссылки, разметка, маркер и точка не в счёт) убран; короткие пункты
    повторять можно."""
    answer = "\n".join(
        [
            "Кварцевые жилы пронизывают девонские известняки и сланцы [1].",
            "- пирит [1]",
            "- пирит [2]",
            "- **Кварцевые жилы** пронизывают девонские известняки и сланцы [3]",
        ]
    )
    assert drop_repeats(answer) == "\n".join(
        [
            "Кварцевые жилы пронизывают девонские известняки и сланцы [1].",
            "- пирит [1]",
            "- пирит [2]",
        ]
    )


def test_drop_repeats_keeps_repeated_headings() -> None:
    """Заголовки не сравниваются: пустой раздел потом уберёт drop_uncited."""
    answer = "**Золото и его спутники в рудах**\nтекст\n**Золото и его спутники в рудах**"
    assert drop_repeats(answer) == answer


def test_drop_repeats_near_copy_without_new_words() -> None:
    """Длинная строка без единого нового слова против сказанной — повтор (вопрос r02)."""
    first = (
        "Оруденение контролируется глубинными корово-мантийными разломами северо-восточного "
        "простирания и их оперяющими структурами [1]."
    )
    second = (
        "Оруденение контролируется глубинными разломами северо-восточного простирания и их "
        "оперяющими структурами [1]."
    )
    assert drop_repeats(f"{first}\n{second}") == first


def test_drop_repeats_keeps_line_with_new_word() -> None:
    """Пункты по одному шаблону («…пирит», «…антимонит») — не повтор: слово новое."""
    answer = (
        "Кварцевые жилы содержат пирит, арсенопирит и халькопирит в зальбандах и в "
        "центральных частях жил [1].\n"
        "Кварцевые жилы содержат антимонит, арсенопирит и халькопирит в зальбандах и в "
        "центральных частях жил [2]."
    )
    assert drop_repeats(answer) == answer


def test_drop_repeats_collapses_looped_inline_list() -> None:
    """Перечень в строке, где кусок повторён по кругу: повторы пунктов уходят, точка
    в конце."""
    looped = (
        "Золото ассоциирует с пиритом [1], арсенопиритом [2], халькопиритом [3], "
        "галенитом [4], сфалеритом [4], галенитом [4], сфалеритом [4], галенитом [4]"
    )
    assert drop_repeats(looped) == (
        "Золото ассоциирует с пиритом [1], арсенопиритом [2], халькопиритом [3], "
        "галенитом [4], сфалеритом [4]."
    )


def test_drop_repeats_short_inline_list_untouched() -> None:
    """Меньше шести пунктов — не перечень по кругу: строка как была."""
    answer = "Хлорит ассоциирует с эпидотом [1], кальцитом [2], эпидотом [3]"
    assert drop_repeats(answer) == answer


def test_drop_repeats_removes_sentence_repeated_in_other_items() -> None:
    """Одна и та же фраза в нескольких пунктах — остаётся в первом (вопрос r11)."""
    same = "Это также отмечено в составе жильных минералов района [2]."
    answer = (
        f"- Пирит встречается в кварцевых жилах района. {same}\n"
        f"- Арсенопирит встречается в зальбандах жил. {same}"
    )
    assert drop_repeats(answer) == (
        f"- Пирит встречается в кварцевых жилах района. {same}\n"
        "- Арсенопирит встречается в зальбандах жил."
    )


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("", id="empty"),
        pytest.param("a\n\nb\n\n", id="blank-lines"),
        pytest.param(
            "Золото приурочено к зонам березитизации в гранитоидах массива [1].\n\n"
            "Серебро связано с полиметаллическими жилами северного фланга [2].",
            id="no-repeats",
        ),
    ],
)
def test_drop_repeats_without_repeats_is_identity(answer: str) -> None:
    assert drop_repeats(answer) == answer


# --------------------------------------------------------------------------- #
#  Зацикливание
# --------------------------------------------------------------------------- #
PHRASE = "Плотность линеаментов совпадает с месторождениями рудного узла по данным [1]. "


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "Начало ответа [1]. " + PHRASE + "Ещё одна мысль [1]. " + PHRASE + "Да", id="repeat"
        ),
        pytest.param(
            "Начало ответа [1]. " + PHRASE + "Ещё одна мысль [1]. "
            "**Плотность линеаментов совпадает с месторождениями рудного узла по данным** [3]. Да",
            id="markup-and-other-citation",
        ),
        pytest.param(PHRASE + PHRASE, id="finished-repeat-at-end"),
    ],
)
def test_is_looping_detects_repeated_sentence(text: str) -> None:
    assert is_looping(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param(PHRASE, id="one-sentence"),
        pytest.param(
            "Начало [1]. " + PHRASE + "Другая фраза о рудных узлах [1]. Да", id="different"
        ),
        pytest.param("Пирит [1]. Пирит [1]. Пирит [1]. Да", id="short-repeat"),
        pytest.param(
            "Начало ответа [1]. " + PHRASE + "Ещё одна мысль [1]. " + PHRASE.strip(),
            id="repeat-not-finished",
        ),
    ],
)
def test_is_looping_false(text: str) -> None:
    assert not is_looping(text)
