"""Начало ответа не срезается проверками (прогон 11.11.2026 21:45: «она приурочена…»,
«Также литологические ловушки…», «Вассоевичу (1967 г.) [1].», «Соболох-Майян, и на юге…»)."""

from __future__ import annotations

from georag.chat.checks import _split_sentences, drop_uncited, rewrite
from georag.chat.quotes import cite_by_words

DEFINITION = (
    "Главная зона нефтеобразования (ГЗН) — зона в разрезе нефтегазоносного бассейна, "
    "где проявляется главная фаза нефтеобразования."
)
DICTIONARY = {
    "n": 10,
    "kind": "словарь",
    "title": "Геологический словарь",
    "text": "Главная зона нефтеобразования (ГЗН) [main oil formation zone] — зона в разрезе "
    "нефтегазоносного бассейна, где проявляется главная фаза нефтеобразования. Во всех "
    "бассейнах ГЗН приурочена к градациям катагенеза РОВ от конца ПК3 до конца МК2.",
}
OTHER = {"n": 3, "kind": "статья", "title": "Отчёт", "text": "Залежи нефти Тимано-Печоры."}


def test_initials_do_not_split_sentence() -> None:
    text = "Зрелость отвечает стадии по Н.Б. Вассоевичу (1967 г.) [1]. Далее другое."
    assert _split_sentences(text) == [
        "Зрелость отвечает стадии по Н.Б. Вассоевичу (1967 г.) [1].",
        "Далее другое.",
    ]


def test_place_abbreviation_does_not_split_sentence() -> None:
    text = "Земли охватывают пос. Соболох-Майян и правобережье р. Алдан [5]."
    assert _split_sentences(text) == [text]


def test_year_still_ends_sentence() -> None:
    assert _split_sentences("Работы шли в 2015 г. Затем их продолжили.") == [
        "Работы шли в 2015 г.",
        "Затем их продолжили.",
    ]


def test_definition_copied_from_fragment_gets_citation() -> None:
    answer = f"{DEFINITION} По Геологическому словарю [10], она приурочена к градациям МК2."
    text, cited = cite_by_words(answer, [OTHER, DICTIONARY])
    assert text.startswith(DEFINITION[:-1] + " [10].")
    assert cited == [DEFINITION]


def test_own_words_get_no_citation() -> None:
    answer = "Нефть образуется в недрах при нагреве органики на больших глубинах планеты."
    assert cite_by_words(answer, [OTHER, DICTIONARY]) == (answer, [])


def test_pronoun_after_intro_keeps_previous_sentence() -> None:
    first = "Это зона, где идёт главная фаза образования нефти в осадочном бассейне."
    second = "По Геологическому словарю [10], она приурочена к градациям МК1–МК2."
    text, removed = drop_uncited(f"{first} {second}")
    assert text == f"{first[:-1]} [10]. {second}"
    assert removed == []


def test_connective_with_pronoun_keeps_previous_sentence() -> None:
    first = "Глинистые породы не являются хорошими коллекторами нефти и газа."
    second = "Наоборот, они служат покрышками [6]."
    assert drop_uncited(f"{first} {second}")[0] == f"{first[:-1]} [6]. {second}"


def test_connective_dropped_on_next_line() -> None:
    answer = (
        "### Как образуются ловушки?\n\n"
        "Литологические ловушки образуются при выклинивании пластов-коллекторов в песках.\n\n"
        "Также литологические ловушки связаны с фациальным замещением [11]."
    )
    text, removed = drop_uncited(answer)
    assert text.endswith("\n\nЛитологические ловушки связаны с фациальным замещением [11].")
    assert len(removed) == 1


def test_connective_kept_after_kept_line() -> None:
    answer = "Первая фраза со ссылкой стоит здесь [1].\n\nОднако вторая тоже верна [2]."
    assert drop_uncited(answer) == (answer, [])


def test_rewrite_drops_connective_on_next_line() -> None:
    answer = "Первая фраза убрана сверкой целиком [3].\n\nОднако вторая остаётся [4]."
    text = rewrite(answer, lambda p: None if p.startswith("Первая") else p)
    assert text == "Вторая остаётся [4]."


def test_citation_after_final_period_stays_with_sentence() -> None:
    text = "Глины — покрышки. Нередко покрышками служат сульфатные породы. [4]"
    assert _split_sentences(text) == [
        "Глины — покрышки.",
        "Нередко покрышками служат сульфатные породы. [4]",
    ]


def test_paragraph_citation_keeps_last_sentence() -> None:
    answer = "Нередко высокими экранирующими свойствами обладают сульфатные породы. [4]"
    assert drop_uncited(answer) == (answer, [])


def test_retold_fragment_gets_citation() -> None:
    answer = (
        "Литологические ловушки ограничены плохопроницаемыми породами за счёт выклинивания "
        "или фациального замещения пластов-коллекторов."
    )
    source = {
        "n": 6,
        "kind": "словарь",
        "title": "Геологический словарь",
        "text": "Литологические ловушки — залежи ограничены выклиниванием или фациальным "
        "замещением пластов-коллекторов плохопроницаемыми отложениями.",
    }
    text, cited = cite_by_words(answer, [OTHER, source])
    assert text.endswith("пластов-коллекторов [6].")
    assert cited == [answer]


def test_service_intro_and_stub_lines_removed() -> None:
    """Вводная о происхождении фрагментов, строка-заглушка и «---» — не ответ (11.11.2026)."""
    from georag.chat.checks import drop_service

    answer = (
        "Все фрагменты взяты из одной статьи «115b» (Николаевич, 2025, отчёт 44), но фрагмент "
        "[3] — из словаря.\n\n---\n\n### Главная зона\n\nПо словарю [3], ГЗН — зона в разрезе "
        "бассейна [3].\nПо Геологическому словарю [3][4]."
    )
    assert drop_service(answer) == (
        "### Главная зона\n\nПо словарю [3], ГЗН — зона в разрезе бассейна [3]."
    )
    # От ответа из одной вводной не остаётся ничего: дальше — повтор или отказ.
    assert drop_service("Все фрагменты [1]–[4] взяты из разных источников: [1] — из статьи.") == ""
    plain = "Березит — метасоматическая порода [1].  \nВсе фрагменты об этом согласны [2]."
    assert drop_service(plain) == plain


def test_bare_refusal_next_to_real_answer_removed() -> None:
    from georag.chat.checks import drop_bare_refusal

    answer = "Кероген III типа генерирует газ [1].\n\nВ базе знаний нет ответа на этот вопрос."
    assert drop_bare_refusal(answer) == "Кероген III типа генерирует газ [1]."
    refusal = "В базе знаний нет ответа на этот вопрос."
    assert drop_bare_refusal(refusal) == refusal
    partial = "Типы названы [1]. В базе знаний нет ответа на вопрос о том, как они связаны."
    assert drop_bare_refusal(partial) == partial


def test_premise_note_only_for_questions_that_assume_something() -> None:
    """«Почему X…», «Раз X…» — указание проверить посылку; обычный вопрос и второй заход — нет."""
    from georag.chat import messages as M

    src = [{"n": 1, "doc_id": "a", "title": "Статья", "text": "Кероген III типа генерирует газ."}]
    last = lambda q, **kw: M.build_messages(q, src, [], **kw)[-1]["content"]  # noqa: E731
    assert M.PREMISE_NOTE in last("Почему кероген III типа даёт в основном нефть?")
    assert M.PREMISE_NOTE in last("Тюменская свита юрская. Почему она сложена известняками?")
    assert M.PREMISE_NOTE in last("Раз задонский горизонт сложен гранитами, какие там коллекторы?")
    assert M.PREMISE_NOTE not in last("Что такое кероген?")
    assert M.PREMISE_NOTE not in last("Какие породы, если не считать глин, служат покрышками?")
    assert M.PREMISE_NOTE not in last("Почему кероген III типа даёт нефть?", insist=True)
    assert M.PREMISE_NOTE not in last("Почему кероген III типа даёт нефть?", style="quote")


def test_premise_correction_keeps_and_gets_citation() -> None:
    """«Это не так: …» без ссылки не убирается: ссылка — от следующей фразы (g06)."""
    from georag.chat.checks import cite_correction, drop_uncited

    answer = (
        "Это не так: линеаменты тесно связаны с разломами. Линеаменты — поверхностные "
        "выражения подземных структур, таких как разломы и зоны сдвигов [1]."
    )
    text, removed = drop_uncited(cite_correction(answer))
    assert text.startswith("Это не так: линеаменты тесно связаны с разломами [1].")
    assert not removed
    plain = "Березит — метасоматическая порода с кварцем и серицитом, без ссылки. Он светлый [2]."
    assert cite_correction(plain) == plain
    alone = "Это не так: линеаменты тесно связаны с разломами."
    assert cite_correction(alone) == alone


def test_second_attempt_note_is_constructive() -> None:
    """Второй заход: изложить сказанное по вопросу, а в сомнении — отказ."""
    from georag.chat import messages as M

    src = [{"n": 1, "doc_id": "a", "title": "Статья", "text": "Кероген III типа генерирует газ."}]
    for q in ("Почему кероген III типа даёт нефть?", "Какие минералы-спутники у золота?"):
        last = M.build_messages(q, src, [], insist=True)[-1]["content"]
        assert "изложи, что в" in last and "неверный ответ хуже отказа" in last
        assert "не пересказывай их" in last


def test_statement_followed_by_question_is_a_premise() -> None:
    from georag.chat.messages import PREMISE_RE

    assert PREMISE_RE.search(
        "Баженовская свита нефтематеринская. Какой процент гравелитов в ней содержится?"
    )
    assert not PREMISE_RE.search("Какие структуры контролируют оруденение на Анабарском щите?")


def test_gold_run_fixes_13_11() -> None:
    """Ошибки обработки, найденные на золотом наборе 13.11.2026: d05, k01, h01, f11, h06."""
    from georag.chat.checks import (
        _split_sentences,
        drop_invented_gaps,
        drop_uncited,
        keep_correction,
    )
    from georag.chat.support import cut_link

    # d05: точка внутри названия в кавычках — не конец фразы.
    quoted = "Он упомянут в статье «Геологический словарь. Т. 2 (К–П). ВСЕГЕИ, 2017» [3]. Да [1]."
    assert _split_sentences(quoted)[0].endswith("2017» [3].")
    # k01: фраза, на которую следующая ссылается в середине («для этих минералов»), остаётся.
    item = (
        "1. **Аргиллизация** — характеризуется наличием глинистых минералов, таких как каолинит "
        "и алунит. Диагностическая полоса поглощения для этих минералов — 2,17 мкм [1]."
    )
    text, removed = drop_uncited(item)
    assert "**Аргиллизация**" in text and not removed
    # h01: вместе с отрезанной связкой уходит союз перед ней.
    cut = cut_link(
        "Плотность линеаментов указывает на зоны с высокой активностью и, следовательно, "
        "повышенным потенциалом для минерализации [4][8]."
    )
    assert cut == "Плотность линеаментов указывает на зоны с высокой активностью [4][8]."
    # f11: исправление посылки, убранное сверкой, возвращается.
    before = "Это не так: оруденение не всегда в центре массивов [2]. Его контролируют разломы [2]."
    after = "Его контролируют разломы [2]."
    assert keep_correction(
        before, after, ["Это не так: оруденение не всегда в центре массивов [2]."]
    ) == (
        before,
        [],
    )
    # h06: «чего нет» о том, о чём не спрашивали, убирается; о спрошенном — остаётся.
    q = "Как растительный покров мешает дистанционному выделению изменённых пород?"
    gap = (
        "Сигнал растительности доминирует [8].\n\nВ базе знаний нет ответа на вопрос о том, как "
        "покров влияет на спектральные характеристики в разных климатических зонах."
    )
    assert drop_invented_gaps(gap, q) == "Сигнал растительности доминирует [8]."
    asked = "Сигнал растительности доминирует [8]. Во фрагментах не описано, как покров мешает выделению изменённых пород."
    assert drop_invented_gaps(asked, q) == asked


def test_gold_second_run_fixes_13_11() -> None:
    """Вторая проба на золотом наборе: k03, c02, k01, t03."""
    from georag.chat.checks import _split_sentences, drop_uncited, rewrite

    # k03: точка сокращения перед ссылкой — не конец фразы.
    parts = _split_sentences(
        "Используют индексы кварца, оксидов железа и др. [1][2]. Также NDVI [6]."
    )
    assert parts[0].endswith("и др. [1][2].")
    # c02: цепочка — первая фраза остаётся, раз на неё ссылается оставшаяся следующая.
    chain = (
        "Орогенные месторождения золота связаны с орогенезом и формируются в зонах субдукции. "
        "Они характеризуются связью с гранитоидными интрузиями и метаморфизмом. Такие "
        "месторождения включают золото-сульфидные, например Майское [1][2]."
    )
    text, removed = drop_uncited(chain)
    assert text.startswith("Орогенные месторождения золота") and not removed
    # k01: первая фраза пункта списка — его заголовок.
    item = (
        "1. **Аргиллизация** — характеризуется наличием глинистых минералов, каолинита и "
        "алунита. Диагностическая полоса поглощения — 2,17 мкм, канал ASTER 5 [4]."
    )
    text, removed = drop_uncited(item)
    assert "**Аргиллизация**" in text and not removed
    # t03: фраза, которая держится за убранную местоимением, уходит вместе с ней.
    answer = "На платформе известен Сухой Лог [1]. В этой области прогнозируются рассолы [5]."
    assert rewrite(answer, lambda p: None if "Сухой Лог" in p else p) == ""


def test_gold_third_run_fixes_13_11() -> None:
    """Третья проба на золотом наборе: h01, f01, c02, w03, f04."""
    from georag.chat.checks import _split_sentences, drop_repeats, drop_uncited
    from georag.chat.plan import term_hints

    # h01: пробел в конце абзаца не мешает снять связку со следующего.
    text, _ = drop_uncited(
        "Плотность линеаментов служит индикатором раздробленности горного массива. \n\n"
        "В частности, зоны с высокой плотностью совпадают с известными объектами [2]."
    )
    assert text.startswith("Зоны с высокой плотностью")
    # f01: запятая в числе — не граница пункта перечня.
    line = (
        "ASTER включает шесть каналов в диапазоне 1,60–2,43 мкм, которые позволяют различать "
        "минералы, слюды, глины, карбонаты, хлориты [1]; в SWIR — шесть каналов от 1,6 до 2,43 мкм."
    )
    assert drop_repeats(line).endswith("до 2,43 мкм.")
    # c02: «мас. %» — не конец фразы.
    assert len(_split_sentences("Солёность флюидов 3,1–5,9 мас. % NaCl экв. [5]. Далее [1].")) == 2
    # w03: фраза перед «Например:» и списком остаётся со ссылками пунктов.
    lead = (
        "Вокруг рудных тел выделяют зоны с разными признаками, включая метасоматические "
        "изменения и аномалии элементов. Например:\n\n- Зоны альбитизации [1].\n- Зоны дробления [2]."
    )
    text, removed = drop_uncited(lead)
    assert text.startswith("Вокруг рудных тел") and "[1, 2]" in text and not removed
    # f04: английское название метода — из словаря переводов проекта.
    assert any(
        "weights of evidence" in t for t in term_hints("Как применяют метод весов доказательств?")
    )


def test_paragraph_end_citation_covers_previous_sentence() -> None:
    """Ссылка в конце абзаца относится и к фразе перед ней — если та не отделена связкой (c03)."""
    from georag.chat.checks import drop_uncited

    para = (
        "Метод весов доказательств основывается на оценке статистической связи признаков с "
        "известными месторождениями. Веса меняются от 1 до 0 в зависимости от расстояния [6][8]."
    )
    text, removed = drop_uncited(para)
    assert text.startswith("Метод весов доказательств основывается") and "[6, 8]" in text
    assert not removed
    example = (
        "Гидротермальные изменения широко распространены в изученных районах мира. Например, "
        "аргиллизация выявляется по полосе 2,17 мкм [7]."
    )
    text, removed = drop_uncited(example)
    assert text == "Аргиллизация выявляется по полосе 2,17 мкм [7]." and len(removed) == 1


def test_strict_cleanup_14_11() -> None:
    """Строгая чистка: догадки модели, пустая подводка к списку (t03), пересказ без ссылки (d05)."""
    from georag.chat.checks import cut_guesses, drop_empty_leads, drop_uncited

    guess = (
        "Платформа сложена архейскими комплексами, что может быть связано с образованием руд [10]."
    )
    assert cut_guesses(guess) == "Платформа сложена архейскими комплексами [10]."
    lead = "На севере платформы известны следующие ископаемые:\n\nПлатформа сложена гнейсами [10]."
    assert drop_empty_leads(lead) == "Платформа сложена гнейсами [10]."
    listed = "Известны следующие:\n- золото [1]"
    assert drop_empty_leads(listed) == listed
    retold = "Лиственит упоминается во фрагментах статей базы как одна из рудоносных формаций."
    assert drop_uncited(retold) == ("", [retold])
    absent = "Во фрагментах статей базы о листвените ничего не сказано."
    assert drop_uncited(absent) == (absent, [])


def test_mislabelled_items_and_definition_term_14_11() -> None:
    """w01: «Пояс X — кратон» уходит; s03/f04: термин определения — до второй части вопроса."""
    from georag.chat.binding import defined_term
    from georag.chat.checks import drop_mislabelled

    answer = (
        "1. **Пояс Свеконорвежский** — орогенный пояс в Балтийском щите [7].\n"
        "2. **Пояс Рио-де-ла-Плата** — кратон в южной части Бразилии [1].\n"
        "3. **Пояс Осло** — грабен в Балтийском щите [7]."
    )
    text, removed = drop_mislabelled(answer)
    assert text == "1. **Пояс Свеконорвежский** — орогенный пояс в Балтийском щите [7]."
    assert len(removed) == 2
    term = defined_term("Что такое сдвиговые зоны и как с ними связано золотое оруденение?")
    assert len(term) == 2 and term[0].startswith("сдвиг") and term[1].startswith("зон")
