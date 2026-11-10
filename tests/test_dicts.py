"""Словари (georag/dicts.py): разбор статьи справочного словаря и рабочие словари проекта.

python -m unittest discover -s tests — без базы и модели."""

import unittest

from georag import dicts as D


class EntryTests(unittest.TestCase):
    def test_etymology_and_english(self):
        e = D._entry(
            "Адуляр [по горам Адула, Швейцария; adularia] – прозрач. разновид. ортоклаза.", 31
        )
        self.assertEqual(
            (e.term, e.etym, e.en), ("Адуляр", "по горам Адула, Швейцария", "adularia")
        )
        self.assertEqual(e.definition, "прозрач. разновид. ортоклаза.")
        self.assertEqual(e.page, 31)

    def test_english_only(self):
        e = D._entry("Адмитанс [admittance] – величина, обратная импедансу.", 31)
        self.assertEqual((e.term, e.en, e.etym), ("Адмитанс", "admittance", ""))

    def test_dash_inside_brackets(self):
        raw = (
            "Декольман [от фр. décollement – отклеивание; Haug E., 1907; décollement] – "
            "пологий надвиг."
        )
        e = D._entry(raw, 315)
        self.assertEqual(e.term, "Декольман")
        self.assertEqual(e.en, "décollement")
        self.assertEqual(e.definition, "пологий надвиг.")

    def test_reference_without_brackets(self):
        e = D._entry("β-ось – см. Структурная ось β.", 22)
        self.assertEqual((e.term, e.definition), ("β-ось", "см. Структурная ось β."))

    def test_no_dash_is_not_entry(self):
        self.assertIsNone(D._entry("ДЛЯ ЗАМЕТОК", 431))


class JoinTests(unittest.TestCase):
    def test_hyphenation_and_soft_hyphen(self):
        self.assertEqual(
            D._join(["гидрогеоло-", "гическая структура"]), "гидрогеологическая структура"
        )
        self.assertEqual(D._join(["издаю\x10 щая при ударе"]), "издающая при ударе")
        self.assertEqual(D._join(["Брюстерит · 10H2O��. Мон."]), "Брюстерит · 10H2O. Мон.")

    def test_real_hyphen_kept(self):
        self.assertEqual(D._join(["медно-", "Порфировое"]), "медно- Порфировое")


class WorkingTests(unittest.TestCase):
    """Рабочие словари — из своих файлов во временной папке, не из config проекта."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        folder = Path(self.tmp.name)
        terms = "# комментарий\nкварц\nкварцевые жилы\npyrite\n"
        (folder / "geo-terms.txt").write_text(terms, "utf-8")
        synonyms = "имена:\n  Донецкий бассейн: [Donetsk basin, Донбасс]\n"
        (folder / "synonyms.yaml").write_text(synonyms, "utf-8")
        self.old, D.CONFIG = D.CONFIG, folder

    def tearDown(self):
        D.CONFIG = self.old
        self.tmp.cleanup()

    def test_terms_file_saved_with_bom(self):
        """Блокнот сохраняет UTF-8 с BOM: первый термин — без невидимого знака, а
        комментарий с отступом — не термин (так же читает граф, entities.load_terms)."""
        from georag.graph.entities import load_terms

        path = D.CONFIG / "geo-terms.txt"
        path.write_text("\ufeffкварц\n  # с отступом\npyrite\n", "utf-8")
        terms = [e["term"] for e in D.working_entries("geo-terms")]
        self.assertEqual(terms, ["кварц", "pyrite"])
        self.assertEqual(load_terms(path), terms)

    def test_listed_only_existing_files(self):
        ids = {d["id"] for d in D.working_listing()}
        self.assertEqual(ids, {"geo-terms", "synonyms"})

    def test_find_and_missing_file(self):
        got = D.working_find("geo-terms", "кварц")
        self.assertEqual([r["term"] for r in got["entries"]], ["кварц", "кварцевые жилы"])
        syn = D.working_find("synonyms", "донбасс")
        self.assertEqual(syn["entries"][0]["term"], "Донецкий бассейн")
        self.assertEqual(D.working_find("term-translations", "")["total"], 0)


PAIRS = D.abbreviation_pairs(
    [
        D.Entry("м-л, м-лы", "", "", "минерал, минералы", 20),
        D.Entry("к-л, к-лы", "", "", "кристалл, кристаллы", 20),
        D.Entry("тв.", "", "", "твердость (при цифре)", 21),
        D.Entry("мон.", "", "", "моноклинный", 21),
        D.Entry("T, t", "", "", "температура", 21),
    ]
)


class AbbreviationTests(unittest.TestCase):
    def test_pairs(self):
        self.assertIn(("к-лы", "кристаллы"), PAIRS)
        self.assertIn(("тв.", "твердость"), PAIRS)  # без пояснения в скобках
        self.assertNotIn("T", [s for s, _ in PAIRS])  # однобуквенное — как в формулах

    def test_expand_forms_and_dots(self):
        got = D.expand("Мон. Удлиненные к-лы. Тв. 2–3. В ассоц. с м-лами, м-лов нет.", PAIRS)
        self.assertEqual(
            got,
            "Моноклинный. Удлиненные кристаллы. Твердость 2–3. В ассоц. с "
            "минералами, минералов нет.",
        )

    def test_headword_initials(self):
        self.assertEqual(
            D.expand_headword("Термин Д. часто применяют.", "Декольман"),
            "Термин Декольман часто применяют.",
        )
        self.assertEqual(
            D.expand_headword("А. с. используется для очистки.", "Адсорбция селективная"),
            "Адсорбция селективная используется для очистки.",
        )
        self.assertEqual(
            D.expand_headword("по Д.С. Коржинскому и Д. Smith", "Декольман"),
            "по Д.С. Коржинскому и Д. Smith",
        )

    def test_no_expand_inside_words(self):
        self.assertEqual(D.expand("сульфидами Tm-лантаноиды", PAIRS), "сульфидами Tm-лантаноиды")


def _index(*entries, pairs=None):
    rows = [
        (e["dict_id"], e["ord"], e["term"], e["en"], e["etym"], e["definition"], e["page"])
        for e in entries
    ]
    return D.build_index(("t",), rows, {"d": "Словарь"}, {"d": pairs or []})


def _e(term, definition, en="", ord_=0):
    return {
        "dict_id": "d",
        "ord": ord_,
        "term": term,
        "en": en,
        "etym": "",
        "definition": definition,
        "page": 5,
    }


class FakeCursor:
    def __init__(self, counts):
        self.counts, self.last = counts, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.last = (sql, params)

    def fetchone(self):
        sql, params = self.last
        if "FROM chunks WHERE" in sql:
            return (self.counts.get(params[0], 0),)
        return (1000,)


class FakeConn:
    def __init__(self, counts=None):
        self.counts = counts or {}

    def cursor(self):
        return FakeCursor(self.counts)


class TermsTests(unittest.TestCase):
    def setUp(self):
        self.idx = _index(
            _e("Зона сдвига", "зона пластических деформаций.", "shear zone", 1),
            _e("Брюхоногие", "син. термина гастроподы.", "", 2),
            _e("Гастроподы (Gastropoda)", "класс моллюсков.", "gastropod", 3),
            _e("Золото", "самородный металл.", "gold", 4),
            _e("Зона", "общий термин.", "", 5),
        )
        self.old, D.index = D.index, lambda conn: self.idx

    def tearDown(self):
        D.index = self.old

    def test_longest_first_and_common_skipped(self):
        got = D.terms_in(FakeConn({"Золото": 500}), "Как зоны сдвига контролируют золото?")
        self.assertEqual([e["term"] for e in got], ["Зона сдвига"])  # не «Зона», не «Золото»

    def test_reference_followed(self):
        got = D.terms_in(FakeConn(), "Где встречаются брюхоногие?")
        self.assertEqual(got[0]["term"], "Гастроподы (Gastropoda)")
        self.assertEqual(got[0]["via"], "Брюхоногие")

    def test_query_variants(self):
        self.assertIn(
            "Как выделяют shear zone", D.query_variants(FakeConn(), "Как выделяют зону сдвига?")
        )
        self.assertIn(
            "Где встречаются gastropod", D.query_variants(FakeConn(), "Где встречаются брюхоногие")
        )
        # Термин, куда ведёт отсылка, — без латыни в скобках: она ничего не находит.
        self.assertIn(
            "Где встречаются Гастроподы",
            D.query_variants(FakeConn(), "Где встречаются брюхоногие"),
        )

    def test_query_variants_english(self):
        # Английский термин вопроса — и русское название статьи (10.11.2026).
        self.assertIn(
            "How to map a Зона сдвига", D.query_variants(FakeConn(), "How to map a shear zone")
        )
        self.assertIn(
            "Where do Гастроподы occur", D.query_variants(FakeConn(), "Where do gastropod occur")
        )

    def test_terms_english_question(self):
        got = D.terms_in(FakeConn(), "How do shear zone structures control gold?")
        self.assertEqual(got[0]["term"], "Зона сдвига")

    def test_no_dictionary(self):
        D.index = lambda conn: None
        self.assertEqual(D.terms_in(FakeConn(), "зона сдвига"), [])
        self.assertEqual(D.query_variants(FakeConn(), "зона сдвига"), [])


class MatchingTests(unittest.TestCase):
    """Как вопрос находит термин, написанный в словаре иначе."""

    def setUp(self):
        self.idx = _index(
            _e("Слоистость косая", "слоистость с наклонными слойками.", "cross-bedding", 1),
            _e("Ловушка нефти и газа", "часть резервуара, где скапливаются нефть и газ.", "", 2),
            _e("Нефть", "жидкое горючее полезное ископаемое.", "oil", 3),
            _e("Нефтематеринская свита", "свита, где образуется нефть.", "", 4),
            _e("Породы мишени", "породы, на которые нацелен поиск.", "", 5),
            _e("ГЗН", "главная зона нефтеобразования.", "", 6),
            _e("Главная зона нефтеобразования (ГЗН)", "зона в разрезе бассейна.", "", 7),
            _e("Катагенез (биол.)", "упрощение организации организмов.", "", 8),
            _e("Катагенез (литол.)", "преобразование осад. п. и орг. в-ва при погружении.", "", 9),
            pairs=[("орг.", "органический"), ("в-во", "вещество")],
        )
        self.old, D.index = D.index, lambda conn: self.idx

    def tearDown(self):
        D.index = self.old

    def terms(self, q):
        return [e["term"] for e in D.terms_in(FakeConn(), q)]

    def test_word_order(self):
        self.assertEqual(self.terms("Что такое косая слоистость?"), ["Слоистость косая"])

    def test_compound_before_single_word(self):
        self.assertEqual(self.terms("Что такое ловушка нефти?"), ["Ловушка нефти и газа"])

    def test_long_specific_head_without_other_words(self):
        self.assertIn("Нефтематеринская свита", self.terms("Что такое нефтематеринская толща?"))

    def test_common_head_needs_more_words(self):
        self.assertNotIn("Породы мишени", self.terms("Какие породы считаются кварцевыми?"))

    def test_abbreviation_followed(self):
        got = D.terms_in(FakeConn(), "Что такое ГЗН?")
        self.assertEqual(got[0]["term"], "Главная зона нефтеобразования (ГЗН)")
        self.assertEqual(got[0]["via"], "ГЗН")

    def test_sense_by_definition_with_abbreviations(self):
        self.assertEqual(
            self.terms("Что такое катагенез органического вещества?"), ["Катагенез (литол.)"]
        )
        self.assertEqual(self.terms("Что такое катагенез?"), ["Катагенез (литол.)"])  # не биол.


class DefinitionTests(unittest.TestCase):
    def test_definition_questions(self):
        for q in (
            "Что такое адуляр?",
            "что значит декольман",
            "Дай определение термина зона сдвига",
            "What is adularia?",
            "Кто такие брюхоногие?",
            "Что представляет собой адуляр",
            "What are gastropods?",
            "What's adularia?",
            "Define adularia",
        ):
            self.assertTrue(D.is_definition(q), q)
        self.assertFalse(D.is_definition("Где встречается адуляр и что такое ортоклаз"))
        # «What are the …» — вопрос о составе или списке, а не об определении.
        self.assertFalse(D.is_definition("What are the main ore-controlling structures?"))

    def test_prompt_notes(self):
        from georag.chat.messages import build_messages

        src = {
            "n": 1,
            "doc_id": "словарь:d",
            "ord": 0,
            "title": "Словарь: «Адуляр»",
            "year": None,
            "authors": [],
            "pages": [31],
            "text": "Адуляр — разновидность ортоклаза.",
            "kind": "словарь",
        }
        user = build_messages("Что такое адуляр?", [src], [])[-1]["content"]
        self.assertIn("статьи Геологического словаря", user)
        self.assertIn("начни ответ с определения по словарю", user)
        self.assertIn("поставь [1]", user)
        user = build_messages("Где встречается адуляр?", [src], [])[-1]["content"]
        self.assertNotIn("начни ответ с определения", user)


class SchemaConn:
    """Заглушка базы для запросов вкладки «Словари»: какие таблицы и колонки есть и что
    выполнялось. DDL (CREATE/ALTER) и commit — ошибка: чтение схему не меняет."""

    def __init__(self, tables=True, columns=("abbreviations",), pairs=None):
        self.tables, self.columns, self.pairs = tables, set(columns), pairs or []
        self.sql = []

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        raise AssertionError("commit при чтении словарей")

    def execute(self, sql, params=None):
        if "CREATE" in sql or "ALTER" in sql:
            raise AssertionError(f"DDL при чтении словарей: {sql[:40]}")
        self.sql.append(sql)
        self.params = params

    def fetchone(self):
        last = self.sql[-1]
        if "to_regclass" in last:
            return (self.tables,)
        if "SELECT abbreviations" in last:
            return (self.pairs,)
        return (0,)

    def fetchall(self):
        last = self.sql[-1]
        if "information_schema.columns" in last:
            return [(c,) for c in self.params[1] if c in self.columns]
        return []


class ReadOnlyTests(unittest.TestCase):
    def test_no_tables_empty_without_ddl(self):
        conn = SchemaConn(tables=False)
        self.assertEqual(D.listing(conn), [])
        self.assertEqual(D.find(conn, None, "адуляр"), {"total": 0, "entries": []})

    def test_tables_read_without_ddl(self):
        conn = SchemaConn()
        self.assertEqual(D.listing(conn), [])
        self.assertEqual(D.find(conn, "d", "", letter="А")["entries"], [])

    def test_abbreviations_without_column(self):
        """Словарь загружен до колонки abbreviations: пусто, а не UndefinedColumn,
        который ронял ответ чата (answer.py зовёт as_sources без перехвата)."""
        conn = SchemaConn(columns=(), pairs=[["м-л", "минерал"]])
        self.assertEqual(D.abbreviations(conn, "d"), [])
        self.assertFalse(any("SELECT abbreviations" in s for s in conn.sql))
        entry = {**_e("Адуляр", "м-л группы ортоклаза."), "title": "Словарь"}
        got = D.as_sources(conn, [entry], start=3)
        self.assertEqual((got[0]["n"], got[0]["kind"]), (3, "словарь"))
        self.assertIn("м-л группы", got[0]["text"])

    def test_abbreviations_with_column(self):
        conn = SchemaConn(pairs=[["м-л", "минерал"]])
        self.assertEqual(D.abbreviations(conn, "d"), [("м-л", "минерал")])
        entry = {**_e("Адуляр", "м-л группы ортоклаза."), "title": "Словарь"}
        self.assertIn("минерал группы", D.as_sources(conn, [entry], start=1)[0]["text"])


class BudgetTests(unittest.TestCase):
    """Статьи словаря идут после фрагментов и не урезаются, но окно модели они занимают:
    fit_budget считает их (tail), и уступают последние фрагменты статей."""

    def setUp(self):
        from georag.chat.settings import Settings

        self.settings = Settings()  # окно как у модели: NUM_CTX
        word = "слово " * 400  # 2400 знаков — как обрезанный фрагмент
        self.sources = [
            {"n": i, "doc_id": f"d{i}", "title": f"Статья {i}", "text": word, "authors": []}
            for i in range(1, 11)
        ]
        self.glossary = [
            {**s, "doc_id": "словарь:d", "kind": "словарь", "n": k}
            for k, s in enumerate(self.sources[:3], start=1)
        ]

    def size(self, sources):
        from georag.chat.messages import _tokens, build_messages

        return _tokens(build_messages("Что такое адуляр?", sources, [], self.settings.model))

    def test_glossary_counted_in_window(self):
        from georag.chat.messages import ANSWER_RESERVE, fit_budget

        alone, _ = fit_budget("Что такое адуляр?", self.sources, [], self.settings)
        kept, _ = fit_budget(
            "Что такое адуляр?", self.sources, [], self.settings, tail=self.glossary
        )
        self.assertLess(len(kept), len(alone))
        self.assertGreaterEqual(len(kept), 3)
        budget = self.settings.num_ctx - ANSWER_RESERVE
        self.assertLessEqual(self.size(kept + self.glossary), budget)
        # Без учёта словаря то же окно переполнялось бы.
        self.assertGreater(self.size(alone + self.glossary), budget)

    def test_no_tail_same_as_before(self):
        from georag.chat.messages import fit_budget

        a = fit_budget("Что такое адуляр?", self.sources, [], self.settings)
        b = fit_budget("Что такое адуляр?", self.sources, [], self.settings, tail=[])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
