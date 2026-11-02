"""Словари (georag/dicts.py): разбор статьи справочного словаря и рабочие словари проекта.

python -m unittest discover -s tests — без базы и модели."""

import unittest

from georag import dicts as D


class EntryTests(unittest.TestCase):
    def test_etymology_and_english(self):
        e = D._entry("Адуляр [по горам Адула, Швейцария; adularia] – прозрач. разновид. ортоклаза.", 31)
        self.assertEqual((e.term, e.etym, e.en), ("Адуляр", "по горам Адула, Швейцария", "adularia"))
        self.assertEqual(e.definition, "прозрач. разновид. ортоклаза.")
        self.assertEqual(e.page, 31)

    def test_english_only(self):
        e = D._entry("Адмитанс [admittance] – величина, обратная импедансу.", 31)
        self.assertEqual((e.term, e.en, e.etym), ("Адмитанс", "admittance", ""))

    def test_dash_inside_brackets(self):
        raw = ("Декольман [от фр. décollement – отклеивание; Haug E., 1907; décollement] – "
               "пологий надвиг.")
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
        self.assertEqual(D._join(["гидрогеоло-", "гическая структура"]), "гидрогеологическая структура")
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

    def test_listed_only_existing_files(self):
        ids = {d["id"] for d in D.working_listing()}
        self.assertEqual(ids, {"geo-terms", "synonyms"})

    def test_find_and_missing_file(self):
        got = D.working_find("geo-terms", "кварц")
        self.assertEqual([r["term"] for r in got["entries"]], ["кварц", "кварцевые жилы"])
        syn = D.working_find("synonyms", "донбасс")
        self.assertEqual(syn["entries"][0]["term"], "Донецкий бассейн")
        self.assertEqual(D.working_find("term-translations", "")["total"], 0)


PAIRS = D.abbreviation_pairs([
    D.Entry("м-л, м-лы", "", "", "минерал, минералы", 20),
    D.Entry("к-л, к-лы", "", "", "кристалл, кристаллы", 20),
    D.Entry("тв.", "", "", "твердость (при цифре)", 21),
    D.Entry("мон.", "", "", "моноклинный", 21),
    D.Entry("T, t", "", "", "температура", 21),
])


class AbbreviationTests(unittest.TestCase):
    def test_pairs(self):
        self.assertIn(("к-лы", "кристаллы"), PAIRS)
        self.assertIn(("тв.", "твердость"), PAIRS)  # без пояснения в скобках
        self.assertNotIn("T", [s for s, _ in PAIRS])  # однобуквенное — как в формулах

    def test_expand_forms_and_dots(self):
        got = D.expand("Мон. Удлиненные к-лы. Тв. 2–3. В ассоц. с м-лами, м-лов нет.", PAIRS)
        self.assertEqual(got, "Моноклинный. Удлиненные кристаллы. Твердость 2–3. В ассоц. с "
                              "минералами, минералов нет.")

    def test_headword_initials(self):
        self.assertEqual(D.expand_headword("Термин Д. часто применяют.", "Декольман"),
                         "Термин Декольман часто применяют.")
        self.assertEqual(D.expand_headword("А. с. используется для очистки.", "Адсорбция селективная"),
                         "Адсорбция селективная используется для очистки.")
        self.assertEqual(D.expand_headword("по Д.С. Коржинскому и Д. Smith", "Декольман"),
                         "по Д.С. Коржинскому и Д. Smith")

    def test_no_expand_inside_words(self):
        self.assertEqual(D.expand("сульфидами Tm-лантаноиды", PAIRS), "сульфидами Tm-лантаноиды")


def _index(*entries):
    by_key = {}
    for e in entries:
        by_key.setdefault(D.name_key(e["term"].split(" (")[0]), []).append(e)
        for name in e["en"].split(","):
            if name.strip():
                by_key.setdefault(D.name_key(name), []).append(e)
    return D.Index(("t",), by_key, {"d": "Словарь"})


def _e(term, definition, en="", ord_=0):
    return {"dict_id": "d", "ord": ord_, "term": term, "en": en, "etym": "", "definition": definition,
            "page": 5}


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
        self.assertIn("Как выделяют shear zone", D.query_variants(FakeConn(), "Как выделяют зону сдвига?"))
        self.assertIn("Где встречаются gastropod", D.query_variants(FakeConn(), "Где встречаются брюхоногие"))

    def test_no_dictionary(self):
        D.index = lambda conn: None
        self.assertEqual(D.terms_in(FakeConn(), "зона сдвига"), [])
        self.assertEqual(D.query_variants(FakeConn(), "зона сдвига"), [])


class DefinitionTests(unittest.TestCase):
    def test_definition_questions(self):
        for q in ("Что такое адуляр?", "что значит декольман", "Дай определение термина зона сдвига",
                  "What is adularia?"):
            self.assertTrue(D.is_definition(q), q)
        self.assertFalse(D.is_definition("Где встречается адуляр и что такое ортоклаз"))

    def test_prompt_notes(self):
        from georag.chat.messages import build_messages

        src = {"n": 1, "doc_id": "словарь:d", "ord": 0, "title": "Словарь: «Адуляр»", "year": None,
               "authors": [], "pages": [31], "text": "Адуляр — разновидность ортоклаза.",
               "kind": "словарь"}
        user = build_messages("Что такое адуляр?", [src], [])[-1]["content"]
        self.assertIn("статьи Геологического словаря", user)
        self.assertIn("начни ответ с определения по словарю", user)
        self.assertIn("поставь [1]", user)
        user = build_messages("Где встречается адуляр?", [src], [])[-1]["content"]
        self.assertNotIn("начни ответ с определения", user)


if __name__ == "__main__":
    unittest.main()
