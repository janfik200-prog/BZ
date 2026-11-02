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
    def test_working_dictionaries_listed(self):
        ids = {d["id"] for d in D.working_listing()}
        self.assertLessEqual({"geo-terms", "synonyms", "term-translations"}, ids)

    def test_working_find(self):
        got = D.working_find("term-translations", "лиственит")
        self.assertTrue(any("listvenit" in r["en"] for r in got["entries"]))


if __name__ == "__main__":
    unittest.main()
