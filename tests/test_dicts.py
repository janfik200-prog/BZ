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


if __name__ == "__main__":
    unittest.main()
