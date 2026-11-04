"""Названия документов из таблицы отчётов (index/titles.py), отчёт по вопросам
(chat/report.py) и отрезание неподтверждённой связи «поскольку…» (chat/support.py).

python -m unittest discover -s tests — без базы и модели."""

import tempfile
import unittest
from pathlib import Path

from georag.chat import report as R
from georag.chat import support as S
from georag.index import titles as T

UUID_DOC = "06a6f50a-ae30-4ab4-a55b-ceb62eb673a8"
UUID_REP = "710006ff-ae2f-4532-9656-6000e233f1b6"


class ColumnsTests(unittest.TestCase):
    def test_header_words(self):
        cols = T.columns(["№", "Код отчёта", "Название отчёта", "Авторы", "Год", "Площадь работ"])
        self.assertEqual(cols, {"title": 2, "authors": 3, "year": 4, "place": 5})

    def test_row_keys(self):
        keys = T.row_keys([1, UUID_REP, f"{UUID_DOC}.docx", "текст"])
        self.assertEqual(keys, {UUID_REP, UUID_DOC})


class MatchTests(unittest.TestCase):
    def test_document_code_before_report_code(self):
        docs = {UUID_DOC: {UUID_DOC, UUID_REP}, "other": {"other", UUID_REP}}
        by_doc = T.Row({UUID_DOC}, {"title": "Отчёт по Линденской площади"})
        by_report = T.Row({UUID_REP}, {"title": "Общий отчёт"})
        found = T.match(docs, [by_report, by_doc])
        self.assertIs(found[UUID_DOC], by_doc)  # свой код частнее общего кода отчёта
        self.assertIs(found["other"], by_report)

    def test_title_with_place(self):
        self.assertEqual(
            T.title_of({"title": "Отчёт о работах", "place": "Линденская площадь"}),
            "Отчёт о работах (Линденская площадь)",
        )
        self.assertEqual(
            T.title_of({"title": "Работы на Линденской площади", "place": "Линденской"}),
            "Работы на Линденской площади",
        )

    def test_read_table(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.append(["Реестр отчётов"])
            ws.append(["Файл", "Наименование", "Год"])
            ws.append([f"{UUID_DOC}.docx", "Отчёт А", 2018])
            ws.append(["", "без кода", 2019])
            wb.save(path)
            header, cols, rows = T.read_table(path)
        self.assertEqual(cols, {"title": 1, "year": 2})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].values, {"title": "Отчёт А", "year": "2018"})


class ReportTests(unittest.TestCase):
    def test_questions_and_reaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.yaml"
            path.write_text(
                "вопросы:\n  - {id: f1, вид: ложный, вопрос: 'Почему X?'}\n"
                "  - 'Простой вопрос?'\n",
                "utf-8",
            )
            qs = R.load_questions(path)
        self.assertEqual([(q["id"], q["kind"]) for q in qs], [("f1", "ложный"), ("2", "обычный")])
        self.assertEqual(R.reaction("Это неверно: во фрагментах сказано другое [1]."), "опроверг")
        self.assertEqual(R.reaction("В базе знаний нет ответа на этот вопрос."), "нет данных")
        self.assertTrue(R.reaction("Потому что так [1].").startswith("согласился"))

    def test_bundled_question_set(self):
        qs = R.load_questions(Path(__file__).resolve().parents[1] / "config" / "questions-oil.yaml")
        self.assertEqual(sum(q["kind"] == "обычный" for q in qs), 13)
        self.assertEqual(sum(q["kind"] == "ложный" for q in qs), 20)


class LinkTests(unittest.TestCase):
    def test_cut_link(self):
        claim = "Баженовская свита — нефтематеринская толща, поскольку в ней выявлены залежи [1]."
        self.assertEqual(S.cut_link(claim), "Баженовская свита — нефтематеринская толща [1].")

    def test_weak_link(self):
        claim = "Свита нефтематеринская, что подтверждает приток нефти [2]."
        self.assertTrue(
            S._weak_link({"ответ": "частично", "чего_нет": "связь: нет во фрагменте"}, claim)
        )
        self.assertFalse(S._weak_link({"ответ": "частично", "чего_нет": "другое место"}, claim))
        self.assertFalse(
            S._weak_link({"ответ": "частично", "чего_нет": "связь"}, "Свита есть [1].")
        )

    def test_rules_in_prompt(self):
        from georag.chat.messages import SYSTEM

        self.assertIn("Не связывай факты причиной", SYSTEM)
        self.assertIn("не выдавай частный случай за определение", SYSTEM)


if __name__ == "__main__":
    unittest.main()


class DictionaryEnvTests(unittest.TestCase):
    """--compare-dict: «без словаря» выключает и написания словаря в поиске (GEORAG_DICT)."""

    def test_env_follows_mode_and_is_restored(self) -> None:
        import os
        from unittest import mock

        from georag.chat.settings import dictionary_on

        with mock.patch.dict(os.environ, {"GEORAG_DICT": "1"}):
            with R._dictionary_env(False):
                self.assertFalse(dictionary_on())
            self.assertEqual(os.environ["GEORAG_DICT"], "1")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEORAG_DICT", None)
            with R._dictionary_env(True):
                self.assertTrue(dictionary_on())
            self.assertNotIn("GEORAG_DICT", os.environ)
