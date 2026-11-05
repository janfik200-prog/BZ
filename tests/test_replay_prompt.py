"""Короткий промт ответа (chat/messages.py, GEORAG_PROMPT=short) и повтор ответов на
сохранённых фрагментах (chat/replay.py).

python -m unittest discover -s tests — без базы и модели."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from georag.chat import messages as M
from georag.chat import replay as R
from georag.chat.checks import NO_ANSWER
from georag.chat.plan import Part, Plan
from georag.chat.settings import Settings, prompt_style

SRC = {"n": 1, "doc_id": "d", "ord": 0, "title": "Отчёт", "year": 2018, "authors": [],
       "pages": [12], "text": "Баженовская свита сложена кремнисто-глинистыми породами."}


class PromptTests(unittest.TestCase):
    def test_style_from_environment(self):
        with mock.patch.dict(os.environ, {"GEORAG_PROMPT": "short"}):
            self.assertEqual(prompt_style(), "short")
            self.assertEqual(Settings().prompt, "short")
        with mock.patch.dict(os.environ, {"GEORAG_PROMPT": "что-то"}):
            self.assertEqual(prompt_style(), "full")

    def test_short_keeps_refusal_phrase(self):
        self.assertIn(NO_ANSWER, M.SHORT_SYSTEM)  # по ней код узнаёт отказ

    def test_correction_is_an_answer(self):
        """Исправление ложной посылки — ответ, отказ — только когда нет ни ответа, ни
        исправления (сравнение replay 05.11.2026: полный промт отказывался на f19, oil60)."""
        self.assertIn("Исправление предпосылки — это уже ответ, а не отказ", M.SHORT_SYSTEM)
        self.assertIn("ни исправить его предпосылку", M.SHORT_SYSTEM)
        self.assertIn("только если фрагменты подтверждают их для этого объекта", M.SHORT_SYSTEM)

    def test_system_by_style(self):
        full = M.build_messages("Вопрос?", [SRC], [])
        short = M.build_messages("Вопрос?", [SRC], [], style="short")
        self.assertEqual(full[0]["content"], M.SYSTEM)
        self.assertEqual(short[0]["content"], M.SHORT_SYSTEM)

    def test_rule_numbers_follow_style(self):
        plan = Plan([Part("а", []), Part("б", [])])
        full = M.build_messages("а и б?", [SRC], [], plan=plan, insist=True)[-1]["content"]
        short = M.build_messages("а и б?", [SRC], [], plan=plan, insist=True,
                                 style="short")[-1]["content"]
        self.assertIn("по правилу 6", full)
        self.assertIn("фраза из правила 3", full)
        self.assertIn("по правилу 4", short)
        self.assertIn("фраза из правила 4", short)

    def test_situational_rules(self):
        self.assertIn(M.SUPERLATIVE_RULE,
                      M.situational_rules("Какие породы основные покрышки?", [SRC]))
        self.assertIn(M.GENERAL_RULE, M.situational_rules("Что такое кероген?", [SRC]))
        self.assertNotIn(M.GENERAL_RULE,
                         M.situational_rules("Какие залежи в Западной Сибири?", [SRC]))
        self.assertIn(M.FORMULA_RULE,
                      M.situational_rules("Как считают?", [{**SRC, "text": "по $$a=b$$ (2)"}]))
        # Правила по случаю — только в коротком промте.
        full = M.build_messages("Что такое кероген?", [SRC], [])[-1]["content"]
        self.assertNotIn(M.GENERAL_RULE, full)


class ReplayTests(unittest.TestCase):
    def test_read_report_and_sources(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Ответы"
            ws.append(["№", "Вид", "Вопрос", "Словарь", "Ответ бота"])
            ws.append(["f01", "ложный", "Почему много золота?", "да", "…"])
            fs = wb.create_sheet("Фрагменты")
            fs.append(["№ вопроса", "Словарь", "[n]", "Использован в ответе",
                       "Название статьи / отчёта", "Файл", "Год", "Страницы", "Раздел",
                       "Текст фрагмента"])
            fs.append(["f01", "да", "[2]", "нет", "Геологический словарь. Т. 1: «Золото»", None,
                       None, "31", None, "Золото — металл."])
            fs.append(["f01", "да", "[1]", "да", "Отчёт", "a.docx", "2018", "12, 13", "Глава 1",
                       "Текст."])
            wb.save(path)
            answers, frags = R.read_report(path)
        self.assertEqual(answers[0]["№"], "f01")
        sources = R.to_sources(frags[("f01", "да")])
        self.assertEqual([s["n"] for s in sources], [1, 2])  # по номерам, как в исходном
        self.assertEqual(sources[0]["pages"], [12, 13])
        self.assertEqual(sources[0]["year"], 2018)
        self.assertEqual(sources[1].get("kind"), "словарь")

    def test_prompt_record(self):
        self.assertEqual(R.system_prompt("short"), M.SHORT_SYSTEM)
        self.assertEqual(R.system_prompt("full"), M.SYSTEM)
        self.assertEqual(len(R.fingerprint(M.SHORT_SYSTEM)), 12)
        self.assertNotEqual(R.fingerprint(M.SHORT_SYSTEM), R.fingerprint(M.SYSTEM))
        self.assertEqual(R.rule_names("Что такое кероген?", [SRC], "short"), "общий вопрос")
        self.assertEqual(R.rule_names("Что такое кероген?", [SRC], "full"), "")
        self.assertEqual(R.TEMPERATURE, 0.0)

    def test_not_a_report(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.xlsx"
            Workbook().save(path)
            with self.assertRaises(ValueError):
                R.read_report(path)


if __name__ == "__main__":
    unittest.main()
