"""Regressions from live review; run: python -m unittest discover -s tests."""

import time
import unittest
from unittest.mock import patch

from georag.chat.answer import _answer_from
from georag.chat.collect import check_items, rest_of_summary
from georag.chat.dataset import proper_stems
from georag.chat.messages import build_messages
from georag.chat.plan import Plan
from georag.chat.settings import Settings
from georag.evaluation import IN_BASE, Question, check_answer
from georag.graph.synonyms import parse


class ReliabilityTests(unittest.TestCase):
    def test_endpoint_must_be_in_quote(self):
        chunk = {"chunk_id": 1, "text": "Хлорит ассоциирует с эпидотом."}
        for name, expected in [("алмаз", 0), ("эпидот", 1), ("горнblend", 0)]:
            item = {"с_чем": name, "цитата": chunk["text"], "фрагмент": "c1"}
            with self.subTest(name=name):
                kept, dropped = check_items({"пункты": [item]}, [chunk], ["хлорит"])
                self.assertEqual(len(kept), expected)
                self.assertEqual(dropped, 1 - expected)

    def test_dictionary_translation_and_original_name_are_allowed(self):
        syn = parse({"имена": {"эпидот": ["epidote"]}})
        chunk = {"chunk_id": 1, "text": "Chlorite with epidote."}
        with patch("georag.graph.synonyms.load", return_value=syn):
            for name in ["эпидот", "epidote"]:
                item = {"с_чем": name, "цитата": chunk["text"], "фрагмент": "c1"}
                kept, _ = check_items({"пункты": [item]}, [chunk], ["chlorite"])
                self.assertEqual(len(kept), 1)

    def test_sentence_initial_instruction_is_not_a_place(self):
        self.assertEqual(proper_stems("С чем связан хлорит? Собери сведения."), set())
        self.assertEqual(proper_stems("Хлорит. Билляхская зона разломов."), {"билля"})
        self.assertEqual(proper_stems("Что известно об Озерном?"), {"озерн"})

    def test_summary_tail_only_cites_retained_sources(self):
        summary = {
            "n": 1,
            "kind": "сводка",
            "groups": [
                {"name": "эпидот", "sources": [2, 25]},
                {"name": "кальцит", "sources": [26]},
            ],
        }
        tail = rest_of_summary("", [summary, {"n": 2}])
        self.assertIn("эпидот [2]", tail)
        self.assertNotIn("25", tail)
        self.assertNotIn("кальцит", tail)

    def test_uncited_or_unknown_answer_is_cleared(self):
        for answer in [
            "На Анабарском щите есть гигантские залежи алмазов под каждым выходом гранита.",
            "Хлорит замещает биотит [99].",
            "В базе знаний нет информации. " + "В найденных статьях другая территория. " * 8,
        ]:

            def llm(*args, response=answer):
                yield {"text": response}
                yield {"done": True}

            events = list(
                _answer_from(
                    llm,
                    "Что известно?",
                    [{"n": 1, "doc_id": "d", "title": "Статья", "text": "Гранит"}],
                    [],
                    Plan(parts=[]),
                    [],
                    Settings(),
                    {},
                    time.monotonic(),
                )
            )
            with self.subTest(answer=answer):
                self.assertFalse(any(e["type"] == "done" for e in events))
                if any(e["type"] == "token" for e in events):
                    self.assertEqual(events[-1]["text"], "")

    def test_model_summary_does_not_reference_removed_sources(self):
        summary = {
            "n": 1,
            "doc_id": "summary",
            "title": "хлорит",
            "kind": "сводка",
            "text": "кальцит [25]",
            "groups": [{"name": "кальцит", "sources": [25]}],
        }
        sources = [summary, {"n": 2, "doc_id": "d", "title": "Статья", "text": "Хлорит"}]
        messages = build_messages("Минералы?", sources, [])
        self.assertNotIn("[25]", messages[-1]["content"])
        self.assertNotIn("кальцит", messages[-1]["content"])
        self.assertEqual(summary["groups"][0]["sources"], [25])

    def test_evaluation_rejects_long_answer_without_citations(self):
        result = check_answer(
            Question("x", "Территория?", IN_BASE),
            [
                {"type": "token", "text": "В найденных статьях другая территория. " * 12},
                {"type": "done", "mode": "база", "used": [], "coverage": [0, 0]},
            ],
        )
        self.assertFalse(result["ok"])
        self.assertIn("нет ссылки на фрагмент", result["problems"])

    def test_evaluation_uses_stream_after_retraction(self):
        result = check_answer(
            Question("x", "Хлорит?", IN_BASE, short=True),
            [
                {"type": "token", "text": "Ошибка"},
                {"type": "revised", "text": ""},
                {"type": "token", "text": "Биотит [1]."},
                {"type": "done", "mode": "база", "used": [1], "coverage": [1, 1]},
            ],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["chars"], len("Биотит [1]."))


if __name__ == "__main__":
    unittest.main()
