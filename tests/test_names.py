"""Ключ имени (georag/text.py, name_key): греческая буква — часть имени.

«β-кварц» и «α-кварц» — разные минералы (полиморфы), а ключ у них был один —
«кварц»: регулярка слов ключа знала только латиницу, кириллицу и цифры, и буква
терялась. Граф сводил их в одну сущность, факт «α-кварц переходит в β-кварц»
отбрасывался как «связь с самим собой». Здесь — и это, и то, что сводилось
правильно (падежи, множественное число), сводится как раньше.

python -m pytest tests/test_names.py — без базы и модели."""

import unittest

from georag.graph import autosyn, facts
from georag.graph.api import Graph
from georag.graph.synonyms import Synonyms
from georag.text import mentions, name_key


class GreekLetterTests(unittest.TestCase):
    def test_polymorphs_differ(self):
        """Разные полиморфы и величины — разные ключи."""
        pairs = [
            ("β-кварц", "α-кварц"),
            ("β-quartz", "α-quartz"),
            ("β кварц", "α кварц"),  # буква отдельным словом через пробел
            ("β-кварц", "кварц"),
            ("α-quartz", "quartz"),
            ("фаза α", "фаза β"),  # буква в конце
            ("γ-излучение", "излучение"),
            ("δ18O", "18O"),  # одиночная буква-индекс перед числом
            ("εNd", "Nd"),  # и перед символом элемента
            ("δEu", "Eu"),
            ("ΣREE", "REE"),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertNotEqual(name_key(a), name_key(b))

    def test_same_mineral_any_form(self):
        """Падеж, регистр, дефис или пробел, другое начертание буквы — один ключ."""
        same = ["β-кварц", "β-кварца", "β-кварцу", "β-кварцем", "β кварц", "Β-КВАРЦ", "ϐ-кварц"]
        self.assertEqual({name_key(n) for n in same}, {"β кварц"})
        self.assertEqual(name_key("β-quartz"), name_key("β quartz"))
        self.assertEqual(name_key("µ-кварц"), name_key("μ-кварц"))  # знак микро и мю

    def test_greek_only_name_has_key(self):
        """Имя из одних греческих букв — не пустой ключ (пустой — «число, а не предмет»)."""
        self.assertEqual(name_key("αβ"), "αβ")


class UnchangedKeysTests(unittest.TestCase):
    def test_cases_and_plural(self):
        groups = [
            ["Донецкий бассейн", "Донецкого бассейна", "донецкий  бассейн"],
            ["Donetsk basins", "Donetsk basin"],
            ["рудный узел", "рудного узла"],
            ["разлом", "разломы", "разлома", "разломов"],
            ["processes", "process"],
        ]
        for names in groups:
            with self.subTest(names=names):
                self.assertEqual(len({name_key(n) for n in names}), 1)

    def test_keys_without_greek_as_before(self):
        """Имена без греческих букв — те же ключи, что и до правки."""
        expected = {
            "Донецкого бассейна": "донецк бассейн",
            "Donetsk basins": "donetsk basin",
            "рудного узла": "рудн узл",
            "18O": "18o",
            "I-type граниты": "i type гранит",
            "²⁰⁶Pb/²³⁸U": "206pb 238u",
            "Fe³⁺": "fe3",
        }
        for name, key in expected.items():
            with self.subTest(name=name):
                self.assertEqual(name_key(name), key)

    def test_isotopes_differ(self):
        keys = [name_key(n) for n in ("δ18O", "δ13C", "δ34S", "δ33S", "34S", "δD")]
        self.assertEqual(len(set(keys)), len(keys))

    def test_latin_letter_prefixes(self):
        self.assertEqual(len({name_key(n) for n in ("I-type", "S-type", "A-type")}), 3)
        self.assertNotEqual(name_key("зона A"), name_key("зона B"))


class MentionsTests(unittest.TestCase):
    def test_greek_name_in_text(self):
        self.assertTrue(mentions("β-кварц", "в образце β-кварца немного"))
        self.assertFalse(mentions("β-кварц", "в образце α-кварца немного"))
        self.assertTrue(mentions("рудный узел", "в пределах рудного узла"))


class GraphTests(unittest.TestCase):
    TEXT = "При нагреве выше 573 °C α-кварц переходит в β-кварц."

    def test_fact_between_polymorphs_kept(self):
        """Факт «α-кварц переходит в β-кварц» — не «связь с самим собой»."""
        checked = facts.check({"факты": [[1, "α-кварц", "переходит в", "β-кварц"]]}, self.TEXT)
        self.assertEqual(checked.rejected, [])
        self.assertEqual([(f.src, f.dst) for f in checked.facts], [("α-кварц", "β-кварц")])
        self.assertNotEqual(checked.facts[0].src_key, checked.facts[0].dst_key)

    def test_two_entities(self):
        row = {"doc_id": "d1", "chunk_id": 1, "quote": self.TEXT}
        rows = [
            {**row, "id": 1, "src": "α-кварц", "relation": "переходит в", "dst": "β-кварц"},
            {**row, "id": 2, "src": "β-кварца", "relation": "содержит", "dst": "титан"},
        ]
        graph = Graph(rows, Synonyms())
        self.assertIn("α-кварц", graph.entities)
        self.assertIn("β-кварц", graph.entities)
        self.assertEqual(graph.entities["β-кварц"]["facts"], 2)  # «β-кварца» — она же
        self.assertEqual(graph.resolve("α-кварца"), "α-кварц")
        self.assertEqual(graph.resolve("β-кварцем"), "β-кварц")


class AutosynGuardTests(unittest.TestCase):
    def test_greek_letters_stop_merge(self):
        """Векторы «β-кварц» и «α-кварц» почти одни — сливать без человека нельзя."""
        for a, b in [("β-кварц", "α-кварц"), ("α-кварц", "кварц"), ("β-quartz", "α-quartz")]:
            with self.subTest(a=a, b=b):
                self.assertEqual(autosyn.guard(a, b), "разные греческие буквы")

    def test_same_letters_pass(self):
        self.assertIsNone(autosyn.guard("β-кварц", "Β-кварц"))
        self.assertIsNone(autosyn.guard("β-кварц", "β кварц"))
        self.assertIsNone(autosyn.guard("Landsat8 OLI", "Landsat-8 OLI"))
