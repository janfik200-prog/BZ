"""Ключевые понятия вопроса (georag/index/concepts.py): чистка ответа модели, словарь
переводов, доля понятий во фрагменте и порядок кандидатов.

python -m unittest discover -s tests — без базы и модели."""

import unittest
from dataclasses import dataclass

from georag.index import concepts as C

TABLE = {
    C.name_key("лиственитизация"): ["listvenitization", "listvenite"],
    C.name_key("Сухой Лог"): ["Sukhoi Log", "Sukhoy Log"],
}


@dataclass
class Hit:
    chunk_id: int
    text: str


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer

    def chat_json(self, system, user, **kw):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class CleanTests(unittest.TestCase):
    def test_dictionary_replaces_model_translation(self):
        got = C.clean([["лиственитизация", "lignitization", "Lg"]], TABLE)
        self.assertEqual(got, [["лиственитизация", "listvenitization", "listvenite"]])

    def test_short_and_generic_spellings_dropped(self):
        got = C.clean([["месторождение", "deposit"], ["линеаменты", "lineaments", "L"]], TABLE)
        self.assertEqual(got, [["линеаменты", "lineaments"]])

    def test_concept_with_general_word_kept(self):
        # «структурный контроль» — не общее слово, хотя «контроль» общее.
        self.assertEqual(C.clean([["структурный контроль"]], {}), [["структурный контроль"]])

    def test_at_most_three(self):
        raw = [["кварц"], ["пирит"], ["галенит"], ["сфалерит"]]
        self.assertEqual(len(C.clean(raw, {})), C.MAX_CONCEPTS)

    def test_bad_answer_is_empty(self):
        self.assertEqual(C.clean(None, {}), [])
        self.assertEqual(C.extract(FakeLLM(RuntimeError("down")), "вопрос"), [])
        self.assertEqual(C.extract(FakeLLM({"must": "не список"}), "вопрос"), [])


class ShareTests(unittest.TestCase):
    def test_any_form_and_language(self):
        concepts = [["плотность линеаментов", "lineament density"], ["золото", "gold"]]
        self.assertEqual(C.share(concepts, "Высокая плотность линеаментов на участке"), 0.5)
        self.assertEqual(C.share(concepts, "Lineament density and gold occurrences"), 1.0)
        self.assertEqual(C.share(concepts, "Список литературы"), 0.0)

    def test_ratio_with_slash(self):
        self.assertEqual(C.share([["K/Th"]], "the K/Th ratio map"), 1.0)
        self.assertEqual(C.share([["K/Th"]], "potassium and thorium maps"), 0.0)


class RerankTests(unittest.TestCase):
    def test_concepts_first_ties_keep_order(self):
        hits = [Hit(1, "общие слова"), Hit(2, "Euler deconvolution depth"), Hit(3, "тоже общее"),
                Hit(4, "метод Эйлера")]
        got = [h.chunk_id for h in C.rerank(hits, [["метод Эйлера", "Euler deconvolution"]])]
        self.assertEqual(got, [2, 4, 1, 3])

    def test_no_concepts_no_change(self):
        hits = [Hit(1, "a"), Hit(2, "b")]
        self.assertIs(C.rerank(hits, []), hits)



class PlaceTests(unittest.TestCase):
    def test_distinct_part_of_name(self):
        self.assertEqual(C._distinct("Anabar Shield"), "Anabar")
        self.assertEqual(C._distinct("массив Колдар"), "Колдар")
        self.assertEqual(C._distinct("Shield"), "Shield")  # одно общее слово — как есть

    def test_long_spelling_by_main_words(self):
        words = C._words("Prospects for large gold ore deposits of the Yenisei Ridge")
        self.assertTrue(C.has("large-scale gold ore deposits", words))  # без «scale»
        self.assertFalse(C.has("tin-bearing granites", C._words("granites of the area")))

    def test_parse_splits_place(self):
        llm = FakeLLM({"place": ["Алтай", "Altai"], "must": [["колчеданные месторождения", "VMS"]]})
        parsed = C.parse(llm, "Колчеданные месторождения Алтая")
        self.assertEqual(parsed.place[:1], ["Алтай"])
        self.assertEqual(parsed.concepts[0][:1], ["Алтай"])
        self.assertEqual(C.parse(FakeLLM({"place": None, "must": []}), "x").place, [])


class FilterTests(unittest.TestCase):
    def test_doc_list(self):
        from georag.index.search import _filters

        clause, params = _filters(None, None, ["a", "b"])
        self.assertIn("ANY", clause)
        self.assertEqual(params["doc_id"], ["a", "b"])
        clause, params = _filters(None, None, "a")
        self.assertIn("= %(doc_id)s", clause)


class PlaceMatchTests(unittest.TestCase):
    def test_same_alphabet_needs_close_spelling(self):
        self.assertFalse(C._close_enough("Хакасия", "Якутия", 0.7))
        self.assertTrue(C._close_enough("Лудзин", "Lujing", 0.67))

    def test_place_named_in_full(self):
        words = set(C._words("ore deposits of the field"))
        self.assertFalse(C._named("Seligdar Ore Field", words))
        self.assertTrue(C._named("Seligdar Ore Field", set(C._words("Seligdar deposits"))))


if __name__ == "__main__":
    unittest.main()
