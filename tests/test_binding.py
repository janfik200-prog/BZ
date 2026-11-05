"""Привязка к объекту вопроса (georag/chat/binding.py): утверждение о названном в вопросе
объекте остаётся, только если процитированный фрагмент этот объект называет.

Случаи — из сравнения replay 05.11.2026: oil59 (Линденская впадина), oil36
(Тимано-Печорская провинция), oil55 (задонский горизонт — частичный ответ полезен).

python -m unittest discover -s tests — без базы и модели."""

import unittest

from georag.chat.binding import bind_objects, question_objects


def src(n, text, title="Отчёт"):
    return {"n": n, "title": title, "headings": [], "text": text}


class ObjectsTests(unittest.TestCase):
    def test_names_from_question(self):
        self.assertEqual(question_objects("Какие баженовские отложения есть в Линденской впадине?"),
                         [["линденс"]])
        self.assertEqual(question_objects("Какие комплексы выделяют в Тимано-Печорской провинции?"),
                         [["тимано-печорс"]])
        self.assertEqual(question_objects("Что такое кероген?"), [])  # первое слово — не имя

    def test_two_word_name(self):
        self.assertEqual(question_objects("Какие отложения продуктивны в Западной Сибири?"),
                         [["запад", "сиби"]])


class BindTests(unittest.TestCase):
    def test_claim_about_object_from_other_area_removed(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [src(1, "В Западной Сибири баженовская свита сложена кремнистыми аргиллитами."),
                   src(2, "Линденская впадина выполнена юрскими отложениями.")]
        answer = ("В Линденской впадине баженовские отложения представлены кремнистыми "
                  "аргиллитами [1]. Линденская впадина выполнена юрскими отложениями [2].")
        text, removed = bind_objects(q, answer, sources)
        self.assertEqual(removed, ["В Линденской впадине баженовские отложения представлены "
                                   "кремнистыми аргиллитами [1]."])
        self.assertIn("юрскими отложениями [2]", text)

    def test_dictionary_classification_not_applied_to_province(self):
        q = "Какие нефтегазоносные комплексы выделяют в Тимано-Печорской провинции?"
        sources = [src(1, "Нефтегазоносный комплекс — совокупность пород…",
                       title="Геологический словарь: «Нефтегазоносный комплекс»")]
        _, removed = bind_objects(q, "В Тимано-Печорской провинции выделяют комплексы [1].",
                                  sources)
        self.assertEqual(len(removed), 1)

    def test_object_named_in_title_or_any_case(self):
        q = "Какие коллекторы в Линденской впадине?"
        sources = [src(1, "Коллекторы порового типа.", title="Отчёт по Линденской впадине")]
        text, removed = bind_objects(q, "В Линденской впадине коллекторы порового типа [1].",
                                     sources)
        self.assertEqual(removed, [])

    def test_negation_kept(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [src(1, "В Западной Сибири баженовская свита…")]
        answer = "Фрагменты не указывают, что в Линденской впадине есть баженовские отложения [1]."
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_sentence_not_about_object_kept(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [src(1, "В Западной Сибири баженовская свита…")]
        answer = "В Западной Сибири баженовская свита сложена аргиллитами [1]."
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_general_question_untouched(self):
        answer = "Кероген I типа богат водородом [1]."
        self.assertEqual(bind_objects("Что такое кероген?", answer, [src(1, "…")]), (answer, []))


if __name__ == "__main__":
    unittest.main()
