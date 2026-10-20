"""Запрос поиска по словам (georag/index/search.py, build_tsquery): русская форма слова
в обе стороны. Стеммер Postgres беглую гласную не знает: «узел» → основа «узел»,
«узла» → «узл», — поэтому запрос несёт оба написания, от какой бы формы ни шёл.

python -m unittest discover -s tests — без базы и модели."""

import re
import unittest

from georag.index.search import build_tsquery


def words(query):
    return set(re.findall(r"\w+", build_tsquery(query)))


class SearchWordsTests(unittest.TestCase):
    def test_both_directions(self):
        cases = [
            ("рудного узла", {"узла", "узел"}),  # косвенная → начальная: «Рудный узел»
            ("рудный узел", {"узел", "узл"}),  # начальная → косвенная: «рудного узла»
            ("песка", {"песок"}),
            ("узлы", {"узел"}),
            ("конца", {"конец"}),
            ("угла", {"угол"}),
            ("разлом", {"разлома"}),  # -ом не окончание: вариант с «-а»
        ]
        for query, must in cases:
            with self.subTest(query=query):
                self.assertLessEqual(must, words(query))

    def test_no_vowel_where_none_drops(self):
        """Основа не на «согласная + л, к, ц, н, р» — вариантов нет."""
        for word in ["метода", "района", "окварцевания", "rocks"]:
            with self.subTest(word=word):
                self.assertEqual(build_tsquery(word), word.lower())

    def test_all_and_any_words(self):
        self.assertEqual(build_tsquery("рудного узла"), "рудного & (узла | узел | узол)")
        self.assertEqual(
            build_tsquery("рудного узла", any_word=True), "рудного | (узла | узел | узол)"
        )


if __name__ == "__main__":
    unittest.main()
