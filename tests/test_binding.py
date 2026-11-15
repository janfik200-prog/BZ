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
        self.assertEqual(
            question_objects("Какие баженовские отложения есть в Линденской впадине?"),
            [["линденс"]],
        )
        self.assertEqual(
            question_objects("Какие комплексы выделяют в Тимано-Печорской провинции?"),
            [["тимано-печорс"]],
        )
        self.assertEqual(question_objects("Что такое кероген?"), [])  # первое слово — не имя

    def test_two_word_name(self):
        self.assertEqual(
            question_objects("Какие отложения продуктивны в Западной Сибири?"), [["запад", "сиби"]]
        )


class BindTests(unittest.TestCase):
    def test_claim_about_object_from_other_area_removed(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [
            src(1, "В Западной Сибири баженовская свита сложена кремнистыми аргиллитами."),
            src(2, "Линденская впадина выполнена юрскими отложениями."),
        ]
        answer = (
            "В Линденской впадине баженовские отложения представлены кремнистыми "
            "аргиллитами [1]. Линденская впадина выполнена юрскими отложениями [2]."
        )
        text, removed = bind_objects(q, answer, sources)
        self.assertEqual(
            removed,
            [
                "В Линденской впадине баженовские отложения представлены "
                "кремнистыми аргиллитами [1]."
            ],
        )
        self.assertIn("юрскими отложениями [2]", text)

    def test_dictionary_classification_not_applied_to_province(self):
        q = "Какие нефтегазоносные комплексы выделяют в Тимано-Печорской провинции?"
        sources = [
            src(
                1,
                "Нефтегазоносный комплекс — совокупность пород…",
                title="Геологический словарь: «Нефтегазоносный комплекс»",
            )
        ]
        _, removed = bind_objects(
            q, "В Тимано-Печорской провинции выделяют комплексы [1].", sources
        )
        self.assertEqual(len(removed), 1)

    def test_object_named_in_title_or_any_case(self):
        q = "Какие коллекторы в Линденской впадине?"
        sources = [src(1, "Коллекторы порового типа.", title="Отчёт по Линденской впадине")]
        text, removed = bind_objects(
            q, "В Линденской впадине коллекторы порового типа [1].", sources
        )
        self.assertEqual(removed, [])

    def test_negation_kept(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [src(1, "В Западной Сибири баженовская свита…")]
        answer = "Фрагменты не указывают, что в Линденской впадине есть баженовские отложения [1]."
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_english_negation_kept(self):
        q = "What controls gold in the Wamba district?"
        sources = [src(1, "Berezites of the Middle Urals.", title="Middle Urals")]
        answer = "It is not mentioned that the Wamba district hosts berezites [1]."
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_english_question_object(self):
        q = "What controls gold in the Wamba district?"
        sources = [
            src(1, "Shear zones control gold.", title="Gold in the Wamba district"),
            src(2, "Berezites of the Middle Urals.", title="Middle Urals"),
        ]
        answer = (
            "Gold in the Wamba district is controlled by shear zones [1]. "
            "Wamba hosts berezites [2]."
        )
        _, removed = bind_objects(q, answer, sources)
        self.assertEqual(removed, ["Wamba hosts berezites [2]."])

    def test_element_symbols_not_object(self):
        # «Au-Ag», «Fe-Mn» — химия, а не объект: верное утверждение не убирается (10.11.2026).
        self.assertEqual(question_objects("Где в Сибири встречается Au-Ag оруденение?"), [["сиби"]])
        self.assertEqual(
            question_objects("What is known about Au-Ag-Te ores at Natalka?"), [["nata"]]
        )
        q = "Какие Fe-Mn конкреции известны?"
        answer = "Fe-Mn конкреции развиты на дне океанов [1]."
        sources = [src(1, "Железомарганцевые конкреции развиты на дне океанов.")]
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_sentence_not_about_object_kept(self):
        q = "Какие баженовские отложения встречаются в Линденской впадине?"
        sources = [src(1, "В Западной Сибири баженовская свита…")]
        answer = "В Западной Сибири баженовская свита сложена аргиллитами [1]."
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_definition_needs_term_in_cited_fragment(self):
        """«Что такое X?»: фраза по фрагменту, где X не назван, — пересказ чужого (d05)."""
        answer = "Кероген I типа богат водородом [1]."
        self.assertEqual(
            bind_objects("Что такое кероген?", answer, [src(1, "Кероген I типа богат H.")]),
            (answer, []),
        )
        text, removed = bind_objects("Что такое кероген?", answer, [src(1, "Альбититы…")])
        self.assertEqual((text, len(removed)), ("", 1))


if __name__ == "__main__":
    unittest.main()


class AdjacentNameTests(unittest.TestCase):
    """Имя из нескольких слов названо, только если слова стоят подряд (oil40, 11.11.2026)."""

    def test_two_regions_from_coordinated_adjectives(self):
        self.assertEqual(
            question_objects("Чем различаются условия в Западной и Восточной Сибири?"),
            [["запад", "сиби"], ["восточ", "сиби"]],
        )

    def test_words_of_name_apart_do_not_name_it(self):
        q = "Чем различаются условия нефтеобразования в Западной и Восточной Сибири?"
        answer = (
            "В Западной Сибири минерализация вод растёт к северу [1]. В Восточной Сибири "
            "плотность ресурсов составляет 14–16 тыс. т/км² [2]."
        )
        sources = [
            src(1, "В северных районах Западно-Сибирской низменности минерализация вод растёт."),
            src(
                2,
                "В Восточно-Уральской области и Березовском районе Западно-Сибирской "
                "провинции плотность ресурсов 14–16 тыс. т/км².",
            ),
        ]
        text, removed = bind_objects(q, answer, sources)
        self.assertEqual(text, "В Западной Сибири минерализация вод растёт к северу [1].")
        self.assertEqual(len(removed), 1)

    def test_hyphenated_form_names_it(self):
        q = "Каковы перспективы рифейских отложений Восточной Сибири?"
        answer = "В Восточной Сибири рифейский комплекс продуктивен [1]."
        sources = [src(1, "Рифейский комплекс Восточно-Сибирской платформы продуктивен.")]
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))


class SideOfObjectTests(unittest.TestCase):
    """Вопрос о части объекта: сторона света должна стоять во фрагменте рядом с именем (t03)."""

    Q = "Какие рудные полезные ископаемые известны на севере Сибирской платформы?"

    def test_other_part_of_object_removed(self):
        answer = "На Сибирской платформе известно месторождение Сухой Лог [1]."
        sources = [src(1, "Сухой Лог расположен в южном обрамлении Сибирской платформы.")]
        text, removed = bind_objects(self.Q, answer, sources)
        self.assertEqual(text, "")
        self.assertEqual(len(removed), 1)

    def test_same_part_stays(self):
        answer = "На севере Сибирской платформы известны алмазы Анабара [1]."
        sources = [src(1, "Анабарский район на севере Сибирской платформы известен алмазами.")]
        self.assertEqual(bind_objects(self.Q, answer, sources), (answer, []))

    def test_whole_object_does_not_answer_its_part(self):
        """«Сибирская платформа сложена…» — об объекте целиком, не о его севере (t03)."""
        answer = "Сибирская платформа сложена архейскими комплексами [1]."
        sources = [
            src(
                1,
                "На севере Сибирской платформы — Анабарский щит; платформа сложена архейскими комплексами.",
            )
        ]
        text, removed = bind_objects(self.Q, answer, sources)
        self.assertEqual((text, len(removed)), ("", 1))

    def test_definition_checks_every_cited_sentence(self):
        """«Что такое X?»: фраза без X по фрагменту без X — пересказ чужого (d05)."""
        q = "Что такое лиственит?"
        answer = (
            "Лиственит — метасоматическая порода [2]. В фрагменте [1] отмечается, что в "
            "альбититах проявлены прожилки анкерита [1]."
        )
        sources = [
            src(1, "Формации: альбититы, листвениты. В альбититах проявлены прожилки анкерита."),
            src(2, "Лиственит — метасоматическая горная порода.", "Геологический словарь"),
        ]
        text, removed = bind_objects(q, answer, sources)
        self.assertEqual(text, "Лиственит — метасоматическая порода [2].")

    def test_question_without_side_as_before(self):
        q = "Какие полезные ископаемые известны на Сибирской платформе?"
        answer = "На Сибирской платформе известно месторождение Сухой Лог [1]."
        sources = [src(1, "Сухой Лог расположен в южном обрамлении Сибирской платформы.")]
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))


class NamedMethodTests(unittest.TestCase):
    """Утверждение о названном методе — только по фрагментам, где он назван (f04, h02)."""

    Q = "Почему метод весов доказательств не требует известных месторождений для обучения?"

    def test_other_method_under_its_name_removed(self):
        answer = "Метод весов доказательств не требует месторождений: веса задаёт эксперт [1]."
        sources = [
            src(1, "In weighted overlay the weights are assigned by expert judgment or AHP.")
        ]
        text, removed = bind_objects(self.Q, answer, sources)
        self.assertEqual(text, "")
        self.assertEqual(len(removed), 1)

    def test_fragment_naming_the_method_keeps_claim(self):
        answer = "Метод весов доказательств использует известные месторождения [1]."
        sources = [src(1, "The weights-of-evidence model uses presence or absence of deposits.")]
        self.assertEqual(bind_objects(self.Q, answer, sources), (answer, []))

    def test_question_without_listed_method_untouched(self):
        q = "Как выбирают отрицательные точки при обучении модели?"
        answer = "Отрицательные точки выбирают случайно [1]."
        self.assertEqual(bind_objects(q, answer, [src(1, "Random negatives.")]), (answer, []))


class DefinitionTermFormsTests(unittest.TestCase):
    """Термин определения — в любом падеже, порядке и на английском (s03, 14.11.2026)."""

    def test_adjective_and_noun_forms_match(self):
        q = "Что такое сдвиговые зоны и как с ними связано золотое оруденение?"
        answer = "Сдвиговые зоны — уплощённые структурные зоны простого сдвига [1]."
        sources = [src(1, "Зона сдвига — уплощенная структурная зона простого сдвига.")]
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))

    def test_fragment_in_english_is_not_rejected_for_language(self):
        q = "Что такое сдвиговые зоны?"
        answer = "В Менге золото связано со сдвиговыми зонами [1]."
        sources = [src(1, "At Menge gold is hosted by quartz veins along shear zones.")]
        self.assertEqual(bind_objects(q, answer, sources), (answer, []))
