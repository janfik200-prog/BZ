"""Новый конвейер графа (план v2, А1–А4): дубли, язык, предметы до модели, запрос со
списком предметов. Без базы и модели; spaCy не нужен — первый этап по словарю."""

from __future__ import annotations

from georag.graph import dedup, entities, facts

TEXT_EN = (
    "Gold mineralization in the Kibali district is hosted by quartz veins along the "
    "shear zone. Pyrite and arsenopyrite accompany gold in the sericite-altered schist."
)
TEXT_RU = (
    "Золотое оруденение Кекуры приурочено к кварцевым жилам в зоне смятия. Пирит и "
    "арсенопирит сопровождают золото в березитах."
)


def test_language() -> None:
    assert entities.language(TEXT_RU) == "ru"
    assert entities.language(TEXT_EN) == "en"
    assert entities.language("Кварц и pyrite, золото и gold ore, разлом fault") == "mixed"


def test_terms_in_any_case() -> None:
    found = entities.Finder(use_spacy=False).find(TEXT_RU)
    terms = {t.lower() for t in found.terms}
    assert {"пирит", "арсенопирит", "золото", "зоне смятия"} <= terms
    assert found.lang == "ru" and found.enough(3)


def test_terms_keep_article_spelling() -> None:
    found = entities.Finder(use_spacy=False).find(TEXT_EN)
    assert "Gold" in found.terms and "quartz veins" in found.terms
    assert "shear zone" in found.terms


def test_sparse_fragment_not_enough() -> None:
    found = entities.Finder(use_spacy=False).find(
        "The authors thank the reviewers for helpful comments on the manuscript."
    )
    assert not found.enough(entities.MIN_TERMS)


def test_entities_unique_and_capped() -> None:
    found = entities.Found("en", terms=["Gold", "gold", "pyrite"] + [f"t{i}" for i in range(60)])
    assert found.entities[:2] == ["Gold", "pyrite"]
    assert len(found.entities) == entities.MAX_ENTITIES


def test_duplicate_signature() -> None:
    a = dedup.signature(TEXT_EN * 3)
    b = dedup.signature(TEXT_EN * 3 + " Figure 3 shows the map.")
    c = dedup.signature(TEXT_RU * 3)
    assert a is not None and b is not None and c is not None
    assert dedup.similarity(a, b) >= dedup.DUP_SIMILARITY
    assert dedup.similarity(a, c) < 0.2
    # похожие подписи делят хоть одну полосу — найдутся индексом
    assert set(dedup.bands(a)) & set(dedup.bands(b))


def test_short_fragment_has_no_signature() -> None:
    assert dedup.signature("Short text of a few words.") is None


def test_messages_list_entities() -> None:
    _system, user = facts.messages("правила " * 20, TEXT_EN, ["Gold", "pyrite"])
    assert "Предметы, найденные в этом фрагменте: «Gold»; «pyrite»" in user
    _system, plain = facts.messages("правила " * 20, TEXT_EN)
    assert "Предметы" not in plain


def test_extra_names() -> None:
    fact = facts.Fact("Kibali district", "содержит", "gold", "q")
    assert facts.extra_names([fact], ["gold", "quartz"]) == ["Kibali district"]
    # «разлом» в списке, «Персияновский разлом» — тоже он (одно содержит другое)
    known = facts.Fact("Персияновский разлом", "контролирует", "gold", "q")
    assert facts.extra_names([known], ["разлом", "gold"]) == []


def test_truncated_answer_keeps_whole_facts() -> None:
    import pytest

    from georag.llm import extract_json

    cut = '{"факты": [[1, "gold", "содержит", "Ag"], [2, "pyrite", "содержит", "gold"], [3, "go'
    assert extract_json(cut, repair=True) == {
        "факты": [[1, "gold", "содержит", "Ag"], [2, "pyrite", "содержит", "gold"]]
    }
    with pytest.raises(ValueError):  # без repair — как раньше: ошибка
        extract_json(cut)
    # кавычки и скобки внутри строк не путают
    tricky = '{"факты": [[1, "zone [A]", "связано с", "a \\"b\\""], [2, "x'
    assert extract_json(tricky, repair=True)["факты"] == [[1, "zone [A]", "связано с", 'a "b"']]
    # целый ответ не трогается
    assert extract_json('{"факты": []}', repair=True) == {"факты": []}


def test_forms_of_one_term_count_once() -> None:
    """«пирит», «пирита», «пиритом» — один термин словаря: порог k и список для модели
    считают термины, а не формы слова."""
    finder = entities.Finder(use_spacy=False)
    found = finder.find("Пирит, пирита и пиритом; с пиритами.")
    assert len(found.terms) == 4
    assert found.distinct_terms == 1
    assert not found.enough(2)
    assert found.entities == ["Пирит"]


def test_term_keys_follow_dictionary() -> None:
    found = entities.Finder(use_spacy=False).find(TEXT_RU)
    assert len(found.keys) == len(found.terms)
    assert found.distinct_terms == len(set(found.keys))


def test_spacy_name_same_as_term_not_repeated() -> None:
    found = entities.Found("en", terms=["gold"], names=["Gold", "Kibali"])
    assert found.entities == ["gold", "Kibali"]


def test_released_duplicate_back_in_todo() -> None:
    """Почти-дубль, чей оригинал больше не пройден, — снова в работе; с --no-filter —
    все отсеянные."""
    from fakedb import FakeConn

    from georag.graph import store

    conn = FakeConn(rows=[])
    store.todo(conn, limit=5)
    sql = conn.log[-1][0]
    assert f"left(p.skipped, {len(store.DUPLICATE_OF)}) = '{store.DUPLICATE_OF}'" in sql
    assert "NOT EXISTS (SELECT 1 FROM facts_pass o" in sql
    store.todo(conn, limit=5, retry_skipped=True)
    assert "OR p.skipped IS NOT NULL" in conn.log[-1][0]
