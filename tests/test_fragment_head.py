"""Заголовок фрагмента в задании модели (chat/messages.fragment_block): авторы, год, страницы.
Без базы и модели."""

from __future__ import annotations

from georag.chat.messages import SYSTEM, fragment_block, surnames


def test_surnames() -> None:
    assert surnames(["Daiei Inoue", "Yukihiro Mizuochi", "Hiroyuki SAKURADA", "Ещё Один"]) == (
        "Inoue, Mizuochi, SAKURADA"
    )
    assert surnames(["Г. Н. Иванченко"]) == "Иванченко"
    assert surnames(["Smith, John", "van Dijk, A."]) == "Smith, Dijk"  # «Фамилия, Имя»
    assert surnames(None) == "" and surnames(["", " "]) == ""


def head(source: dict[str, object]) -> str:
    base = {"n": 7, "title": "Geological Characteristics", "text": "Lineaments coincide."}
    return fragment_block({**base, **source}).splitlines()[0]


def test_authors_in_fragment_head() -> None:
    authors = ["Daiei Inoue", "Yukihiro Mizuochi", "Hiroyuki Sakurada", "Taro Yamada"]
    assert head({"authors": authors, "year": 1992, "pages": [3]}) == (
        "[7] «Geological Characteristics» (Inoue, Mizuochi, Sakurada, 1992, с. 3)"
    )
    assert head({"year": 1992}) == "[7] «Geological Characteristics» (1992)"
    assert head({}) == "[7] «Geological Characteristics»"


def test_note_keeps_authors_summary_does_not() -> None:
    note = fragment_block(
        {"n": 2, "title": "T", "authors": ["Daiei Inoue"], "kind": "заметка", "note": "суть"}
    )
    assert note.splitlines() == ["[2] «T» (Inoue)", "(заметка по фрагменту) суть"]
    for kind in ("сводка", "датасет"):
        assert head({"kind": kind, "authors": ["Daiei Inoue"]}) == (
            "[7] «Geological Characteristics»"
        )


def test_rules_mention_authors_and_translation() -> None:
    assert "Смит — Smith" in SYSTEM and "переводи" in SYSTEM
