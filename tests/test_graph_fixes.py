"""Граф по тест-плану: связь по-русски (16.25), имена как в статье (16.26), причины
отказов по видам (16.27), граф в чате и при сборе по статьям (11.31, 11.33).
Без базы и модели."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fakedb import FakeConn

from georag.chat import answer as chat
from georag.chat.plan import Part, Plan
from georag.chat.retrieval import Retrieval
from georag.chat.settings import Settings
from georag.graph import facts, store

EN = "Gold is associated with pyrite in the Kekura deposit. Rutile occurs in the Moeda Formation."
RU = "Золото ассоциирует с пиритом на месторождении Кекура."


def _one(text: str, item: list[Any]) -> facts.Checked:
    return facts.check({"факты": [item]}, text)


def test_relation_in_english_rejected() -> None:
    checked = _one(EN, [1, "gold", "is associated with", "pyrite"])
    assert not checked.facts and checked.rejected[0].endswith("связь не по-русски")


def test_relation_with_latin_letter_inside_rejected() -> None:
    checked = _one(EN, [1, "gold", "тrending", "pyrite"])  # латинская «r» в русском слове
    assert not checked.facts and checked.rejected[0].endswith("связь не по-русски")


def test_russian_relation_kept() -> None:
    checked = _one(EN, [1, "gold", "ассоциирует с", "pyrite"])
    assert [(f.src, f.relation, f.dst) for f in checked.facts] == [
        ("gold", "ассоциирует с", "pyrite")
    ]


def test_half_translated_name_rejected() -> None:
    checked = _one(EN, [2, "rutile из Moeda Formation", "залегает в", "Moeda Formation"])
    assert not checked.facts
    assert checked.rejected[0].endswith("имя переведено, а в статье оно по-английски")


def test_russian_name_in_russian_quote_kept() -> None:
    checked = _one(RU, [1, "золото", "ассоциирует с", "пиритом"])
    assert len(checked.facts) == 1


def test_reason_kind_groups_names() -> None:
    assert store.reason_kind("в цитате нет «золото»") == store.reason_kind("в цитате нет «разломы»")
    assert store.reason_kind("нет предложения 12") == "нет предложения с таким номером"
    assert store.reason_kind("связь не по-русски") == "связь не по-русски"


def test_rejected_reasons_counts_and_examples() -> None:
    rows = [
        ("«a» — x — «золото»: в цитате нет «золото»",),
        ("«b» — y — «разломы»: в цитате нет «разломы»",),
        ("«c» — built out from — «d»: связь не по-русски",),
        ("«вода (CO3: pH > 7)» — z — «e»: в цитате нет «вода (CO3: pH > 7)»",),
    ]
    conn = FakeConn(rows)
    found = store.rejected_reasons(conn)
    assert found[0][:2] == ("в цитате нет одного из концов", 3)  # двоеточие в имени — тоже
    assert found[0][2][0] == "«a» — x — «золото»: в цитате нет «золото»"
    assert found[1] == ("связь не по-русски", 1, ["«c» — built out from — «d»"])


# --------------------------------------------------------------------------- #
#  Граф в чате
# --------------------------------------------------------------------------- #
GOLD = {
    "name": "золото",
    "documents": 12,
    "includes": [],
    "pass": {"passed": 10, "chunks": 10},
    "facts": [
        {
            "about": "золото",
            "relation": "ассоциирует с",
            "other": "пирит",
            "direction": "→",
            "documents": 7,
            "chunk_ids": [1],
        }
    ],
}


class Gathered:
    sources = [{"n": 1, "doc_id": "a", "ord": 0, "text": "выписки по статьям"}]

    def info(self) -> dict[str, Any]:
        return {"name": "золото", "collected": {"articles": 30, "read": 30}}


def _prepare(
    monkeypatch: pytest.MonkeyPatch, question: str, plan: Plan
) -> tuple[Any, list[Any], list[str]]:
    """prepare со сбором по статьям (как в базе) и подменённым графом."""
    seen_extra: list[Any] = []
    hints: list[str] = []

    def collect(*a: Any, **k: Any) -> Iterator[Any]:
        yield {"type": "status", "text": "собираю"}
        return Gathered()

    def retrieve(*a: Any, **k: Any) -> Iterator[Any]:
        seen_extra.extend(a[9] if len(a) > 9 else k.get("extra") or [])
        yield {"type": "status", "text": "ищу"}
        return Retrieval([], judged=True)

    def graph_hits(conn: Any, hint: str, q: str) -> list[str]:
        hints.append(hint)
        return ["фрагмент 1 из графа", "фрагмент 2 из графа"]

    monkeypatch.setattr(chat, "collect", collect)
    monkeypatch.setattr(chat, "retrieve", retrieve)
    monkeypatch.setattr(chat.db, "is_real", lambda conn: True)
    monkeypatch.setattr(chat, "question_subject", lambda conn, hint, q: "золото")
    gen = chat.prepare(
        object(),
        None,
        question,
        [],
        Settings(),
        lambda q, h, s: plan,
        judge=lambda *a, **k: None,
        datasets=lambda conn, hint, q: GOLD,
        graph_hits=graph_hits,
        catalog=None,
        article=None,
    )
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value, seen_extra, hints


def test_collect_also_takes_graph_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    """11.31: сбор по статьям сработал — фрагменты из графа всё равно в кандидатах."""
    plan = Plan([Part("Что известно о золоте?", [])], subject="золото", collect=True)
    prepared, extra, hints = _prepare(monkeypatch, "Что известно о золоте?", plan)
    assert hints == ["золото"] and len(extra) == 2
    assert prepared.graph == 2
    assert [s["doc_id"] for s in prepared.sources] == ["a"]  # не вопрос о связях — без сводки


def test_link_question_gets_graph_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """11.33: «С какими минералами связано золото?» — модель предмет не назвала, а сводка
    фактов графа со счётом статей идёт модели рядом с выписками по статьям."""
    question = "С какими минералами связано золото?"
    plan = Plan([Part(question, [])], subject=None, collect=True)
    prepared, extra, hints = _prepare(monkeypatch, question, plan)
    assert hints == ["золото"] and len(extra) == 2
    summary = prepared.sources[1]
    assert summary["doc_id"] == "датасет:золото" and summary["n"] == 2
    assert "пирит (7 статей)" in summary["text"]
    assert prepared.graph == 3
