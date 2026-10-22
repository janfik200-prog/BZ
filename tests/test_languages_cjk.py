"""Статьи на китайском и японском: вопрос в переводе (GEORAG_TRANSLATE) — своим правилом
разбора и своей небольшой добавкой кандидатов (plan, retrieval). Без базы и модели."""

from __future__ import annotations

from typing import Any

import pytest

from georag.chat import plan as planmod
from georag.chat import retrieval
from georag.chat.answer import Prepared
from georag.chat.settings import Settings
from georag.index.search import Hit

DATA = {
    "вопрос_целиком": "Масштаб аэромагнитной съёмки бассейна",
    "части": [{"вопрос": "Масштаб съёмки", "запросы": ["аэромагнитная съёмка масштаб"]}],
    "перевод": ["盆地 航磁测量 比例尺", "盆地 空中磁気探査"],
}


def make() -> planmod.Plan:
    return planmod.make_plan("?", [], Settings(), lambda q, h, s: planmod.parse_plan(DATA, q, s))


def test_translations_kept_apart_from_queries() -> None:
    plan = make()
    assert plan.translated == DATA["перевод"]
    assert not set(plan.translated) & set(plan.queries)
    info = Prepared(plan, [], retrieval.Retrieval([])).info()
    assert info["queries"][-2:] == DATA["перевод"]  # в «Искал» переводы видны
    one = planmod.parse_plan({**DATA, "перевод": "盆地 航磁"}, "?", Settings())
    assert one.translated == ["盆地 航磁"]  # строкой, а не списком
    assert planmod.parse_plan({"части": []}, "?", Settings()).translated == []


def test_translations_searched_as_small_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Переводы ищутся отдельно, TRANSLATED кандидатов, после основных."""
    plan = make()
    asked: list[tuple[list[str], int]] = []
    zh = Hit(99, "zh-doc", 1, "盆地 1∶5万 高精度航磁测量", title="盆地", similarity=0.72)
    en = Hit(1, "en-doc", 1, "Aeromagnetic survey at 1:50 000.", title="Survey", similarity=0.6)

    def candidates(
        conn: Any, emb: Any, queries: list[str], cache: Any, limit: int, facts: bool, doc: Any
    ) -> list[Any]:
        asked.append((list(queries), limit))
        return [zh] if queries == plan.translated else [en]

    def judge(question: str, plan: Any, batch: list[dict[str, Any]], *a: Any) -> Any:
        return {"подходят": [c["id"] for c in batch], "не_хватает": [], "запросы": []}

    monkeypatch.setattr(retrieval, "candidates", candidates)
    found = retrieval.run(retrieval.retrieve(None, None, "?", plan, Settings(), judge=judge))
    assert asked[0][0] == plan.queries and asked[-1] == (plan.translated, retrieval.TRANSLATED)
    assert [s["doc_id"] for s in found.sources] == ["en-doc", "zh-doc"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("  ", []),
        ("по-китайски,по-японски", ["по-китайски", "по-японски"]),
        (" по-китайски , ,", ["по-китайски"]),
    ],
)
def test_translate_languages(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("GEORAG_TRANSLATE", raw)
    assert planmod.translate_languages() == expected


def test_translation_rule_only_when_languages_set(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    class Client:
        def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
            asked.append(user)
            return {"части": [{"вопрос": "q", "запросы": ["q"]}]}

    monkeypatch.setattr(Settings, "client", lambda self: Client())
    for langs in ("", "по-китайски,по-японски"):
        monkeypatch.setenv("GEORAG_TRANSLATE", langs)
        planmod.plan_question("Вопрос?", [], Settings())
    without, with_langs = asked
    assert "«перевод»" not in without and '"перевод"' not in without
    assert (
        "по-китайски (" in with_langs and "по-японски (" in with_langs and '"перевод"' in with_langs
    )
    assert "НЕ весь вопрос" in with_langs and "корейск" in with_langs


def test_broken_translations_dropped() -> None:
    """Живая проверка 21.10.2026: корейский вместо японского и русский вместо перевода."""
    got = ["盆地 航磁", "盆地 空中磁気", "투라수 지역 광맥", "ДЗЗ прогнозирование молибдена"]
    langs = ["по-китайски", "по-японски"]
    assert planmod.keep_translations(got, langs) == ["盆地 航磁", "盆地 空中磁気"]
    assert "투라수 지역 광맥" in planmod.keep_translations(got, [*langs, "по-корейски"])


def test_rule_names_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    class Client:
        def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
            asked.append(user)
            return {}

    monkeypatch.setattr(Settings, "client", lambda self: Client())
    monkeypatch.setenv("GEORAG_TRANSLATE", "по-китайски,по-японски")
    planmod.plan_question("Вопрос?", [], Settings())
    assert "по-китайски (упрощёнными иероглифами" in asked[0] and "кандзи и кана" in asked[0]
