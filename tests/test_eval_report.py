"""Отчёт оценки: режим прогона в шапке и сравнение только со сравнимым прогоном."""

from __future__ import annotations

import json
from pathlib import Path

from georag.evaluation import previous_run


def _run(logs: Path, stamp: str, ids: list[str], graph: bool, value: float) -> None:
    data = {
        "meta": {"when": stamp, "graph": graph, "facts": False, "with_answer": True},
        "summary": {"search_ok": value},
        "results": [{"id": i} for i in ids],
    }
    (logs / f"eval-{stamp}.json").write_text(json.dumps(data), encoding="utf-8")


def test_previous_same_mode_and_questions(tmp_path: Path) -> None:
    _run(tmp_path, "20261001-1000", ["s01", "x01"], True, 0.5)  # сравнимый
    _run(tmp_path, "20261002-1000", ["s01", "x01"], False, 0.9)  # без графа
    _run(tmp_path, "20261003-1000", ["s01"], True, 0.7)  # другие вопросы
    meta = {"graph": True, "facts": False, "with_answer": True}
    summary, when = previous_run(tmp_path, "20261010-1000", meta, ["x01", "s01"])
    assert (summary, when) == ({"search_ok": 0.5}, "20261001-1000")


def test_previous_none(tmp_path: Path) -> None:
    _run(tmp_path, "20261001-1000", ["s01"], True, 0.5)
    meta = {"graph": False, "facts": False, "with_answer": True}
    assert previous_run(tmp_path, "20261010-1000", meta, ["s01"]) == (None, "")
