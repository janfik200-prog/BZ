"""Переключатели и числа из окружения (georag/env.py) — одни правила у всех флагов."""

from __future__ import annotations

import pytest

from georag import env
from georag.chat.settings import Settings, dictionary_on, env_flag
from georag.evidence import gate
from georag.graph.entities import ENTITY_LIST_ENV, entity_list_enabled


@pytest.mark.parametrize(
    "value, default, expected",
    [
        ("1", False, True),
        ("TRUE", False, True),
        (" да ", False, True),
        ("on", False, True),
        ("0", True, False),
        ("Нет", True, False),
        ("off", True, False),
        ("", True, True),
        ("", False, False),
        ("может быть", True, True),
        ("может быть", False, False),
    ],
)
def test_flag(monkeypatch: pytest.MonkeyPatch, value: str, default: bool, expected: bool) -> None:
    monkeypatch.setenv("GEORAG_TEST_FLAG", value)
    assert env.flag("GEORAG_TEST_FLAG", default) is expected


@pytest.mark.parametrize("value, expected", [("12", 12), ("0", 5), ("-3", 5), ("abc", 5), ("", 5)])
def test_positive_int(monkeypatch: pytest.MonkeyPatch, value: str, expected: int) -> None:
    monkeypatch.setenv("GEORAG_TEST_INT", value)
    assert env.positive_int("GEORAG_TEST_INT", 5) == expected


@pytest.mark.parametrize("value", ["true", "да", "YES"])
def test_flags_read_alike(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Раньше «true» не включал GEORAG_SKIP_CAPTIONS, «да» — GEORAG_SCOPE_GATE."""
    for name in ("GEORAG_SCOPE_GATE", "GEORAG_CONCEPTS", "GEORAG_SKIP_CAPTIONS"):
        monkeypatch.setenv(name, value)
    assert gate.enabled() and env_flag("GEORAG_SKIP_CAPTIONS") and Settings().concepts


def test_default_on_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Словарь в чате и список предметов графа включены, пока их явно не выключат."""
    monkeypatch.delenv("GEORAG_DICT", raising=False)
    monkeypatch.delenv(ENTITY_LIST_ENV, raising=False)
    assert dictionary_on() and entity_list_enabled()
    monkeypatch.setenv("GEORAG_DICT", "нет")
    monkeypatch.setenv(ENTITY_LIST_ENV, "off")
    assert not dictionary_on() and not entity_list_enabled()


def test_graph_workers_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEORAG_CONFIRM_WORKERS=abc ронял graph ValueError ещё до разбора ключей."""
    from georag.graph import cli

    monkeypatch.setenv("GEORAG_CONFIRM_WORKERS", "abc")
    monkeypatch.setenv("GEORAG_WORKERS", "0")
    args = cli.build_parser().parse_args(["build"])
    assert (args.workers, args.confirm_workers) == (cli.WORKERS, cli.CONFIRM_WORKERS)
