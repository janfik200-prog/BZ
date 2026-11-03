"""T9 / E3: опция «отсев первого этапа без списка предметов в задании».

По умолчанию facts.run отдаёт модели список предметов (как было); entity_list=False или
GEORAG_GRAPH_ENTITY_LIST=0 — отсев тот же, а задание — как без первого этапа. Без базы и
модели: хранилище и проверка дублей подменены, модель — запись заданий."""

from __future__ import annotations

from typing import Any

import pytest

from georag.graph import dedup, entities, facts, store
from georag.graph.entities import Found

TEXTS = {
    1: "Пирит и халцедон в кварцевых жилах с золотом. Сульфиды в зоне смятия.",
    2: "Таблица 3.",  # беден терминами — отсеивается до модели
}


class FakeFinder:
    def find(self, text: str) -> Found:
        if text.startswith("Пирит"):
            return Found(
                "ru",
                terms=["Пирит", "халцедон", "кварцевых жилах", "золотом"],
                names=["Ключевское"],
            )
        return Found("ru", terms=[])


class FakeLLM:
    model = "fake"

    def __init__(self) -> None:
        self.users: list[str] = []

    def chat_json(self, system: str, user: str, **_kw: Any) -> dict:
        self.users.append(user)
        return {"факты": []}


@pytest.fixture
def patched(monkeypatch):
    skipped: list[int] = []
    saved: list[int] = []
    monkeypatch.setattr(
        store,
        "todo",
        lambda conn, limit=None, retry_skipped=False: [(i, "d", t) for i, t in TEXTS.items()],
    )
    monkeypatch.setattr(
        store, "skip", lambda conn, chunk_id, reason, lang=None: skipped.append(chunk_id)
    )
    monkeypatch.setattr(store, "save", lambda conn, chunk_id, *a, **kw: saved.append(chunk_id))
    monkeypatch.setattr(dedup, "find_duplicate", lambda conn, chunk_id, text: None)
    monkeypatch.delenv(entities.ENTITY_LIST_ENV, raising=False)

    class Conn:
        def commit(self) -> None:
            pass

    return Conn(), skipped, saved


def _run(conn: Any, **kw: Any) -> tuple[FakeLLM, dict]:
    llm = FakeLLM()
    stats = facts.run(conn, llm, log=lambda *_: None, finder=FakeFinder(), min_terms=3, **kw)
    return llm, stats


def test_default_keeps_entity_list(patched) -> None:
    conn, skipped, saved = patched
    llm, stats = _run(conn)
    assert skipped == [2] and saved == [1] and stats["sparse"] == 1
    assert len(llm.users) == 1 and "Предметы, найденные в этом фрагменте" in llm.users[0]


def test_option_drops_list_but_keeps_filter(patched) -> None:
    conn, skipped, saved = patched
    llm, stats = _run(conn, entity_list=False)
    assert skipped == [2] and saved == [1] and stats["sparse"] == 1  # отсев — как прежде
    rules = facts.load_rules()
    assert llm.users == [facts.messages(rules, TEXTS[1], None)[1]]  # задание — как у E0
    assert "Предметы" not in llm.users[0]


def test_env_flag(patched, monkeypatch) -> None:
    conn, _skipped, _saved = patched
    monkeypatch.setenv(entities.ENTITY_LIST_ENV, "0")
    assert not entities.entity_list_enabled()
    llm, _ = _run(conn)
    assert "Предметы" not in llm.users[0]
    llm, _ = _run(conn, entity_list=True)  # явный аргумент сильнее флага
    assert "Предметы" in llm.users[0]
    monkeypatch.setenv(entities.ENTITY_LIST_ENV, "1")
    assert entities.entity_list_enabled()
