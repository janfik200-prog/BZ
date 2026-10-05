"""Где базе не хватает статей — по графу, и темы для добычи, чтобы добрать.

Два вида пробелов:

* **в словаре есть, фактов нет.** Название попало в config/synonyms.yaml — значит,
  оно вам нужно, — а в статьях базы о нём ничего не сказано;
* **всё известно из одной статьи.** О сущности много фактов, но все из одной
  статьи: подтвердить или опровергнуть нечем.

Каждый пробел — тема для добычи в том же виде, что config/topics.yaml: имя и,
если в словаре есть английское написание, оно же отдельной темой — открытых
статей на английском больше.

    python georag.py gaps                             → config/topics-graph.yaml
    python georag.py all --topics topics-graph.yaml   добыть
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "config" / "topics-graph.yaml"

MIN_FACTS = 3  # «много фактов» — от стольких
MAX_WORDS = 5  # длинные имена — пересказ, а не название: темой не годятся
_DIGITS = re.compile(r"\d")
_LATIN = re.compile(r"^[A-Za-z][A-Za-z0-9 \-]+$")


def _facts(n: int) -> str:
    last2, last = n % 100, n % 10
    word = (
        "факт"
        if last == 1 and last2 != 11
        else "факта" if 2 <= last <= 4 and not 12 <= last2 <= 14 else "фактов"
    )
    return f"{n} {word}"


def _topic_name(name: str) -> bool:
    return len(name.split()) <= MAX_WORDS and not _DIGITS.search(name)


def _mentioned(spellings: list[str], names: list[str]) -> bool:
    """Встречается ли хоть одно написание внутри имени какой-нибудь сущности:
    «Донбасс» — в «Восточный Донбасс», «Донбасс (Донецкий угольный бассейн)»."""
    from .synonyms import _spelling_pattern

    for spelled in spellings:
        if len(spelled) < 3:
            continue
        try:
            pattern = _spelling_pattern(spelled)
        except re.error:
            continue
        if any(pattern.search(name) for name in names):
            return True
    return False


def find(g: Any, syn: Any, limit: int = 25) -> list[dict[str, Any]]:
    """Пробелы: [{"name", "why", "topics": [...]}]. Поровну: половина — словарь без
    фактов, половина — всё из одной статьи (чего-то одного не хватит — добирает другим)."""
    from ..text import normalize

    names = [normalize(e) for e in g.entities]
    missing = []
    for canon, variants in syn.groups.items():
        if canon in g.entities or g.related(canon) or _mentioned([canon, *variants], names):
            continue
        english = [v for v in variants if _LATIN.match(v)][:1]
        missing.append(
            {
                "name": canon,
                "why": "в словаре синонимов есть, в статьях базы фактов нет",
                "topics": [canon, *english],
            }
        )
    lonely_rows = []
    lonely = [
        e
        for e in g.entities.values()
        if e["facts"] >= MIN_FACTS and len(e["docs"]) == 1 and _topic_name(e["name"])
    ]
    lonely.sort(key=lambda e: -e["facts"])
    for e in lonely:
        english = [v for v in syn.groups.get(e["name"], []) if _LATIN.match(v)][:1]
        lonely_rows.append(
            {
                "name": e["name"],
                "why": f"{_facts(e['facts'])}, и все из одной статьи — подтвердить нечем",
                "topics": [e["name"], *english],
            }
        )
    half = (limit + 1) // 2
    take_missing = max(half, limit - len(lonely_rows))
    out = missing[:take_missing]
    return out + lonely_rows[: limit - len(out)]


def to_yaml(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ТЕМЫ ДЛЯ ДОБЫЧИ ПО ПРОБЕЛАМ ГРАФА — сделаны командой python georag.py gaps,",
        "# пересоздаются ею же. Лишние темы удалите, нужные перенесите в config/topics.yaml.",
        "#",
        "# Добыть:  python georag.py all --topics topics-graph.yaml",
        "#          python georag.py ingest",
        "#          python georag.py graph",
        "",
        "topics:",
    ]
    if not rows:
        lines.append("  # пробелов не найдено")
    for row in rows:
        lines.append(f"  # {row['name']}: {row['why']}")
        for topic in row["topics"]:
            lines.append(f"  - {_quote(topic)}")
    return "\n".join(lines) + "\n"


def _quote(text: str) -> str:
    if re.search(r"[:#\[\]{},&*!|>'\"%@`]", text) or text != text.strip():
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text
