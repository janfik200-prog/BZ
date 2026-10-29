"""Ключевые понятия вопроса: найденные фрагменты, где они есть, — выше.

Поиск по смыслу и по словам находит и фрагменты «рядом с темой»: тот же термин в другом
значении, общие слова. Модель выделяет из вопроса одно–три обязательных понятия (метод,
показатель, минерал или процесс, тип оруденения, полезное ископаемое, место) с русским и
английским написанием; фрагменты упорядочиваются по доле найденных в них понятий, а при
равной доле — как нашёл поиск. Ничего не выбрасывается: фрагмент без понятий просто ниже.

Проверка 30.10.2026 (research/v15, 25 вопросов, разметка «полезен ли фрагмент»): доля
полезных среди пяти лучших 0,50 → 0,72. Включается GEORAG_CONCEPTS=1.

Модель переводит термины сама и ошибается; где понятие есть в словаре
config/term-translations.yaml, английские написания берутся оттуда.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..paths import ROOT
from ..text import name_key

TRANSLATIONS = ROOT / "config" / "term-translations.yaml"
MAX_CONCEPTS = 3
MIN_SPELLING = 3  # «L», «F», «Au» — совпали бы почти с любым текстом

SYSTEM = (
    "Ты помогаешь искать сведения в геологических статьях на русском и английском языках. "
    "Выдели от одного до трёх обязательных понятий: без каждого из них фрагмент не может отвечать на "
    "вопрос. Это метод, показатель, минерал или процесс, тип оруденения, полезное ископаемое, название "
    "места. Общие слова («признак», «участок», «месторождение», «связь», «формирование», "
    "«картирование», «метод», «как выделяют») обязательными не бывают. Для каждого понятия дай "
    "написания: русское, английское, общепринятые сокращения. Не придумывай сокращений, которых нет в "
    "литературе. Не дроби понятие на части и не заменяй его частными примерами.\n"
    "Примеры:\n"
    "«Отношение калия к торию как признак изменений» → "
    '{"must": [["отношение калия к торию", "K/Th", "potassium-thorium ratio"]]}\n'
    "«Минералы-индикаторы золото-кварцевого оруденения» → "
    '{"must": [["минералы-индикаторы", "indicator minerals"], ["золото-кварцевое оруденение", '
    '"gold-quartz"]]}\n'
    "«Колчеданные месторождения Алтая» → "
    '{"must": [["колчеданные месторождения", "VMS", "massive sulfide"], ["Алтай", "Altai"]]}\n'
    'Ответ — только JSON: {"must": [["написание", "написание", ...], ...]}'
)

# Общие слова: понятие из одних таких слов отбрасывается (основы, как в _stem).
GENERIC = {
    "признак", "участ", "месторожд", "deposit", "связ", "relat", "формиров", "format",
    "картир", "mapp", "метод", "method", "выделен", "контрол", "contr", "зона", "zone",
    "район", "area", "данн", "data", "изучен", "study", "анализ", "analys", "оценк",
    "оруденен", "mineraliz", "mineralis", "порода", "rock",
}  # fmt: skip

_WORD = re.compile(r"[\w/]+", re.U)


def _stem(word: str) -> str:
    """Начало слова для сравнения: падеж и число не мешают («линеаментов» — «линеамент»)."""
    w = word.lower().replace("ё", "е")
    return w if len(w) <= 4 or "/" in w else w[: max(4, len(w) - 3)]


def _words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text or "")]


def _generic(spelling: str) -> bool:
    """Написание из одних общих слов («месторождения», «mapping»); короткие слова не в счёт."""
    words = [_stem(w) for w in _WORD.findall(spelling) if len(w) > 2]
    return bool(words) and all(
        any(w.startswith(g) or (len(w) >= 4 and g.startswith(w)) for g in GENERIC) for w in words
    )


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[str, list[str]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text("utf-8")) or {}
    return {name_key(str(k)): [str(x) for x in (v or [])] for k, v in data.items()}


def translations(path: Path = TRANSLATIONS) -> dict[str, list[str]]:
    """Словарь переводов (ключ русского понятия → английские написания); нет файла — пусто."""
    if not path.exists():
        return {}
    return _load(str(path), path.stat().st_mtime)


def clean(raw: Any, table: dict[str, list[str]] | None = None) -> list[list[str]]:
    """Ответ модели → понятия: без коротких и общих написаний, со словарными переводами."""
    table = translations() if table is None else table
    out: list[list[str]] = []
    for concept in raw if isinstance(raw, list) else []:
        spellings = [str(s).strip() for s in (concept if isinstance(concept, list) else [concept])]
        spellings = [s for s in spellings if len(s) >= MIN_SPELLING and not _generic(s)]
        if not spellings:
            continue
        known = next((table[name_key(s)] for s in spellings if name_key(s) in table), None)
        if known is not None:
            # Перевод — из словаря: кириллические написания модели остаются, латинские — нет.
            spellings = [s for s in spellings if re.search("[а-яё]", s, re.I)] + known
        out.append(list(dict.fromkeys(spellings)))
    return out[:MAX_CONCEPTS]


def extract(llm: Any, question: str) -> list[list[str]]:
    """Понятия вопроса; модель молчит или ответ не разобрать — пусто (поиск как раньше)."""
    try:
        data = llm.chat_json(SYSTEM, question, temperature=0.0, max_tokens=400)
    except Exception:  # noqa: BLE001 — без понятий поиск работает как прежде
        return []
    return clean(data.get("must") if isinstance(data, dict) else None)


def has(spelling: str, words: list[str]) -> bool:
    """Все слова написания есть в тексте (по началу слова)."""
    parts = [_stem(w) for w in _WORD.findall(spelling)]
    return bool(parts) and all(any(t.startswith(p) for t in words) for p in parts)


def share(concepts: list[list[str]], text: str) -> float:
    """Доля понятий, хотя бы одно написание которых есть в тексте."""
    if not concepts:
        return 0.0
    words = _words(text)
    return sum(any(has(s, words) for s in c) for c in concepts) / len(concepts)


def rerank(hits: list[Any], concepts: list[list[str]]) -> list[Any]:
    """Сначала фрагменты с большей долей понятий; при равной — прежний порядок."""
    if not concepts:
        return hits
    scored = [(share(concepts, h.text), i, h) for i, h in enumerate(hits)]
    return [h for _, _, h in sorted(scored, key=lambda x: (-x[0], x[1]))]
