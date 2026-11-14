"""Сокращения во фрагментах и их расшифровки в ответе.

Фрагменты пишут «РОВ», «ФЕС», «ДМНГ» без расшифровки, и модель расшифровывала их сама:
«растительных остатков (РОВ)», «ФЕС (фазовые эффективные свойства)», три разных «ДМНГ»
(прогон 11.11.2026: 12 неверных расшифровок из 70). Правило 4 промта это запрещает — модель
его не выполняет. Здесь:

1. В задание идёт список: сокращения фрагментов с известной расшифровкой
   (config/abbreviations.yaml и «слова (СОКР)» в самих фрагментах) и сокращения без
   расшифровки, которые нужно писать как есть (note).
2. В готовом ответе неверная или ничем не подтверждённая расшифровка убирается — остаётся
   сокращение, как во фрагменте (fix). Расшифровка, слова которой стоят подряд во фрагменте,
   не трогается.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from ..paths import CONFIG

PATH = CONFIG / "abbreviations.yaml"
LISTED = 10  # сокращений каждого вида в задании — не больше

_ABBR = re.compile(r"(?<![\w-])[А-ЯЁ]{2,6}(?![\w-])")
_CAPS = re.compile(r"[А-ЯЁ]{4,}")
_WORDS = re.compile(r"[А-Яа-яЁёA-Za-z]+")
_LETTERS = r"[А-Яа-яЁё][А-Яа-яЁё-]*"
# «слова (СОКР)» и «СОКР (слова)».
_BEFORE = re.compile(rf"((?:{_LETTERS}\s+){{1,6}})\(([А-ЯЁ]{{2,6}})\)")
_AFTER = re.compile(r"(?<![\w-])([А-ЯЁ]{2,6})\s+\(([^()\d]{4,80})\)")


@lru_cache(maxsize=1)
def load() -> dict[str, str]:
    """Расшифровки из config/abbreviations.yaml; файла нет или он не читается — пусто."""
    try:
        import yaml

        data = yaml.safe_load(PATH.read_bytes().decode("utf-8-sig")) or {}
    except Exception:  # noqa: BLE001 — без списка проверяются только расшифровки фрагментов
        return {}
    return {
        str(k).strip(): " ".join(str(v).split())
        for k, v in data.items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }


def _stem(word: str) -> str:
    return word.lower().replace("ё", "е")[:5]


def _stems(text: str) -> list[str]:
    """Основы слов по порядку; составное через дефис — по частям."""
    return [_stem(w) for w in _WORDS.findall(text or "")]


def _texts(sources: list[dict[str, Any]]) -> list[str]:
    return [str(s.get("note") or s.get("text") or "") for s in sources]


def in_fragments(texts: list[str]) -> list[str]:
    """Сокращения во фрагментах, частые первыми. Слово заголовка, набранного прописными
    («ГЕОЛОГИЯ РАЙОНА»), — не сокращение: рядом с ним стоит другое слово прописными."""
    seen: Counter[str] = Counter()
    for text in texts:
        tokens = text.split()
        for i, token in enumerate(tokens):
            word = token.strip(".,;:()«»\"'!?–—")
            if not _ABBR.fullmatch(word):
                continue
            around = tokens[max(0, i - 1) : i] + tokens[i + 1 : i + 2]
            if any(_CAPS.fullmatch(t.strip(".,;:()«»\"'!?–—")) for t in around):
                continue
            seen[word] += 1
    return [a for a, _ in seen.most_common()]


def defined_in(texts: list[str]) -> dict[str, str]:
    """Расшифровки из самих фрагментов: «структурно-литологические ловушки (СЛЛ)». Начало
    расшифровки — ближайшее к скобке слово на первую букву сокращения."""
    out: dict[str, str] = {}
    for text in texts:
        for m in _BEFORE.finditer(text):
            abbr, words = m.group(2), m.group(1).split()
            letters = abbr.lower().replace("ё", "е")
            starts = [
                k
                for k in range(1, min(len(words), len(abbr) + 1) + 1)
                if words[-k].lower().replace("ё", "е").startswith(letters[0])
            ]
            # Лучше — где первые буквы слов (и частей через дефис) складываются в
            # сокращение: «литологически ограниченные ловушки (ЛЛ)», а не «ловушки».
            spelled = [k for k in starts if _spells(words[-k:], letters)]
            if spelled or starts:
                out.setdefault(abbr, " ".join(words[-(spelled or starts)[0] :]))
    return out


def _spells(words: list[str], letters: str) -> bool:
    """Буквы сокращения идут по порядку среди первых букв слов."""
    initials = iter(_stem(part)[:1] for w in words for part in w.split("-") if part)
    return all(letter in initials for letter in letters)


def known_and_unknown(sources: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Сокращения фрагментов: с расшифровкой (словарь проекта, затем сами фрагменты) и без."""
    texts = _texts(sources)
    found = in_fragments(texts)
    expansions = {**defined_in(texts), **load()}
    known = {a: expansions[a] for a in found if a in expansions}
    return known, [a for a in found if a not in expansions]


def note(sources: list[dict[str, Any]]) -> str:
    """Указание в задание: что как расшифровывается и что не расшифровывать; пусто — нечего."""
    known, unknown = known_and_unknown(sources)
    lines = []
    if known:
        listed = "; ".join(f"{a} — {e}" for a, e in list(known.items())[:LISTED])
        lines.append(f"Сокращения во фрагментах: {listed}. Расшифровывай их только так.")
    if unknown:
        lines.append(
            f"Сокращения {', '.join(unknown[:LISTED])} во фрагментах не расшифрованы: пиши их "
            "как есть, свою расшифровку не придумывай."
        )
    return " ".join(lines)


def _in_sources(words: list[str], sources: list[list[str]]) -> bool:
    """Слова стоят во фрагменте подряд и по порядку (по основам)."""
    need = [s for w in words for s in _stems(w)]
    if not need:
        return True
    n = len(need)
    return any(
        text[i : i + n] == need for text in sources for i in range(len(text) - n + 1)
    )


def _same(words: list[str], expansion: str) -> bool:
    """Расшифровка та же (или её часть): по основам, падеж и число не мешают."""
    got, want = set(_stems(" ".join(words))), set(_stems(expansion))
    return bool(got) and (got <= want or want <= got)


def fix(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Ответ без выдуманных расшифровок: (текст, что исправлено). «растительных остатков
    (РОВ)» → «РОВ», «ФЕС (фазовые эффективные свойства)» → «ФЕС (фильтрационно-ёмкостные
    свойства)», «КДП (контур давления)» → «КДП», если во фрагментах такой расшифровки нет."""
    texts = _texts(sources)
    expansions = {**defined_in(texts), **load()}
    stems = [_stems(t) for t in texts]
    fixed: list[str] = []

    def after(m: re.Match[str]) -> str:
        abbr, inside = m.group(1), m.group(2).strip()
        words = inside.split()
        first = abbr[0].lower().replace("ё", "е")
        if not words[0].lower().replace("ё", "е").startswith(first):
            return m.group()  # пояснение в скобках, а не расшифровка
        expansion = expansions.get(abbr)
        if expansion:
            if _same(words, expansion):
                return m.group()
            fixed.append(m.group())
            return f"{abbr} ({expansion})"
        if _in_sources(words, stems):
            return m.group()
        fixed.append(m.group())
        return abbr

    def before(m: re.Match[str]) -> str:
        words, abbr = m.group(1).split(), m.group(2)
        expansion = expansions.get(abbr)
        if expansion and _same(words[-len(expansion.split()) :], expansion):
            return m.group()
        # Сколько слов перед скобкой — попытка расшифровки: справа налево, пока слово
        # начинается на букву сокращения («органического вещества (РОВ)» — два слова).
        letters = set(abbr.lower().replace("ё", "е"))
        k = 0
        while k < len(words) and words[-k - 1][0].lower().replace("ё", "е") in letters:
            k += 1
        if not k:
            return m.group()
        if not expansion and _in_sources(words[-k:], stems):
            return m.group()
        fixed.append(" ".join(words[-k:]) + f" ({abbr})")
        starts = [w.start() for w in re.finditer(_LETTERS, m.group(1))]
        return m.group(1)[: starts[-k]] + abbr

    text = _AFTER.sub(after, answer)
    text = _BEFORE.sub(before, text)
    return text, fixed
