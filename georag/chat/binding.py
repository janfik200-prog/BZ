"""Привязка к объекту вопроса: утверждение о названном в вопросе объекте — только по
фрагментам, где этот объект назван.

Сравнение replay 05.11.2026 (93 вопроса, те же фрагменты, температура 0): короткий промт
приписывал Линденской впадине баженовские отложения и свойства из материалов о других
участках (oil59), а Тимано-Печорской провинции — классификацию комплексов из словаря (oil36).
Правило «ссылка подтверждает утверждение для указанного объекта» в промте было — модель его
нарушила. Здесь это проверяет код: объект вопроса — имя собственное («Линденская впадина»,
«Тимано-Печорская провинция»); предложение, которое о нём говорит, остаётся, только если хотя
бы один процитированный в нём фрагмент называет этот объект (в тексте, заголовке или
названии статьи, в любом падеже). Отрицание («во фрагментах не указано, что…», «посылка не
подтверждается») остаётся: это не утверждение об объекте.
"""

from __future__ import annotations

import re
from typing import Any

from .checks import cited_numbers, rewrite

_WORD = re.compile(r"[A-Za-zА-ЯЁа-яё][\w-]*")
# Имя собственное — слово с заглавной не в начале вопроса и не после точки.
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZА-ЯЁ][a-zа-яё][\w-]*(?:\s+[A-ZА-ЯЁ][a-zа-яё][\w-]*)*")
_NEGATION = re.compile(
    r"не\s+(?:подтвержд|указ|упомина|содерж|описан|сказано|привод|назван|следует|связыва)|"
    r"нет\s+(?:сведений|данных|информации|ответа)|отсутству|неизвестно|не\s+удалось",
    re.IGNORECASE,
)


def _stem(word: str) -> str:
    """Начало слова: падеж не мешает («Линденской» — «линденск», «Сибири» — «сиби»)."""
    w = word.lower().replace("ё", "е")
    return w if len(w) <= 4 else w[: max(4, len(w) - 3)]


def question_objects(question: str) -> list[list[str]]:
    """Объекты вопроса — имена собственные, каждое списком основ слов."""
    out = []
    for m in _NAME.finditer(question.strip()):
        stems = [_stem(w) for w in _WORD.findall(m.group()) if len(w) >= 3]
        if stems and stems not in out:
            out.append(stems)
    return out


def _names(stems: list[str], text: str) -> bool:
    words = [w.lower().replace("ё", "е") for w in _WORD.findall(text or "")]
    return all(any(w.startswith(s) for w in words) for s in stems)


def _source_text(source: dict[str, Any]) -> str:
    return " ".join(
        [str(source.get("title") or ""), *map(str, source.get("headings") or []),
         str(source.get("text") or "")]
    )


def bind_objects(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Ответ без предложений об объекте вопроса, которые не подтверждены фрагментами,
    называющими этот объект: (текст, убранные)."""
    objects = question_objects(question)
    if not objects:
        return answer, []
    by_n = {s["n"]: s for s in sources}
    removed: list[str] = []

    def change(part: str) -> str | None:
        cited = [by_n[n] for n in cited_numbers(part) if n in by_n]
        if not cited or _NEGATION.search(part):
            return part
        for stems in objects:
            if _names(stems, part) and not any(_names(stems, _source_text(s)) for s in cited):
                removed.append(part.strip())
                return None
        return part

    return rewrite(answer, change), removed
