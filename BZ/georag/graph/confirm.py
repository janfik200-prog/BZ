"""Подтверждение фактов: модель отвечает на вопросы по цитате, код сверяет ответ с фактом.

Спросить модель «верен ли факт?» не работает: на 100 размеченных вручную фактах
Qwen3-14B поймала 5 ошибок из 24 и отбросила бы 10 верных. Отвечать на вопрос по
предложению модель умеет лучше, а сравнивает ответ с фактом уже код. По каждому
факту четыре проверки — два вопроса и два пропуска, вперёд и назад:

    «Согласно предложению, „X“ — приурочено к — что именно?»   → должно быть Y
    «Согласно предложению, что именно — приурочено к — „Y“?»   → должно быть X
    «X приурочено к ___»                                        → Y
    «___ приурочено к Y»                                        → X

Число совпавших ответов (0–4) пишется в facts.votes. На независимой выборке из
100 фактов (logs/quality-20261005): без проверки верных 73 %; среди фактов с ≥ 3
подтверждениями — 89 %, с 4 — 95 %. Граф берёт факты с ≥ MIN_VOTES; остальные
остаются в базе (порог можно поменять, ничего не прогоняя заново).

Перепутанное направление ловится само: на вопрос «что приурочено к X?» модель
отвечает тем, что в предложении действительно приурочено к X.
"""

from __future__ import annotations

import re
import time
from typing import Any

from . import store

MIN_VOTES = store.MIN_VOTES

SYSTEM = "Ты внимательно читаешь предложения из геологических статей. Отвечай только JSON."
_RULE = (
    "Отвечай только по этому предложению, словами из него. Если предложение этого прямо "
    "не утверждает (только перечисляет рядом, говорит о другом, говорит обратное) — "
    "ответь «не сказано»."
)
QUESTIONS = (
    "Предложение: «{quote}»\n\n" + _RULE + "\n\n"
    "1. Согласно предложению, «{src}» — {rel} — что именно?\n"
    "2. Согласно предложению, что именно — {rel} — «{dst}»?\n\n"
    'Верни JSON: {{"1": "...", "2": "..."}}'
)
BLANKS = (
    "Предложение: «{quote}»\n\n"
    "Заполни пропуск «___» так, как это утверждает предложение. " + _RULE + "\n\n"
    "1. {src} {rel} ___\n"
    "2. ___ {rel} {dst}\n\n"
    'Верни JSON: {{"1": "...", "2": "..."}}'
)
NOT_SAID = "не сказано"


def _stems(text: str) -> set[str]:
    return {w.lower()[:5] for w in re.findall(r"[^\W\d_]{3,}", text)}


def matches(answer: Any, name: str) -> bool:
    """Ответ называет то же, что имя в факте: половина основ имени — или ответ уже, но из него."""
    text = str(answer or "")
    found, want = _stems(text), _stems(name)
    if not found or not want or NOT_SAID in text.lower():
        return False
    common = found & want
    return len(common) / len(want) >= 0.5 or bool(common and found <= want)


def votes(llm: Any, src: str, relation: str, dst: str, quote: str) -> int:
    """Сколько из четырёх проверок факт прошёл (0–4)."""
    total = 0
    for template in (QUESTIONS, BLANKS):
        prompt = template.format(quote=quote, src=src, rel=relation, dst=dst)
        answer = llm.chat_json(SYSTEM, prompt, temperature=0)
        if not isinstance(answer, dict):
            continue
        total += matches(answer.get("1"), dst) + matches(answer.get("2"), src)
    return total


def run(
    conn: Any, llm: Any, limit: int | None = None, log: Any = print, stop_after_errors: int = 3
) -> dict[str, int]:
    """Проверить факты, которых ещё не проверяли. Возобновляемо: прервали — продолжит."""
    rows = store.unconfirmed(conn, limit=limit)
    stats = {"todo": len(rows), "done": 0, "kept": 0, "errors": 0}
    errors_in_row = 0
    started = time.monotonic()
    for i, (fact_id, src, relation, dst, quote) in enumerate(rows, start=1):
        try:
            got = votes(llm, src, relation, dst, quote)
            errors_in_row = 0
        except Exception as exc:  # noqa: BLE001 — модель молчит: этот факт потом
            stats["errors"] += 1
            errors_in_row += 1
            if log:
                log(f"  модель не ответила: {type(exc).__name__}: {exc}")
            if errors_in_row >= stop_after_errors:
                if log:
                    log(
                        "  Модель молчит несколько раз подряд — останавливаюсь; продолжится отсюда."
                    )
                break
            continue
        store.set_votes(conn, fact_id, got)
        stats["done"] += 1
        stats["kept"] += got >= MIN_VOTES
        if i % 50 == 0 or i == len(rows):
            conn.commit()
            if log:
                pace = (time.monotonic() - started) / i
                log(
                    f"  [{i}/{len(rows)}] подтверждено ≥{MIN_VOTES} из 4: {stats['kept']} "
                    f"из {stats['done']} · {pace:.1f} с на факт"
                )
    conn.commit()
    return stats
