"""Переключатели и числа из переменных окружения — одними правилами для всего проекта.

Раньше каждый модуль читал свою переменную по-своему: GEORAG_GENERAL понимал «да», а
GEORAG_SCOPE_GATE — нет; GEORAG_SKIP_CAPTIONS включался только «1», и «true» молча
оставлял его выключенным; GEORAG_CONFIRM_WORKERS=abc ронял graph трассировкой.

Включено — 1, true, yes, да, on; выключено — 0, false, no, нет, off (без учёта регистра).
Пусто или что-то другое — значение по умолчанию. Только стандартная библиотека: модуль
берут и запускатель, и лёгкие модули до перехода в .venv.
"""

from __future__ import annotations

import os

ON = frozenset({"1", "true", "yes", "да", "on"})
OFF = frozenset({"0", "false", "no", "нет", "off"})


def flag(name: str, default: bool = False) -> bool:
    """Переключатель: явное «да» или «нет»; иначе default."""
    value = os.environ.get(name, "").strip().lower()
    if value in ON:
        return True
    if value in OFF:
        return False
    return default


def positive_int(name: str, default: int) -> int:
    """Целое больше нуля; пусто, ноль или не число — default."""
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else default
