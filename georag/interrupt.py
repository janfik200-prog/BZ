"""Запуск модуля из запускателя: Ctrl+C — одна строка вместо трассировки.

    python -m georag.interrupt georag.acquire.local --limit 5

Прерванная команда останавливается как раньше (что разобрано и записано — остаётся),
но вместо двух экранов KeyboardInterrupt печатается «Прервано», код выхода — 130.
Свою обработку Ctrl+C у команд (graph, collect: «продолжится с этого места») это не
трогает: сюда доходит только то, что команда не поймала сама.
"""

from __future__ import annotations

import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Нужно имя модуля: python -m georag.interrupt georag.index.cli stats", file=sys.stderr
        )
        return 2
    module = sys.argv.pop(1)
    try:
        # Как python -m module: тот же __main__, те же аргументы (sys.argv[0] — файл модуля).
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except KeyboardInterrupt:
        print("\nПрервано (Ctrl+C).", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
