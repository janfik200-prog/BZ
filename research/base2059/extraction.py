"""Опыт v18 (двухэтапное А против прямого Б выделения связей) на этой базе.

research/v18/extraction_compare.py вызывается как есть; подменены только адрес базы (эта
база, только чтение — в пилоте georag_v14) и папки результатов (research/base2059/v18 —
пилотные файлы v18 не трогаются). Выборка, зерно, модель (Qwen3-14B, GEORAG_LLM),
правила и проверки — те же.

    python research/base2059/extraction.py run --n 100
    python research/base2059/extraction.py sheet      # лист слепой разметки
    python research/base2059/extraction.py score      # итог по размеченному листу
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "v18"))
sys.path.insert(0, str(ROOT))

import extraction_compare as X  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
X.DSN = DSN + "?options=-c%20default_transaction_read_only%3Don"
X.HERE = HERE / "v18"
X.OUT = X.HERE / "results"
X.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    X.main()
