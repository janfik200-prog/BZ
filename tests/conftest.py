"""Для pytest: корень проекта и папка проверок — в пути импорта, как у
python -m unittest discover -s tests (tests/README.md)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
