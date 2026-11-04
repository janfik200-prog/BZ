"""Где что лежит.

В репозитории — код, config и docs. Всё, что живёт только на этой машине, — в
одной папке .local (в git не идёт): разобранные статьи, логи, ключи и токены,
файлы, которые пересоздают команды (questions, gaps, synonyms), и проверки кода
(tests — они у разработчика). Только стандартная библиотека: georag.py импортирует
модуль и до перехода в .venv."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
LOCAL = ROOT / ".local"
ARTICLES = LOCAL / "acquired"  # разобранные статьи: текст, фрагменты, разметка Docling
LOGS = LOCAL / "logs"


def files_in(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Файлы папки с этими расширениями (без учёта регистра), по имени. Через os.scandir:
    на Windows вид записи приходит вместе со списком, без обращения к диску на каждый
    файл. С Path.iterdir() и is_file() список input на внешнем USB-диске с 39 803 PDF
    читался больше 13 минут, os.scandir — 0,2 с (закрытая БЗ, 29.10.2026)."""
    if not folder.is_dir():
        return []
    with os.scandir(folder) as entries:
        return sorted(
            Path(e.path) for e in entries if e.name.lower().endswith(suffixes) and e.is_file()
        )
