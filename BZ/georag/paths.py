"""Где что лежит.

В репозитории — код, config и docs. Всё, что живёт только на этой машине, — в
одной папке .local (в git не идёт): разобранные статьи, логи, ключи и токены,
файлы, которые пересоздают команды (questions, gaps, synonyms), и проверки кода
(tests — они у разработчика). Только стандартная библиотека: georag.py импортирует
модуль и до перехода в .venv."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
LOCAL = ROOT / ".local"
ARTICLES = LOCAL / "acquired"  # разобранные статьи: текст, фрагменты, разметка Docling
LOGS = LOCAL / "logs"
