"""Общие ключи командной строки.

Адрес базы, устройство для векторов и сервер языковой модели нужны в пяти местах:
загрузка, разметка, чат-бот, веб-сервер, оценка. Раньше каждое объявляло их
само, и значения по умолчанию расходились (где-то был mps, где-то нет). Теперь
ключи объявляются здесь и везде одинаковы.
"""

from __future__ import annotations

import argparse

from .llm import DEFAULT_MODEL, LLM_HOST

DEVICES = ["auto", "cuda", "cpu", "mps"]


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dsn", default=None, help="адрес базы, иначе GEORAG_DSN")


def add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="где считать векторы BGE-M3: auto — на видеокарте, если она есть",
    )


def add_llm_args(parser: argparse.ArgumentParser, model: str = DEFAULT_MODEL) -> None:
    parser.add_argument("--model", default=model, help="языковая модель (имя на сервере vLLM)")
    parser.add_argument("--llm-host", default=LLM_HOST, help="адрес сервера модели (vLLM)")
