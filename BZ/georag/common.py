"""Общие ключи командной строки.

Адрес базы, выбор считалки векторов и модели в Ollama нужны в пяти местах:
загрузка, разметка, чат-бот, веб-сервер, оценка. Раньше каждое объявляло их
само, и значения по умолчанию расходились (где-то был mps, где-то нет). Теперь
ключи объявляются здесь и везде одинаковы.
"""

from __future__ import annotations

import argparse

from .llm import DEFAULT_MODEL, OLLAMA_HOST

DEVICES = ["auto", "cuda", "cpu", "mps"]


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dsn", default=None, help="адрес базы, иначе GEORAG_DSN")


def add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedder", choices=["local", "ollama"], default="local",
                        help="где считать векторы: на видеокарте (local) или в Ollama")
    parser.add_argument("--device", choices=DEVICES, default="auto")


def add_llm_args(parser: argparse.ArgumentParser, model: str = DEFAULT_MODEL) -> None:
    parser.add_argument("--model", default=model, help="языковая модель в Ollama")
    parser.add_argument("--ollama-host", default=OLLAMA_HOST, help="адрес Ollama")
