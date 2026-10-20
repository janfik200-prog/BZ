"""Подставная база для проверок без Postgres: записывает запросы и отвечает тем, что
скажет проверка.

    conn = FakeConn(rows=[("de", 50)])              на любой запрос — эти строки
    conn = FakeConn(answer=lambda sql, params: …)    ответ по запросу
    conn.log                                         [(запрос одной строкой, параметры)]
    conn.queries("INSERT INTO formulas")             параметры запросов, что так начинаются

Каждый запрос проверяется, как его проверил бы psycopg: именованные подстановки
%(имя)s — все есть в словаре параметров, позиционные %s — по числу параметров.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

Answer = Callable[[str, Any], Iterable[Any]]


def check_placeholders(sql: str, params: Any) -> None:
    names = set(re.findall(r"%\((\w+)\)s", sql))
    positional = len(re.findall(r"%s", sql))
    if names:
        assert isinstance(params, dict), sql
        assert not names - set(params), (names - set(params), sql)
        assert not positional, sql
    elif positional:
        assert params is not None and len(params) == positional, (params, sql)


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.rows: list[Any] = []
        self.rowcount = 0

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        check_placeholders(sql, params)
        flat = " ".join(sql.split())
        self.conn.log.append((flat, params))
        self.rows = list(self.conn.answer(flat, params))
        self.rowcount = len(self.rows)

    def fetchall(self) -> list[Any]:
        return self.rows

    def fetchone(self) -> Any:
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, rows: list[Any] | None = None, answer: Answer | None = None) -> None:
        self.rows: list[Any] = list(rows or [])
        self.answer: Answer = answer or (lambda sql, params: self.rows)
        self.log: list[tuple[str, Any]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def queries(self, prefix: str) -> list[Any]:
        """Параметры записанных запросов, которые начинаются с prefix."""
        return [params for sql, params in self.log if sql.startswith(prefix)]

