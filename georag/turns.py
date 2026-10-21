"""Очередь ответов чат-бота — одна на процесс: страница и Телеграм-бот (web --tg).

Ответ — это несколько запросов к модели подряд (разбор вопроса, отбор, сверка, сам
ответ) и соединение с базой на всё время, пока он пишется. Сверх MAX_PARALLEL ответов
сразу вопрос ждёт здесь, по порядку прихода: человек видит своё место, а соединение с
базой берёт, только когда очередь подошла, — ждущие не занимают пул.

Раньше очередь была только у Телеграм-бота, а страница пускала к модели всех сразу.
Сколько заданий модель вмещает по токенам, следит ещё и llm.py (Budget) — для любых
вызовов модели, не только ответов.

    with turns.turn(on_wait=lambda place: ...):
        ... ответ ...
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

# Ответов сразу: GEORAG_PARALLEL, по умолчанию 4 (docs/КОМАНДЫ.md, «Несколько
# пользователей»). Память модели сверх этого бережёт бюджет токенов в llm.py.
# Не число (пустое «GEORAG_PARALLEL=» в .env) — по умолчанию, а не падение страницы и бота
# при импорте; так же читаются GEORAG_DB_POOL (index/db.py) и GEORAG_KV_TOKENS (llm.py).
_PARALLEL_RAW = os.environ.get("GEORAG_PARALLEL", "").strip()
MAX_PARALLEL = max(1, int(_PARALLEL_RAW) if _PARALLEL_RAW.lstrip("-").isdigit() else 4)
POLL = 2.0  # с: как часто ждущему напоминать его место (и проверять, не ушёл ли он)


class Turns:
    """Места у модели: не больше size ответов сразу, остальные ждут по порядку прихода.

    acquire/release — как у семафора; turn — то же блоком with."""

    def __init__(self, size: int = MAX_PARALLEL) -> None:
        self.size = max(1, size)
        self._cond = threading.Condition()
        self._busy = 0
        self._queue: deque[object] = deque()

    @property
    def waiting(self) -> int:
        """Сколько ответов ждут очереди."""
        with self._cond:
            return len(self._queue)

    def acquire(
        self,
        blocking: bool = True,
        timeout: float | None = None,
        on_wait: Callable[[int], None] | None = None,
        poll: float | None = None,
    ) -> bool:
        """Занять место. Мест нет — встать в конец очереди и ждать, пока подойдёт черёд.

        on_wait(место) зовётся сразу и потом раз в poll секунд, пока ждёт; место — номер
        в очереди, считая самого ждущего (первый в очереди — 1). Исключение из on_wait
        (человек ушёл со страницы) снимает с очереди. Не дождался за timeout — False."""
        import time

        poll = POLL if poll is None else poll
        with self._cond:
            if self._busy < self.size and not self._queue:
                self._busy += 1
                return True
            if not blocking:
                return False
            me = object()
            self._queue.append(me)
        deadline = None if timeout is None else time.monotonic() + timeout
        got = False
        try:
            while True:
                with self._cond:
                    if self._queue[0] is me and self._busy < self.size:
                        self._queue.popleft()
                        self._busy += 1
                        got = True
                        self._cond.notify_all()
                        return True
                    place = self._queue.index(me) + 1
                if on_wait:
                    on_wait(place)  # вне блокировки: тут пишут в сеть
                with self._cond:
                    if self._queue[0] is me and self._busy < self.size:
                        continue
                    wait = poll
                    if deadline is not None:
                        wait = min(wait, deadline - time.monotonic())
                        if wait <= 0:
                            return False
                    self._cond.wait(wait)
        finally:
            if not got:
                with self._cond:
                    if me in self._queue:
                        self._queue.remove(me)
                        self._cond.notify_all()

    def release(self) -> None:
        with self._cond:
            if self._busy <= 0:
                raise ValueError("место в очереди отдают больше раз, чем брали")
            self._busy -= 1
            self._cond.notify_all()

    @contextmanager
    def turn(
        self, on_wait: Callable[[int], None] | None = None, poll: float | None = None
    ) -> Iterator[None]:
        """Место у модели на время блока (acquire, в конце release)."""
        self.acquire(on_wait=on_wait, poll=poll)
        try:
            yield
        finally:
            self.release()


SHARED = Turns()


def turn(
    on_wait: Callable[[int], None] | None = None, poll: float | None = None
) -> AbstractContextManager[None]:
    """Место в общей очереди процесса (SHARED)."""
    return SHARED.turn(on_wait, poll)
