"""Очередь ответов (georag/turns.py) и бюджет токенов модели (georag/llm.py, Budget).

python -m unittest discover -s tests — без базы и модели."""

import threading
import time
import unittest

from georag.llm import Budget
from georag.turns import Turns


def _start(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


class TurnsTests(unittest.TestCase):
    def test_free_place_is_taken_at_once(self):
        turns = Turns(2)
        self.assertTrue(turns.acquire(blocking=False))
        self.assertTrue(turns.acquire(blocking=False))
        self.assertFalse(turns.acquire(blocking=False))
        turns.release()
        self.assertTrue(turns.acquire(blocking=False))

    def test_waiters_get_their_turn_in_arrival_order(self):
        turns = Turns(1)
        turns.acquire()
        order, threads = [], []
        lock = threading.Lock()

        def wait(name):
            turns.acquire(poll=0.05)
            with lock:
                order.append(name)
            time.sleep(0.02)
            turns.release()

        for name in ("первый", "второй", "третий"):
            threads.append(_start(wait, name))
            time.sleep(0.05)  # приходят по очереди
        self.assertEqual(turns.waiting, 3)
        turns.release()
        for thread in threads:
            thread.join(2)
        self.assertEqual(order, ["первый", "второй", "третий"])

    def test_place_moves_up_as_queue_goes(self):
        turns = Turns(1)
        turns.acquire()
        seen = []
        first = _start(lambda: (turns.acquire(poll=0.05), time.sleep(0.1), turns.release()))
        time.sleep(0.05)
        second = _start(lambda: (turns.acquire(on_wait=seen.append, poll=0.05), turns.release()))
        time.sleep(0.15)
        turns.release()  # первый занимает место — второй теперь первый в очереди
        first.join(2)
        second.join(2)
        self.assertEqual(seen[0], 2)
        self.assertIn(1, seen)

    def test_leaving_the_queue_frees_it(self):
        turns = Turns(1)
        turns.acquire()

        def gone(place):
            raise BrokenPipeError("страницу закрыли")

        with self.assertRaises(BrokenPipeError):
            turns.acquire(on_wait=gone)
        self.assertEqual(turns.waiting, 0)
        turns.release()
        self.assertTrue(turns.acquire(blocking=False))  # место не утекло

    def test_timeout(self):
        turns = Turns(1)
        turns.acquire()
        self.assertFalse(turns.acquire(timeout=0.1, poll=0.05))
        self.assertEqual(turns.waiting, 0)

    def test_release_without_acquire_is_an_error(self):
        with self.assertRaises(ValueError):
            Turns(1).release()


class BudgetTests(unittest.TestCase):
    def test_task_waits_until_it_fits(self):
        budget = Budget(100)
        entered = threading.Event()

        def second():
            with budget.hold(60):
                entered.set()

        with budget.hold(60):
            thread = _start(second)
            self.assertFalse(entered.wait(0.1))  # 60 + 60 > 100 — ждёт
        self.assertTrue(entered.wait(1))
        thread.join(1)
        self.assertEqual(budget.used, 0)

    def test_task_bigger_than_budget_runs_alone(self):
        budget = Budget(100)
        with budget.hold(500):
            self.assertEqual(budget.used, 100)


if __name__ == "__main__":
    unittest.main()
