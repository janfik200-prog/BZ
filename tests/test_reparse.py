"""Повторная загрузка статьи не теряет граф: фрагменты сопоставляются с прежними по
тексту, а не по номеру (georag/index/db.py: match_chunks, replace_chunks).

python -m unittest discover -s tests — без базы и модели."""

import unittest

from fakedb import FakeConn

from georag.index import db


def row(ord_, text):
    return {
        "doc_id": "d1",
        "ord": ord_,
        "text": text,
        "headings": [],
        "pages": [],
        "n_tokens": 1,
        "has_table": False,
        "embedding": "[0]",
    }


OLD = [(11, 0, "Введение."), (12, 1, "Методы."), (13, 2, "Выводы.")]  # (id, ord, текст)


def facts_conn(old):
    """Подставная база: прежние фрагменты статьи (id, ord, текст), таблица фактов есть."""

    def answer(sql, params):
        if sql.startswith("SELECT ord, id, text FROM chunks"):
            return [(o, i, t) for i, o, t in old]
        if "to_regclass('facts')" in sql:
            return [(True,)]
        return []

    return FakeConn(answer=answer)


class MatchChunksTests(unittest.TestCase):
    def test_new_paragraph_at_start_keeps_every_chunk(self):
        """Абзац в начале: номера всех фрагментов сдвинулись, но тексты те же."""
        new = [row(0, "Новый абзац."), row(1, "Введение."), row(2, "Методы."), row(3, "Выводы.")]
        self.assertEqual(
            db.match_chunks(OLD, new), [(None, False), (11, True), (12, True), (13, True)]
        )

    def test_edited_text_keeps_place_but_not_facts(self):
        new = [row(0, "Введение."), row(1, "Методы, исправлено."), row(2, "Выводы.")]
        self.assertEqual(db.match_chunks(OLD, new), [(11, True), (12, False), (13, True)])

    def test_spaces_and_formula_marks_are_not_changes(self):
        old = [(21, 0, "Плотность: rho = 1")]
        self.assertEqual(db.match_chunks(old, [row(0, "Плотность:  $$rho = 1$$")]), [(21, True)])

    def test_duplicate_texts_go_to_nearest(self):
        old = [(1, 0, "Таблица 1."), (2, 5, "Таблица 1.")]
        new = [row(4, "Таблица 1."), row(9, "Таблица 1."), row(10, "Таблица 1.")]
        self.assertEqual(db.match_chunks(old, new), [(2, True), (1, True), (None, False)])

    def test_reorder_and_removal(self):
        new = [row(0, "Выводы."), row(1, "Введение.")]  # «Методы» убраны, порядок другой
        self.assertEqual(db.match_chunks(OLD, new), [(13, True), (11, True)])


class ReplaceChunksTests(unittest.TestCase):
    def test_after_shift_graph_stays(self):
        conn = facts_conn(OLD)
        new = [row(0, "Новый абзац."), row(1, "Введение."), row(2, "Методы."), row(3, "Выводы.")]
        self.assertEqual(db.replace_chunks(conn, "d1", new), 4)
        self.assertEqual(conn.queries("UPDATE chunks SET ord = -1 - ord"), [("d1",)])
        moved = [(p["chunk_id"], p["ord"]) for p in conn.queries("UPDATE chunks SET ord = %(ord)s")]
        self.assertEqual(moved, [(11, 1), (12, 2), (13, 3)])
        self.assertEqual([p["text"] for p in conn.queries("INSERT INTO chunks")], ["Новый абзац."])
        self.assertEqual(conn.queries("DELETE FROM facts"), [])  # граф статьи цел
        self.assertEqual(conn.queries("DELETE FROM chunks"), [])

    def test_drops_only_changed_and_gone(self):
        conn = facts_conn(OLD)
        db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, "Методы, исправлено.")])
        self.assertEqual(conn.queries("DELETE FROM chunks WHERE id = ANY"), [([13],)])
        self.assertEqual(conn.queries("DELETE FROM facts WHERE chunk_id"), [([12],)])

    def test_first_load_inserts_without_renumbering(self):
        conn = facts_conn([])
        db.replace_chunks(conn, "d1", [row(0, "Введение.")])
        self.assertEqual(conn.queries("UPDATE chunks"), [])
        self.assertEqual(len(conn.queries("INSERT INTO chunks")), 1)


if __name__ == "__main__":
    unittest.main()
