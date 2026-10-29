"""T13. Происхождение цитат при переразборе: миграция, версия парсера, пометка «устарел».

Без Postgres — на заглушке (tests/fakedb.py): база с колонками T13 и без них.
Живой прогон на копии базы — research/v14/results/stale_audit_t13.json.
"""

from __future__ import annotations

import re
from typing import Any

from fakedb import FakeConn

from georag.evidence.provenance import ChunkSnapshot, DocumentSnapshot, snapshot_from_db
from georag.graph import store
from georag.index import db, ingest
from georag.parse.models import DOCLING, PROBE, PYMUPDF, parser_version

QUOTE = "Золоторудная минерализация приурочена к зонам рассланцевания."
OLD = [(11, 0, "Введение."), (12, 1, f"Методы. {QUOTE} Конец абзаца.")]  # (id, ord, текст)
FACTS = [(101, 12, QUOTE)]  # (id, chunk_id, цитата)


def row(ord_: int, text: str) -> dict[str, Any]:
    return {"doc_id": "d1", "ord": ord_, "text": text, "headings": [], "pages": [1],
            "n_tokens": 1, "has_table": False, "embedding": "[0]"}


def base(old=OLD, facts=FACTS, stale=(), columns=True) -> FakeConn:
    """Подставная база: фрагменты статьи, факты; columns — колонки T13 есть."""
    marked = set(stale)

    def answer(sql: str, params: Any) -> list[Any]:
        if sql.startswith("SELECT ord, id, text FROM chunks"):
            return [(o, i, t) for i, o, t in old]
        if "to_regclass('facts')" in sql:
            return [(True,)]
        if "information_schema.columns" in sql:
            return [(c,) for c in params[1]] if columns else []
        if sql.startswith("SELECT id, chunk_id, quote, stale_since IS NOT NULL FROM facts"):
            return [(f, c, q, f in marked) for f, c, q in facts if c in params[0]]
        return []

    return FakeConn(answer=answer)


# --------------------------------------------------------------------------- #
#  Миграция
# --------------------------------------------------------------------------- #
def test_migration_is_idempotent_and_has_rollback() -> None:
    up = db.PROVENANCE_SQL.read_text("utf-8")
    down = db.PROVENANCE_ROLLBACK_SQL.read_text("utf-8")
    statements = [s for s in re.split(r";\s*", re.sub(r"--[^\n]*", "", up)) if s.strip()]
    assert statements and all("ADD COLUMN IF NOT EXISTS" in s for s in statements)
    for col in (*db.PARSE_COLUMNS, *db.STALE_COLUMNS):
        assert f"ADD COLUMN IF NOT EXISTS {col}" in up
        assert f"DROP COLUMN IF EXISTS {col}" in down
    assert "ALTER TABLE IF EXISTS facts" in up  # база без графа — миграция не падает
    conn = FakeConn()
    db.migrate_provenance(conn)
    db.migrate_provenance(conn)  # второй раз — те же запросы, без ошибки
    assert conn.log[0] == conn.log[1]


def test_init_db_applies_migration() -> None:
    conn = FakeConn()
    db.init_db(conn, with_hnsw=False)
    assert any("ADD COLUMN IF NOT EXISTS stale_since" in sql for sql, _ in conn.log)


def test_has_columns_needs_every_name() -> None:
    assert db.has_columns(base(), "facts", db.STALE_COLUMNS)
    assert not db.has_columns(base(columns=False), "facts", db.STALE_COLUMNS)
    # заглушка, что на любой запрос отвечает (True,), колонок не «заводит»
    assert not db.has_columns(FakeConn(rows=[(True,)]), "facts", db.STALE_COLUMNS)


# --------------------------------------------------------------------------- #
#  Переразбор: изменённый текст помечает факт, неизменный — нет
# --------------------------------------------------------------------------- #
def stale_updates(conn: FakeConn) -> list[Any]:
    return conn.queries("UPDATE facts SET stale_since = now()")


def test_reparse_with_changed_text_marks_fact() -> None:
    conn = base()
    got: dict[str, list[int]] = {}
    new = [row(0, "Введение."), row(1, "Методы. Минерализация описана иначе. Конец абзаца.")]
    db.replace_chunks(conn, "d1", new, parser_version="docling@2.140.0", stale=got)
    assert got == {"marked": [101], "cleared": []}
    [(reason, ids)] = stale_updates(conn)
    assert ids == [101] and reason == "QUOTE_NOT_IN_PARSED_TEXT: переразбор docling@2.140.0"
    assert conn.queries("DELETE FROM facts WHERE") == []  # факт не стёрт, а помечен
    assert conn.queries("DELETE FROM facts_pass") == [([12],)]  # модель пройдёт фрагмент заново


def test_reparse_with_same_text_marks_nothing() -> None:
    conn = base()
    got: dict[str, list[int]] = {}
    db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, OLD[1][2])], stale=got)
    assert got == {"marked": [], "cleared": []}
    assert stale_updates(conn) == [] and conn.queries("DELETE FROM facts") == []


def test_reparse_with_new_layout_but_quote_kept_marks_nothing() -> None:
    """Текст фрагмента другой (переносы, соседний абзац), но цитата в нём есть — факт прямой."""
    conn = base()
    got: dict[str, list[int]] = {}
    new = [row(0, "Введение."), row(1, f"Методы (уточнено).\n{QUOTE.replace(' к ', '  к\n')} Конец.")]
    db.replace_chunks(conn, "d1", new, stale=got)
    assert got["marked"] == [] and stale_updates(conn) == []


def test_quote_back_clears_mark() -> None:
    conn = base(stale=[101])
    got: dict[str, list[int]] = {}
    db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, OLD[1][2])], stale=got)
    assert got == {"marked": [], "cleared": [101]}
    assert conn.queries("UPDATE facts SET stale_since = NULL") == [([101],)]


def test_already_stale_keeps_its_time() -> None:
    conn = base(stale=[101])
    got: dict[str, list[int]] = {}
    db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, "Другой текст.")], stale=got)
    assert got == {"marked": [], "cleared": []} and stale_updates(conn) == []


def test_old_base_without_columns_deletes_as_before() -> None:
    conn = base(columns=False)
    db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, "Другой текст.")])
    assert conn.queries("DELETE FROM facts WHERE chunk_id") == [([12],)]
    assert stale_updates(conn) == []


# --------------------------------------------------------------------------- #
#  Версия парсера и отпечаток разбора
# --------------------------------------------------------------------------- #
def test_parser_version_from_package() -> None:
    assert re.fullmatch(r"docling@\d+\.\d+.*", parser_version(DOCLING))
    assert re.fullmatch(r"pymupdf4llm@\d+\.\d+.*", parser_version(PYMUPDF))
    assert parser_version(PROBE) == f"{PROBE}@unrecorded"


def test_ingest_takes_recorded_version_or_says_unrecorded() -> None:
    assert ingest.parser_version({"parser": "docling", "parser_version": "docling@2.140.0"}) == "docling@2.140.0"
    assert ingest.parser_version({"parser": "docling"}) == "docling@unrecorded"


def test_parse_print_equals_snapshot_print() -> None:
    rows = [row(1, "Второй  фрагмент."), row(0, "Первый\nфрагмент.")]
    snap = DocumentSnapshot("d1", "", "x", chunks={
        str(i): ChunkSnapshot(str(i), r["text"], ord=r["ord"]) for i, r in enumerate(rows)})
    conn = base()
    assert db.record_parse(conn, "d1", rows, "docling@2.140.0") == snap.parsed_sha256
    [(version, digest, doc_id)] = conn.queries("UPDATE documents SET parser_version")
    assert (version, digest, doc_id) == ("docling@2.140.0", snap.parsed_sha256, "d1")
    assert conn.queries("UPDATE chunks SET parser_version") == [("docling@2.140.0", "d1")]
    assert db.record_parse(base(columns=False), "d1", rows, "v") is None  # база до T13
    conn = base()
    db.record_parse(conn, "d1", rows)  # чистка текста: версия прежняя, фрагменты не трогаются
    assert conn.queries("UPDATE chunks SET parser_version") == []


def test_snapshot_takes_parser_version_column() -> None:
    def answer(version: str | None):
        def a(sql: str, params: Any) -> list[Any]:
            if sql.startswith("SELECT doc_id, sha256, parser"):
                assert "to_jsonb(d) ->> 'parser_version'" in sql  # и в базе без колонки
                return [("d1", "a" * 64, "docling", "T", None, "ru", version)]
            return []
        return a

    assert snapshot_from_db(FakeConn(answer=answer("docling@2.140.0")), "d1").parser_version == "docling@2.140.0"
    assert snapshot_from_db(FakeConn(answer=answer(None)), "d1").parser_version == "docling@unrecorded"


# --------------------------------------------------------------------------- #
#  Граф: устаревший факт — не в графе, а среди кандидатов
# --------------------------------------------------------------------------- #
def test_accepted_fresh_excludes_stale() -> None:
    assert "stale_since" not in store.accepted()  # база до T13 — условие прежнее
    sql = store.accepted(fresh=True)
    assert sql.startswith("(f.stale_since IS NULL AND ")
    assert "o.stale_since IS NULL" in sql  # устаревший не подтверждает чужой факт повтором


def test_all_facts_uses_stale_column_only_if_present() -> None:
    conn = base()
    store.all_facts(conn)
    assert "f.stale_since IS NULL" in conn.log[-1][0]
    conn = base()
    store.all_facts(conn, candidates=True)
    assert "WHERE NOT (f.stale_since IS NULL AND" in conn.log[-1][0]
    conn = base(columns=False)
    store.all_facts(conn)
    assert "stale_since" not in conn.log[-1][0]
