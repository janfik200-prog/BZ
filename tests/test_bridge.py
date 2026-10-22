"""Мост для базы данных проекта: у цитат графа — страницы и DOI, /api/health, поиск в
границах одной статьи (/api/search?doc_id=…). Без базы и модели."""

from __future__ import annotations

from typing import Any

from fakedb import FakeConn

from georag.graph import api, store
from georag.web import server

ROW = (7, 501, "d", "Кекура", "вмещает", "золото", "Кекура вмещает золото", "Title", 2024,
       "", "10.1/x", 4, None, None, None, None, None, [5, 6])  # fmt: skip


def test_fact_quote_has_pages_and_doi() -> None:
    conn = FakeConn(rows=[ROW])
    facts = store.all_facts(conn)
    sql = conn.log[-1][0]
    assert sql.index("c.pages") > sql.index("f.votes")  # последней колонкой
    assert "JOIN chunks c ON c.id = f.chunk_id" in sql
    assert facts[0]["pages"] == [5, 6] and facts[0]["votes"] == 4
    quote = api._quote(facts[0])
    assert quote["pages"] == [5, 6] and quote["doi"] == "10.1/x"
    assert quote["url"] == "https://doi.org/10.1/x"


def test_version_is_short_commit_or_empty() -> None:
    version = server._version()
    assert version == "" or (4 <= len(version) <= 12 and version.isalnum())


def test_search_passes_doc_id(monkeypatch: Any) -> None:
    asked: list[Any] = []

    def hybrid(conn: Any, emb: Any, text: str, **kw: Any) -> list[Any]:
        asked.append(kw.get("doc_id"))
        return []

    class Conn:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(server, "hybrid_search", hybrid)
    monkeypatch.setattr(server.db, "connect", lambda dsn: Conn())
    monkeypatch.setattr(server.State, "synonyms", staticmethod(lambda: {}))
    monkeypatch.setattr(server, "synonym_variants", lambda text, syn: [])
    handler = server.Handler.__new__(server.Handler)
    handler._search({"q": ["золото"], "doc_id": ["d1"]})
    handler._search({"q": ["золото"]})
    assert asked == ["d1", None]
