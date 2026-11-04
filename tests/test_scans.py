"""Страницы распознанного скана картинками (georag/index/scans.py): сохранение рядом со
статьёй, отдача по адресу без выхода из папки статей, ссылки у источников чата."""

from __future__ import annotations

from pathlib import Path

import pytest

from georag.index import scans

pymupdf = pytest.importorskip("pymupdf")


def _pdf(pages: int) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    data: bytes = doc.tobytes()
    return data


def test_pages_saved_and_found(tmp_path: Path) -> None:
    assert scans.save(_pdf(3), "local-a-1", tmp_path) == 3
    assert scans.pages("local-a-1", tmp_path) == [1, 2, 3]
    path = scans.page_file("local-a-1", 2, tmp_path)
    assert path is not None and path.read_bytes()[:2] == b"\xff\xd8"  # JPEG
    assert scans.page_file("local-a-1", 4, tmp_path) is None
    # заново — прежние страницы стёрты
    assert scans.save(_pdf(1), "local-a-1", tmp_path) == 1
    assert scans.pages("local-a-1", tmp_path) == [1]


@pytest.mark.parametrize("doc_id", ["", "../x", "a/b", "a" + chr(92) + "b", "датасет:золото", ".."])
def test_doc_id_cannot_leave_articles(tmp_path: Path, doc_id: str) -> None:
    assert scans.folder(doc_id, tmp_path) is None
    assert scans.pages(doc_id, tmp_path) == []
    assert scans.page_file(doc_id, 1, tmp_path) is None


def test_broken_pdf_saves_nothing(tmp_path: Path) -> None:
    assert scans.save(b"%PDF-1.4 oops", "d", tmp_path) == 0
    assert scans.pages("d", tmp_path) == []


def test_ocr_record_gets_page_images(tmp_path: Path) -> None:
    """Добыча и свои статьи: у распознанного скана страницы ложатся рядом с записью."""
    import json
    from types import SimpleNamespace

    from georag.acquire.models import ACQUIRED, AcquiredRecord
    from georag.acquire.pipeline import _write_doc

    parsed = SimpleNamespace(docling_doc=None, text="текст")
    ocr = AcquiredRecord(doc_id="s1", status=ACQUIRED, parser_version="docling@2+ocr:rapidocr@3")
    _write_doc(tmp_path, ocr, parsed, [], _pdf(2))
    assert json.loads((tmp_path / "s1.json").read_text("utf-8"))["scan_pages"] == 2
    assert scans.pages("s1", tmp_path) == [1, 2]
    plain = AcquiredRecord(doc_id="p1", status=ACQUIRED, parser_version="docling@2")
    _write_doc(tmp_path, plain, parsed, [], _pdf(2))
    assert "scan_pages" not in json.loads((tmp_path / "p1.json").read_text("utf-8"))
    assert scans.pages("p1", tmp_path) == []


def test_chat_sources_marked_with_scan_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from georag.web import server

    scans.save(_pdf(3), "s1", tmp_path)
    original = scans.pages
    monkeypatch.setattr(server.scans, "pages", lambda doc_id: original(doc_id, tmp_path))
    out = server.with_scans(
        [
            {"n": 1, "doc_id": "s1", "pages": [2, 5]},
            {"n": 2, "doc_id": "other", "pages": [1]},
            {"n": 3, "doc_id": "датасет:золото", "kind": "датасет"},
        ]
    )
    assert out[0]["scan_pages"] == [2]
    assert "scan_pages" not in out[1] and "scan_pages" not in out[2]
