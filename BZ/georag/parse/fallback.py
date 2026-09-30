"""Запасные парсеры: PyMuPDF4LLM и ручной manual.txt.

Docling — основной путь. Эти два включаются, только когда основной не справился:
PyMuPDF4LLM заметно быстрее и проще, но хуже держит сложную вёрстку и таблицы.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .config import Settings, ocr_langs_for
from .models import FAILED, OK, ParsedDoc, ParseInput

_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


def parse_with_pymupdf(source: "Path | str | ParseInput", settings: Settings) -> ParsedDoc:
    inp = ParseInput.of(source)
    started = time.monotonic()
    result = ParsedDoc(
        doc_id=inp.doc_id,
        source_path=inp.origin or str(inp.path or inp.name),
        parser="pymupdf4llm",
        status=FAILED,
    )

    try:
        import pymupdf
        import pymupdf4llm
    except ImportError as exc:
        result.errors.append(f"pymupdf4llm не установлен: {exc}")
        return result

    doc = None
    try:
        if inp.data is not None:
            doc = pymupdf.open(stream=inp.data, filetype="pdf")
        else:
            doc = pymupdf.open(inp.path)
        kwargs = {"page_chunks": True, "show_progress": False}
        # OCR-параметры появились не во всех версиях: пробуем с ними, при TypeError — без.
        ocr_kwargs = {
            "ocr_language": "+".join(ocr_langs_for("tesseract", settings.ocr_langs)),
        }
        try:
            pages = pymupdf4llm.to_markdown(doc, **kwargs, **ocr_kwargs)
        except TypeError:
            pages = pymupdf4llm.to_markdown(doc, **kwargs)

        pages_text: dict[int, str] = {}
        tables = 0
        for i, page in enumerate(pages, start=1):
            meta = page.get("metadata") or {}
            page_no = int(meta.get("page_number") or i)
            pages_text[page_no] = (page.get("text") or "").strip()
            tables += len(page.get("tables") or [])

        markdown = "\n\n".join(pages_text[p] for p in sorted(pages_text))
        result.status = OK
        result.markdown = markdown
        result.pages_text = pages_text
        result.page_count = len(pages_text) or int(doc.page_count)
        result.table_count = tables
        result.sections = [m.strip() for m in _HEADING_RE.findall(markdown) if len(m) < 300]
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result.duration_sec = round(time.monotonic() - started, 2)
        if doc is not None:
            try:
                doc.close()
            except Exception:  # noqa: BLE001, S110
                pass

    return result


def manual_text_path(source: "Path | str | ParseInput", settings: Settings) -> Path | None:
    """manual.txt ищем рядом с PDF и в data/manual/ по имени документа."""
    inp = ParseInput.of(source)
    candidates = [settings.manual_dir / f"{inp.doc_id}.txt"]
    if inp.path is not None:
        candidates.insert(0, inp.path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_manual(source: "Path | str | ParseInput", settings: Settings) -> ParsedDoc | None:
    """Последний рубеж: текст, положенный человеком рядом с PDF."""
    inp = ParseInput.of(source)
    txt = manual_text_path(inp, settings)
    if txt is None:
        return None

    started = time.monotonic()
    content = txt.read_text(encoding="utf-8", errors="replace")
    result = ParsedDoc(
        doc_id=inp.doc_id,
        source_path=inp.origin or str(inp.path or inp.name),
        parser="manual",
        status=OK,
        markdown=content,
        pages_text={1: content},
        page_count=1,
        sections=[m.strip() for m in _HEADING_RE.findall(content) if len(m) < 300],
        duration_sec=round(time.monotonic() - started, 2),
    )
    result.errors.append(f"использован ручной текст: {txt}")
    return result
