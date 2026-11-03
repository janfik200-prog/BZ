"""WP01. Неизменяемое происхождение цитаты: отпечатки статьи, разбора и цитаты.

Цитату берёт код, а не модель (facts.py — по номеру предложения), и здесь она
привязывается к замороженному разбору: смещения в нормализованном тексте фрагмента,
отпечаток цитаты, отпечаток разбора всей статьи. verify_evidence повторяет проверку
по снимку статьи: изменённая цифра, чужая страница, другой отпечаток, битый OCR —
INVALID_EVIDENCE с причиной, а не тихое «верно».

Нормализация одна (normalize): пробельные знаки подряд — один пробел, края без
пробелов. Больше ничего не меняется: регистр, ё, кавычки, цифры — как в разборе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .contracts import EvidenceSpan, sha256_text

NORMALIZATION = "collapse-whitespace-v1"
# Битый текстовый слой: знаки замены, «буквы» из областей частного использования,
# сплошные одиночные буквы через пробел («g o l d»).
_BROKEN = re.compile(r"[�-]")
_SPACED = re.compile(
    r"(?:(?<![^\W\d_])[^\W\d_]\s){6,}"
)  # буквы через пробел; цифры таблиц — не сбой
OCR_BROKEN_SHARE = 0.02


def normalize(text: str) -> str:
    return " ".join((text or "").split())


@dataclass
class ChunkSnapshot:
    chunk_id: str
    text: str  # как в базе (chunks.text)
    pages: tuple[int, ...] = ()
    headings: tuple[str, ...] = ()
    ord: int = 0

    @property
    def normalized(self) -> str:
        return normalize(self.text)


@dataclass
class DocumentSnapshot:
    """Замороженный разбор статьи: то, по чему проверяется каждая цитата."""

    doc_id: str
    pdf_sha256: str
    parser_version: str
    title: str = ""
    chunks: dict[str, ChunkSnapshot] = field(default_factory=dict)
    license_status: str = "UNKNOWN"
    doi: str | None = None
    language: str | None = None

    @property
    def parsed_sha256(self) -> str:
        ordered = sorted(self.chunks.values(), key=lambda c: c.ord)
        return sha256_text("\n".join(c.normalized for c in ordered))


@dataclass(frozen=True)
class EvidenceValidation:
    valid: bool
    reason_codes: tuple[str, ...]


def broken_text(text: str) -> bool:
    """Похоже на сбой OCR или шрифта: доля битых знаков или буквы через пробел."""
    if not text:
        return True
    bad = len(_BROKEN.findall(text))
    return bad / max(len(text), 1) >= OCR_BROKEN_SHARE or bool(_SPACED.search(text))


def make_evidence(
    doc: DocumentSnapshot,
    chunk_id: str,
    quote: str,
    sentence_indices: tuple[int, ...] = (),
    evidence_id: str | None = None,
) -> EvidenceSpan | None:
    """Доказательство по цитате, найденной в нормализованном тексте фрагмента.
    Цитаты в нём нет — None: такое доказательство не создаётся."""
    chunk = doc.chunks.get(str(chunk_id))
    if chunk is None:
        return None
    text, wanted = chunk.normalized, normalize(quote)
    start = text.find(wanted) if wanted else -1
    if start < 0:
        return None
    pages = chunk.pages or (0,)
    page_start, page_end = max(min(pages), 1), max(max(pages), 1)
    return EvidenceSpan(
        evidence_id=evidence_id or f"E:{doc.doc_id}:{chunk_id}:{start}",
        doc_id=doc.doc_id,
        pdf_sha256=doc.pdf_sha256,
        parsed_sha256=doc.parsed_sha256,
        parser_version=doc.parser_version or "unknown",
        chunk_id=str(chunk_id),
        page_start=page_start,
        page_end=page_end,
        offset_start=start,
        offset_end=start + len(wanted),
        exact_quote=wanted,
        quote_sha256=sha256_text(wanted),
        sentence_indices=tuple(sentence_indices) or (1,),
        license_status=doc.license_status,
    )


def verify_evidence(
    evidence: EvidenceSpan, document: DocumentSnapshot | None
) -> EvidenceValidation:
    """Повторная проверка доказательства по снимку статьи (ТЗ WP01, приёмка):
    подменённая цифра, страница, отпечаток или пропавший фрагмент — не проходит."""
    reasons: list[str] = []
    if document is None:
        return EvidenceValidation(False, ("SOURCE_MISSING", "SOURCE_HASH_INVALID"))
    if evidence.doc_id != document.doc_id:
        reasons.append("DOC_ID_MISMATCH")
    if not document.pdf_sha256:
        reasons.append("PDF_HASH_MISSING")
    elif evidence.pdf_sha256 != document.pdf_sha256:
        reasons.append("PDF_HASH_MISMATCH")
    if evidence.parsed_sha256 != document.parsed_sha256:
        reasons.append("PARSED_HASH_MISMATCH")
    chunk = document.chunks.get(str(evidence.chunk_id))
    if chunk is None:
        reasons.append("SPAN_LOST")
    else:
        text = chunk.normalized
        if text[evidence.offset_start : evidence.offset_end] != evidence.exact_quote:
            reasons.append("QUOTE_SPAN_MISMATCH")
        pages = set(chunk.pages)
        if pages and not {evidence.page_start, evidence.page_end} <= pages:
            reasons.append("PAGE_MISMATCH")
        if broken_text(evidence.exact_quote):
            reasons.append("OCR_BROKEN")
    if sha256_text(evidence.exact_quote) != evidence.quote_sha256:
        reasons.append("QUOTE_HASH_MISMATCH")
    if reasons:
        return EvidenceValidation(False, ("SOURCE_HASH_INVALID", *reasons))
    return EvidenceValidation(True, ("SOURCE_SPAN_EXACT",))


def snapshot_from_db(conn: Any, doc_id: str) -> DocumentSnapshot | None:
    """Снимок статьи из базы (documents + chunks) — только чтение.

    Версия парсера — из documents.parser_version (T13), если колонка есть и заполнена;
    иначе «имя@unrecorded». to_jsonb(d) читает колонку и из базы до миграции, где её нет:
    отдельной проверки схемы на каждый снимок не нужно."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, sha256, parser, title, doi, language, "
            "to_jsonb(d) ->> 'parser_version' FROM documents d WHERE doc_id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(
            "SELECT id, text, pages, headings, ord FROM chunks WHERE doc_id = %s ORDER BY ord",
            (doc_id,),
        )
        chunks = {
            str(cid): ChunkSnapshot(str(cid), text or "", tuple(pages or ()), tuple(heads or ()), o)
            for cid, text, pages, heads, o in cur.fetchall()
        }
    return DocumentSnapshot(
        doc_id=row[0],
        # Отпечаток PDF не выдумывается: нет его в базе — пусто, и проверка цитаты не пройдёт.
        pdf_sha256=(row[1] or "").lower() if re.fullmatch(r"[0-9a-fA-F]{64}", row[1] or "") else "",
        # Версия парсера не записана (статья загружена до T13) — так и помечается.
        parser_version=(row[6] if len(row) > 6 and row[6] else f"{row[2] or 'unknown'}@unrecorded"),
        title=row[3] or "",
        chunks=chunks,
        doi=row[4],
        language=row[5],
    )


def audit(conn: Any, evidence: list[EvidenceSpan]) -> dict[str, Any]:
    """provenance_audit.json: сколько доказательств воспроизводится по базе."""
    cache: dict[str, DocumentSnapshot | None] = {}
    rows = []
    for ev in evidence:
        if ev.doc_id not in cache:
            cache[ev.doc_id] = snapshot_from_db(conn, ev.doc_id)
        result = verify_evidence(ev, cache[ev.doc_id])
        rows.append(
            {
                "evidence_id": ev.evidence_id,
                "valid": result.valid,
                "reasons": list(result.reason_codes),
            }
        )
    return {
        "normalization": NORMALIZATION,
        "n": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "invalid": [r for r in rows if not r["valid"]],
    }
