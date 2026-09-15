"""Загрузка добытого в базу: чанки → векторы → строки в Postgres.

Источник данных — папка добычи. Рядом лежат <doc_id>.json (метаданные и статус)
и <doc_id>.chunks.json (нарезка). Ни один PDF для этого не нужен: всё, что надо
для индекса, уже снято на этапе добычи.

Повторный запуск не плодит дубли: чанки документа переписываются целиком,
а документы, у которых число чанков не изменилось, пропускаются.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import db

# Список литературы — это чужие названия и ссылки, а не содержание статьи.
# В поиске такие чанки выглядят убедительно (там все нужные слова!), но читать
# в них нечего: на живом прогоне половину выдачи занимали именно библиографии.
REFS_HEADING = re.compile(
    r"^\s*(список\s+(литератур\w*|источник\w*)|литератур\w*|библиограф\w*|"
    r"references?|bibliography|works\s+cited)\s*:?\s*$",
    re.IGNORECASE,
)
# Даже без заголовка библиографию выдаёт плотность ссылок и DOI.
_CITATION_MARK = re.compile(r"https?://|doi:|doi\.org|//\s*[A-ZА-Я]|\bpp\.\s*\d", re.IGNORECASE)
MIN_MARKS_PER_KB = 4


def is_reference_chunk(chunk: dict) -> bool:
    """Похож ли чанк на список литературы, а не на текст статьи."""
    for heading in chunk.get("headings") or []:
        if REFS_HEADING.match(str(heading)):
            return True

    text = chunk.get("text") or ""
    if len(text) < 200:
        return False
    marks = len(_CITATION_MARK.findall(text))
    return marks / (len(text) / 1000) >= MIN_MARKS_PER_KB


@dataclass
class IngestReport:
    seen: int = 0
    indexed: int = 0
    skipped: int = 0
    empty: int = 0
    chunks: int = 0
    refs_dropped: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _document_row(meta: dict) -> dict:
    return {
        "doc_id": meta.get("doc_id") or "",
        "title": meta.get("title") or "",
        "authors": list(meta.get("authors") or []),
        "year": meta.get("year"),
        "journal": meta.get("journal"),
        "doi": meta.get("doi"),
        "url": meta.get("url") or "",
        "source": meta.get("source") or "",
        "language": meta.get("language"),
        "status": meta.get("status") or "",
        "sha256": meta.get("sha256") or "",
        "pages": int(meta.get("pages") or 0),
        "parser": meta.get("parser") or "",
        "accuracy": float(meta.get("accuracy") or 0.0),
        "fetched_at": meta.get("fetched_at") or "",
    }


def _chunk_rows(doc_id: str, chunks: list[dict], vectors: list[list[float]]) -> list[dict]:
    rows = []
    for chunk, vector in zip(chunks, vectors):
        rows.append(
            {
                "doc_id": doc_id,
                "ord": int(chunk.get("index", 0)),
                "text": chunk.get("text") or "",
                "headings": list(chunk.get("headings") or []),
                "pages": [int(p) for p in (chunk.get("pages") or [])],
                "n_tokens": int(chunk.get("n_tokens") or 0),
                "has_table": bool(chunk.get("has_table")),
                "embedding": db.vector_literal(vector),
            }
        )
    return rows


def ingest_dir(
    conn,
    embedder,
    acquired_dir: Path,
    force: bool = False,
    verbose: bool = True,
) -> IngestReport:
    report = IngestReport()
    known = {} if force else db.indexed_docs(conn)

    for chunks_path in sorted(acquired_dir.glob("*.chunks.json")):
        doc_id = chunks_path.name[: -len(".chunks.json")]
        meta_path = acquired_dir / f"{doc_id}.json"
        report.seen += 1

        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"{doc_id}: {exc}")
            continue

        if not chunks:
            report.empty += 1
            continue


        # Библиографию в базу не кладём: места занимает много, читать нечего.
        kept = [c for c in chunks if not is_reference_chunk(c)]
        report.refs_dropped += len(chunks) - len(kept)
        if not kept:
            report.empty += 1
            continue
        if not force and known.get(doc_id) == len(kept):
            report.skipped += 1
            continue
        chunks = kept

        meta.setdefault("doc_id", doc_id)
        # embed_text несёт заголовки разделов сверху — модель видит, из какой
        # части статьи кусок. Показываем потом человеку всё равно text.
        texts = [c.get("embed_text") or c.get("text") or "" for c in chunks]

        try:
            vectors = embedder.encode(texts, progress=verbose)
            db.upsert_document(conn, _document_row(meta))
            written = db.replace_chunks(conn, doc_id, _chunk_rows(doc_id, chunks, vectors))
            conn.commit()
        except Exception as exc:  # noqa: BLE001 — одна статья не должна ронять загрузку
            conn.rollback()
            report.errors.append(f"{doc_id}: {type(exc).__name__}: {exc}")
            continue

        report.indexed += 1
        report.chunks += written
        if verbose:
            title = (meta.get("title") or doc_id)[:60]
            print(f"  + {title} — {written} чанков")

    return report
