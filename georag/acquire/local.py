"""Свои статьи из папки: PDF и документы Word (.docx) кладут в input, разобранные
переезжают в output.

    python -m georag.acquire.local                      # input → output в папке проекта
    python -m georag.acquire.local --input D:\\статьи   # другая папка

Разбор тот же, что у статей из интернета (process_document), и текст ложится туда же —
в .local/acquired, откуда его берёт ingest. Каждый файл переезжает сразу после разбора:
прервали прогон (Ctrl+C, выключили компьютер) — в input остаётся ровно то, до чего
очередь не дошла, и следующий запуск продолжает с него. Статья, которую разобрать не
вышло (скан, сбой парсера), уходит в input/не разобрались — чтобы не разбирать её
каждый раз заново и чтобы в input не путалась с ещё не тронутыми.

Документ Word лучше PDF той же статьи: формулы в нём — разметка OMath, и docling
переводит её в LaTeX точно и без модели формул (в PDF формула — картинка из знаков,
её распознаёт модель, 2–4 с на формулу на процессоре). Рисунки из .docx не вырезаются:
их вырезка — по страницам PDF.

Файл, в отличие от добычи из интернета, не пропадает: он и есть статья, лежит в output.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..parse.config import Settings
from ..parse.models import ParseInput
from ..parse.parsers.worker import DoclingWorker
from ..parse.pipeline import StepLogger, process_document, say
from ..paths import ARTICLES, ROOT, files_in
from .models import ACQUIRED, SCANNED, AcquiredRecord, AcquireReport
from .pipeline import (
    _save_report,
    _write_doc,
    append_index,
    failed_note,
    load_index,
    record_status,
)

INBOX = ROOT / "input"
OUTBOX = ROOT / "output"
FAILED_DIR = "не разобрались"  # внутри input
SUFFIXES = (".pdf", ".docx")  # старый .doc docling не читает — сохранить его как .docx


def local_doc_id(path: Path, sha256: str) -> str:
    """Идентификатор статьи из папки: имя файла латиницей и начало хэша. Хэш — потому
    что русское имя латиницей пустеет, а два «article.pdf» из разных журналов — разные."""
    name = re.sub(r"[^a-z0-9._-]+", "_", path.stem.lower()).strip("_.")[:60]
    return f"local-{name}-{sha256[:10]}" if name else f"local-{sha256[:10]}"


def _pdf_meta(path: Path) -> dict[str, Any]:
    """Название, авторы и год из свойств PDF или документа Word; чего нет — название по
    имени файла."""
    meta: dict[str, Any] = {"title": path.stem, "authors": [], "year": None}
    try:
        title, author, created = _word_props(path) if is_word(path) else _pdf_props(path)
    except Exception:  # noqa: BLE001 — битые свойства не мешают разбору
        return meta
    title = title.strip()
    # «Microsoft Word - отчёт.docx», «untitled» — это не название статьи.
    if len(title) > 8 and not re.match(r"(?i)(microsoft word|untitled|\S+\.(docx?|pdf)$)", title):
        meta["title"] = title
    author = author.strip()
    if author:
        meta["authors"] = [a.strip() for a in re.split(r"[;,]| and ", author) if a.strip()]
    year = re.search(r"(?:19|20)\d{2}", created)
    if year:
        meta["year"] = int(year.group())
    return meta


def is_word(path: Path) -> bool:
    return path.suffix.lower() == ".docx"


def _pdf_props(path: Path) -> tuple[str, str, str]:
    import pymupdf

    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        info = doc.metadata or {}
    created = re.match(r"D:(\d{4})", info.get("creationDate") or "")
    return info.get("title") or "", info.get("author") or "", created.group(1) if created else ""


_CORE = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


def _word_props(path: Path) -> tuple[str, str, str]:
    """Свойства документа Word — docProps/core.xml внутри .docx (это zip)."""
    import zipfile
    from xml.etree import ElementTree

    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("docProps/core.xml"))

    def field(name: str) -> str:
        node = root.find(name, _CORE)
        return (node.text or "") if node is not None else ""

    return field("dc:title"), field("dc:creator"), field("dcterms:created")


def _move(path: Path, folder: Path) -> Path:
    """Переложить файл, не затирая одноимённый: «статья.pdf» → «статья (2).pdf»."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    n = 2
    while target.exists():
        target = folder / f"{path.stem} ({n}){path.suffix}"
        n += 1
    shutil.move(str(path), str(target))
    return target


def parse_one(
    path: Path, settings: Settings, worker: Any, logger: Any, out_dir: Path
) -> AcquiredRecord:
    """Один PDF или документ Word из папки → запись и файлы в out_dir, как у добытой статьи."""
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    doc_id = local_doc_id(path, sha)
    meta = _pdf_meta(path)
    inp = ParseInput(doc_id=doc_id, path=path, display_name=path.name, origin=str(path))
    doc_result, parsed, chunks = process_document(
        inp, settings, worker, logger, write_outputs=False
    )

    status = record_status(doc_result)
    record = AcquiredRecord(
        doc_id=doc_id,
        status=status,
        source="папка",
        title=meta["title"],
        authors=meta["authors"],
        year=meta["year"],
        sha256=sha,
        bytes_len=len(data),
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        parser=doc_result.parser,
        parser_version=doc_result.parser_version,
        accuracy=doc_result.accuracy,
        pages=doc_result.pages,
        sections=doc_result.sections,
        tables=doc_result.tables,
        chunks=len(chunks),
        duration_sec=doc_result.duration_sec,
        failed_checks=doc_result.failed_checks,
        attempts=doc_result.attempts,
        cases=doc_result.cases,
        filtered_by="человек",
        note={ACQUIRED: "", SCANNED: "скан: текстового слоя нет — не разбирается"}.get(
            status, failed_note(doc_result)
        ),
    )
    # рисунки вырезаются из страниц PDF; у документа Word их нет
    pdf = None if is_word(path) else data
    _write_doc(out_dir, record, None if status == SCANNED else parsed, chunks, pdf)
    return record


def process_folder(
    inbox: Path,
    outbox: Path,
    settings: Settings,
    out_dir: Path = ARTICLES,
    limit: int | None = None,
    force: bool = False,
) -> AcquireReport:
    """Все PDF и .docx из inbox по очереди; каждый переезжает сразу после своего разбора."""
    started = time.monotonic()
    report = AcquireReport(topic=f"статьи из папки {inbox}")
    pdfs = files_in(inbox, SUFFIXES)
    if limit:
        pdfs = pdfs[:limit]
    report.found = len(pdfs)
    if not pdfs:
        return report

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(settings.log_dir, prefix="local")
    known = {} if force else load_index(out_dir)
    worker = DoclingWorker(settings)
    try:
        for i, path in enumerate(pdfs, start=1):
            say(f"\n[{i}/{len(pdfs)}] {path.name}")
            # Разобрана в прошлый раз, но переложить не успели — второй раз не разбираем.
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            done = known.get(local_doc_id(path, sha))
            if done and done.get("status") == ACQUIRED:
                report.already_known += 1
                say(f"  уже разобрана → {_move(path, outbox).relative_to(outbox.parent)}")
                continue
            record = parse_one(path, settings, worker, logger, out_dir)
            append_index(out_dir, record)
            report.records.append(record)
            folder = outbox if record.status == ACQUIRED else inbox / FAILED_DIR
            say(f"  → {_move(path, folder).relative_to(folder.parent)}")
    finally:
        worker.close()
        report.duration_sec = time.monotonic() - started
        _save_report(out_dir, logger.run_id, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Разбор своих PDF и документов Word (.docx) из папки")
    p.add_argument("--input", type=Path, default=INBOX, help="папка с PDF и .docx для разбора")
    p.add_argument("--output", type=Path, default=OUTBOX, help="куда перекладывать разобранные")
    p.add_argument("--out", type=Path, default=ARTICLES, help="куда писать разобранный текст")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    p.add_argument("--limit", type=int, default=None, help="разобрать только первые N файлов")
    p.add_argument("--force", action="store_true", help="разбирать и уже разобранные")
    p.add_argument("--no-formulas", action="store_true", help="без распознавания формул")
    p.add_argument(
        "--ocr",
        action="store_true",
        help="распознавать сканы (OCR, прототип; то же — GEORAG_OCR=1)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.input.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    if args.device:
        settings.device = args.device
    if args.ocr:
        settings.ocr = True
    if args.no_formulas:
        settings.formulas = False
    else:
        from ..parse.parsers.docling import formulas_problem

        problem = formulas_problem(settings)
        if problem:
            print(f"Разбор не начат: {problem}", file=sys.stderr)
            return 1

    report = process_folder(
        args.input, args.output, settings, args.out, limit=args.limit, force=args.force
    )
    if not report.found:
        print(f"В папке {args.input} нет PDF и .docx — положите статьи туда и запустите снова.")
        return 0

    failed = [r for r in report.records if r.status != ACQUIRED]
    print("\n" + "=" * 72)
    print(
        f"Разобрано: {len(report.acquired)} | уже были: {report.already_known} | "
        f"не разобрались: {len(failed)} | чанков: {report.total_chunks} | "
        f"время: {report.duration_sec:.0f} с"
    )
    if failed:
        print(f"\nНе разобрались (лежат в {args.input / FAILED_DIR}):")
        for rec in failed:
            print(f"  • {rec.title[:60]} — {rec.note}")
    if report.acquired:
        print("\nДальше: python georag.py ingest — загрузить в базу")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
