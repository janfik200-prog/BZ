"""Пайплайн парсинга: Docling → валидация → OCR-пересборка → fallback → manual → ручная проверка.

Каждый документ считается в отдельном процессе. Это не паранойя: у docling есть
известные случаи зависания в нативном парсере, где ни document_timeout, ни Ctrl+C
не помогают — процесс просто перестаёт отвечать. Для ночного cron это означало бы
остановку всей очереди, поэтому родитель ждёт результат и убивает зависший процесс.

Логи пишутся построчно в JSONL: шаг, сколько занял, результат, ошибки — ровно то,
что потом понадобится, чтобы понять, где узкое место.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .fallback import parse_manual, parse_with_pymupdf
from .models import (
    FAILED,
    MANUAL_REVIEW,
    OK,
    PARTIAL,
    TIMEOUT,
    Chunk,
    ParsedDoc,
    ParseInput,
    ValidationReport,
)
from .validate import accuracy, load_golden, validate


# --------------------------------------------------------------------------- #
#  Логирование
# --------------------------------------------------------------------------- #
class StepLogger:
    def __init__(self, log_dir: Path, run_id: str | None = None, prefix: str = "parse"):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"{prefix}-{self.run_id}.jsonl"

    def log(self, doc_id: str, step: str, status: str, duration: float, **extra: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "doc": doc_id,
            "step": step,
            "status": status,
            "duration_sec": round(duration, 2),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"  [{step}] {status} за {duration:.1f}с"
            + (f" — {extra.get('detail')}" if extra.get("detail") else "")
        )


# --------------------------------------------------------------------------- #
#  Изоляция парсинга в отдельном процессе
# --------------------------------------------------------------------------- #
def _worker_loop(
    tasks: mp.Queue[Any], results: mp.Queue[Any], settings_dict: dict[str, Any]
) -> None:  # pragma: no cover
    """Живёт в дочернем процессе: держит модели docling в памяти между документами."""
    from .config import Settings as _Settings

    settings = _Settings.from_dict(settings_dict)

    parser = None
    setup_error: str | None = None
    try:
        from .docling import DoclingParser

        parser = DoclingParser(settings)
    except Exception as exc:  # noqa: BLE001 — docling может быть не установлен
        setup_error = f"{type(exc).__name__}: {exc}"

    while True:
        task = tasks.get()
        if task is None:
            break
        inp = ParseInput(
            doc_id=task["doc_id"],
            path=Path(task["path"]) if task.get("path") else None,
            data=task.get("data"),
            display_name=task.get("display_name", ""),
            origin=task.get("origin", ""),
        )
        base = {
            "doc_id": inp.doc_id,
            "source_path": inp.origin or str(inp.path or inp.name),
            "parser": "docling+ocr" if task["full_page_ocr"] else "docling",
            "status": FAILED,
        }
        if parser is None:
            results.put({**base, "errors": [f"docling недоступен: {setup_error}"]})
            continue
        try:
            parsed = parser.parse(inp, full_page_ocr=task["full_page_ocr"])
            results.put(parsed.to_dict())
        except Exception as exc:  # noqa: BLE001
            results.put({**base, "errors": [f"{type(exc).__name__}: {exc}"]})


class DoclingWorker:
    """Один дочерний процесс + жёсткий таймаут на документ."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ctx = mp.get_context("spawn")
        self._proc: mp.process.BaseProcess | None = None

    def _start(self) -> None:
        self.tasks = self._ctx.Queue()
        self.results = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self.tasks, self.results, self.settings.to_dict()),
            daemon=True,
        )
        self._proc.start()

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            self._start()

    def restart(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=10)
        self._start()

    def parse(self, source: Path | str | ParseInput, full_page_ocr: bool = False) -> ParsedDoc:
        inp = ParseInput.of(source)
        self._ensure_alive()
        timeout = (
            self.settings.ocr_doc_timeout_sec if full_page_ocr else self.settings.doc_timeout_sec
        )
        parser_name = "docling+ocr" if full_page_ocr else "docling"
        self.tasks.put(
            {
                "doc_id": inp.doc_id,
                "path": str(inp.path) if inp.path else None,
                "data": inp.data,
                "display_name": inp.display_name,
                "origin": inp.origin,
                "full_page_ocr": full_page_ocr,
            }
        )

        started = time.monotonic()
        deadline = started + timeout
        while True:
            try:
                payload = self.results.get(timeout=2)
                return ParsedDoc.from_dict(payload)
            except queue.Empty:
                # Процесс мог умереть сам: не установлен docling, OOM, сегфолт в
                # нативном парсере. Ждать в этом случае весь таймаут бессмысленно.
                if self._proc is None or not self._proc.is_alive():
                    exitcode = getattr(self._proc, "exitcode", None)
                    self.restart()
                    return ParsedDoc(
                        doc_id=inp.doc_id,
                        source_path=inp.origin or str(inp.path or inp.name),
                        parser=parser_name,
                        status=FAILED,
                        errors=[f"дочерний процесс завершился (exitcode={exitcode})"],
                        duration_sec=round(time.monotonic() - started, 2),
                    )
                if time.monotonic() > deadline:
                    # Завис — убиваем и поднимаем заново, очередь не встаёт.
                    self.restart()
                    return ParsedDoc(
                        doc_id=inp.doc_id,
                        source_path=inp.origin or str(inp.path or inp.name),
                        parser=parser_name,
                        status=TIMEOUT,
                        errors=[f"парсер не ответил за {timeout} с, процесс убит"],
                        duration_sec=float(timeout),
                    )

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            try:
                self.tasks.put(None)
                self._proc.join(timeout=10)
            finally:
                if self._proc.is_alive():
                    self._proc.kill()


# --------------------------------------------------------------------------- #
#  Результат по документу
# --------------------------------------------------------------------------- #
@dataclass
class DocResult:
    doc_id: str
    source_path: str
    parser: str
    status: str
    accuracy: float = 0.0
    pages: int = 0
    sections: int = 0
    tables: int = 0
    chunks: int = 0
    duration_sec: float = 0.0
    failed_checks: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "parser": self.parser,
            "status": self.status,
            "accuracy": self.accuracy,
            "pages": self.pages,
            "sections": self.sections,
            "tables": self.tables,
            "chunks": self.chunks,
            "duration_sec": round(self.duration_sec, 2),
            "failed_checks": self.failed_checks,
            "attempts": self.attempts,
        }


# --------------------------------------------------------------------------- #
#  Основной проход по документу
# --------------------------------------------------------------------------- #
def process_document(
    source: Path | str | ParseInput,
    settings: Settings,
    worker: DoclingWorker,
    logger: StepLogger,
    make_chunks: bool = True,
    write_outputs: bool = True,
) -> tuple[DocResult, ParsedDoc, list[Chunk]]:
    """Прогон одного документа по цепочке попыток.

    Возвращает сводку, лучший разобранный документ и чанки — режиму добычи нужны
    сами данные, а не только статистика.
    """
    inp = ParseInput.of(source)
    started = time.monotonic()
    golden = load_golden(inp.doc_id, settings)
    attempts: list[str] = []
    best: tuple[ParsedDoc, ValidationReport] | None = None

    def consider(parsed: ParsedDoc, report: ValidationReport) -> None:
        """Запоминаем лучший результат — на случай, если ни один не пройдёт валидацию."""
        nonlocal best
        if best is None or accuracy(report) > accuracy(best[1]):
            best = (parsed, report)

    # --- попытка 1: docling как есть ----------------------------------------
    parsed = worker.parse(inp, full_page_ocr=False)
    attempts.append(f"docling:{parsed.status}")
    report = validate(parsed, settings, golden)
    logger.log(
        inp.doc_id,
        "docling",
        parsed.status,
        parsed.duration_sec,
        detail=f"страниц {parsed.page_count}, разделов {len(parsed.sections)}, таблиц {parsed.table_count}",
        validation=report.to_dict(),
        errors=parsed.errors,
    )
    consider(parsed, report)

    # --- попытка 2: полностраничный OCR, если текста нет ---------------------
    # Только когда парсер отработал, но текста не набралось (скан). Если docling
    # упал или завис, второй заход тем же парсером — потерянное время.
    if parsed.status in (OK, PARTIAL) and not report.ok and report.suggestion == "rerun_ocr":
        parsed_ocr = worker.parse(inp, full_page_ocr=True)
        attempts.append(f"docling+ocr:{parsed_ocr.status}")
        report_ocr = validate(parsed_ocr, settings, golden)
        logger.log(
            inp.doc_id,
            "docling+ocr",
            parsed_ocr.status,
            parsed_ocr.duration_sec,
            detail=f"страниц {parsed_ocr.page_count}, символов {len(parsed_ocr.text)}",
            validation=report_ocr.to_dict(),
            errors=parsed_ocr.errors,
        )
        consider(parsed_ocr, report_ocr)
        parsed, report = parsed_ocr, report_ocr

    # --- попытка 3: другой парсер -------------------------------------------
    if not (report.ok and parsed.status in (OK, PARTIAL)):
        parsed_fb = parse_with_pymupdf(inp, settings)
        attempts.append(f"pymupdf4llm:{parsed_fb.status}")
        report_fb = validate(parsed_fb, settings, golden)
        logger.log(
            inp.doc_id,
            "pymupdf4llm",
            parsed_fb.status,
            parsed_fb.duration_sec,
            detail=f"страниц {parsed_fb.page_count}, символов {len(parsed_fb.text)}",
            validation=report_fb.to_dict(),
            errors=parsed_fb.errors,
        )
        consider(parsed_fb, report_fb)
        parsed, report = parsed_fb, report_fb

    # --- попытка 4: ручной текст --------------------------------------------
    if not (report.ok and parsed.status in (OK, PARTIAL)):
        parsed_manual = parse_manual(inp, settings)
        if parsed_manual is not None:
            attempts.append(f"manual:{parsed_manual.status}")
            report_manual = validate(parsed_manual, settings, golden)
            logger.log(
                inp.doc_id,
                "manual",
                parsed_manual.status,
                parsed_manual.duration_sec,
                detail="взят ручной текст",
                validation=report_manual.to_dict(),
            )
            consider(parsed_manual, report_manual)
            parsed, report = parsed_manual, report_manual

    assert best is not None
    parsed, report = best
    passed = report.ok and parsed.status in (OK, PARTIAL)
    status = parsed.status if passed else MANUAL_REVIEW

    # --- чанкинг -------------------------------------------------------------
    chunks: list[Chunk] = []
    if passed and make_chunks:
        t0 = time.monotonic()
        try:
            from .chunking import chunk_parsed_doc

            chunks = chunk_parsed_doc(parsed, settings)
            logger.log(
                inp.doc_id,
                "chunking",
                OK,
                time.monotonic() - t0,
                detail=f"{len(chunks)} чанков, медиана {_median([c.n_tokens for c in chunks])} токенов",
            )
        except Exception as exc:  # noqa: BLE001
            logger.log(
                inp.doc_id,
                "chunking",
                FAILED,
                time.monotonic() - t0,
                errors=[f"{type(exc).__name__}: {exc}"],
            )

    if write_outputs:
        _write_outputs(parsed, report, chunks, settings, status)

    result = DocResult(
        doc_id=parsed.doc_id,
        source_path=parsed.source_path,
        parser=parsed.parser,
        status=status,
        accuracy=accuracy(report),
        pages=parsed.page_count,
        sections=len(parsed.sections),
        tables=parsed.table_count,
        chunks=len(chunks),
        duration_sec=time.monotonic() - started,
        failed_checks=[f"{c.name}: {c.detail}" for c in report.failed],
        attempts=attempts,
    )
    logger.log(
        inp.doc_id,
        "document",
        status,
        result.duration_sec,
        detail=f"парсер {parsed.parser}, точность {result.accuracy:.0%}, чанков {len(chunks)}",
    )
    return result, parsed, chunks


def _median(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[len(values) // 2]


def _write_outputs(
    parsed: ParsedDoc,
    report: ValidationReport,
    chunks: list[Chunk],
    settings: Settings,
    status: str,
) -> None:
    out = settings.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = parsed.doc_id

    (out / f"{stem}.md").write_text(parsed.text, encoding="utf-8")

    payload = parsed.to_dict(with_doc=False)
    payload["final_status"] = status
    payload["validation"] = report.to_dict()
    (out / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # DoclingDocument нужен на следующих шагах (перечанкинг без повторного парсинга).
    if parsed.docling_doc:
        (out / f"{stem}.docling.json").write_text(
            json.dumps(parsed.docling_doc, ensure_ascii=False), encoding="utf-8"
        )

    if chunks:
        (out / f"{stem}.chunks.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def process_all(paths: list[Path], settings: Settings, make_chunks: bool = True) -> list[DocResult]:
    settings.ensure_dirs()
    logger = StepLogger(settings.log_dir)
    worker = DoclingWorker(settings)
    results: list[DocResult] = []

    try:
        for i, path in enumerate(paths, start=1):
            print(f"\n[{i}/{len(paths)}] {path.name}")
            doc_result, _parsed, _chunks = process_document(
                path, settings, worker, logger, make_chunks=make_chunks
            )
            results.append(doc_result)
    finally:
        worker.close()

    report_path = settings.out_dir / f"report-{logger.run_id}.json"
    report_path.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nОтчёт: {report_path}\nЛог:    {logger.path}")
    return results
