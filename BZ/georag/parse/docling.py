"""Парсинг PDF через Docling.

Отличия от типового кода из статей и туториалов — см. README, раздел «Что поправлено».
Коротко: язык OCR, современный OcrMode вместо устаревшего force_full_page_ocr,
режим TableFormer, ускоритель, разбор статуса конверсии и провенанс страниц.
"""

from __future__ import annotations

import time
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .config import Settings, ocr_langs_for
from .models import FAILED, OK, PARTIAL, ParsedDoc, ParseInput

# --- совместимость версий -----------------------------------------------------
# OcrMode появился взамен force_full_page_ocr; на старых версиях его нет.
try:
    from docling.datamodel.pipeline_options import OcrMode  # docling >= 2.5x
except ImportError:  # pragma: no cover
    OcrMode = None  # type: ignore[assignment]

try:
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
except ImportError:  # pragma: no cover - старая раскладка модулей
    from docling.datamodel.pipeline_options import (  # type: ignore[no-redef]
        AcceleratorDevice,
        AcceleratorOptions,
    )


def _docling_source(inp: ParseInput):
    """Путь к файлу либо поток байт — docling принимает и то, и другое."""
    if inp.data is not None:
        from io import BytesIO

        from docling.datamodel.base_models import DocumentStream

        return DocumentStream(name=inp.name, stream=BytesIO(inp.data))
    if inp.path is None:
        raise ValueError(f"нечего парсить: у {inp.doc_id} нет ни пути, ни данных")
    return inp.path


def _ocr_options(settings: Settings, full_page: bool):
    """Опции OCR c правильными языками и режимом."""
    langs = ocr_langs_for(settings.ocr_engine, settings.ocr_langs)

    if settings.ocr_engine == "tesseract":
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        opts = TesseractCliOcrOptions(lang=langs)
    elif settings.ocr_engine == "rapidocr":
        from docling.datamodel.pipeline_options import RapidOcrOptions

        opts = RapidOcrOptions(lang=langs)
    else:
        from docling.datamodel.pipeline_options import EasyOcrOptions

        opts = EasyOcrOptions(lang=langs)

    if full_page:
        # Новый способ; force_full_page_ocr помечен deprecated, но нужен для старых версий.
        if OcrMode is not None and hasattr(opts, "mode"):
            opts.mode = OcrMode.FULL_PAGE
        else:  # pragma: no cover
            opts.force_full_page_ocr = True
    return opts


def build_converter(settings: Settings, full_page_ocr: bool = False) -> DocumentConverter:
    device = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
        "mps": getattr(AcceleratorDevice, "MPS", AcceleratorDevice.AUTO),
    }.get(settings.device, AcceleratorDevice.AUTO)

    pipeline_options = PdfPipelineOptions()
    # Полностраничный проход всегда означает OCR; в обычном разбор идёт по настройке.
    pipeline_options.do_ocr = bool(full_page_ocr or settings.use_ocr)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE if settings.table_mode == "accurate" else TableFormerMode.FAST,
        do_cell_matching=settings.do_cell_matching,
    )
    pipeline_options.ocr_options = _ocr_options(settings, full_page=full_page_ocr)
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=settings.num_threads, device=device
    )
    # Таймаут внутри docling: не панацея (см. README), но дешёвая первая линия обороны.
    if hasattr(pipeline_options, "document_timeout"):
        pipeline_options.document_timeout = float(
            settings.ocr_doc_timeout_sec if full_page_ocr else settings.doc_timeout_sec
        )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _label_of(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def _extract_structure(doc) -> tuple[list[str], dict[int, str], int]:
    """Заголовки разделов, текст по страницам и число страниц."""
    sections: list[str] = []
    pages_text: dict[int, str] = {}

    for item in getattr(doc, "texts", []) or []:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        label = _label_of(item)
        if label in {"section_header", "title"} and len(text) < 300:
            sections.append(text)
        for prov in (getattr(item, "prov", None) or []):
            page_no = getattr(prov, "page_no", None)
            if page_no is None:
                continue
            pages_text[page_no] = (pages_text.get(page_no, "") + "\n" + text).strip()

    pages = getattr(doc, "pages", None) or {}
    page_count = len(pages) if pages else (max(pages_text) if pages_text else 0)
    return sections, pages_text, page_count


class DoclingParser:
    """Держит конвертеры в памяти: модели грузятся один раз на процесс."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._converters: dict[bool, DocumentConverter] = {}

    def converter(self, full_page_ocr: bool) -> DocumentConverter:
        if full_page_ocr not in self._converters:
            self._converters[full_page_ocr] = build_converter(self.settings, full_page_ocr)
        return self._converters[full_page_ocr]

    def parse(self, source: "Path | str | ParseInput", full_page_ocr: bool = False) -> ParsedDoc:
        inp = ParseInput.of(source)
        started = time.monotonic()
        parser_name = "docling+ocr" if full_page_ocr else "docling"

        result = ParsedDoc(
            doc_id=inp.doc_id,
            source_path=inp.origin or str(inp.path or inp.name),
            parser=parser_name,
            status=FAILED,
        )

        try:
            conv = self.converter(full_page_ocr).convert(
                _docling_source(inp),
                raises_on_error=False,          # разбираем статус сами, а не ловим исключение
                max_num_pages=self.settings.max_pages,
            )
        except Exception as exc:  # noqa: BLE001 — любое падение парсера = повод для fallback
            result.errors.append(f"{type(exc).__name__}: {exc}")
            result.duration_sec = round(time.monotonic() - started, 2)
            return result

        status = getattr(conv, "status", None)
        for err in (getattr(conv, "errors", None) or []):
            result.errors.append(str(getattr(err, "error_message", err)))

        if status == ConversionStatus.FAILURE or getattr(conv, "document", None) is None:
            result.status = FAILED
            result.duration_sec = round(time.monotonic() - started, 2)
            return result

        doc = conv.document
        sections, pages_text, page_count = _extract_structure(doc)

        result.status = OK if status == ConversionStatus.SUCCESS else PARTIAL
        result.markdown = doc.export_to_markdown()
        result.sections = sections
        result.pages_text = pages_text
        result.page_count = page_count
        result.table_count = len(getattr(doc, "tables", []) or [])
        result.docling_doc = doc.export_to_dict()
        result.duration_sec = round(time.monotonic() - started, 2)
        return result
