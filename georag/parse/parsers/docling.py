"""Парсинг PDF через Docling: текстовый слой PDF → абзацы, заголовки, таблицы, формулы.

Только конвертация — в дочернем процессе (worker). Доводка результата (подписи, знаки,
формулы, слова, дефисы) — в основном процессе, у оркестратора (parse/orchestrator).

Отличия от типового кода из статей и туториалов — см. docs/УСТРОЙСТВО.md, раздел
«Что поправлено». Коротко: без распознавания картинок (OCR), режим TableFormer,
ускоритель, рамка и пачка модели формул, разбор статуса конверсии и провенанс страниц.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from ..config import Settings
from ..diag import describe, full_traceback
from ..models import DOCLING, FAILED, OK, PARTIAL, ParsedDoc, ParseInput
from .structure import extract_structure

log = logging.getLogger(__name__)

try:
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
except ImportError:  # pragma: no cover - старая раскладка модулей
    from docling.datamodel.pipeline_options import (  # type: ignore[attr-defined, unused-ignore]
        AcceleratorDevice,
        AcceleratorOptions,
    )


def _docling_source(inp: ParseInput) -> Any:
    """Путь к файлу либо поток байт — docling принимает и то, и другое."""
    if inp.data is not None:
        from io import BytesIO

        from docling.datamodel.base_models import DocumentStream

        return DocumentStream(name=inp.name, stream=BytesIO(inp.data))
    if inp.path is None:
        raise ValueError(f"нечего парсить: у {inp.doc_id} нет ни пути, ни данных")
    return inp.path


def build_converter(settings: Settings, notes: list[str] | None = None) -> DocumentConverter:
    """Конвертер docling по настройкам; что применить не удалось — в notes (и в лог)."""
    device = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
        "mps": getattr(AcceleratorDevice, "MPS", AcceleratorDevice.AUTO),
    }.get(settings.device, AcceleratorDevice.AUTO)

    pipeline_options = PdfPipelineOptions()
    # Распознавание картинок (OCR) у docling включено по умолчанию — выключаем: сканы не
    # разбираются (их останавливает зонд, profile.SCAN), а на PDF с текстовым слоем текст
    # с OCR и без — тот же (замер 13.10.2026), OCR только добавлял время.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=(
            TableFormerMode.ACCURATE if settings.table_mode == "accurate" else TableFormerMode.FAST
        ),
        do_cell_matching=True,  # текст ячеек — из текстового слоя PDF, а не с картинки
    )
    pipeline_options.do_formula_enrichment = settings.formulas
    if settings.formulas and not _set_formula_model(
        settings.formula_expansion, settings.formula_batch_size
    ):
        note = (
            "модель формул docling не найдена (CodeFormulaVlmModel, CodeFormulaModel) — "
            "рамка и пачка не применены, формулы медленнее и с мусором; проверьте версию docling"
        )
        log.warning(note)
        if notes is not None:
            notes.append(note)
    if settings.figures:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = settings.figures_scale
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=settings.num_threads, device=device
    )
    # Таймаут внутри docling: не панацея (см. README), но дешёвая первая линия обороны.
    if hasattr(pipeline_options, "document_timeout"):
        pipeline_options.document_timeout = float(settings.doc_timeout_sec)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


_FORMULA_MODELS = (
    ("docling.models.stages.code_formula.code_formula_vlm_model", "CodeFormulaVlmModel"),
    ("docling.models.stages.code_formula.code_formula_model", "CodeFormulaModel"),
)


def _set_formula_model(
    factor: float, batch: int, models: tuple[tuple[str, str], ...] = _FORMULA_MODELS
) -> bool:
    """Расширение рамки формулы и размер пачки для модели — у docling это поля класса,
    без опции. Обычный конвейер PDF берёт CodeFormulaVlmModel, старый — CodeFormulaModel.
    False — ни одного класса нет (docling сменил устройство): настройки не применены."""
    import importlib

    applied = False
    for module, name in models:
        try:
            cls = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
        cls.expansion_factor = factor
        cls.elements_batch_size = batch
        applied = True
    return applied


class DoclingParser:
    """Держит конвертер в памяти: модели грузятся один раз на процесс."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._converter: DocumentConverter | None = None
        self.warnings: list[str] = []  # о настройке docling — в ошибки каждого документа

    def parse(self, source: Path | str | ParseInput) -> ParsedDoc:
        """Любой сбой — статус FAILED с причиной, местом и стеком, а не исключение."""
        inp = ParseInput.of(source)
        started = time.monotonic()
        result = ParsedDoc.failed(inp, DOCLING)
        try:
            self._convert(inp, result)
        except Exception as exc:  # noqa: BLE001 — любое падение парсера = повод для запасного
            result.status = FAILED
            result.errors += [describe(exc), full_traceback(exc)]
        result.errors += [f"предупреждение: {w}" for w in self.warnings]
        result.duration_sec = round(time.monotonic() - started, 2)
        return result

    def _convert(self, inp: ParseInput, result: ParsedDoc) -> None:
        if self._converter is None:
            self._converter = build_converter(self.settings, self.warnings)
        conv = self._converter.convert(
            _docling_source(inp),
            raises_on_error=False,  # разбираем статус сами, а не ловим исключение
            max_num_pages=self.settings.max_pages,
        )
        status = getattr(conv, "status", None)
        for err in getattr(conv, "errors", None) or []:
            where = getattr(err, "module_name", "") or getattr(err, "component_type", "")
            message = str(getattr(err, "error_message", err))
            result.errors.append(f"{where}: {message}" if where else message)

        if status == ConversionStatus.FAILURE or getattr(conv, "document", None) is None:
            if not result.errors:
                result.errors.append(f"docling вернул статус {status} без описания ошибки")
            return  # статус остаётся FAILED

        doc = conv.document  # как выдал docling: доводка — у оркестратора
        sections, pages_text, page_count = extract_structure(doc)
        markdown = doc.export_to_markdown()
        docling_doc = doc.export_to_dict()

        # Поля — только когда всё посчитано: сбой посередине не оставит полуготовый результат.
        result.markdown = markdown
        result.sections = sections
        result.pages_text = pages_text
        result.page_count = page_count
        result.table_count = len(getattr(doc, "tables", []) or [])
        result.docling_doc = docling_doc
        result.status = OK if status == ConversionStatus.SUCCESS else PARTIAL
