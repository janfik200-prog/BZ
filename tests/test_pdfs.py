"""Разбор эталонных PDF (tests/data, пересоздаются make_pdfs.py).

Зонд и запасной парсер (PyMuPDF) идут везде, docling — только где он установлен:
без него (проверки в CI, requirements-dev.txt) эти проверки пропускаются."""

from __future__ import annotations

from pathlib import Path

import pytest

from georag.parse.models import OK, SCAN, ParseInput
from georag.parse.orchestrator import hyphens, profile

pymupdf = pytest.importorskip("pymupdf")

VARIANTS = {
    "plain_ru.pdf": (profile.PLAIN, "ru", 2),
    "hyphenated_ru.pdf": (profile.HYPHENATED, "ru", 1),
    "formulas_ru.pdf": (profile.PLAIN, "ru", 1),
    "greek_en.pdf": (profile.PLAIN, "en", 1),
    "scan.pdf": (profile.SCAN, "en", 1),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_probe_variant(data: Path, name: str) -> None:
    variant, language, pages = VARIANTS[name]
    found = profile.probe(ParseInput.of(data / name))
    assert (found.variant, found.pages) == (variant, pages), found.reason
    if variant != profile.SCAN:
        assert found.language == language


def test_text_layer_is_exact(data: Path) -> None:
    """Эталон читается так, как напечатан: обычные дефисы и пробелы, греческие буквы."""
    text = pymupdf.open(data / "greek_en.pdf")[0].get_text()
    assert "σ-meson" in text and "ω-meson" in text
    assert "­" not in text and " " not in text


def test_line_hyphens_joined(data: Path) -> None:
    """Вёрстка с переносами: «преде- лах» — одно слово (так текст отдаёт парсер)."""
    text = " ".join(pymupdf.open(data / "hyphenated_ru.pdf")[0].get_text().split())
    fixed, count = hyphens.fix_line_hyphens(text, profile.HYPHENATED)
    assert count == 8
    assert "в пределах всей рудной зоны" in fixed
    assert "северо-западного" in fixed  # настоящий дефис остаётся


def test_fallback_parser(data: Path) -> None:
    pytest.importorskip("pymupdf4llm")
    from georag.parse.parsers.pymupdf import parse_with_pymupdf

    parsed = parse_with_pymupdf(data / "plain_ru.pdf")
    assert parsed.status == OK and parsed.page_count == 2
    assert sorted(parsed.pages_text) == [1, 2]
    assert parsed.sections == ["Строение Северного блока", "Методы"]
    assert "Линеаментный анализ" in parsed.pages_text[2]


def test_scan_is_not_parsed(data: Path, tmp_path: Path) -> None:
    """Скан конвейер не разбирает: статус scan, без попыток парсеров."""
    from georag.parse import pipeline
    from georag.parse.config import Settings

    settings = Settings(ocr=False)  # без OCR — даже если задан GEORAG_OCR
    settings.log_dir = tmp_path
    logger = pipeline.StepLogger(tmp_path)
    result, parsed, chunks = pipeline.process_document(
        data / "scan.pdf", settings, worker=None, logger=logger, write_outputs=False
    )
    assert result.status == SCAN and parsed.status == SCAN and not chunks


def test_scan_with_ocr_goes_to_docling_only(data: Path, tmp_path: Path) -> None:
    """С OCR скан идёт к docling с пометкой ocr, запасной PyMuPDF не пробуется (текста у
    скана нет); в версии парсера — движок OCR."""
    from georag.parse import pipeline
    from georag.parse.config import Settings
    from georag.parse.models import DOCLING, ParsedDoc

    seen: list[ParseInput] = []

    class Worker:
        def parse(self, inp: ParseInput) -> ParsedDoc:
            seen.append(inp)
            return ParsedDoc.failed(inp, DOCLING, "проверка")

    called: list[object] = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(pipeline, "parse_with_pymupdf", lambda inp: called.append(inp))
    try:
        result, _parsed, chunks = pipeline.process_document(
            data / "scan.pdf",
            Settings(ocr=True, log_dir=tmp_path),
            Worker(),
            pipeline.StepLogger(tmp_path),
            write_outputs=False,
        )
    finally:
        monkey.undo()
    assert [i.ocr for i in seen] == [True] and not called and not chunks
    assert "+ocr:rapidocr@" in result.parser_version


def test_ocr_converter_options() -> None:
    pytest.importorskip("docling")
    from georag.parse.config import Settings
    from georag.parse.parsers.docling import ocr_options

    opts = ocr_options(Settings(ocr_lang="cyrillic"))
    assert opts.lang == ["cyrillic"] and str(opts.mode.value) == "full_page"


@pytest.mark.slow
def test_ocr_reads_russian_scan(tmp_path: Path) -> None:
    """Скан русской страницы (картинка без текстового слоя) — текст распознан."""
    pytest.importorskip("docling")
    pytest.importorskip("rapidocr")
    from georag.parse.config import Settings
    from georag.parse.parsers.docling import DoclingParser

    src = pymupdf.open(Path(__file__).parent / "data" / "plain_ru.pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
    scan = tmp_path / "scan_ru.pdf"
    out.save(scan)
    inp = ParseInput.of(scan)
    inp.ocr = True
    parsed = DoclingParser(Settings(device="cpu", formulas=False)).parse(inp)
    assert parsed.status == OK, parsed.errors[:2]
    assert "Северный блок сложен гранитогнейсами" in parsed.text


@pytest.mark.slow
def test_docling_formulas(data: Path) -> None:
    """Docling (если установлен): формулы со ссылками «(1)», «(2)» разбираются."""
    pytest.importorskip("docling")
    from georag.parse.config import Settings
    from georag.parse.parsers.docling import DoclingParser

    parsed = DoclingParser(Settings()).parse(data / "formulas_ru.pdf")
    assert parsed.status == OK, parsed.errors[:2]
    assert "Плотность пород" in parsed.text


def test_truncated_pdf_is_broken_not_scan(data: Path, tmp_path: Path) -> None:
    """Обрезанный PDF — «повреждён», а не «скан»: у скана текста нет, а этот просто
    не дочитывается (нет конца файла, таблицы ссылок)."""
    whole = (data / "plain_ru.pdf").read_bytes()
    cut = tmp_path / "cut.pdf"
    cut.write_bytes(whole[: len(whole) // 3])
    found = profile.probe(ParseInput.of(cut))
    assert found.variant == profile.BROKEN, found.reason
    assert "обрезан" in found.reason


def test_broken_needs_repair() -> None:
    """Мало текста без восстановления файла — по-прежнему скан."""
    scan = profile.decide(profile.Profile("", "", pages=3, chars_per_page=2))
    assert scan.variant == profile.SCAN
    broken = profile.decide(profile.Profile("", "", pages=3, chars_per_page=2, repaired=True))
    assert broken.variant == profile.BROKEN


def test_repaired_scan_with_images_is_scan() -> None:
    """Восстановленный PyMuPDF, но со страницами-картинками — скан, а не повреждённый."""
    p = profile.Profile("", "", pages=3, chars_per_page=2, repaired=True, images=3)
    assert profile.decide(p).variant == profile.SCAN
    assert profile.decide(profile.Profile("", "", pages=0)).variant == profile.BROKEN
