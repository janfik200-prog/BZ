"""Валидация после парсинга.

Две группы проверок:

1. Эвристики, которые работают без эталона (есть ли вообще текст, не битая ли
   кодировка, нет ли патологических повторов). Они ловят самые частые провалы:
   скан без текстового слоя и PDF с поломанными шрифтами, из которого извлекается
   мусор вида (cid:34).
2. Сверка с эталоном (golden standard) из tests/golden/<имя>.json: разделы, ключевые
   сущности, число таблиц и страниц. Это и есть метрика «точность парсинга».

Сопоставление сущностей — с поправкой на русскую морфологию: эталонное
«Анабарский щит» находится и в «Анабарского щита».
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from ..text import normalize, phrase_pattern, ru_stem, stem_variants
from .config import Settings
from .models import ParsedDoc, ValidationReport

# Символы, которые мы считаем нормальными для геологической статьи ru/en.
_ALLOWED_RE = re.compile(
    r"[0-9a-zA-Zа-яА-ЯёЁ\s\.,;:!\?\-–—‑'\"«»()\[\]{}/\\%°±×·′″†‡&@#\$€₽\+=<>\|~\^_\*"
    r"Ͱ-Ͽ"      # греческие буквы — формулы, минералогия
    r"‐-‧‰-⁞"  # типографика
    r"]"
)
_CID_RE = re.compile(r"\(cid:\d+\)")

# Нормализация и основы слов — общие, в georag/text.py. Старые имена оставлены:
# на них опираются проверки.
_normalize = normalize
_stem = ru_stem
_word_variants = stem_variants
_entity_pattern = phrase_pattern


def load_golden(doc: "Path | str", settings: Settings) -> dict | None:
    """Эталон для документа: tests/golden/<doc_id>.json."""
    stem = doc.stem if isinstance(doc, Path) else str(doc)
    candidate = settings.golden_dir / f"{stem}.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"битый эталон {candidate}: {exc}") from exc


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    cid_hits = len(_CID_RE.findall(text))
    cleaned = _CID_RE.sub("", text)
    allowed = len(_ALLOWED_RE.findall(cleaned))
    total = max(len(cleaned), 1)
    # каждый (cid:NN) считаем как один «мусорный» символ сверху
    return round(1 - allowed / total + cid_hits / total, 4)


def thin_pages(parsed: ParsedDoc, min_chars: int) -> list[int]:
    """Страницы, с которых почти ничего не извлеклось."""
    return [p for p, t in sorted(parsed.pages_text.items()) if len(t.strip()) < min_chars // 2]


def validate(parsed: ParsedDoc, settings: Settings, golden: dict | None = None) -> ValidationReport:
    report = ValidationReport()
    text = parsed.text
    norm = _normalize(text)
    pages = max(parsed.page_count, 1)

    # --- 1. Есть ли вообще текст --------------------------------------------
    chars_per_page = len(text.strip()) / pages
    has_text = chars_per_page >= settings.min_chars_per_page
    report.add(
        "has_text",
        has_text,
        critical=True,
        detail=f"{chars_per_page:.0f} символов на страницу (порог {settings.min_chars_per_page})",
    )
    if not has_text:
        report.suggestion = "rerun_ocr"

    # --- 2. Кодировка и мусор ------------------------------------------------
    ratio = garbage_ratio(text)
    ok_garbage = ratio <= settings.max_garbage_ratio
    report.add(
        "encoding_sane",
        ok_garbage,
        critical=True,
        detail=f"доля мусорных символов {ratio:.1%} (порог {settings.max_garbage_ratio:.0%})",
    )
    if not ok_garbage and report.suggestion == "none":
        # Битые шрифты OCR не лечит — тут нужен другой парсер.
        report.suggestion = "fallback"

    # --- 3. Патологические повторы ------------------------------------------
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    repeats = Counter(lines).most_common(1)
    worst = repeats[0][1] if repeats else 0
    report.add(
        "no_pathological_repeats",
        worst <= settings.max_line_repeats,
        critical=False,
        detail=f"самая частая строка встречается {worst} раз",
    )

    # --- 4. Пустые страницы (информационно) ----------------------------------
    thin = thin_pages(parsed, settings.min_chars_per_page)
    report.add(
        "pages_covered",
        len(thin) <= pages * 0.2,
        critical=False,
        detail=f"почти пустых страниц: {len(thin)} из {pages}" + (f" → {thin[:10]}" if thin else ""),
    )

    # --- 5. Сверка с эталоном ------------------------------------------------
    if not golden:
        report.add("golden_standard", True, critical=False, detail="эталон не задан, сверка пропущена")
        return report

    expected_sections = [s for s in (golden.get("sections") or []) if s.strip()]
    if expected_sections:
        found = [s for s in expected_sections if _entity_pattern(s).search(norm)]
        ratio_sections = len(found) / len(expected_sections)
        missing = [s for s in expected_sections if s not in found]
        report.add(
            "sections",
            ratio_sections >= settings.min_sections_ratio,
            critical=True,
            detail=f"найдено {len(found)}/{len(expected_sections)}"
            + (f", нет: {', '.join(missing[:5])}" if missing else ""),
        )

    expected_entities = [e for e in (golden.get("entities") or []) if e.strip()]
    if expected_entities:
        found_e = [e for e in expected_entities if _entity_pattern(e).search(norm)]
        ratio_entities = len(found_e) / len(expected_entities)
        missing_e = [e for e in expected_entities if e not in found_e]
        report.add(
            "entities",
            ratio_entities >= settings.min_entities_ratio,
            critical=True,
            detail=f"найдено {len(found_e)}/{len(expected_entities)}"
            + (f", нет: {', '.join(missing_e[:5])}" if missing_e else ""),
        )

    if golden.get("tables") is not None:
        expected_tables = int(golden["tables"])
        delta = abs(parsed.table_count - expected_tables)
        report.add(
            "tables",
            delta <= settings.table_tolerance,
            critical=False,
            detail=f"найдено {parsed.table_count}, ожидалось {expected_tables}",
        )

    if golden.get("pages") is not None:
        expected_pages = int(golden["pages"])
        report.add(
            "page_count",
            parsed.page_count == expected_pages,
            critical=False,
            detail=f"страниц {parsed.page_count}, ожидалось {expected_pages}",
        )

    # Если эталон не сошёлся, а текст при этом тонкий — сначала пробуем OCR.
    if not report.ok and report.suggestion == "none":
        report.suggestion = "rerun_ocr" if chars_per_page < settings.min_chars_per_page * 2 else "fallback"

    return report


def accuracy(report: ValidationReport) -> float:
    """Доля пройденных проверок — грубая метрика «точность парсинга» из плана."""
    if not report.checks:
        return 0.0
    return round(sum(1 for c in report.checks if c.ok) / len(report.checks), 3)
