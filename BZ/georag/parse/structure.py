"""Разметка docling: разделы, текст по страницам, подписи, принятые за колонтитул.

Отдельно от docling.py: здесь только docling-core (модели данных), без самого docling
с его моделями — функции нужны и в основном процессе (восстановление текста,
textlayer), куда тяжёлый docling не тянем.
"""

from __future__ import annotations

import re
from typing import Any

FURNITURE = {"page_header", "page_footer"}  # колонтитулы: в Markdown и фрагменты не идут
# Подпись рисунка или таблицы: «FIG. 4.», «Fig. 2», «Рис. 3», «TABLE IV», «Таблица 2».
_CAPTION = re.compile(
    r"^\s*(?:FIG\.?|Fig\.?|Figure|Рис\.?|Рисунок|TABLE|Table|Таблица)\s*[\dIVXLC]+\b"
)


def label_of(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def extract_structure(doc: Any) -> tuple[list[str], dict[int, str], int]:
    """Заголовки разделов, текст по страницам (без колонтитулов — как в Markdown) и
    число страниц."""
    sections: list[str] = []
    pages_text: dict[int, str] = {}

    for item in getattr(doc, "texts", []) or []:
        text = (getattr(item, "text", "") or "").strip()
        label = label_of(item)
        if not text or label in FURNITURE:
            continue
        if label in {"section_header", "title"} and len(text) < 300:
            sections.append(text)
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is None:
                continue
            pages_text[page_no] = (pages_text.get(page_no, "") + "\n" + text).strip()

    pages = getattr(doc, "pages", None) or {}
    page_count = len(pages) if pages else (max(pages_text) if pages_text else 0)
    return sections, pages_text, page_count


def rescue_captions(doc: Any) -> int:
    """Подпись рисунка внизу страницы docling бывает принимает за колонтитул — тогда она
    пропадает из текста (статья FMM, 14.10.2026: «FIG. 4. Lists 1–4 for box b» под
    векторным рисунком). Такая подпись возвращается в текст — в конец своей страницы.
    Возвращает, сколько подписей вернули."""
    from docling_core.types.doc.labels import DocItemLabel

    rescued = 0
    for item in list(getattr(doc, "texts", []) or []):
        if label_of(item) not in FURNITURE or not item.prov or not _CAPTION.match(item.text or ""):
            continue
        page_no = item.prov[0].page_no
        on_page = [
            node
            for node, _ in doc.iterate_items()
            if getattr(node, "prov", None)
            and node.prov[0].page_no == page_no
            and label_of(node) not in FURNITURE
        ]
        if not on_page:
            continue
        doc.insert_text(
            sibling=on_page[-1],
            label=DocItemLabel.CAPTION,
            text=item.text,
            orig=item.orig,
            prov=item.prov[0],
            after=True,
        )
        rescued += 1
    return rescued
