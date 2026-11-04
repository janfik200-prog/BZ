"""Страницы скана картинками: что распознал OCR, можно сверить с самой страницей.

Скан (PDF без текстового слоя) с OCR (parse/config.py, Settings.ocr) разбирается в
текст, а сама статья — PDF — после разбора не хранится (добыча) или лежит отдельно
(свои статьи, папка output). Чтобы ошибку распознавания было видно, каждая страница
скана сохраняется картинкой рядом с файлами статьи:

    .local/acquired/<doc_id>.scan/page-001.jpg …

Страница чата показывает у источника из скана ссылки «скан с. N» — по адресу
/api/scan?doc_id=…&page=N (web/server.py) открывается картинка страницы.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..paths import ARTICLES

SCAN_DIR = "{doc_id}.scan"  # папка страниц скана — рядом с файлами статьи
DPI = 150  # мелкий шрифт и индексы читаются; страница A4 — около 0,3–0,6 МБ
JPEG_QUALITY = 80
MAX_PAGES = 400  # как Settings.max_pages: дальше разбор всё равно не идёт
_PAGE = re.compile(r"page-(\d{3,4})\.jpg")


def folder(doc_id: str, articles: Path = ARTICLES) -> Path | None:
    """Папка страниц скана; doc_id с «/», «\\» или «..» — None: путь не выходит из папки
    статей (doc_id приходит в адресе страницы)."""
    # «:» — у сводок и статей словаря («датасет:…»): не статья, и на Windows это не имя файла.
    if not doc_id or any(bad in doc_id for bad in ("/", "\\", ":", "..")):
        return None
    path = (articles / SCAN_DIR.format(doc_id=doc_id)).resolve()
    return path if articles.resolve() in path.parents else None


def save(pdf: Path | bytes, doc_id: str, articles: Path = ARTICLES) -> int:
    """Страницы PDF — картинками в <doc_id>.scan. Прежние картинки статьи стираются.
    Сколько страниц сохранено; PDF не открылся — 0 (текст статьи от этого не страдает)."""
    import pymupdf

    target = folder(doc_id, articles)
    if target is None:
        return 0
    if target.is_dir():
        for old in target.glob("page-*.jpg"):
            old.unlink()
    try:
        if isinstance(pdf, Path):
            opened = pymupdf.open(str(pdf))  # type: ignore[no-untyped-call]
        else:
            opened = pymupdf.open(stream=pdf, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception:  # noqa: BLE001 — битый PDF: статья без картинок страниц
        return 0
    saved = 0
    with opened:
        for number in range(1, min(opened.page_count, MAX_PAGES) + 1):
            try:
                pix = opened[number - 1].get_pixmap(dpi=DPI)
                data = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)  # type: ignore[no-untyped-call]
            except Exception:  # noqa: BLE001 — не отрисовалась одна страница: остальные идут
                continue
            target.mkdir(parents=True, exist_ok=True)
            (target / f"page-{number:03d}.jpg").write_bytes(data)
            saved += 1
    return saved


def pages(doc_id: str, articles: Path = ARTICLES) -> list[int]:
    """Номера страниц, картинки которых есть; не скан или картинок нет — пусто."""
    target = folder(doc_id, articles)
    if target is None or not target.is_dir():
        return []
    found = (_PAGE.fullmatch(p.name) for p in target.iterdir())
    return sorted(int(m.group(1)) for m in found if m)


def page_file(doc_id: str, page: int, articles: Path = ARTICLES) -> Path | None:
    """Картинка страницы скана; нет — None."""
    target = folder(doc_id, articles)
    if target is None or not 1 <= page <= MAX_PAGES:
        return None
    path = target / f"page-{page:03d}.jpg"
    return path if path.is_file() else None
