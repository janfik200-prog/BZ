"""Рисунки статьи → фрагменты базы: подпись и описание моделью со зрением.

Подписи рисунков в базе и так есть — docling кладёт их в текст, и поиск их находит
(проверено 11.10.2026: подписи всех 1006 крупных рисунков базы — во фрагментах).
Нет того, что на самом рисунке: легенды карты, названий на схеме, осей графика,
координат. Это пишет модель со зрением — по рисунку, вырезанному
docling (`generate_picture_images`), — и получается ещё один фрагмент статьи:

    Рисунок (с. 9). Рис. 8. Схема прогноза нефтегазоносности тюменской свиты…
    Что на рисунке (описание модели по изображению): изолинии абсолютных отметок…

Проба 11.10.2026 (qwen3.5:9b) на 8 рисунках (карты, схема алгоритма, графики): по 1–2 с на
рисунок; описания по существу верны, в двух — по одной ошибке (перепутаны условные
знаки, «синий — выше» без основания); со схемы алгоритма прочитаны названия методов,
которых в подписи нет (CatBoost, LightGBM, SHAP). Включается флагом --figures при
добыче: PDF после разбора не хранится, поэтому рисунки описываются сразу.

Модель со зрением — отдельный сервер vLLM (у основной Qwen3-14B зрения нет): адрес
GEORAG_FIGURE_LLM (по умолчанию тот же, что у основной модели), имя модели
GEORAG_FIGURE_MODEL. Как запустить — docs/КОМАНДЫ.md, «Рисунки».
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from ..llm import LLM_HOST
from .models import Chunk

# Модель со зрением на сервере vLLM (--served-model-name), например Qwen/Qwen3-VL-8B-Instruct-FP8.
FIGURE_MODEL = os.environ.get("GEORAG_FIGURE_MODEL", "").strip() or "qwen3-vl"
FIGURE_HOST = os.environ.get("GEORAG_FIGURE_LLM", "").strip() or LLM_HOST
MIN_SIDE = 300  # меньше — значок, логотип, формула
MAX_SIDE = 1280  # больше модели не нужно: дольше и не точнее
HEADING = "Рисунки"
MARK = "Что на рисунке (описание модели по изображению):"

PROMPT = """Это рисунок из геологической статьи. Подпись под ним: «{caption}».

Опиши рисунок так, чтобы по описанию его можно было найти поиском:
1) что это: карта, разрез, график, диаграмма, схема, фотография;
2) что изображено: район, объекты, породы, разломы, месторождения, скважины;
3) легенда, подписи осей и условные знаки;
4) названия и числа, которые на рисунке читаются.
Пиши только то, что действительно видно на рисунке; неразборчивое не додумывай и
не пересказывай подпись. По-русски, 3–6 предложений."""


@dataclass
class Figure:
    page: int | None
    caption: str
    png: bytes
    size: tuple[int, int]


def _ref_text(doc: dict[str, Any], ref: dict[str, Any]) -> str:
    """Текст по ссылке docling вида {"$ref": "#/texts/12"}."""
    path = str(ref.get("$ref", "")).lstrip("#/").split("/")
    if len(path) != 2 or not path[1].isdigit():
        return ""
    items = doc.get(path[0]) or []
    index = int(path[1])
    return str(items[index].get("text", "")).strip() if index < len(items) else ""


def extract(doc: dict[str, Any]) -> list[Figure]:
    """Крупные рисунки из DoclingDocument (в виде словаря) с подписями.

    Мелкие (значки, логотипы, формулы) отбрасываются; без подписи на первой
    странице — тоже: это обложка или логотип журнала."""
    from PIL import Image

    out = []
    for pic in doc.get("pictures") or []:
        uri = str((pic.get("image") or {}).get("uri", ""))
        if not uri.startswith("data:image/") or "," not in uri:
            continue
        image: Image.Image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        if min(image.size) < MIN_SIDE:
            continue
        caption = " ".join(_ref_text(doc, c) for c in pic.get("captions") or []).strip()
        prov = pic.get("prov") or []
        page = prov[0].get("page_no") if prov else None
        if not caption and page == 1:
            continue
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        image.save(buf, "PNG")
        out.append(Figure(page, " ".join(caption.split()), buf.getvalue(), image.size))
    return out


def describe(figure: Figure, client: Any) -> str:
    """Описание рисунка моделью со зрением; client — llm.LLM (stream)."""
    picture = "data:image/png;base64," + base64.b64encode(figure.png).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT.format(caption=figure.caption or "нет")},
                {"type": "image_url", "image_url": {"url": picture}},
            ],
        }
    ]
    parts = [p.get("text", "") for p in client.stream(messages, temperature=0)]
    return " ".join("".join(parts).split())


def as_chunk(doc_id: str, index: int, figure: Figure, description: str, source: str) -> Chunk:
    where = f" (с. {figure.page})" if figure.page else ""
    lines = [f"Рисунок{where}. {figure.caption}".rstrip(), f"{MARK} {description}"]
    text = "\n".join(lines)
    return Chunk(
        doc_id=doc_id,
        index=index,
        text=text,
        embed_text=f"{HEADING}\n{text}",
        headings=[HEADING],
        pages=[figure.page] if figure.page else [],
        n_tokens=len(text) // 3,
        has_table=False,
        source_path=source,
    )


def figure_chunks(
    doc_id: str, doc: dict[str, Any], start: int, client: Any, source: str = "", log: Any = None
) -> list[Chunk]:
    """Фрагменты из рисунков статьи; номера — после текстовых (с start).
    Модель не ответила на рисунок — он пропускается, остальные идут."""
    chunks: list[Chunk] = []
    for figure in extract(doc):
        try:
            description = describe(figure, client)
        except Exception as exc:  # noqa: BLE001 — рисунок без описания не повод терять статью
            if log:
                log(f"рисунок с. {figure.page}: {type(exc).__name__}: {exc}")
            continue
        if description:
            chunks.append(as_chunk(doc_id, start + len(chunks), figure, description, source))
    return chunks


def strip_images(doc: dict[str, Any]) -> None:
    """Убрать картинки из DoclingDocument перед записью на диск: с ними файл статьи
    весит мегабайты, а описания уже во фрагментах."""
    for pic in doc.get("pictures") or []:
        if isinstance(pic.get("image"), dict):
            pic["image"].pop("uri", None)
    for page in (doc.get("pages") or {}).values():
        if isinstance(page, dict) and isinstance(page.get("image"), dict):
            page["image"].pop("uri", None)
