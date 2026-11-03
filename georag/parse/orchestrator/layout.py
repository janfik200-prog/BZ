"""Разметка docling: что модель разметки поняла не так — по геометрии и тексту.

Три случая из прогона тест-плана (27.10.2026):

1. ФИО автора — заголовком раздела. «Мишенин Сергей Григорьевич 1» (цифра — сноска на
   место работы) docling считает заголовком, и весь текст статьи ложится «в раздел»
   последнего автора: в поиске и у чат-бота путь к фрагменту — его имя. Заголовок из
   одного ФИО (с отчеством или инициалами, сноской) становится обычным текстом.
2. Строка таблицы — абзацами под таблицей. Модель таблиц не видит последнюю строку
   («КП-4 | 1,2 | 5,5 | 31» — четыре абзаца). Возвращается строка, которая стоит на шаг
   строки ниже последней и чьи куски начинаются ровно у левых краёв столбцов.
3. Абзац, разорванный стыком страниц или столбцов: «…В об-» на одной странице и «щем
   случае…» на следующей — два абзаца, и фрагмент начинается с обрывка слова (7 из 26
   в реальной статье). Абзац со строчной буквы после текста без конца предложения —
   продолжение: склеивается, дефис на стыке («об- щем») разбирают правила дефисов.
"""

from __future__ import annotations

import re
from typing import Any

from ..parsers.structure import FURNITURE, label_of

# --- 1. ФИО вместо заголовка ------------------------------------------------- #
_CAP = r"[А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z][а-яёa-z]+)?"
_INITIALS = r"(?:[А-ЯЁA-Z]\.\s?){1,2}"
_PATRONYMIC = re.compile(r"(?:вич|вна|ична|чна|ич|оглы|кызы)$")
_MARKS = r"[\s\d,*†‡¹²³⁴⁵⁶⁷⁸⁹⁰]*"
# «И. И. Иванов», «Иванов И. И.»
_INITIALED = re.compile(rf"^(?:{_INITIALS}\s?{_CAP}|{_CAP},?\s{_INITIALS}){_MARKS}$")
_THREE_WORDS = re.compile(rf"^({_CAP})\s+({_CAP})\s+({_CAP}){_MARKS}$")


def is_person(text: str) -> bool:
    """Строка — одно ФИО или список ФИО через запятую (со сносками), а не заголовок."""
    parts = [p.strip() for p in re.split(r",\s*(?=[А-ЯЁA-Z])", text.strip()) if p.strip()]
    if not parts:
        return False
    for part in parts:
        three = _THREE_WORDS.match(part)
        if three and any(_PATRONYMIC.search(w.lower()) for w in three.groups()):
            continue  # «Мишенин Сергей Григорьевич 1»
        if _INITIALED.match(part):
            continue  # «И. И. Иванов¹»
        return False
    return True


def demote_authors(doc: Any) -> int:
    """Заголовки из одних ФИО — обычным текстом. Сколько исправлено."""
    from docling_core.types.doc.items.text import TextItem
    from docling_core.types.doc.labels import DocItemLabel

    fixed = 0
    for item in list(doc.texts):
        if label_of(item) == "section_header" and is_person(item.text or ""):
            new = TextItem(
                self_ref=item.self_ref,
                parent=item.parent,
                label=DocItemLabel.TEXT,
                prov=item.prov,
                orig=item.orig,
                text=item.text,
            )
            doc.replace_item(new_item=new, old_item=item)
            fixed += 1
    return fixed


# --- общее: порядок чтения ---------------------------------------------------- #
def _reading_order(doc: Any) -> list[Any]:
    """Элементы тела документа в порядке чтения, без колонтитулов."""
    return [
        item
        for item, _level in doc.iterate_items(with_groups=False)
        if label_of(item) not in FURNITURE
    ]


def _top_left(bbox: Any, page_height: float) -> Any:
    origin = str(getattr(bbox, "coord_origin", "")).upper()
    return bbox.to_top_left_origin(page_height) if "BOTTOM" in origin else bbox


# --- 2. Строка таблицы под таблицей -------------------------------------------- #
ROW_TEXT = 40  # кусок строки таблицы короткий: «КП-4», «1,2»
COLUMN_SLACK = 6.0  # пт: левый край куска — у левого края столбца
ROW_SLACK = 3.0  # пт: куски одной строки — на одной высоте


def rescue_table_rows(doc: Any) -> int:
    """Строки таблицы, которые модель таблиц не увидела и отдала абзацами под таблицей,
    — обратно в таблицу. Сколько строк возвращено."""
    from docling_core.types.doc.items.table.table_data import TableCell

    order = _reading_order(doc)
    rescued = 0
    for table in list(getattr(doc, "tables", []) or []):
        data, prov = getattr(table, "data", None), getattr(table, "prov", None)
        if data is None or not prov or not data.table_cells or table not in order:
            continue
        page = prov[0].page_no
        height = doc.pages[page].size.height
        cells = [(c, _top_left(c.bbox, height)) for c in data.table_cells if c.bbox is not None]
        if not cells:
            continue
        last = max(c.start_row_offset_idx for c, _ in cells)
        # Верх строки — самый высокий верх её ячеек: текст в ячейках начинается на разной
        # высоте (центровка, ячейка в две строки), и шаг по двум нижним верхам ячеек
        # выходил в несколько пунктов — строка под таблицей тогда не возвращалась.
        row_tops: dict[int, float] = {}
        for c, b in cells:
            row = c.start_row_offset_idx
            row_tops[row] = min(row_tops.get(row, b.t), b.t)
        row_top = row_tops[last]  # отступ до следующей строки — тоже от верха строки
        step = row_tops[last] - row_tops[last - 1] if last - 1 in row_tops else 0.0
        if step <= 0:
            step = max(b.b - b.t for _, b in cells) * 1.8
        # Левые края всех ячеек столбца: у заголовка он свой («Глубина, м» шире чисел).
        lefts: dict[int, list[float]] = {}
        for c, b in cells:
            if c.start_col_offset_idx == c.end_col_offset_idx - 1:
                lefts.setdefault(c.start_col_offset_idx, []).append(b.l)
        following = order[order.index(table) + 1 :]
        taken: list[Any] = []
        while True:
            row = _next_row(following[len(taken) :], page, height, row_top, step, lefts)
            if not row:
                break
            last += 1
            for col, item, box in row:
                data.table_cells.append(
                    TableCell(
                        bbox=box,
                        row_span=1,
                        col_span=1,
                        start_row_offset_idx=last,
                        end_row_offset_idx=last + 1,
                        start_col_offset_idx=col,
                        end_col_offset_idx=col + 1,
                        text=item.text,
                        column_header=False,
                        row_header=False,
                        row_section=False,
                    )
                )
            data.num_rows = last + 1
            row_top = row[0][2].t
            taken += [item for _col, item, _box in row]
            rescued += 1
        if taken:
            doc.delete_items(node_items=taken)
    return rescued


def _next_row(
    items: list[Any],
    page: int,
    height: float,
    row_top: float,
    step: float,
    lefts: dict[int, list[float]],
) -> list[tuple[int, Any, Any]]:
    """Следующая строка таблицы из абзацев сразу за ней: [(столбец, абзац, рамка)] или []."""
    row: list[tuple[int, Any, Any]] = []
    for item in items:
        if label_of(item) not in ("text", "paragraph") or not item.prov:
            break
        text = (item.text or "").strip()
        if not text or len(text) > ROW_TEXT or item.prov[0].page_no != page:
            break
        box = _top_left(item.prov[0].bbox, height)
        if row and abs(box.t - row[0][2].t) > ROW_SLACK:
            break  # следующая строка — в следующий заход
        col = next(
            (c for c, edges in lefts.items() if any(abs(box.l - e) <= COLUMN_SLACK for e in edges)),
            None,
        )
        if col is None or any(c == col for c, _i, _b in row):
            break
        row.append((col, item, box))
    if not row:
        return []
    gap = row[0][2].t - row_top
    cols = {c for c, _i, _b in row}
    enough = 0 in cols and len(cols) >= max(2, len(lefts) - 1)
    return row if enough and 0.5 * step <= gap <= 1.6 * step else []


# --- 3. Абзац, разорванный стыком страниц ------------------------------------- #
_SENTENCE_END = re.compile(r"[.!?:;…]\s*$")
_LOWER_START = re.compile(r"^\s*[a-zа-яё]")
_PROSE = ("text", "paragraph", "list_item")
# Между кусками абзаца бывают рисунок с подписью, таблица, сноска — они его не рвут;
# заголовок, формула, новый пункт списка — рвут.
_TRANSPARENT = ("picture", "table", "caption", "footnote", "chart")


def join_split_paragraphs(doc: Any) -> int:
    """Продолжение абзаца на следующей странице или в следующем столбце — к нему.
    Сколько стыков склеено."""
    joined = 0
    target = None
    gone: list[Any] = []
    for item in _reading_order(doc):
        label = label_of(item)
        if label in _TRANSPARENT:
            continue
        text = (item.text or "") if label in _PROSE else ""
        if (
            target is not None
            and label in ("text", "paragraph")
            and _LOWER_START.match(text)
            and not _SENTENCE_END.search(target.text or "")
        ):
            target.text = f"{target.text.rstrip()} {text.lstrip()}"
            if getattr(target, "orig", None) is not None and getattr(item, "orig", None):
                target.orig = f"{target.orig.rstrip()} {item.orig.lstrip()}"
            target.prov = list(target.prov or []) + list(item.prov or [])
            gone.append(item)
            joined += 1
            continue
        target = item if label in _PROSE and text else None
    if gone:
        doc.delete_items(node_items=gone)
    return joined
