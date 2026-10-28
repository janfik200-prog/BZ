"""Ошибки разбора PDF из тест-плана 27.10.2026: 6.08, 7.02, 7.05, 8.10, 8.20, 8.29, 9.02."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from georag.index import figures, noise
from georag.parse.orchestrator import hyphens, layout
from georag.parse.orchestrator.formulas import fix_formula
from georag.parse.orchestrator.words import join_soft_hyphens

SOFT = "­"


# --- 6.08, 7.02: мягкий перенос на месте настоящего дефиса ------------------------ #
@pytest.mark.parametrize(
    ("text", "joined"),
    [
        (f"госу{SOFT} дарственных", "государственных"),
        (f"ра{SOFT} - бот", "работ"),
        (f"недоста{SOFT} то{SOFT} чно", "недостаточно"),
        (f"Северо{SOFT} Западный", "Северо-Западный"),  # 6.08
        (f"Au{SOFT} содержащие", "Au-содержащие"),
        (f"геолого{SOFT} геофизический", "геолого-геофизический"),
        (f"кварц{SOFT} серицитовые", "кварц-серицитовые"),
        (f"КП{SOFT}1", "КП-1"),  # 7.02: у букв с цифрой переноса не бывает
        (f"hydro{SOFT} thermal", "hydrothermal"),
    ],
)
def test_soft_hyphen(text: str, joined: str) -> None:
    assert join_soft_hyphens(text) == joined


def test_mineral_compound_keeps_hyphen() -> None:
    """«кварц- серицитовые» — составное с минералом; «труд- нодоступных» — перенос."""
    assert hyphens.line_hyphen("кварц", "серицитовые") == "кварц-серицитовые"
    assert hyphens.line_hyphen("хлорит", "эпидотовых") == "хлорит-эпидотовых"
    assert hyphens.line_hyphen("труд", "нодоступных") == "труднодоступных"
    assert hyphens.line_hyphen("кварц", "ами") == "кварцами"


# --- 8.10: модель формул выдала чужую формулу --------------------------------------- #
def test_wrong_formula_rebuilt_from_text() -> None:
    orig = "Σ = n∑ i =1 xi = ∫ L 0 g(x) dx (3)"
    fix = fix_formula(r"\hat { a }", orig)
    assert fix.from_text
    assert fix.text == r"\Sigma = \sum _{i=1}^{n} xi = \int _{0}^{L} g(x) dx \quad (3)"
    again = fix_formula(fix.text, orig)  # перечанкинг не портит собранное
    assert again.text == fix.text and not again.from_text


@pytest.mark.parametrize(
    ("latex", "orig"),
    [
        (r"\rho = \rho _ { 0 } \left ( 1 - \varphi \right )", "ρ = ρ0 (1 - φ) (1)"),
        (r"S = \frac { a } { b }", "S = a b (2)"),
        (r"\sum _ { i } x _ { i }", "n∑ i =1 xi (4)"),
    ],
)
def test_right_formula_kept(latex: str, orig: str) -> None:
    fix = fix_formula(latex, orig)
    assert not fix.from_text and fix.text.startswith(latex)


# --- 8.20: PDF статьи из папки лежит под исходным именем ------------------------------ #
def test_find_pdf_by_hash(tmp_path: Path) -> None:
    data = b"%PDF-1.4 kekura"
    sha = hashlib.sha256(data).hexdigest()
    (tmp_path / "other.pdf").write_bytes(b"%PDF-1.4 other")
    pdf = tmp_path / "kekura_formulas.pdf"
    pdf.write_bytes(data)
    folders = {p.stem: p for p in tmp_path.glob("*.pdf")}
    doc_id = f"local-kekura_formulas-{sha[:10]}"
    hashes: dict[Path, str] = {}
    assert figures.find_pdf(doc_id, {"sha256": sha}, folders, hashes) == pdf
    assert list(hashes) == [pdf]  # имя из doc_id — первым: другие файлы не читались
    assert figures.find_pdf(doc_id, {}, folders) is None
    assert figures.find_pdf("other", {}, folders) == tmp_path / "other.pdf"


# --- 8.29: фрагмент из одних авторов и мест работы ---------------------------------- #
def test_authors_and_affiliations_dropped() -> None:
    text = (
        "- И. И. Иванов¹, П. П. Петров²\n"
        "- ¹ Институт геологии рудных месторождений РАН, Москва, Россия\n"
        "- ² Северо-Восточный комплексный научно-исследовательский институт ДВО РАН, Магадан"
    )
    assert noise.is_affiliations(text)


@pytest.mark.parametrize(
    "text",
    [
        # список литературы: авторы и «РАН», но у записей — страницы
        "28. Некрасов И. Я. Особенности месторождения Альфа // Докл. РАН. – 1997. – Т. 353, "
        "№ 1. – С. 97–99. 29. Горячев Н. А. Геология жильных поясов. – Магадан : СВКНИИ ДВО "
        "РАН, 1998. – 210 с.",
        # методика: институт, «ИГМ СО РАН» и пара фамилий среди обычного текста
        "Возраст определён Ar-Ar методом в лаборатории ИГМ СО РАН (аналитик А. В. Травин). "
        "Образцы облучались в кадмированном канале реактора вместе с монитором МСА-11; "
        "среднее отношение изотопов аргона за период измерений составило 295,5 ± 0,5, "
        "нагревание шло ступенчато в кварцевом реакторе с внешним нагревом.",
    ],
)
def test_text_with_institutes_kept(text: str) -> None:
    assert not noise.is_affiliations(text)


# --- 7.05, 7.02, 9.02: разметка docling --------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "person"),
    [
        ("Мишенин Сергей Григорьевич 1", True),
        ("И. И. Иванов¹, П. П. Петров²", True),
        ("Иванов И. И.", True),
        ("Геологическое строение района", False),
        ("Методика работ", False),
        ("Северный Ледовитый Океан", False),
    ],
)
def test_is_person(text: str, person: bool) -> None:
    assert layout.is_person(text) is person


def _doc() -> Any:
    pytest.importorskip("docling_core")
    from docling_core.types.doc.document import DoclingDocument

    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=_size(600, 800))
    doc.add_page(page_no=2, size=_size(600, 800))
    return doc


def _size(w: float, h: float) -> Any:
    from docling_core.types.doc.base import Size

    return Size(width=w, height=h)


def _prov(page: int, l: float, t: float, r: float, b: float, n: int = 1) -> Any:  # noqa: E741
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    from docling_core.types.doc.common.reference import ProvenanceItem

    box = BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=page, bbox=box, charspan=(0, n))


def _text(doc: Any, text: str, page: int = 1, l: float = 50, t: float = 100) -> Any:  # noqa: E741
    from docling_core.types.doc.labels import DocItemLabel

    return doc.add_text(
        label=DocItemLabel.TEXT, text=text, prov=_prov(page, l, t, l + 10 * len(text), t + 12)
    )


def test_demote_authors() -> None:
    from docling_core.types.doc.labels import DocItemLabel

    doc = _doc()
    doc.add_heading("Введение", prov=_prov(1, 50, 50, 200, 62))
    doc.add_heading("Мишенин Сергей Григорьевич 1", prov=_prov(1, 50, 70, 300, 82))
    assert layout.demote_authors(doc) == 1
    labels = [(t.text, t.label) for t in doc.texts]
    assert labels == [
        ("Введение", DocItemLabel.SECTION_HEADER),
        ("Мишенин Сергей Григорьевич 1", DocItemLabel.TEXT),
    ]


def test_rescue_table_row() -> None:
    from docling_core.types.doc.items.table.table_data import TableCell, TableData

    doc = _doc()
    lefts, tops = (80, 210, 310, 400), (100, 120, 140)
    rows = (("Проба", "Au", "Ag", "Cu"), ("КП-1", "1,1", "5,0", "30"), ("КП-2", "0,9", "4,8", "28"))
    # заголовок «Глубина» шире чисел — его левый край левее, чем у столбца чисел
    cells = [
        TableCell(
            bbox=_prov(1, left - 20 * ((r, c) == (0, 3)), top, left + 60, top + 12).bbox,
            row_span=1,
            col_span=1,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            text=rows[r][c],
            column_header=r == 0,
        )
        for r, top in enumerate(tops)
        for c, left in enumerate(lefts)
    ]
    data = TableData(num_rows=3, num_cols=4, table_cells=cells)
    doc.add_table(data=data, prov=_prov(1, 80, 100, 460, 152))
    for left, text in zip(lefts, ("КП-4", "1,2", "5,5", "31"), strict=True):
        _text(doc, text, l=left, t=160)
    _text(doc, "Таблица показывает содержания металлов в пробах.", l=50, t=200)
    assert layout.rescue_table_rows(doc) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 4
    last = sorted(
        (c.start_col_offset_idx, c.text)
        for c in table.data.table_cells
        if c.start_row_offset_idx == 3
    )
    assert [t for _c, t in last] == ["КП-4", "1,2", "5,5", "31"]
    assert [t.text for t in doc.texts] == ["Таблица показывает содержания металлов в пробах."]


def test_join_split_paragraphs() -> None:
    doc = _doc()
    _text(doc, "Работы ведутся по единой методике. В об-", page=1, t=700)
    _text(doc, "щем случае они делятся на два этапа.", page=2, t=60)
    _text(doc, "Новый абзац начинается с заглавной.", page=2, t=80)
    assert layout.join_split_paragraphs(doc) == 1
    assert [t.text for t in doc.texts] == [
        "Работы ведутся по единой методике. В об- щем случае они делятся на два этапа.",
        "Новый абзац начинается с заглавной.",
    ]
    assert [p.page_no for p in doc.texts[0].prov] == [1, 2]
