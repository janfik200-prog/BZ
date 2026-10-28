"""Проверочный документ Word с формулами OMath (OMML) — tests/data/omath_ru.docx.

    python tests/data/make_docx.py

Формулы — как их пишет редактор формул Word: дробь, сумма с пределами, индекс по-русски,
строчная формула внутри абзаца; рядом заголовки, абзацы и таблица.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml import parse_xml

M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


def _r(text: str) -> str:
    return f"<m:r><m:t>{text}</m:t></m:r>"


def _sub(base: str, sub: str) -> str:
    return f"<m:sSub><m:e>{_r(base)}</m:e><m:sub>{_r(sub)}</m:sub></m:sSub>"


def _sup(base: str, sup: str) -> str:
    return f"<m:sSup><m:e>{_r(base)}</m:e><m:sup>{_r(sup)}</m:sup></m:sSup>"


def _frac(num: str, den: str) -> str:
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def _sum(lower: str, upper: str, body: str) -> str:
    return (
        '<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr>'
        f"<m:sub>{lower}</m:sub><m:sup>{upper}</m:sup><m:e>{body}</m:e></m:nary>"
    )


# (1) плотность с пористостью: ρ = ρ₀(1 − φ)
DENSITY = _r("ρ=") + _sub("ρ", "0") + _r("(1−φ)")
# (2) коэффициент пористости: K_п = V_пор / V_обр · 100 %
POROSITY = _sub("K", "п") + _r("=") + _frac(_sub("V", "пор"), _sub("V", "обр")) + _r("·100 %")
# (3) среднее: x̄ = 1/n ∑_{i=1}^{n} x_i
MEAN = _r("x̄=") + _frac(_r("1"), _r("n")) + _sum(_r("i=1"), _r("n"), _sub("x", "i"))
# строчная: c²
INLINE = _sup("c", "2")


def _display(p: Any, omml: str) -> None:
    p._p.append(parse_xml(f"<m:oMathPara {M}><m:oMath>{omml}</m:oMath></m:oMathPara>"))


def _inline(p: Any, omml: str) -> None:
    p._p.append(parse_xml(f"<m:oMath {M}>{omml}</m:oMath>"))


def build(path: Path) -> Path:
    doc = Document()
    doc.core_properties.title = "Петрофизика рудного поля: формулы в Word"
    doc.core_properties.author = "Иванов И. И.; Петров П. П."
    doc.add_heading("Петрофизика рудного поля", level=1)
    doc.add_paragraph(
        "Плотность пород рудного поля определялась по керну скважин. Связь плотности с "
        "пористостью описывается формулой (1), где ρ₀ — плотность минерального скелета."
    )
    _display(doc.add_paragraph(), DENSITY)
    doc.add_paragraph("(1)")
    doc.add_heading("Пористость", level=2)
    doc.add_paragraph(
        "Коэффициент пористости вычислялся как отношение объёма пор к объёму образца по "
        "формуле (2). Значения приведены в таблице 1."
    )
    _display(doc.add_paragraph(), POROSITY)
    table = doc.add_table(rows=3, cols=3)
    for r, row in enumerate(
        (("Проба", "Kп, %", "ρ, г/см³"), ("КП-1", "4,6", "2,71"), ("КП-2", "7,9", "2,64"))
    ):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    doc.add_heading("Среднее по профилю", level=2)
    p = doc.add_paragraph("Среднее содержание по профилю — формула (3); энергия пропорциональна ")
    _inline(p, INLINE)
    p.add_run(", как и в формуле Эйнштейна.")
    _display(doc.add_paragraph(), MEAN)
    doc.add_paragraph(
        "Среднее содержание золота по профилю составило 2,4 г/т, что выше бортового "
        "содержания, принятого для месторождений этого типа в Магаданской области."
    )
    doc.save(str(path))
    return path


if __name__ == "__main__":
    print(build(Path(__file__).with_name("omath_ru.docx")))
