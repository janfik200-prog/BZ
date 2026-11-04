"""Проверка случаев по этапам: до разбора, попытки, после разбора, нарезка.

Здесь — порядок и учёт: что проверить на каком этапе, что записать в журнал. Сами
обработки — по модулю на случай (profile, hyphens, textlayer, formulas, words,
captions). Каждая — под защитой: упала — у её случаев «не вышло» с причиной,
остальные идут дальше.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from ..diag import describe, full_traceback
from ..models import OK, PARTIAL, ParsedDoc, ParseInput, ValidationReport
from ..parsers.structure import FURNITURE, extract_structure, label_of
from . import hyphens, layout, profile, textlayer
from .captions import rescue_captions
from .cases import Finding
from .formulas import fix_formula, fix_omath
from .words import fix_homoglyphs, join_soft_hyphens, shifted_letters, tidy_text, unshift_cyrillic


# --------------------------------------------------------------------------- #
#  До разбора и попытки
# --------------------------------------------------------------------------- #
def before(kind: profile.Profile) -> list[Finding]:
    """Что сказал зонд."""
    scan, unknown = kind.variant == profile.SCAN, kind.variant == profile.UNKNOWN
    readable = not scan and not unknown
    return [
        Finding("скан", scan, kind.reason if scan else ""),
        Finding("не читается", unknown, kind.reason if unknown else ""),
        Finding("переносы", readable, f"{kind.variant}: {kind.reason}" if readable else ""),
    ]


def attempts(tried: list[tuple[ParsedDoc, ValidationReport]]) -> Finding:
    """Понадобился ли запасной парсер и почему."""
    first, report = tried[0]
    if len(tried) == 1:
        return Finding("docling не справился", False)
    if first.status not in (OK, PARTIAL):
        why = f"статус {first.status}"
        if first.errors:
            why += f" ({first.errors[0].splitlines()[0][:150]})"
    else:
        failed = [f"{c.name}: {c.detail}" for c in report.failed if c.critical]
        why = "не прошёл проверку — " + "; ".join(failed)
    return Finding("docling не справился", True, f"{why} → {tried[-1][0].parser}")


def chunks(problem: str) -> Finding:
    return Finding("нет фрагментов", bool(problem), problem)


# --------------------------------------------------------------------------- #
#  После разбора
# --------------------------------------------------------------------------- #
def after(parsed: ParsedDoc, inp: ParseInput, kind: profile.Profile) -> list[Finding]:
    """Доводка выбранной попытки: у разметки docling — все случаи этапа, у текста без
    разметки (PyMuPDF4LLM) — слова и дефисы на концах строк."""
    return _docling(parsed, inp, kind) if parsed.docling_doc else _plain(parsed, kind)


Step = Callable[[Any, ParseInput], list[Finding]]


def _docling(parsed: ParsedDoc, inp: ParseInput, kind: profile.Profile) -> list[Finding]:
    from docling_core.types.doc.document import DoclingDocument

    try:
        doc = DoclingDocument.model_validate(parsed.docling_doc)
    except Exception as exc:  # noqa: BLE001
        return [_failed("доводка не удалась", exc, "разметка docling не читается")]
    findings: list[Finding] = []
    for cases, step in _DOCLING_STEPS:
        findings += _guarded(cases, partial(step, doc, inp))
    # дефисы — после слов (мягкие переносы уже склеены); формулы не трогаются
    findings += _guarded(("дефисы на концах строк",), lambda: _hyphens(_targets(doc), kind))
    try:
        sections, pages_text, page_count = extract_structure(doc)
        markdown, docling_doc = doc.export_to_markdown(), doc.export_to_dict()
    except Exception as exc:  # noqa: BLE001
        return findings + [_failed("доводка не удалась", exc, "не собрать разметку обратно")]
    parsed.markdown, parsed.sections, parsed.pages_text = markdown, sections, pages_text
    parsed.page_count = page_count or parsed.page_count
    parsed.docling_doc = docling_doc
    return findings


def _captions(doc: Any, inp: ParseInput) -> list[Finding]:
    n = rescue_captions(doc)
    return [Finding("подпись в колонтитуле", n > 0, f"возвращено в текст: {n}" if n else "")]


def _symbols(doc: Any, inp: ParseInput) -> list[Finding]:
    if profile.is_word(inp):
        return [Finding("испорченные знаки", False, "документ Word: текстового слоя PDF нет")]
    if inp.ocr:  # сверять не с чем: весь текст — из распознавания
        return [Finding("испорченные знаки", False, "скан: текстового слоя PDF нет")]
    share = textlayer.garbled_share(doc, inp)
    if share < textlayer.GARBLED_SHARE:
        return [Finding("испорченные знаки", False, f"лишних знаков {share:.1%}")]
    s = textlayer.restore(doc, inp)
    return [
        Finding(
            "испорченные знаки",
            True,
            f"лишних знаков {share:.1%} — текст из PyMuPDF: абзацев {s['абзацев']}, "
            f"ячеек {s['ячеек']}, остался текст docling у {s['оставлено docling']}",
        )
    ]


def _formulas(doc: Any, inp: ParseInput) -> list[Finding]:
    items = [t for t in doc.texts if label_of(t) == "formula" and t.text]
    if profile.is_word(inp):
        return _word_formulas(doc, items)
    tails = numbers = russian = rebuilt = 0
    for item in items:
        fix = fix_formula(item.text, getattr(item, "orig", "") or "")
        item.text = fix.text
        tails += fix.tail
        numbers += bool(fix.number)
        russian += fix.russian
        rebuilt += fix.from_text
    total, none = len(items), "" if items else "формул нет"
    return [
        Finding("хвост формулы", tails > 0, f"убран у {tails} из {total}" if tails else none),
        Finding("номер формулы", numbers > 0, f"{numbers} из {total}" if numbers else none),
        Finding("русские обозначения", russian > 0, f"у {russian} из {total}" if russian else none),
        Finding(
            "не та формула", rebuilt > 0, f"из текста PDF {rebuilt} из {total}" if rebuilt else none
        ),
    ]


_WORD_NUMBER = re.compile(r"\(\s*(\d{1,3})\s*\)")


def _word_formulas(doc: Any, items: list[Any]) -> list[Finding]:
    """Формулы Word — LaTeX из OMath (docling): русские буквы и знаки — показываемыми;
    номер «(N)», который в Word стоит отдельным абзацем или после табуляции, — к формуле,
    как у формул PDF («\\quad (N)»)."""
    fixed = numbered = 0
    for item in items:
        text = fix_omath(item.text)
        fixed += text != item.text
        item.text = text
        item.orig = ""  # текста PDF у формулы Word нет (index/formulas.py: pdf_text)
    order = [item for item, _level in doc.iterate_items(with_groups=False)]
    gone = []
    for item, following in zip(order, order[1:], strict=False):
        number = _WORD_NUMBER.fullmatch((getattr(following, "text", "") or "").strip())
        if label_of(item) == "formula" and label_of(following) in ("text", "paragraph") and number:
            item.text = f"{item.text} \\quad ({number.group(1)})"
            gone.append(following)
            numbered += 1
    if gone:
        doc.delete_items(node_items=gone)
    total = len(items)
    detail = f"из OMath {total}: исправлено {fixed}, номер приписан {numbered}" if total else ""
    return [Finding("формулы Word", total > 0, detail or "формул нет")]


def _targets(doc: Any) -> list[Any]:
    """Абзацы (без формул) и ячейки таблиц разметки docling — всё, где текст чинится."""
    return [t for t in doc.texts if label_of(t) != "formula" and t.text] + [
        cell
        for table in getattr(doc, "tables", []) or []
        for cell in getattr(getattr(table, "data", None), "table_cells", None) or []
        if cell.text
    ]


def _shifted(doc: Any, inp: ParseInput) -> list[Finding]:
    targets = _targets(doc)
    letters = shifted_letters([t.text for t in targets])
    if not letters:
        return [Finding("буквы со сдвигом", False)]
    fixed = 0
    for target in targets:
        text = unshift_cyrillic(target.text)
        fixed += text != target.text
        target.text = text
    detail = f"букв {letters}, исправлено абзацев и ячеек: {fixed}"
    return [Finding("буквы со сдвигом", fixed > 0, detail)]


def _words(doc: Any, inp: ParseInput) -> list[Finding]:
    soft = homoglyphs = 0
    for target in _targets(doc):
        soft += target.text.count("\u00ad")
        joined = join_soft_hyphens(target.text)
        fixed = fix_homoglyphs(joined)
        homoglyphs += fixed != joined
        target.text = fixed
    return [
        Finding("мягкие переносы", soft > 0, f"склеено {soft}" if soft else ""),
        Finding(
            "буквы-двойники", homoglyphs > 0, f"исправлено в {homoglyphs}" if homoglyphs else ""
        ),
    ]


def _authors(doc: Any, inp: ParseInput) -> list[Finding]:
    n = layout.demote_authors(doc)
    return [Finding("ФИО вместо заголовка", n > 0, f"стали текстом: {n}" if n else "")]


def _table_rows(doc: Any, inp: ParseInput) -> list[Finding]:
    n = layout.rescue_table_rows(doc)
    return [Finding("строка таблицы абзацами", n > 0, f"возвращено в таблицу: {n}" if n else "")]


def _page_breaks(doc: Any, inp: ParseInput) -> list[Finding]:
    if profile.is_word(inp):  # абзац Word страницей не рвётся
        return [Finding("абзац на стыке страниц", False, "документ Word: страниц нет")]
    n = layout.join_split_paragraphs(doc)
    return [Finding("абзац на стыке страниц", n > 0, f"склеено: {n}" if n else "")]


def _furniture(doc: Any, inp: ParseInput) -> list[Finding]:
    n = sum(1 for t in doc.texts if label_of(t) in FURNITURE)
    return [Finding("колонтитулы", n > 0, f"не в тексте: {n}" if n else "")]


_FIGURE = re.compile(r"^\s*(?:FIG\.?|Fig\.?|Figure|Рис\.?|Рисунок)\s*(\d+)")


def _figures(doc: Any, inp: ParseInput) -> list[Finding]:
    numbers = {
        m.group(1)
        for t in doc.texts
        if label_of(t) not in FURNITURE and (m := _FIGURE.match(t.text or ""))
    }
    pictures = len(getattr(doc, "pictures", []) or [])
    missing = len(numbers) > pictures
    detail = f"подписей рисунков {len(numbers)}, картинок {pictures}" if missing else ""
    return [Finding("рисунок без картинки", missing, detail)]


# Порядок важен: подпись из колонтитула — до замены знаков (её текст тоже чинится);
# формулы и слова — после (восстановленный текст тоже чистится); рисунки — после подписей.
_DOCLING_STEPS: tuple[tuple[tuple[str, ...], Step], ...] = (
    (("подпись в колонтитуле",), _captions),
    (("испорченные знаки",), _symbols),
    # Сдвиг — после замены знаков (текст из PyMuPDF сдвинут так же) и до слов: буквам-
    # двойникам и дефисам нужны настоящие русские буквы.
    (("буквы со сдвигом",), _shifted),
    (
        ("хвост формулы", "номер формулы", "русские обозначения", "не та формула", "формулы Word"),
        _formulas,
    ),
    (("мягкие переносы", "буквы-двойники"), _words),
    (("колонтитулы",), _furniture),
    (("рисунок без картинки",), _figures),
    # Разметка (layout.py) — после слов: куски строк и абзацев уже чистые; дефис на
    # склеенном стыке страниц («об- щем») разбирают дефисы, они идут после всех шагов.
    (("ФИО вместо заголовка",), _authors),
    (("строка таблицы абзацами",), _table_rows),
    (("абзац на стыке страниц",), _page_breaks),
)


def _plain(parsed: ParsedDoc, kind: profile.Profile) -> list[Finding]:
    """Текст без разметки (PyMuPDF4LLM): слова и дефисы — по страницам, Markdown, разделам."""

    def words() -> list[Finding]:
        pages = parsed.pages_text
        soft = sum(text.count("\u00ad") for text in pages.values())
        joined = {k: join_soft_hyphens(v) for k, v in pages.items()}
        fixed = {k: fix_homoglyphs(v) for k, v in joined.items()}
        homoglyphs = sum(1 for k in fixed if fixed[k] != joined[k])
        parsed.pages_text = fixed
        parsed.markdown = tidy_text(parsed.markdown)
        parsed.sections = [tidy_text(s) for s in parsed.sections]
        return [
            Finding("мягкие переносы", soft > 0, f"склеено {soft}" if soft else ""),
            Finding(
                "буквы-двойники",
                homoglyphs > 0,
                f"исправлено на {homoglyphs} страницах" if homoglyphs else "",
            ),
        ]

    def line_ends() -> list[Finding]:
        pages = {k: _Text(v) for k, v in parsed.pages_text.items()}
        findings = _hyphens(list(pages.values()), kind)
        parsed.pages_text = {k: v.text for k, v in pages.items()}
        parsed.markdown = hyphens.join_line_hyphens(parsed.markdown, kind.variant)
        parsed.sections = [hyphens.join_line_hyphens(s, kind.variant) for s in parsed.sections]
        return findings

    def shifted() -> list[Finding]:
        letters = shifted_letters(list(parsed.pages_text.values()))
        if not letters:
            return [Finding("буквы со сдвигом", False)]
        fixed = {k: unshift_cyrillic(v) for k, v in parsed.pages_text.items()}
        pages = sum(1 for k in fixed if fixed[k] != parsed.pages_text[k])
        parsed.pages_text = fixed
        parsed.markdown = unshift_cyrillic(parsed.markdown)
        parsed.sections = [unshift_cyrillic(s) for s in parsed.sections]
        detail = f"букв {letters}, исправлено страниц: {pages}"
        return [Finding("буквы со сдвигом", pages > 0, detail)]

    return (
        _guarded(("буквы со сдвигом",), shifted)
        + _guarded(("мягкие переносы", "буквы-двойники"), words)
        + _guarded(("дефисы на концах строк",), line_ends)
    )


@dataclass
class _Text:
    text: str


def _hyphens(targets: list[Any], kind: profile.Profile) -> list[Finding]:
    """Дефисы на концах строк по варианту зонда; targets — всё, у чего есть поле text."""
    if kind.variant == profile.WORD:
        return [Finding("дефисы на концах строк", False, "документ Word: строк нет — не трогаются")]
    if kind.variant in (profile.UNKNOWN, profile.SCAN):
        return [Finding("дефисы на концах строк", False, "вариант не известен — не трогаются")]
    fixed = 0
    for target in targets:
        target.text, n = hyphens.fix_line_hyphens(target.text, kind.variant)
        fixed += n
    return [
        Finding("дефисы на концах строк", fixed > 0, f"{kind.variant}: {fixed}" if fixed else "")
    ]


def _guarded(cases: tuple[str, ...], step: Callable[[], list[Finding]]) -> list[Finding]:
    """Обработка случая; упала — у её случаев «не вышло» с причиной, остальные идут дальше."""
    try:
        return step()
    except Exception as exc:  # noqa: BLE001
        return [_failed(case, exc) for case in cases]


def _failed(case: str, exc: BaseException, what: str = "") -> Finding:
    found = case == "доводка не удалась"  # этот случай и есть сбой; у остальных — не проверен
    return Finding(case, found, what, error=describe(exc), trace=full_traceback(exc))
