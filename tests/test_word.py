"""Документы Word (.docx): приём в local, разбор docling, формулы из OMath.

Проверочный документ — tests/data/omath_ru.docx (собирает tests/data/make_docx.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from georag.acquire import local
from georag.acquire.models import ACQUIRED, AcquiredRecord
from georag.parse.config import Settings
from georag.parse.models import ParseInput
from georag.parse.orchestrator import hyphens, profile
from georag.parse.orchestrator.formulas import fix_omath

DOCX = Path(__file__).parent / "data" / "omath_ru.docx"


# --- LaTeX формул Word от docling ---------------------------------------------- #
@pytest.mark.parametrize(
    ("latex", "fixed"),
    [
        (
            r"K_{ \cyrp }=\frac{V_{ \cyrp \cyro \cyrr }}{V_{ \cyro \cyrb \cyrr }} "
            r"\text{ \textperiodcentered } 100 \%",
            r"K_{\text{п}}=\frac{V_{\text{пор}}}{V_{\text{обр}}} \cdot 100 \%",
        ),
        (r"\CYRP \cyrr \cyri \CYRYO", r"\text{ПриЁ}"),
        (r"T = 25 \textdegree C", r"T = 25 ^{\circ} C"),
        (r"\rho = \rho_{0}(1- \varphi )", r"\rho = \rho_{0}(1- \varphi )"),
        (r"\cyrzzz x", r"\cyrzzz x"),  # незнакомая команда — как есть
        ("", ""),
    ],
)
def test_fix_omath(latex: str, fixed: str) -> None:
    assert fix_omath(latex) == fixed
    assert fix_omath(fixed) == fixed  # перечанкинг не портит


# --- зонд и дефисы ------------------------------------------------------------ #
def test_probe_word() -> None:
    kind = profile.probe(ParseInput.of(DOCX))
    assert kind.variant == profile.WORD
    assert profile.is_word(ParseInput(doc_id="x", data=b"PK", display_name="Статья.DOCX"))
    assert not profile.is_word(ParseInput.of(Path("a.pdf")))


def test_word_hyphens_untouched() -> None:
    text = "северо- и юго-западный, рудно- магматический"
    assert hyphens.fix_line_hyphens(text, profile.WORD) == (text, 0)


# --- разбор целиком ------------------------------------------------------------ #
class _InProcess:
    """docling в этом же процессе — для .docx моделей не нужно, разбор доли секунды."""

    def __init__(self, settings: Settings) -> None:
        from georag.parse.parsers.docling import DoclingParser

        self.parser = DoclingParser(settings)

    def parse(self, inp: Any) -> Any:
        return self.parser.parse(inp)


def test_docx_parsed_with_omath_formulas(tmp_path: Path) -> None:
    pytest.importorskip("docling")
    from georag.parse.pipeline import StepLogger, process_document

    settings = Settings(log_dir=tmp_path)
    inp = ParseInput(doc_id="omath", path=DOCX, display_name=DOCX.name)
    worker: Any = _InProcess(settings)
    res, parsed, chunks = process_document(
        inp, settings, worker, StepLogger(tmp_path), write_outputs=False
    )
    assert (res.status, res.parser) == ("ok", "docling")
    md = parsed.markdown
    assert r"$$\rho = \rho_{0}(1- \varphi ) \quad (1)$$" in md  # номер — к формуле
    assert r"K_{\text{п}}=\frac{V_{\text{пор}}}{V_{\text{обр}}} \cdot 100 \%" in md
    assert r"\frac{1}{n}\sum_{i=1}^{n}x_{i}" in md
    assert "\\cyr" not in md and "textperiodcentered" not in md
    assert "\n(1)\n" not in md
    assert "| КП-1" in md and parsed.sections[:1] == ["Петрофизика рудного поля"]
    assert chunks and all(c.pages == [] for c in chunks)
    word = [c for c in res.cases if c.startswith("формулы Word")]
    assert word == ["формулы Word: из OMath 4: исправлено 1, номер приписан 1"]
    assert not any("не вышло" in c for c in res.cases)
    assert parsed.docling_doc is not None
    formulas = [t for t in parsed.docling_doc["texts"] if t["label"] == "formula"]
    assert len(formulas) == 4 and all(t["orig"] == "" for t in formulas)


# --- local: .docx в папке ------------------------------------------------------ #
def test_word_meta() -> None:
    meta = local._pdf_meta(DOCX)
    assert meta["title"] == "Петрофизика рудного поля: формулы в Word"
    assert meta["authors"] == ["Иванов И. И.", "Петров П. П."]
    assert isinstance(meta["year"], int)


def test_local_takes_docx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Worker:
        def __init__(self, settings: Any) -> None:
            pass

        def close(self) -> None:
            pass

    parsed: list[str] = []

    def parse_one(path: Path, *_: Any) -> AcquiredRecord:
        parsed.append(path.name)
        return AcquiredRecord(doc_id=path.stem, status=ACQUIRED, title=path.stem)

    monkeypatch.setattr(local, "DoclingWorker", _Worker)
    monkeypatch.setattr(local, "parse_one", parse_one)
    inbox, outbox = tmp_path / "input", tmp_path / "output"
    inbox.mkdir()
    (inbox / "a.pdf").write_bytes(b"%PDF a")
    (inbox / "Статья.DOCX").write_bytes(DOCX.read_bytes())
    (inbox / "старая.doc").write_bytes(b"old")
    report = local.process_folder(inbox, outbox, Settings(log_dir=tmp_path), tmp_path / "out")
    assert report.found == 2 and sorted(parsed) == ["a.pdf", "Статья.DOCX"]
    assert sorted(p.name for p in outbox.iterdir()) == ["a.pdf", "Статья.DOCX"]
    assert [p.name for p in inbox.iterdir() if p.is_file()] == ["старая.doc"]


def test_docx_figures_not_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Рисунки вырезаются из страниц PDF: у .docx в запись идёт без исходника."""
    from georag.parse.models import OK, ParsedDoc
    from georag.parse.pipeline import DocResult

    sent: list[Any] = []

    def process_document(inp: Any, *_: Any, **__: Any) -> Any:
        res = DocResult(doc_id=inp.doc_id, source_path="", parser="docling", status=OK)
        return res, ParsedDoc.failed(inp, "docling"), []

    monkeypatch.setattr(local, "process_document", process_document)
    monkeypatch.setattr(local, "_write_doc", lambda out, rec, parsed, chunks, pdf: sent.append(pdf))
    for name in ("статья.docx", "статья.pdf"):
        path = tmp_path / name
        path.write_bytes(DOCX.read_bytes() if name.endswith(".docx") else b"%PDF x")
        local.parse_one(path, Settings(log_dir=tmp_path), None, None, tmp_path)
    assert sent == [None, b"%PDF x"]
