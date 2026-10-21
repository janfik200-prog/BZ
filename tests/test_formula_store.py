"""Формулы как объекты (index/formulas.py): пометка во фрагментах, описание словами,
ссылки «(N)», таблица формул, картинка формулы, загрузка статьи."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn, returning_ids

from georag.index import db, formulas

RHO = r"\rho = \rho_0 (1 - \varphi) \quad (1)"
GRAV = r"\Delta g = \frac{4}{3} \pi G \Delta\rho R^{3} / h^{2} \quad (2)"


def item(index: int, latex: str, number: str, **kw: Any) -> dict[str, Any]:
    return {"index": index, "latex": latex, "number": number, "before": "", "after": "", **kw}


# --------------------------------------------------------------------------- #
#  Текст
# --------------------------------------------------------------------------- #
def test_plain_latex() -> None:
    assert formulas.plain_latex(RHO) == "ρ = ρ_0 (1 − φ)"
    assert formulas.plain_latex(r"K_{\mathrm{п}} = \frac{V_{пор}}{V_{обр}} \cdot 100 \%") == (
        "K_п = (V_пор)/(V_обр) · 100 %"
    )
    assert formulas.plain_latex(r"\sqrt{\frac{a}{b}} \leq \infty") == "√((a)/(b)) ≤ ∞"


def test_describe_uses_context() -> None:
    text = formulas.describe(
        item(
            1,
            RHO,
            "1",
            before="Модель простая. Плотность пород выражается через пористость:",
            after="где ρ0 — плотность скелета, φ — пористость. Дальше о другом.",
        )
    )
    assert text.startswith("Формула (1): ρ = ρ_0 (1 − φ).")
    assert "Плотность пород выражается через пористость." in text
    assert "где ρ0 — плотность скелета" in text
    assert "Модель простая" not in text  # только последняя фраза перед формулой


def test_references() -> None:
    text = "Из формулы (2) и уравнения 3 видно, что (7) верно; см. (2015), f(2), рис. (9)."
    assert formulas.references(text, {"2", "3", "7"}) == {"2", "3", "7"}
    assert formulas.references("Eq. (12) and equations 4", {"12", "4"}) == {"12", "4"}
    assert formulas.references("(5)", {"2"}) == set()  # формулы (5) в статье нет
    assert formulas.references(r"$$x = y \quad (5)$$ далее", {"5"}) == set()  # свой номер


def test_mark_formulas_whitespace_and_backslashes() -> None:
    text = "Плотность:\n\\rho = \\rho_0   (1 - \\varphi)\n\\quad (1)\nгде …"
    marked, found = formulas.mark_formulas(text, [item(1, RHO, "1")])
    assert found == [1]
    assert marked == f"Плотность:\n$${RHO}$$\nгде …"  # LaTeX — как есть, с обратными косыми
    again, _ = formulas.mark_formulas(marked, [item(1, RHO, "1")])
    assert again == marked  # уже помеченное — не удваивается
    described, _ = formulas.mark_formulas(text, [item(1, RHO, "1")], replace=lambda f: "[Ф]")
    assert described == "Плотность:\n[Ф]\nгде …"
    assert formulas.mark_formulas("нет формул", [item(1, RHO, "1")]) == ("нет формул", [])


def write_docling(folder: Path, doc_id: str) -> None:
    doc = {
        "texts": [
            {"label": "text", "text": "Плотность пород выражается через пористость:"},
            {"label": "formula", "text": RHO, "prov": [{"page_no": 1, "bbox": {"l": 1}}]},
            {"label": "text", "text": "где ρ0 — плотность скелета, φ — пористость."},
            {"label": "formula", "text": GRAV, "prov": [{"page_no": 1}]},
            {"label": "formula", "text": "", "orig": "картинка без текста"},  # до 13.10
        ]
    }
    (folder / f"{doc_id}.docling.json").write_text(json.dumps(doc), encoding="utf-8")


def test_prepare_and_mark_chunks(tmp_path: Path) -> None:
    write_docling(tmp_path, "d1")
    found = formulas.prepare("d1", tmp_path)
    assert [f["number"] for f in found] == ["1", "2"]  # формула без текста — не в счёт
    assert found[0]["description"].startswith("Формула (1): ρ = ρ_0 (1 − φ).")
    assert formulas.prepare("нет-такой", tmp_path) == []
    chunks = [
        {
            "index": 0,
            "text": f"Плотность пород: {RHO} где ρ0 — плотность.",
            "embed_text": f"Раздел\n{RHO}",
        },
        {"index": 1, "text": "Из формулы (1) следует…", "embed_text": "Из формулы (1) следует…"},
    ]
    where = formulas.mark_chunks(chunks, found)
    assert where == {0: [1]}
    assert f"$${RHO}$$" in chunks[0]["text"]
    assert "\\rho" not in chunks[0]["embed_text"]  # в векторе — словами
    assert "[Формула (1): ρ = ρ_0 (1 − φ)." in chunks[0]["embed_text"]
    assert chunks[1]["text"] == "Из формулы (1) следует…"


# --------------------------------------------------------------------------- #
#  База (подставное соединение)
# --------------------------------------------------------------------------- #
def test_store_formulas_and_refs() -> None:
    chunks = [(0, 11, f"$${RHO}$$"), (1, 12, "Из формулы (1) и (2) следует"), (2, 13, "ничего")]
    conn = FakeConn(answer=returning_ids(chunks=chunks))
    items = [item(1, RHO, "1", description="ф1"), item(2, GRAV, "2", description="ф2")]
    assert formulas.store(conn, "d1", items, {0: [1]}, [[0.1], [0.2]]) == 2
    inserts = conn.queries("INSERT INTO formulas")
    assert [p[1] for p in inserts] == [1, 2]
    assert inserts[0][10] == 11 and inserts[1][10] is None  # где стоит: фрагмент 0, второй — нигде
    refs = conn.queries("INSERT INTO formula_refs")
    assert sorted(refs) == [(12, 101), (12, 102)]  # фрагмент 1 ссылается на (1) и (2)
    assert conn.log[0][0].startswith("DELETE FROM formulas")  # прежние — заново


def test_of_document_with_number_and_refs() -> None:
    row = (5, "d1", 1, "1", RHO, "", 1, None, "", "", "ф1", 11, "Статья", 2024, "")

    def answer(sql: str, params: Any) -> list[Any]:
        if "FROM formula_refs" in sql:
            return [(12, 1, [3], "Из формулы (1) следует")]
        return [row]

    conn = FakeConn(answer=answer)
    [found] = formulas.of_document(conn, "d1", "(1)")
    assert found["index"] == 1 and found["latex"] == RHO and found["title"] == "Статья"
    assert conn.log[0][1] == ["d1", "1"]  # номер — без скобок
    assert found["referenced_by"] == [
        {"chunk_id": 12, "ord": 1, "pages": [3], "text": "Из формулы (1) следует"}
    ]
    assert "Формула (1) — Статья, с. 1" in formulas.show(found)


def test_marking_keeps_graph_facts() -> None:
    """Пометка $$…$$ — не изменение текста: факты фрагмента не стираются."""
    old = [(0, 11, RHO), (1, 12, "старый текст")]  # (ord, id, текст) — как SELECT ord, id, text
    rows = [
        {
            "doc_id": "d1",
            "ord": 0,
            "text": f"$${RHO}$$",
            "headings": [],
            "pages": [],
            "n_tokens": 1,
            "has_table": False,
            "embedding": "[0]",
        },
        {
            "doc_id": "d1",
            "ord": 1,
            "text": "новый текст",
            "headings": [],
            "pages": [],
            "n_tokens": 1,
            "has_table": False,
            "embedding": "[0]",
        },
    ]
    conn = FakeConn(answer=lambda sql, params: old if "FROM chunks" in sql else [(True,)])
    db.replace_chunks(conn, "d1", rows)
    dropped = conn.queries("DELETE FROM facts WHERE")
    assert dropped == [([12],)]  # только фрагмент, где текст правда другой


# --------------------------------------------------------------------------- #
#  Формула в ответе модели (правило 8)
# --------------------------------------------------------------------------- #
def test_formula_line_is_not_a_claim() -> None:
    """Формула отдельной строкой — не утверждение без ссылки: чистка ответа её не
    убирает, а общие слова модели без ссылки — по-прежнему убирает."""
    from georag.chat.checks import coverage, drop_uncited

    one = (
        f"Плотность пород района считают по формуле (1) из статьи [3]:\n\n$${RHO}$$\n\n"
        "где φ — пористость, ρ0 — плотность скелета породы [3]."
    )
    assert drop_uncited(one) == (one, [])
    assert coverage(one) == (2, 2)
    loose = "А это — общие слова модели без всякой ссылки на статью."
    multi = f"По формуле (1) [3]:\n$$\n{RHO}\n$$\nОна верна для песчаников района [3].\n{loose}"
    kept, removed = drop_uncited(multi)
    assert RHO in kept and removed == [loose]


# --------------------------------------------------------------------------- #
#  Картинка формулы
# --------------------------------------------------------------------------- #
def test_telegram_formulas() -> None:
    """Телеграм: формула в тексте — моноширинным в одну строку, под фрагментом —
    кнопки «Формула (N)», подпись к картинке — с LaTeX."""
    from georag.tg import ui

    text = ui.answer_html(f"По формуле (1) [3]:\n$$\n{RHO}\n$$\nгде φ < 1 [3].")
    assert f"<code>{RHO}</code>" in text and "$$" not in text and "&lt; 1" in text
    marks = ui.fragment_keyboard({}, None, [{"id": 5, "number": "2"}, {"id": 6, "number": None}])
    assert marks == {
        "inline_keyboard": [
            [
                {"text": "Формула (2)", "callback_data": "fm:5"},
                {"text": "Формула 2", "callback_data": "fm:6"},
            ]
        ]
    }
    assert ui.fragment_keyboard({}, None, []) is None
    caption = ui.formula_caption({"number": "2", "page": 3, "title": "Статья", "latex": "a<b"})
    assert caption == "<b>Формула (2)</b> — Статья, с. 3\n<code>a&lt;b</code>"


def test_static_stays_inside() -> None:
    """Страница отдаёт KaTeX из web/static — и ничего вне этой папки."""
    from georag.web import server

    sent: list[tuple[int, str]] = []
    handler = server.Handler.__new__(server.Handler)
    handler._send = lambda code, body, kind: sent.append((code, kind))  # type: ignore[method-assign]
    handler._json = lambda payload, code=200: sent.append((code, "json"))  # type: ignore[method-assign]
    handler._static("katex/katex.min.js")
    handler._static("../server.py")
    handler._static("katex/LICENSE")  # не того типа
    assert sent == [(200, "text/javascript; charset=utf-8"), (404, "json"), (404, "json")]


def test_render_png() -> None:
    pytest.importorskip("matplotlib")
    png = formulas.render_png(RHO)
    assert png is not None and png.startswith(b"\x89PNG")
    assert formulas.render_png("") is None
