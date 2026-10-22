"""Рисунки статей (index/figures.py): картинка из PDF по рамке docling, ссылки «рис. N»,
рисунки под ответом на странице и в Телеграме. Без базы и модели: PDF рисуется в проверке."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn, returning_ids

from georag.index import figures
from georag.tg import ui
from georag.web import server

PAGE = Path(__file__).resolve().parent.parent / "georag" / "web" / "index.html"


def docling(box: dict[str, float], caption: str = "Fig. 3. Geological map of the area") -> Any:
    """Разметка docling с одной страницей A4 и рисунком в рамке box (начало — внизу)."""
    return {
        "texts": [{"text": caption}],
        "pictures": [
            {
                "prov": [{"page_no": 1, "bbox": {**box, "coord_origin": "BOTTOMLEFT"}}],
                "captions": [{"$ref": "#/texts/0"}] if caption else [],
            }
        ],
        "pages": {"1": {"size": {"width": 595.0, "height": 842.0}, "page_no": 1}},
    }


def red_square_pdf() -> bytes:
    """Страница A4, красный квадрат 200×200 пт: от (100, 100) сверху-слева."""
    import pymupdf

    pdf = pymupdf.open()  # type: ignore[no-untyped-call]
    page = pdf.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(100, 100, 300, 300), color=(1, 0, 0), fill=(1, 0, 0))  # type: ignore[no-untyped-call]  # noqa: E501
    data: bytes = pdf.tobytes()
    return data


@pytest.mark.parametrize(
    ("caption", "label"),
    [("Рис. 3. Карта", "3"), ("Рисунок 2а — разрез", "2а"), ("Figure 12: map", "12"),
     ("Fig.4 Section", "4"), ("Таблица 1", None)],
)  # fmt: skip
def test_label_from_caption(caption: str, label: str | None) -> None:
    assert figures.label_of(caption) == label


def test_references_in_text() -> None:
    text = "как видно на рис. 2, 3 и 5 и на рисунке 7а (Fig. 4; figs. 8 and 9)"
    labels = {"2", "3", "4", "5", "7а", "8", "9", "10"}
    assert figures.references(text, labels) == labels - {"10"}
    assert figures.references("в 2020 г. по трём рисункам", labels) == set()


def test_only_captioned_and_large_pictures() -> None:
    big = {"l": 100.0, "t": 742.0, "r": 300.0, "b": 542.0}
    assert len(figures.pictures(docling(big))) == 1
    assert figures.pictures(docling(big, caption="")) == []  # логотип, обложка
    small = {"l": 100.0, "t": 742.0, "r": 120.0, "b": 722.0}
    assert figures.pictures(docling(small)) == []  # значок


def test_save_cuts_picture_from_pdf(tmp_path: Path) -> None:
    from PIL import Image

    # Квадрат в PDF — 100..300 сверху; в разметке начало внизу: t = 842-100, b = 842-300.
    box = {"l": 100.0, "t": 742.0, "r": 300.0, "b": 542.0}
    saved = figures.save(docling(box), red_square_pdf(), "doc", tmp_path)
    assert [(f["n"], f["label"], f["page"]) for f in saved] == [(1, "3", 1)]
    path = figures.file_of(saved[0], tmp_path)
    assert path is not None and path.name == "fig-001.jpg"
    with Image.open(path) as img:
        assert img.size == (400, 400)  # 200 пт при 144 точках на дюйм
        r, g, b = img.convert("RGB").getpixel((200, 200))
        assert r > 200 and g < 60 and b < 60  # вырезан сам квадрат, а не пустое место
    # Заново — прежние файлы стираются, номера с начала.
    again = figures.save(docling(box), red_square_pdf(), "doc", tmp_path)
    assert [f["file"] for f in again] == ["doc.figures/fig-001.jpg"]
    assert figures.save(docling(box), b"not a pdf", "bad", tmp_path) == []


def test_file_only_inside_articles(tmp_path: Path) -> None:
    (tmp_path / "secret.jpg").write_bytes(b"x")
    articles = tmp_path / "articles"
    articles.mkdir()
    assert figures.file_of({"file": "../secret.jpg"}, articles) is None


def test_store_links_caption_and_references() -> None:
    chunks = [
        (11, "Fig. 3. Geological map of the area with sampling points."),
        (12, "Sampling grid is shown in Fig. 3 and Fig. 9."),
        (13, "No pictures here."),
    ]
    conn = FakeConn(answer=returning_ids(100, chunks=chunks))
    item = {"n": 1, "label": "3", "file": "d.figures/fig-001.jpg", "page": 2,
            "caption": "Fig. 3. Geological map of the area", "width": 400, "height": 300}  # fmt: skip
    assert figures.store(conn, "d", [item]) == 1
    inserted = conn.queries("INSERT INTO figures")[0]
    assert inserted[0] == "d" and inserted[-1] == 11  # фрагмент с подписью
    refs = conn.queries("INSERT INTO figure_refs")
    assert {r[0] for r in refs} == {11, 12}  # подпись и «Fig. 3»; на 9 рисунка нет


def row(figure_id: int, chunk_id: int, n: int) -> tuple[Any, ...]:
    return (figure_id, "d", n, str(n), f"d.figures/fig-{n:03d}.jpg", 4, None,
            f"Fig. {n}. Map", 400, 300, chunk_id, "Title", 2024, chunk_id)  # fmt: skip


def test_for_answer_only_cited_in_order() -> None:
    conn = FakeConn(rows=[row(1, 501, 1), row(2, 502, 2), row(1, 502, 1)])
    sources = [{"n": 1, "chunk_id": 501}, {"n": 2, "chunk_id": 502}, {"n": 3, "chunk_id": 503}]
    chosen = figures.for_answer(conn, sources, [2, 1])
    assert [(f["id"], f["source"]) for f in chosen] == [(1, 1), (2, 2)]  # без повтора
    assert figures.for_answer(conn, sources, []) == []  # ответ ни на что не сослался
    line = server.answer_figures(conn, sources, [1])[0]
    assert line["url"] == "/api/figure?id=1" and line["line"] == "Рис. 1 к [1], с. 4. Fig. 1. Map"
    assert "Title" in ui.figure_caption(chosen[0]) and "к [1]" in ui.figure_caption(chosen[0])


def test_no_table_no_figures() -> None:
    def broken(sql: str, params: Any) -> Any:
        raise RuntimeError('relation "figures" does not exist')

    assert server.answer_figures(FakeConn(answer=broken), [{"n": 1, "chunk_id": 5}], [1]) == []


def test_page_shows_figures_after_answer() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'ev.type === "figures"' in page and "function renderFigures" in page
    assert "bot.append(renderFigures(figures))" in page


class Embedder:
    """Векторы по словам: вопрос о порфировой меди близок к подписи о ней, а не к карте."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "порфир" in t or "porphyry" in t.lower() else [0.0, 1.0] for t in texts
        ]


def test_own_first_per_doc_and_close_to_question() -> None:
    def fig(figure_id: int, doc: str, own: int, link: int, caption: str) -> tuple[Any, ...]:
        return (figure_id, doc, figure_id, str(figure_id), "f.jpg", 1, None, caption,
                400, 300, own, "T", 2024, link)  # fmt: skip

    rows = [
        fig(1, "a", 900, 501, "Magnetic map"),  # фрагмент [1] на него только ссылается
        fig(2, "a", 501, 501, "Porphyry copper setting"),  # подпись в самом фрагменте [1]
        fig(3, "a", 902, 501, "Porphyry section"),
        fig(4, "b", 502, 502, "Porphyry Cu model"),
    ]
    conn = FakeConn(rows=rows)
    sources = [{"n": 1, "chunk_id": 501}, {"n": 2, "chunk_id": 502}]
    plain = figures.for_answer(conn, sources, [1, 2])
    assert [f["id"] for f in plain] == [2, 4, 1]  # свои — вперёд; из статьи a — два
    close = figures.for_answer(conn, sources, [1, 2], question="порфир", embedder=Embedder())
    assert [f["id"] for f in close] == [2, 4]  # магнитная карта к вопросу не близка
