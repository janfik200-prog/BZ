"""Рисунок 1 отчёта: цепочка происхождения одной связи графа этой базы (факт 140722).

Все звенья — из базы (только чтение): публикация, страница и раздел фрагмента, место
цитаты в тексте фрагмента, связь словами статьи, тип и виды концов, голоса проверки,
территория по справочнику (georag/evidence/scope.py), число публикаций у пары концов.
Оформление — как у рисунка исходного отчёта.

    python research/base2059/figure1.py  →  research/base2059/figure1.png
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import psycopg  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

from georag.evidence import gate  # noqa: E402
from georag.evidence.scope import resolve_scope  # noqa: E402
from georag.graph import store  # noqa: E402

FACT = 140722
DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
OUT = HERE / "figure1.png"
TITLE = {140722: "Discovery, development and study of the Kirovskyi gold ore deposit of Amur province "
                 "(Amur region, Russia)"}

BLUE, VIOLET, GREEN, SAND = "#dce8f5", "#e6dff2", "#d3eae3", "#f5e6cf"
EDGE = "#1a1a1a"
DERIV = {"EXPLICIT": "по цитате", "ANAPHORA": "по соседним предложениям",
         "SECTION": "по заголовку раздела", "TITLE": "по названию статьи"}


def facts() -> dict:
    with psycopg.connect(DSN + "?options=-c%20default_transaction_read_only%3Don") as c:
        r = c.execute(
            """SELECT f.src, f.relation, f.dst, f.rel_type, f.rel_flip, f.src_kind, f.dst_kind, f.votes,
                      f.quote, f.src_key, f.dst_key, d.title, d.authors, d.year, d.doi,
                      ch.headings, ch.pages, ch.text, ch.ord
               FROM facts f JOIN documents d USING (doc_id) JOIN chunks ch ON ch.id = f.chunk_id
               WHERE f.id = %s""", (FACT,)).fetchone()
        keys = ("src relation dst rel_type rel_flip src_kind dst_kind votes quote src_key dst_key title "
                "authors year doi headings pages text ord").split()
        f = dict(zip(keys, r, strict=True))
        a, b = sorted([f["src_key"], f["dst_key"]])
        f["docs"] = c.execute(
            f"SELECT count(DISTINCT f.doc_id) FROM facts f WHERE {store.accepted(fresh=True)} "
            "AND least(f.src_key, f.dst_key) = %s AND greatest(f.src_key, f.dst_key) = %s", (a, b)).fetchone()[0]
    reg = gate.registry()
    sc = resolve_scope(f["quote"], reg, chunk_text=f["text"], headings=f["headings"] or [], title=f["title"])
    area = sorted(sc.region_ids)[0]
    chain = [reg.areas[x].name for x in reg.chain(area)]
    f["territory"] = (f"{chain[0]} ({DERIV[sc.derivation]}) →\n"
                      f"{', '.join(chain[1:])} (по справочнику)")
    start = f["text"].find(f["quote"])
    f["offsets"] = (start, start + len(f["quote"])) if start >= 0 else None
    return f


def author(name: str) -> str:
    """«V. Stepanov» → «Stepanov V.»."""
    parts = name.split()
    return f"{parts[-1]} {' '.join(parts[:-1])}" if len(parts) > 1 else name


def rows(f: dict) -> list[tuple[str, str, str]]:
    kinds = (f["dst_kind"], f["src_kind"]) if f["rel_flip"] else (f["src_kind"], f["dst_kind"])
    # Название в базе набрано заглавными — для подписи регистр восстановлен вручную.
    title = TITLE.get(FACT) or f["title"]
    where = f"С. {', '.join(map(str, f['pages']))} PDF, раздел «{f['headings'][-1]}»."
    if f["offsets"]:
        where += f" Смещения {f['offsets'][0]}–{f['offsets'][1]} в тексте фрагмента № {f['ord'] + 1} статьи."
    one = "Одна публикация служит основанием для этой записи." if f["docs"] == 1 else \
        f"Публикаций с этой парой объектов: {f['docs']}."
    return [
        ("Публикация", f"{', '.join(author(x) for x in f['authors'])} {title}. {f['year']}.\nDOI {f['doi']}", BLUE),
        ("Место цитаты", where, BLUE),
        ("Цитата", f"«{f['quote']}»", BLUE),
        ("Извлечённая\nсвязь", f"{f['src']} — {f['relation']} — {f['dst']}.", VIOLET),
        ("Нормализованный\nтип", f"Тип «{f['rel_type']}», {kinds[0]} → {kinds[1]}.", VIOLET),
        ("Проверка", "Код нашёл цитату дословно и упоминания обоих объектов,\nтип согласован с видами объектов.\n"
                     f"Ответы модели на контрольные вопросы — {f['votes']} совпадения из 4.", GREEN),
        ("Территория", f["territory"], SAND),
        ("Пометки", "В сохранённой связи нет пометок отрицания и предположения.", VIOLET),
        ("Источники", one, VIOLET),
    ]


def wrap(text: str, width: int = 62) -> str:
    return "\n".join(textwrap.fill(p, width) if len(p) > width else p for p in text.split("\n"))


def draw(items: list[tuple[str, str, str]]) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    fs, line, pad, gap = 15, 0.43, 0.32, 0.30
    texts = [wrap(t) for _, t, _ in items]
    heights = [max(t.count("\n") + 1, h.count("\n") + 1) * line + 2 * pad for (h, _, _), t in zip(items, texts)]
    W, total = 18.0, sum(heights) + gap * (len(items) - 1)
    fig = plt.figure(figsize=(W / 2, total / 2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, total)
    ax.axis("off")
    lx, lw, rx, rw = 0.1, 4.0, 4.45, W - 4.55
    y = total
    for i, ((head, _, color), text, h) in enumerate(zip(items, texts, heights)):
        y0 = y - h
        for x, w in ((lx, lw), (rx, rw)):
            ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=color, ec=EDGE, lw=1.6))
        ax.text(lx + lw / 2, y0 + h / 2, head, ha="center", va="center", fontsize=fs, fontweight="bold",
                linespacing=1.15)
        ax.text(rx + 0.22, y0 + h / 2, text, ha="left", va="center", fontsize=fs, linespacing=1.3)
        if i < len(items) - 1:
            cx = lx + lw / 2
            ax.add_patch(Polygon([(cx - 0.12, y0), (cx + 0.12, y0), (cx, y0 - 0.14)], closed=True, fc=EDGE, ec=EDGE))
        y = y0 - gap
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    draw(rows(facts()))
