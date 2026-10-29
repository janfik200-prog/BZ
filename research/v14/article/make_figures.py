"""Рисунки статьи v14: рис. 1 — путь факта от статьи до ответа с тремя раздельными
проверками и гейтом применимости (объединяет рис. 1 и 2 v13); рис. 2 — результаты V
(из results/metrics.json, ничего не вписывается руками)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def box(ax, x, y, w, h, text, bold=False, fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor=fill, edgecolor="black", linewidth=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, fontweight="bold" if bold else "normal")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color="black"))


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    w, h = 2.15, 1.05
    xs = [0.2 + i * 2.55 for i in range(5)]
    top = ["Статьи (PDF)\nразбор Docling,\nфрагменты, с. N", "Первый этап без модели:\nпочти-дубли (MinHash\n+ защищённые отличия),\nсловарь + spaCy",
           "Модель-строитель:\nсвязи между предметами,\nцитата — кодом\nпо № предложения", "(A) Цитата верна:\nсмещения и отпечатки\nв замороженном разборе",
           "(B) Цитата утверждает\nсвязь: confirm.py\n+ правила направления,\nотрицания"]
    for x, t in zip(xs, top):
        box(ax, x, 3.55, w, h, t)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, 4.08), (b, 4.08))
    bot = ["Вопрос → QueryFrame:\nтерритория, тип,\nквантор, режим", "Область действия факта:\nцитата → анафора →\nраздел → название\n(справочник территорий)",
           "(C) Применимость:\nDIRECT / ANALOG /\nOUT_OF_SCOPE /\nUNRESOLVED + причины", "Ответ и граф:\nв ответ — только DIRECT;\nаналоги — отдельно;\nсверх 24 — NOT_CHECKED",
           "Досье геолога:\n4 секции, цитата, DOI,\nстраница, причина,\nжурнал правок"]
    fills = ["white", "white", "#e8e8e8", "white", "white"]
    for x, t, f in zip(xs, bot, fills):
        box(ax, x, 0.95, w, 1.2, t, bold=(f != "white"), fill=f)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, 1.55), (b, 1.55))
    # (B) → (C)
    ax.plot([xs[4] + w / 2, xs[4] + w / 2, xs[2] + w / 2], [3.55, 2.75, 2.75], color="black", linewidth=1.6)
    arrow(ax, (xs[2] + w / 2, 2.75), (xs[2] + w / 2, 2.15))
    ax.text(6.5, 0.35, "Верная цитата (A) и поддержка (B) не делают факт применимым к району вопроса (C): три проверки раздельны",
            ha="center", fontsize=10.5, style="italic")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_pipeline.png", dpi=220)
    plt.close(fig)


def fig2() -> None:
    m = json.loads((RES / "metrics.json").read_text("utf-8"))
    sup = json.loads((RES / "support_rules.json").read_text("utf-8"))
    names = ["V0", "V1", "V2", "V3"]
    rows = [m[f"TEST:{n}"] for n in names]
    v3s = sup["metrics"]["TEST"][sup["chosen_on_dev"]]
    v2s = sup["v2_same_support"]["TEST"][sup["chosen_on_dev"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r, n, mk in zip(rows, names, ("^", "v", "s", "o")):
        ax.scatter(r["direct_coverage"], r["ftr"], s=70, marker=mk, color="black")
        ax.annotate(f"{n}", (r["direct_coverage"], r["ftr"]), textcoords="offset points", xytext=(6, 5), fontsize=10)
    for r, n in ((v2s, "V2+S"), (v3s, "V3+S")):
        ax.scatter(r["direct_coverage"], r["ftr"], s=70, marker="D", facecolors="white", edgecolors="black")
        ax.annotate(n, (r["direct_coverage"], r["ftr"]), textcoords="offset points", xytext=(6, -12), fontsize=10)
    ax.set_xlabel("Охват прямых ответов (TEST, 20 вопросов)")
    ax.set_ylabel("Доля ложного переноса FTR (TEST, 112 чужих пар)")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.03, 0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_results.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("ok")
