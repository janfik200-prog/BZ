"""Рис. 1 редакции v15: последовательность создания базы знаний (без проверок применимости)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="black", linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="black"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h = 2.15, 1.1
    xs = [0.2 + i * 2.55 for i in range(5)]
    top = ["Научные статьи\n(PDF, русский\nи английский)", "Разбор текста:\nпорядок чтения,\nтаблицы, формулы",
           "Фрагменты\nпо абзацам\nи разделам", "Векторы смысла\nи словарный\nиндекс", "Хранилище:\nстатьи, фрагменты,\nграф знаний"]
    for x, t in zip(xs, top):
        box(ax, x, 3.7, w, h, t, "#eeeeee" if t.startswith("Хранилище") else "white")
    for a, b in zip(xs[:4], xs[1:5]):
        arrow(ax, (a + w, 4.25), (b, 4.25))
    bot = ["Отбор фрагментов\nбез модели: повторы,\nязык, термины", "Предметы:\nсловарь терминов\nи названия мест",
           "Языковая модель:\nсвязи между\nпредметами", "Граф знаний: типы,\nединые названия,\nтемы-кластеры", "Поиск: по смыслу,\nпо словам,\nпо графу"]
    for x, t in zip(xs, bot):
        box(ax, x, 1.15, w, h, t, "#eeeeee" if t.startswith("Поиск") else "white")
    for a, b in zip(xs[:4], xs[1:5]):
        arrow(ax, (a + w, 1.7), (b, 1.7))
    # фрагменты → отбор
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[0] + w / 2], [3.7, 3.0, 3.0], color="black", linewidth=1.5)
    arrow(ax, (xs[0] + w / 2, 3.0), (xs[0] + w / 2, 2.25))
    # граф → хранилище; хранилище → поиск
    arrow(ax, (xs[3] + w / 2, 2.25), (xs[4] + 0.45, 3.7))
    arrow(ax, (xs[4] + w - 0.45, 3.7), (xs[4] + w - 0.45, 2.25))
    ax.text(6.5, 0.45, "Всё работает на одном компьютере: база данных, модель векторов, разбор PDF и языковая модель",
            ha="center", fontsize=11, style="italic")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_kb_pipeline.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("ok")
