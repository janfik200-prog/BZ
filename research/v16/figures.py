"""Таблицы и рисунки для статьи по results/*.json и *.csv v16.

    figures.py → results/tables/*.csv|.md, results/figures/*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TAB, FIG = RES / "tables", RES / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # категориальные цвета 1 и 2 (dataviz, palette.md)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"B0": "B0 только название", "B1": "B1 векторный", "B2": "B2 + название статьи",
         "B2t": "B2t + текст статьи", "B3": "B3 предложенный", "B4": "B4 гибридный + место",
         "B3+gf": "B3 + близкие связи графа", "B3+gn": "B3 + обход графа", "B3+gn1": "B3 + обход на шаг",
         "B3+gn_raw": "B3 + обход без нормализации", "B3+gf_untyped": "B3 + связи без типа",
         "B3+gf_cand": "B3 + кандидаты", "B1+gf": "B1 + близкие связи графа", "B1+gn": "B1 + обход графа",
         "B1+gn1": "B1 + обход на шаг"}


def d(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def write(name: str, head: list[str], rows: list[list]) -> None:
    with (TAB / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([head, *rows])
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n", "utf-8")


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})

    audit = json.loads((RES / "relation_pipeline_summary.json").read_text("utf-8"))
    write("table_relation_funnel", ["Этап", "Вход", "Прошло", "Иной итог"], [
        ["Предложено моделью", "—", audit["proposed"], "нижняя граница"],
        ["Проверка кодом", audit["proposed"], audit["stored"], f"{audit['rejected_by_code']} отклонены"],
        ["Подтверждение", audit["stored"], audit["confirmed"], f"{audit['candidates']} кандидаты"],
        ["Типизация", audit["confirmed"], audit["graph"], f"{audit['excluded_typing']} без типа"],
    ])

    r = json.loads((RES / "retrieval_summary.json").read_text("utf-8"))["by_set"]
    al = r["all"]
    variants = ["B0", "B1", "B2", "B2t", "B3", "B4", "B3+gf", "B3+gn", "B3+gn1", "B3+gn_raw", "B3+gf_untyped",
                "B3+gf_cand"]
    write("table_retrieval_territory_graph",
          ["Вариант", "Статья Top-5", "Статья Top-1", "Фрагмент Top-5", "MRR@10", "Время, с (медиана)"],
          [[NAMES[v], d(al[v]["top5"]), d(al[v]["first"]), d(al[v]["frag"]), d(al[v]["mrr10"]),
            f"{al[v]['seconds_median']:.3f}".replace(".", ",")] for v in variants])
    pairs = [k for k in al if " - " in k]
    write("table_retrieval_differences", ["Сравнение", "Метрика", "Разность", "95 % интервал"],
          [[k, m, d(al[k][m][0]), f"{d(al[k][m][1])} … {d(al[k][m][2])}"] for k in pairs
           for m in ("first", "frag", "mrr10")])

    lab = json.loads((RES / "labeling_scores.json").read_text("utf-8"))
    with (RES / "prefilter_evaluation.csv").open(encoding="utf-8") as fh:
        pre = list(csv.DictReader(fh))
    write("table_prefilter_thresholds",
          ["Порог (терминов)", "Отсеяно фрагментов", "Экономия токенов", "Полнота", "95 % интервал",
           "Отсеяно фрагментов со связями (оценка)"],
          [[p["threshold"], p["filtered_chunks"], d(float(p["token_saving"])), d(float(p["relation_recall"])),
            f"{d(float(p['recall_ci_low']))} … {d(float(p['recall_ci_high']))}", p["missed_relations"]]
           for p in pre])
    write("table_prefilter_by_terms", ["Терминов", "Размечено", "Со связями", "Доля", "95 % интервал",
                                       "Фрагментов в корпусе"],
          [[b, v["labelled"], v["with_relations"], d(v["share"]), f"{d(v['ci'][0])} … {d(v['ci'][1])}",
            v["population"]] for b, v in lab["filter_by_terms"].items()])

    # Рис. 1: компромисс фильтра — одна шкала (доли), две серии, подписи значений.
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
    th = [int(p["threshold"]) for p in pre]
    save = [100 * float(p["token_saving"]) for p in pre]
    lost = [100 * (1 - float(p["relation_recall"])) for p in pre]
    w = 0.36
    for off, vals, color, label in ((-w / 2, save, BLUE, "экономия токенов"),
                                    (w / 2, lost, ORANGE, "потеря фрагментов со связями")):
        bars = ax.bar([t + off for t in th], vals, w - 0.03, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f} %", ha="center", fontsize=8, color=INK)
    ax.set_xticks(th, [f"< {t}" for t in th])
    ax.set_xlabel("порог: фрагмент отсеивается, если терминов словаря меньше")
    ax.set_ylabel("%")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "prefilter_tradeoff.png")
    plt.close(fig)

    # Рис. 2: статья Top-1 по вариантам поиска (одна серия — без легенды).
    shown = ["B0", "B4", "B2", "B1", "B2t", "B3", "B3+gf", "B3+gn"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    vals = [al[v]["first"] for v in shown]
    bars = ax.barh(range(len(shown)), vals, 0.6, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, d(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(shown)), [NAMES[v] for v in shown])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"статья Top-1, вопросов — {al['n']}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",")))
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval_top1.png")
    plt.close(fig)
    print("ok", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
