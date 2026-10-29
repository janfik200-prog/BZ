"""WP04 / WP12: качество инструментов на разметке и графики эксперимента V.

1. Поддержка источником: голоса confirm.py (≥ k из 4) против gold_support (ENTAILED vs
   остальное) — по уникальным фактам выборки: точность, полнота, матрица, цена ложного
   подтверждения (доля принятых неверных связей). Признаки цитаты (полярность,
   модальность, квантор) против разметки.
2. Разбор вопроса: gate.frame_question по тексту 35 вопросов против рамок queries.json.
3. Графики: plots/risk_coverage.png (охват прямых ответов против избирательного риска,
   кривые по порогу голосов), plots/ftr_coverage.png (FTR и охват с 95 % интервалами).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from georag.evidence import gate  # noqa: E402
from georag.evidence.claims import features  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.metrics import wilson  # noqa: E402

V14 = ROOT / "research" / "v14"
DATA, RES, PLOTS = V14 / "data", V14 / "results", V14 / "plots"


def support_eval(gold: list[dict], pairs: dict[str, dict]) -> dict:
    facts = {}
    for g in gold:
        fid = g["fact_id"]
        facts.setdefault(fid, (g, pairs[g["sample_id"]]))
    out = {"n_facts": len(facts), "by_threshold": {}}
    for k in (1, 2, 3, 4):
        tp = fp = fn = tn = 0
        for g, p in facts.values():
            ent = g["gold_support"] == "ENTAILED"
            acc = (p["votes"] or 0) >= k
            tp += ent and acc
            fp += (not ent) and acc
            fn += ent and not acc
            tn += (not ent) and not acc
        out["by_threshold"][k] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "precision_ci": wilson(tp, tp + fp),
            "recall": tp / (tp + fn) if tp + fn else None, "recall_ci": wilson(tp, tp + fn),
            "false_confirmation_rate": fp / (fp + tn) if fp + tn else None,
        }
    legacy = Counter()
    for g, p in facts.values():
        legacy[(g["gold_support"], bool(p["accepted"]))] += 1
    out["legacy_graph_acceptance_vs_gold"] = {f"{a}|accepted={b}": v for (a, b), v in sorted(legacy.items())}
    # Признаки цитаты против разметки (по уникальным фактам).
    for field, gold_field in (("polarity", "polarity"), ("modality", "modality"), ("quantifier", "quantifier")):
        conf = Counter()
        for g, p in facts.values():
            f = features(p["quote"], p["relation"])
            conf[(g[gold_field], getattr(f, field))] += 1
        agree = sum(v for (a, b), v in conf.items() if a == b or (field == "modality" and {a, b} <= {"ASSERTED", "REPORTED"}))
        out[f"{field}_agreement"] = {"agree": agree, "n": len(facts), "rate": agree / len(facts),
                                     "confusion": {f"gold={a}|pred={b}": v for (a, b), v in sorted(conf.items())}}
    return out


def frame_eval(queries: list[dict], reg) -> dict:
    rows, ok = [], Counter()
    for q in queries:
        f = gate.frame_question(q["raw_text"], reg, q["query_id"])
        checks = {
            "region": f.target_region_id == q["target_region_id"],
            "quantifier": (f.requested_quantifier in ("TYPICAL", "ALL")) == (q["requested_quantifier"] in ("TYPICAL", "ALL")),
            "mode": f.requested_mode == q["requested_mode"],
            "analogies": f.allow_analogies == q["allow_analogies"],
            "deposit": bool(f.target_deposit_type and "gold" in f.target_deposit_type),
        }
        ok.update(k for k, v in checks.items() if v)
        rows.append({"query_id": q["query_id"], "parsed_region": f.target_region_id, "gold_region": q["target_region_id"],
                     "parsed_quantifier": f.requested_quantifier, "gold_quantifier": q["requested_quantifier"],
                     "parsed_mode": f.requested_mode, "gold_mode": q["requested_mode"], **{f"ok_{k}": v for k, v in checks.items()}})
    with (RES / "frame_parse.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(queries)
    return {k: {"ok": ok[k], "n": n, "rate": ok[k] / n} for k in ("region", "quantifier", "mode", "analogies", "deposit")}


def plots(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, split in zip(axes, ("DEV", "TEST")):
        for fam, color, marker in (("V2", "#4C72B0", "s"), ("V3", "#C44E52", "o")):
            pts = []
            for k in (1, 2, 3, 4):
                name = fam if k == 3 else f"{fam}@votes>={k}"
                r = report[f"{split}:{name}"]
                if r["selective_risk"] is not None:
                    pts.append((r["direct_coverage"], r["selective_risk"], k))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color, label=f"{fam} (порог голосов 1–4)")
            for x, y, k in pts:
                ax.annotate(f"≥{k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)
        for name, color in (("V0", "#555555"), ("V1", "#55A868")):
            r = report[f"{split}:{name}"]
            ax.scatter([r["direct_coverage"]], [r["selective_risk"]], color=color, marker="^", s=60, label=name, zorder=3)
        ax.set_title(f"{split}: охват прямых ответов и риск")
        ax.set_xlabel("DIRECT_coverage (доля вопросов с верным DIRECT)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Избирательный риск (неверные DIRECT / все DIRECT)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "risk_coverage.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    names = ["V0", "V1", "V2", "V3"]
    for ax, metric, ci, title in ((axes[0], "ftr", "ftr_ci", "FTR — ложный перенос (чужие пары)"),
                                  (axes[1], "direct_coverage", "coverage_ci", "DIRECT_coverage — охват")):
        for i, split in enumerate(("DEV", "TEST")):
            vals = [report[f"{split}:{n}"][metric] or 0 for n in names]
            lo = [max(0.0, v - (report[f"{split}:{n}"][ci][0] or 0)) for v, n in zip(vals, names)]
            hi = [max(0.0, (report[f"{split}:{n}"][ci][1] or 0) - v) for v, n in zip(vals, names)]
            xs = [j + (i - 0.5) * 0.36 for j in range(len(names))]
            ax.bar(xs, vals, width=0.34, yerr=[lo, hi], capsize=3, label=split, color=("#8DA0CB" if i == 0 else "#FC8D62"))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "ftr_coverage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    reg = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
    gold = list(csv.DictReader((DATA / "gold_annotations.csv").open(encoding="utf-8")))
    pairs = {json.loads(l)["sample_id"]: json.loads(l) for l in (DATA / "pairs.jsonl").open(encoding="utf-8")}
    queries = json.loads((DATA / "queries.json").read_text("utf-8"))
    report = json.loads((RES / "metrics.json").read_text("utf-8"))
    out = {"support": support_eval(gold, pairs), "frame_parse": frame_eval(queries, reg)}
    (RES / "support_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    with (RES / "support_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["min_votes", "tp", "fp", "fn", "tn", "precision", "recall", "false_confirmation_rate"])
        for k, r in out["support"]["by_threshold"].items():
            w.writerow([k, r["tp"], r["fp"], r["fn"], r["tn"], r["precision"], r["recall"], r["false_confirmation_rate"]])
    plots(report)
    print(json.dumps({"support": {k: {x: v[x] for x in ("tp", "fp", "fn", "tn", "precision", "recall")}
                                  for k, v in out["support"]["by_threshold"].items()},
                      "legacy": out["support"]["legacy_graph_acceptance_vs_gold"],
                      "features": {f: out["support"][f"{f}_agreement"]["rate"] for f in ("polarity", "modality", "quantifier")},
                      "frame_parse": out["frame_parse"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
