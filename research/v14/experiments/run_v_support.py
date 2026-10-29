"""WP04 × WP11: V3 с разными инструментами поддержки; правило выбирается только на DEV.

Кандидаты (S — метка поддержки, которую получает V3):
  votes3        — confirm.py ≥ 3 из 4 (основной протокол v14.1);
  llm           — только run_support_llm.py (ENTAILED с верным направлением);
  votes3_or_llm — любое из двух;
  votesK_and_llm (K = 1, 2) — ENTAILED у модели и хотя бы K голосов confirm.py.

Критерий выбора, заданный до просмотра результатов: наибольший DEV-охват прямых
ответов при DEV-избирательном риске не выше, чем у votes3, + 0,05 и без роста DEV-FTR.
Затем выбранное правило — один раз на TEST. Пишет results/support_rules.json и
results/raw_predictions_support.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg  # noqa: E402

import run_v as R  # noqa: E402
from georag.evidence import applicability as ap  # noqa: E402
from georag.evidence import metrics as M  # noqa: E402
from georag.evidence.claims import claim_from_fact  # noqa: E402
from georag.evidence.contracts import Scope, SupportAssessment  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.provenance import make_evidence, snapshot_from_db  # noqa: E402
from georag.evidence.scope import resolve_scope  # noqa: E402
from georag.evidence.support import assess_support  # noqa: E402

RULES = ("votes3", "llm", "votes3_or_llm", "votes1_and_llm", "votes2_and_llm")


def entailed(rule: str, votes: int, llm: str) -> bool:
    v = votes or 0
    return {
        "votes3": v >= 3,
        "llm": llm == "ENTAILED",
        "votes3_or_llm": v >= 3 or llm == "ENTAILED",
        "votes1_and_llm": v >= 1 and llm == "ENTAILED",
        "votes2_and_llm": v >= 2 and llm == "ENTAILED",
    }[rule]


def main() -> None:
    R.check_freeze()
    reg = load(R.DATA / "regional_registry.csv", R.DATA / "registry_aliases.csv")
    queries = {q["query_id"]: R.frame(q, reg) for q in json.loads((R.DATA / "queries.json").read_text("utf-8"))}
    pairs = [json.loads(line) for line in (R.DATA / "pairs.jsonl").open(encoding="utf-8")]
    gold = list(csv.DictReader((R.DATA / "gold_annotations.csv").open(encoding="utf-8")))
    llm = {r["fact_id"]: r["label"] for r in map(json.loads, (R.RES / "raw_support_llm.jsonl").open(encoding="utf-8"))}
    with psycopg.connect(R.DSN) as conn:
        docs = {d: snapshot_from_db(conn, d) for d in {p["doc_id"] for p in pairs}}
    preds: dict[str, list[dict]] = {r: [] for r in RULES}
    preds_v2: dict[str, list[dict]] = {r: [] for r in RULES}  # та же поддержка у V2 — для честного сравнения
    for p in pairs:
        doc = docs[p["doc_id"]]
        chunk = doc.chunks[str(p["chunk_id"])]
        ev = make_evidence(doc, str(p["chunk_id"]), p["quote"])
        sc = resolve_scope(p["quote"], reg, chunk_text=chunk.text, headings=chunk.headings, title=doc.title)
        sc_naive = resolve_scope(p["quote"], reg, chunk_text=chunk.text, headings=chunk.headings, title=doc.title, mode="naive")
        fact = {"id": p["fact_id"], "src": p["src"], "relation": p["relation"], "dst": p["dst"], "quote": p["quote"],
                "votes": p["votes"], "accepted": p["accepted"], "model": p["model"]}
        claim = claim_from_fact(fact, ev.evidence_id, scope=Scope())
        base = assess_support(claim, ev, document=doc, votes=p["votes"])
        for rule in RULES:
            if base.label == "INVALID_EVIDENCE":
                sup = base
            else:
                ok = entailed(rule, p["votes"], llm[p["fact_id"]])
                sup = SupportAssessment(base.support_id, claim.claim_id, ev.evidence_id, "ENTAILED" if ok else "PARTIAL",
                                        (f"SUPPORT_RULE_{rule}",), "LLM", f"support-{rule}", True, None, p["split"])
            d = ap.decide_applicability(queries[p["query_id"]], claim, sup, reg, scope=sc, config=ap.V3)
            preds[rule].append({"baseline": f"V3[{rule}]", "sample_id": p["sample_id"], "query_id": p["query_id"],
                                "split": p["split"], "predicted_status": d.status, "reason_codes": ";".join(d.reason_codes)})
            d2 = ap.decide_applicability(queries[p["query_id"]], claim, sup, reg, scope=sc_naive, config=ap.V2)
            preds_v2[rule].append({"baseline": f"V2[{rule}]", "sample_id": p["sample_id"], "query_id": p["query_id"],
                                   "split": p["split"], "predicted_status": d2.status, "reason_codes": ";".join(d2.reason_codes)})
    report = {s: {r: M.compute_metrics(preds[r], gold, r, s).row() for r in RULES} for s in ("DEV", "TEST")}
    report_v2 = {s: {r: M.compute_metrics(preds_v2[r], gold, r, s).row() for r in RULES} for s in ("DEV", "TEST")}
    dev = report["DEV"]
    ref = dev["votes3"]
    allowed = [r for r in RULES if (dev[r]["selective_risk"] or 0) <= (ref["selective_risk"] or 0) + 0.05
               and (dev[r]["ftr"] or 0) <= (ref["ftr"] or 0)]
    chosen = max(allowed, key=lambda r: (dev[r]["direct_coverage"] or 0, -(dev[r]["selective_risk"] or 0)))
    # Поддержка по разметке: точность и полнота каждого правила на уникальных фактах DEV и TEST.
    facts = {}
    for g in gold:
        facts.setdefault(g["fact_id"], g)
    vot = {p["fact_id"]: p["votes"] for p in pairs}
    sup_quality = {}
    for s in ("DEV", "TEST", "ALL"):
        for rule in RULES:
            tp = fp = fn = 0
            for fid, g in facts.items():
                if s != "ALL" and g["split"] != s:
                    continue
                e, a = g["gold_support"] == "ENTAILED", entailed(rule, vot[int(fid)], llm[int(fid)])
                tp += e and a
                fp += (not e) and a
                fn += e and not a
            sup_quality[f"{s}:{rule}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                                          "recall": tp / (tp + fn) if tp + fn else None}
    boot = {s: M.paired_bootstrap(preds[chosen], preds_v2[chosen], gold, s) for s in ("DEV", "TEST")}
    out = {"v2_same_support": report_v2, "bootstrap_V3_vs_V2_same_support": boot, "criterion": "max DEV coverage s.t. DEV selective_risk <= votes3 + 0.05 and DEV FTR <= votes3",
           "chosen_on_dev": chosen, "allowed": allowed, "metrics": report, "support_quality": sup_quality}
    (R.RES / "support_rules.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    with (R.RES / "raw_predictions_support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(preds["votes3"][0]))
        w.writeheader()
        for r in RULES:
            w.writerows(preds[r])
            w.writerows(preds_v2[r])
    for s in ("DEV", "TEST"):
        print(s)
        for r in RULES:
            m = report[s][r]
            q = sup_quality[f"{s}:{r}"]
            print(f"  {r:16s} FTR {m['ftr']:.3f} ({m['false_direct_foreign']}/{m['n_foreign']})  cov {m['direct_coverage']:.3f} "
                  f"({m['covered']}/{m['n_answerable']})  SR {m['selective_risk'] if m['selective_risk'] is None else round(m['selective_risk'],3)} "
                  f"({m['n_false_direct']}/{m['n_direct_pred']})  support P={q['precision']:.3f} R={q['recall']:.3f}")
    print("chosen on DEV:", chosen)
    for s in ("DEV", "TEST"):
        m = report_v2[s][chosen]
        print(f"  V2[{chosen}] {s}: FTR {m['ftr']:.3f} ({m['false_direct_foreign']}/{m['n_foreign']}) cov {m['direct_coverage']:.3f} "
              f"({m['covered']}/{m['n_answerable']}) SR {m['selective_risk']:.3f} ({m['n_false_direct']}/{m['n_direct_pred']})")
        print("   bootstrap V3-V2:", {k: boot[s][k] for k in ("delta_ftr", "delta_ftr_ci95", "delta_coverage", "delta_coverage_ci95")})


if __name__ == "__main__":
    main()
