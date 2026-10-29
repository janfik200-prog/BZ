"""WP11–WP12, эксперимент V: применимость на одних и тех же парах (факт, вопрос).

Все методы получают одинаковые пары из data/pairs.jsonl и одинаковые рамки вопросов из
data/queries.json; различается только правило допуска (applicability.PolicyConfig):

  V0 — старый граф: факт в графе (≥ 3 из 4 голосов confirm.py или повтор в другой
       статье, store.accepted) — DIRECT; территория не проверяется;
  V1 — V0 + наивное совпадение названия территории вопроса (с синонимами справочника)
       в цитате, заголовке раздела или названии статьи;
  V2 — справочник (вложенность) + тип оруденения + поддержка источником; область — по
       цитате, иначе по названию статьи без проверки общности;
  V3 — V2 + квантор/модальность/полярность + проверяемый контекст (анафора со span,
       раздел, название статьи только конкретное и без конфликта с фрагментом).

Абляции V3 и кривая «риск — охват» по порогу голосов (1–4). Порог и правила выбираются
по DEV; TEST — один замороженный протокол. Результаты: results/raw_predictions.csv и
.jsonl (все решения построчно), metrics.csv/json, bootstrap.json, confusion.json,
error_cases.csv/md, risk_coverage.csv; plots/*.png.

    .venv/Scripts/python.exe research/v14/experiments/run_v.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.evidence import applicability as ap  # noqa: E402
from georag.evidence import metrics as M  # noqa: E402
from georag.evidence.claims import claim_from_fact  # noqa: E402
from georag.evidence.contracts import QueryFrame, Scope, validate  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.provenance import make_evidence, snapshot_from_db  # noqa: E402
from georag.evidence.scope import resolve_scope  # noqa: E402
from georag.evidence.support import assess_support  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
V14 = ROOT / "research" / "v14"
DATA, RES, PLOTS = V14 / "data", V14 / "results", V14 / "plots"
RUN_ID = "V-" + time.strftime("%Y%m%d-%H%M%S")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_freeze() -> dict:
    freeze = json.loads((RES / "FREEZE.json").read_text("utf-8"))
    paths = {"gold_annotations.csv": DATA / "gold_annotations.csv", "pairs.jsonl": DATA / "pairs.jsonl",
             "queries.json": DATA / "queries.json", "regional_registry.csv": DATA / "regional_registry.csv",
             "registry_aliases.csv": DATA / "registry_aliases.csv"}
    for name, path in paths.items():
        if sha(path) != freeze["sha256"][name]:
            raise SystemExit(f"{name} изменён после заморозки — прогон не считается")
    return freeze


def frame(q: dict, reg) -> QueryFrame:
    return QueryFrame(
        query_id=q["query_id"], raw_text=q["raw_text"], target_region_id=q["target_region_id"],
        target_region_name=reg.areas[q["target_region_id"]].name, target_deposit_type=q["target_deposit_type"],
        requested_polarity=q["requested_polarity"], requested_quantifier=q["requested_quantifier"],
        allow_analogies=q["allow_analogies"], requested_mode=q["requested_mode"],
    )


def methods() -> dict[str, tuple[ap.PolicyConfig, str, int]]:
    """Имя → (политика, режим области действия, порог голосов поддержки)."""
    out = {"V0": (ap.V0, "full", 3), "V1": (ap.V1, "full", 3), "V2": (ap.V2, "naive", 3), "V3": (ap.V3, "full", 3)}
    for name, cfg in ap.ABLATIONS.items():
        mode = {"V3-anaphora": "no_anaphora", "V3-title_rules": "naive"}.get(name, "full")
        out[name] = (cfg, mode, 3)
    for k in (1, 2, 4):  # кривая риск — охват: порог голосов
        out[f"V3@votes>={k}"] = (replace(ap.V3, name=f"V3@votes>={k}"), "full", k)
        out[f"V2@votes>={k}"] = (replace(ap.V2, name=f"V2@votes>={k}"), "naive", k)
    return out


def main() -> None:
    freeze = check_freeze()
    reg = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
    queries = {q["query_id"]: q for q in json.loads((DATA / "queries.json").read_text("utf-8"))}
    frames = {k: frame(q, reg) for k, q in queries.items()}
    for f in frames.values():
        validate("query", f.to_json())
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open(encoding="utf-8")]
    gold = list(csv.DictReader((DATA / "gold_annotations.csv").open(encoding="utf-8")))
    gold_by = {g["sample_id"]: g for g in gold}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    docs = {}
    with psycopg.connect(DSN) as conn:
        for p in pairs:
            if p["doc_id"] not in docs:
                docs[p["doc_id"]] = snapshot_from_db(conn, p["doc_id"])

    meth = methods()
    raw = []
    scope_cache: dict[tuple[str, str], object] = {}
    started = time.perf_counter()
    for p in pairs:
        doc = docs[p["doc_id"]]
        chunk = doc.chunks[str(p["chunk_id"])]
        ev = make_evidence(doc, str(p["chunk_id"]), p["quote"])
        fact = {"id": p["fact_id"], "src": p["src"], "relation": p["relation"], "dst": p["dst"], "quote": p["quote"],
                "votes": p["votes"], "accepted": p["accepted"], "model": p["model"], "source_family": p["source_family"]}
        q = frames[p["query_id"]]
        texts = [p["quote"], " ".join(p["headings"]), p["title"]]
        for name, (cfg, mode, k) in meth.items():
            key = (p["sample_id"], mode)
            if key not in scope_cache:
                scope_cache[key] = resolve_scope(p["quote"], reg, chunk_text=chunk.text, headings=chunk.headings,
                                                 title=doc.title, mode=mode)
            sc = scope_cache[key]
            claim = claim_from_fact(fact, ev.evidence_id, scope=Scope(region_id=(sc.region_ids or (None,))[0]))
            sup = assess_support(claim, ev, document=doc, votes=p["votes"], min_votes=k, split=p["split"])
            t0 = time.perf_counter()
            d = ap.decide_applicability(q, claim, sup, reg, scope=sc, config=cfg, name_texts=texts)
            latency = (time.perf_counter() - t0) * 1000
            raw.append({
                "run_id": RUN_ID, "baseline": name, "sample_id": p["sample_id"], "query_id": p["query_id"],
                "source_family": p["source_family"], "split": p["split"], "predicted_status": d.status,
                "model_support": sup.label, "policy_version": d.policy_version, "registry_version": d.registry_version,
                "model_id": p["model"], "reason_codes": ";".join(d.reason_codes), "evidence_id": ev.evidence_id,
                "latency_ms": round(latency, 3), "tokens_in": 0, "tokens_out": 0,
                "scope_regions": ";".join(sc.region_ids), "scope_derivation": sc.derivation,
                "claim_polarity": claim.polarity, "claim_modality": claim.modality, "claim_quantifier": claim.quantifier,
                "decision_id": d.decision_id, "git_sha": git,
            })
    elapsed = time.perf_counter() - started

    RES.mkdir(parents=True, exist_ok=True)
    cols = list(raw[0])
    with (RES / "raw_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(raw)
    with (RES / "raw_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_method = defaultdict(list)
    for r in raw:
        by_method[r["baseline"]].append(r)
    metrics_rows, report = [], {}
    for split in ("DEV", "TEST"):
        for name, rows in by_method.items():
            m = M.compute_metrics(rows, gold, name, split)
            row = m.row()
            row["run_id"] = RUN_ID
            metrics_rows.append(row)
            report[f"{split}:{name}"] = row
    with (RES / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
        w.writeheader()
        for r in metrics_rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    (RES / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")

    boot = {}
    for split in ("DEV", "TEST"):
        for other in ("V0", "V1", "V2"):
            boot[f"{split}:V3-vs-{other}"] = M.paired_bootstrap(by_method["V3"], by_method[other], gold, split)
        for abl in ap.ABLATIONS:
            boot[f"{split}:V3-vs-{abl}"] = M.paired_bootstrap(by_method["V3"], by_method[abl], gold, split, n_boot=2000)
    (RES / "bootstrap.json").write_text(json.dumps(boot, ensure_ascii=False, indent=1), "utf-8")
    conf = {f"{s}:{n}": M.confusion(by_method[n], gold, s) for s in ("DEV", "TEST") for n in ("V0", "V1", "V2", "V3")}
    (RES / "confusion.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), "utf-8")

    # Кривая «риск — охват»: точки по порогу голосов для V2 и V3, плюс V0/V1.
    with (RES / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "family", "min_votes", "direct_coverage", "selective_risk", "ftr", "n_direct_pred"])
        for split in ("DEV", "TEST"):
            for name in by_method:
                if not (name in ("V0", "V1", "V2", "V3") or "@votes" in name):
                    continue
                r = report[f"{split}:{name}"]
                fam = "V3" if name.startswith("V3") else "V2" if name.startswith("V2") else name
                k = int(name.split(">=")[1]) if "@votes" in name else 3
                w.writerow([split, name, fam, k, r["direct_coverage"], r["selective_risk"], r["ftr"], r["n_direct_pred"]])

    # Ошибки V3 и V0 на TEST — построчно, с причинами (для разбора).
    errors = []
    for name in ("V3", "V2", "V0"):
        for r in by_method[name]:
            g = gold_by[r["sample_id"]]
            if r["predicted_status"] != g["gold_applicability"] and (r["predicted_status"] == "DIRECT" or g["gold_applicability"] == "DIRECT"):
                kind = ("FALSE_DIRECT_FOREIGN" if g["gold_applicability"] in ("OUT_OF_SCOPE", "ANALOG") and r["predicted_status"] == "DIRECT"
                        else "FALSE_DIRECT_OTHER" if r["predicted_status"] == "DIRECT" else "MISSED_DIRECT")
                errors.append({"sample_id": r["sample_id"], "baseline": name, "split": r["split"], "error_category": kind,
                               "gold_label": g["gold_applicability"], "predicted_label": r["predicted_status"],
                               "source_id": g["source_id"], "evidence_id": r["evidence_id"], "reason_codes": r["reason_codes"],
                               "scope_regions": r["scope_regions"], "scope_derivation": r["scope_derivation"],
                               "gold_region": g["region_id"], "gold_support": g["gold_support"], "gold_notes": g["reason_notes"],
                               "quote": g["quote"][:300]})
    with (RES / "error_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0]) if errors else ["sample_id"])
        w.writeheader()
        w.writerows(errors)

    summary = {
        "run_id": RUN_ID, "git_sha": git, "freeze_git_head": freeze["git_head_at_freeze"],
        "n_pairs": len(pairs), "n_methods": len(meth), "decisions": len(raw), "elapsed_s": round(elapsed, 2),
        "registry_version": reg.version, "policy_version": ap.POLICY_VERSION,
        "errors_by_method": {n: dict(Counter((e["split"], e["error_category"]) for e in errors if e["baseline"] == n).items()) for n in ("V0", "V2", "V3")},
    }
    summary["errors_by_method"] = {n: {f"{s}:{c}": v for (s, c), v in d.items()} for n, d in summary["errors_by_method"].items()}
    (RES / "run_v_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for split in ("DEV", "TEST"):
        print(f"\n{split}: method | FTR (k/n) | coverage (k/n) | sel.risk (k/n) | unresolved")
        for name in ["V0", "V1", "V2", "V3", *ap.ABLATIONS, *[n for n in by_method if "@votes" in n]]:
            r = report[f"{split}:{name}"]
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {name:22s} {fmt(r['ftr'])} ({r['false_direct_foreign']}/{r['n_foreign']}) | "
                  f"{fmt(r['direct_coverage'])} ({r['covered']}/{r['n_answerable']}) | "
                  f"{fmt(r['selective_risk'])} ({r['n_false_direct']}/{r['n_direct_pred']}) | {r['unresolved_rate']:.2f}")


if __name__ == "__main__":
    main()
