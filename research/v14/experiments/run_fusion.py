"""WP07. Независимость подтверждений и противоречия по всему графу (только чтение).

1. Связи (концы по ключу без падежей + связь), сказанные в ≥ 2 статьях: сколько из них
   держатся на независимых семействах, а сколько — на копиях одной работы или повторе
   текста. Сравнение со старым буфером графа (store.accepted: «повтор в другой статье —
   уже не случайность»), который считает статьи, а не семейства.
2. Противоречия: пары фактов с теми же концами и связью и разной полярностью; каждая
   пара классифицируется fusion.compare по области действия (V3, resolve_scope) — одна
   территория или разные (DIFFERENT_SCOPE не противоречие).

Пишет results/independence_report.json, results/conflicts.jsonl.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.evidence import fusion  # noqa: E402
from georag.evidence.claims import features  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.scope import resolve_scope  # noqa: E402
from georag.text import name_key, normalize  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
V14 = ROOT / "research" / "v14"
DATA, RES = V14 / "data", V14 / "results"


def main() -> None:
    reg = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
    fam = {r["source_id"]: r["doc_family"] for r in csv.DictReader((DATA / "source_manifest.csv").open(encoding="utf-8"))}
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT f.id, f.doc_id, f.src, f.relation, f.dst, f.quote, f.votes, c.text, c.headings, d.title "
                    "FROM facts f JOIN chunks c ON c.id = f.chunk_id JOIN documents d ON d.doc_id = f.doc_id")
        cols = [c.name for c in cur.description]
        facts = [dict(zip(cols, r)) for r in cur.fetchall()]
    ind = fusion.independent_support(facts, fam)
    multi = {k: v for k, v in ind.items() if v["n_docs"] >= 2}
    statuses = Counter(v["independence_status"] for v in multi.values())
    examples = [{"relation": list(k), "docs": sorted(v["docs"]), "n_independent": v["n_independent"], "status": v["independence_status"]}
                for k, v in multi.items() if v["independence_status"] != "INDEPENDENT"][:30]
    # Старый буфер: «концы связаны и в другой статье» (без связи и без направления).
    ends = defaultdict(set)
    ends_fam = defaultdict(set)
    for f in facts:
        k = tuple(sorted((name_key(f["src"]), name_key(f["dst"]))))
        ends[k].add(f["doc_id"])
        ends_fam[k].add(fam.get(f["doc_id"], f["doc_id"]))
    buffer_promoted = sum(1 for k in ends if len(ends[k]) >= 2)
    buffer_promoted_independent = sum(1 for k in ends if len(ends_fam[k]) >= 2)

    # Противоречия: те же концы и связь, разная полярность (по признакам цитаты).
    by_key = defaultdict(list)
    for f in facts:
        key = (name_key(f["src"]), normalize(f["relation"]).replace("предположительно ", ""), name_key(f["dst"]))
        by_key[key].append(f)
    conflicts = []
    for key, group in by_key.items():
        pols = {features(f["quote"], f["relation"]).polarity for f in group}
        if len(group) < 2 or len(pols) < 2:
            continue
        recs = []
        for f in group:
            sc = resolve_scope(f["quote"], reg, chunk_text=f["text"], headings=tuple(f["headings"] or ()), title=f["title"])
            recs.append({"claim_id": f"F:{f['id']}", "src": f["src"], "relation": f["relation"], "dst": f["dst"],
                         "polarity": features(f["quote"], f["relation"]).polarity, "region_ids": sc.region_ids,
                         "stage": features(f["quote"]).stage, "doc_id": f["doc_id"], "quote": f["quote"][:240]})
        for c in fusion.conflicts(recs):
            a = next(r for r in recs if r["claim_id"] == c.a)
            b = next(r for r in recs if r["claim_id"] == c.b)
            if a["polarity"] != b["polarity"]:
                conflicts.append({"verdict": c.verdict, "reason": c.reason, "a": a, "b": b})
    with (RES / "conflicts.jsonl").open("w", encoding="utf-8") as f:
        for c in conflicts:
            f.write(json.dumps(c, ensure_ascii=False, default=list) + "\n")
    report = {
        "n_facts": len(facts),
        "relations_in_2plus_docs": len(multi),
        "independence_status": dict(statuses),
        "non_independent_examples": examples,
        "legacy_buffer_end_pairs_in_2plus_docs": buffer_promoted,
        "same_counting_by_families": buffer_promoted_independent,
        "polarity_disagreements_same_relation": len(conflicts),
        "conflict_verdicts": dict(Counter(c["verdict"] for c in conflicts)),
    }
    (RES / "independence_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=list), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "non_independent_examples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
