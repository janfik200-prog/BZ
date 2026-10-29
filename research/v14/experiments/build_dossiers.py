"""WP09. Два досье по одним и тем же цитатам: разные территория и квантор вопроса.

Набор фактов один — факты графа о золоте со структурными признаками в цитате
(cli.STRUCTURAL), из всего корпуса. Меняется только вопрос:

  A: «Какие публикации непосредственно подтверждают структурные признаки золотого
     оруденения для Центральной Восточной пустыни Египта? Какие работы дают только
     аналогии для других районов?» — CED, существование, аналогии разрешены;
  B: «Характерен ли для Египта в целом структурный (сдвиговый) контроль золотого
     оруденения?» — Египет, TYPICAL, без аналогий.

Одна и та же цитата меняет статус из-за вопроса, а не из-за нового факта. Пишет
dossier/A и dossier/B (JSON, CSV, MD), dossier/switches.csv — цитаты, у которых статус
в A и B различается, и демонстрацию журнала правок (dossier/overrides_demo.jsonl:
пометка «демонстрация механизма», это не экспертное решение).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")

from georag.evidence import cli, dossier  # noqa: E402
from georag.index.db import connect  # noqa: E402

OUT = ROOT / "research" / "v14" / "dossier"
QA = ("Какие публикации непосредственно подтверждают структурные признаки золотого оруденения для Центральной "
      "Восточной пустыни Египта? Какие работы дают только аналогии для других районов?")
QB = "Характерен ли для Египта в целом структурный (сдвиговый) контроль золотого оруденения?"


def main() -> None:
    with connect() as conn:
        fa, ra = cli.build(conn, QA, cli.STRUCTURAL, "DOSSIER-A")
        fb, rb = cli.build(conn, QB, cli.STRUCTURAL, "DOSSIER-B")
    da = cli.write_dossier(fa, ra, OUT / "A")
    cli.write_dossier(fb, rb, OUT / "B")
    # Демонстрация правки: первая запись UNRESOLVED с REGION_TITLE_ONLY или REGION_UNKNOWN.
    log = OUT / "overrides_demo.jsonl"
    if log.exists():
        log.unlink()
    target = next((r for r in ra if r["decision"].status == "UNRESOLVED"
                   and set(r["decision"].reason_codes) & {"REGION_TITLE_ONLY", "REGION_UNKNOWN"}), None)
    if target is not None:
        dossier.record_override(log, target["decision"], "OUT_OF_SCOPE",
                                "ДЕМОНСТРАЦИЯ механизма журнала: не экспертное решение; проверяется, что исходный статус сохраняется",
                                "demo:claude-agent (не геолог)")
        cli.write_dossier(fa, ra, OUT / "A_with_override", dossier.read_overrides(log))
    by_claim_b = {r["claim"].claim_id: r for r in rb}
    switches = []
    for r in ra:
        b = by_claim_b.get(r["claim"].claim_id)
        if b and b["decision"].status != r["decision"].status:
            switches.append({"claim_id": r["claim"].claim_id, "doc_id": r["source"]["doc_id"], "title": r["source"]["title"],
                             "quote": r["source"]["quote"][:300], "status_A": r["decision"].status,
                             "reasons_A": ";".join(r["decision"].reason_codes), "status_B": b["decision"].status,
                             "reasons_B": ";".join(b["decision"].reason_codes),
                             "scope": ";".join(r["scope"].region_ids), "derivation": r["scope"].derivation})
    with (OUT / "switches.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(switches[0]) if switches else ["claim_id"])
        w.writeheader()
        w.writerows(switches)
    summary = {
        "A": {"frame": fa.to_json(), "sections": Counter(r["decision"].status for r in ra)},
        "B": {"frame": fb.to_json(), "sections": Counter(r["decision"].status for r in rb)},
        "n_facts_same_set": len(ra), "status_switches": len(switches),
        "switch_kinds": Counter(f"{s['status_A']}→{s['status_B']}" for s in switches),
        "override_demo": bool(target),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=dict), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=dict))
    print("A DIRECT:", len(da.entries["DIRECT"]))


if __name__ == "__main__":
    main()
