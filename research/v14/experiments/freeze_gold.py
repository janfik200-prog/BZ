"""WP10. Сборка разметки и заморозка до прогона методов.

labels_part*.csv (разметка по правилам annotation_guidelines.md) + pairs.jsonl →
data/gold_annotations.csv в формате templates/ANNOTATION_GOLD.csv; выборка для
проверки геологом (annotation/expert_review_sample.csv: все DIRECT и ANALOG и
случайная треть остальных, ≥ 30 % пар) и пустая таблица согласия
(annotation/annotator_agreement.csv). results/FREEZE.json — отпечатки разметки,
пар, справочника, вопросов и git HEAD: результаты считаются только по этим версиям.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14 = ROOT / "research" / "v14"
DATA, ANN, RES = V14 / "data", V14 / "annotation", V14 / "results"
ANNOTATOR = "claude-opus-5.5-annotator"
COLUMNS = ["sample_id", "query_id", "source_family", "source_id", "evidence_id", "page", "quote", "subject",
           "predicate", "object", "modality", "polarity", "quantifier", "region_id", "deposit_type",
           "gold_support", "gold_applicability", "annotator_id", "adjudication_status", "split", "reason_notes",
           "stratum", "fact_id", "chunk_id"]


def sha(p: Path) -> str:
    """Отпечаток содержимого без учёта концов строк (git хранит LF, Windows пишет CRLF)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    labels = {}
    for f in sorted(glob.glob(str(ANN / "labels_part*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["sample_id"] in labels:
                raise SystemExit(f"повтор разметки {r['sample_id']}")
            labels[r["sample_id"]] = r
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open(encoding="utf-8")]
    missing = [p["sample_id"] for p in pairs if p["sample_id"] not in labels]
    if missing:
        raise SystemExit(f"нет разметки: {missing[:10]}")
    valid = {"DIRECT", "ANALOG", "OUT_OF_SCOPE", "UNRESOLVED"}
    rows = []
    for p in pairs:
        lab = labels[p["sample_id"]]
        assert lab["gold_applicability"] in valid, lab
        assert lab["gold_support"] in {"ENTAILED", "PARTIAL", "CONTRADICTED", "UNKNOWN", "INVALID_EVIDENCE"}, lab
        if lab["gold_applicability"] == "DIRECT":
            assert lab["gold_support"] == "ENTAILED", lab  # правило разметки
        rows.append({
            "sample_id": p["sample_id"], "query_id": p["query_id"], "source_family": p["source_family"],
            "source_id": p["doc_id"], "evidence_id": f"E:{p['doc_id']}:{p['chunk_id']}", "page": ",".join(map(str, p["pages"])),
            "quote": p["quote"], "subject": p["src"], "predicate": p["relation"], "object": p["dst"],
            "modality": lab["gold_modality"], "polarity": lab["gold_polarity"], "quantifier": lab["gold_quantifier"],
            "region_id": lab["gold_region_of_fact"], "deposit_type": "", "gold_support": lab["gold_support"],
            "gold_applicability": lab["gold_applicability"], "annotator_id": ANNOTATOR,
            "adjudication_status": "SINGLE_ANNOTATOR_UNADJUDICATED", "split": p["split"],
            "reason_notes": lab["reason_notes"], "stratum": p["stratum"], "fact_id": p["fact_id"], "chunk_id": p["chunk_id"],
        })
    with (DATA / "gold_annotations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # Выборка для геолога: все DIRECT/ANALOG + случайная треть остального, зерно фиксировано.
    rng = random.Random(20261031)
    must = [r for r in rows if r["gold_applicability"] in ("DIRECT", "ANALOG")]
    rest = [r for r in rows if r["gold_applicability"] not in ("DIRECT", "ANALOG")]
    rng.shuffle(rest)
    need = max(int(0.3 * len(rows)) - len(must), len(rest) // 3)
    sample = sorted(must + rest[:need], key=lambda r: r["sample_id"])
    with (ANN / "expert_review_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "query_id", "split", "quote", "subject", "predicate", "object", "title_doc",
                    "expert_support", "expert_region", "expert_applicability", "expert_id", "comment"])
        titles = {p["sample_id"]: p["title"] for p in pairs}
        for r in sample:
            # Метка Claude геологу не показывается: проверка вслепую.
            w.writerow([r["sample_id"], r["query_id"], r["split"], r["quote"], r["subject"], r["predicate"],
                        r["object"], titles[r["sample_id"]], "", "", "", "", ""])
    with (ANN / "annotator_agreement.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "annotator1", "annotator2", "label1", "label2", "adjudicated_label", "adjudicator", "notes"])
        for r in sample:
            w.writerow([r["sample_id"], ANNOTATOR, "", r["gold_applicability"], "", "", "", "BLOCKED: второй разметчик-геолог не назначен"])

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    freeze = {
        "frozen_at": "2026-10-29",
        "note": "Отпечатки до прогона V0–V3 на разметке. Менять правила V3 после этого — только с новой заморозкой и пометкой.",
        "git_head_at_freeze": head,
        "hash_rule": "sha256 содержимого файла с заменой CRLF на LF",
        "annotator": ANNOTATOR,
        "n_pairs": len(rows),
        "labels": dict(Counter(r["gold_applicability"] for r in rows)),
        "labels_by_split": {s: dict(Counter(r["gold_applicability"] for r in rows if r["split"] == s)) for s in ("DEV", "TEST")},
        "expert_review_sample": len(sample),
        "sha256": {name: sha(path) for name, path in {
            "gold_annotations.csv": DATA / "gold_annotations.csv",
            "pairs.jsonl": DATA / "pairs.jsonl",
            "queries.json": DATA / "queries.json",
            "split_manifest.json": DATA / "split_manifest.json",
            "regional_registry.csv": DATA / "regional_registry.csv",
            "registry_aliases.csv": DATA / "registry_aliases.csv",
            "annotation_guidelines.md": ANN / "annotation_guidelines.md",
        }.items()},
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: freeze[k] for k in ("n_pairs", "labels", "labels_by_split", "expert_review_sample")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
