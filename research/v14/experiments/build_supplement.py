"""WP10, дополнение выборки (29.10.2026, до прогона методов): после первой разметки
DIRECT-пар оказалось 33 (план ТЗ — не менее 35), а кандидатов в аналоги — ни одного:
стратификация build_dataset.py не отнесла Судан к «соседям» Египта. Добавляются, тем же
способом и с другим зерном, без пересэмплирования уже размеченного:

  inn2   — до 2 фактов на вопрос из статей о его территории (ещё не взятые фрагменты);
  analog — для сравнительных вопросов Q19 и Q25 — до 4 фактов из статей, у области
           исследования которых общий тег геологической обстановки с территорией вопроса
           (справочник, settings). Это целевой «трудный» поднабор: метка аналогии
           ставится разметчиком, а не берётся из справочника.

Пишет дополнение в те же pairs.jsonl / annotation_sheet.csv (строки добавляются) и
обновляет split_manifest.json (отпечатки, счётчики, запись о дополнении).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg  # noqa: E402

import build_dataset as B  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.provenance import normalize  # noqa: E402
from georag.text import split_sentences  # noqa: E402

SEED = 20261030


def main() -> None:
    reg = load(B.DATA / "regional_registry.csv", B.DATA / "registry_aliases.csv")
    pairs = [json.loads(line) for line in (B.DATA / "pairs.jsonl").open(encoding="utf-8")]
    if any(p["stratum"] in ("inn2", "analog") for p in pairs):
        print("дополнение уже есть — ничего не делаю")
        return
    used = defaultdict(set)
    for p in pairs:
        used[p["query_id"]].add(p["chunk_id"])
    fam_split = {f: v["split"] for f, v in json.loads((B.DATA / "split_manifest.json").read_text("utf-8"))["families"].items()}
    import csv as _csv

    fam_of = {r["source_id"]: r["doc_family"] for r in _csv.DictReader((B.DATA / "source_manifest.csv").open(encoding="utf-8"))}
    with psycopg.connect(B.DSN) as conn, conn.cursor() as cur:
        cur.execute(B.GOLD_SQL, (list(B.STUDY),))
        cols = [c.name for c in cur.description]
        facts = [dict(zip(cols, r)) for r in cur.fetchall()]
    by_doc = defaultdict(list)
    for f in facts:
        by_doc[f["doc_id"]].append(f)
    rng = random.Random(SEED)
    extra = []
    for q in B.QUERIES:
        target = q["target_region_id"]
        inn = sorted((r for d, rows in by_doc.items() for r in rows
                      if B.stratum(reg, B.STUDY[d][0], B.STUDY[d][1], target) == "inn"), key=lambda r: r["id"])
        rng.shuffle(inn)
        picks = []
        for r in inn:
            if len(picks) >= 2:
                break
            if r["chunk_id"] in used[q["query_id"]]:
                continue
            picks.append(("inn2", r))
            used[q["query_id"]].add(r["chunk_id"])
        if q["allow_analogies"]:
            pool = []
            for d, rows in by_doc.items():
                regions, kind = B.STUDY[d]
                if B.stratum(reg, regions, kind, target) == "inn":
                    continue
                if any(reg.shared_settings(r, target) for r in regions):
                    pool.extend(rows)
            pool.sort(key=lambda r: r["id"])
            rng.shuffle(pool)
            docs_taken = defaultdict(int)
            for r in pool:
                if len([p for p in picks if p[0] == "analog"]) >= 4:
                    break
                if r["chunk_id"] in used[q["query_id"]] or docs_taken[r["doc_id"]] >= 2:
                    continue
                picks.append(("analog", r))
                docs_taken[r["doc_id"]] += 1
                used[q["query_id"]].add(r["chunk_id"])
        for name, row in picks:
            fam = fam_of[row["doc_id"]]
            sents = split_sentences(" ".join((row["text"] or "")[:4000].split()))
            quote_norm = normalize(row["quote"])
            idx = next((i for i, s in enumerate(sents) if normalize(s) and normalize(s)[:50] in quote_norm), None)
            before = " ".join(sents[max(0, idx - 2): idx]) if idx is not None else ""
            extra.append({
                "sample_id": f"{q['query_id']}-F{row['id']}", "query_id": q["query_id"], "stratum": name,
                "fact_id": row["id"], "chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "source_family": fam,
                "split": fam_split[fam], "src": row["src"], "relation": row["relation"], "dst": row["dst"],
                "quote": row["quote"], "votes": row["votes"], "accepted": bool(row["accepted"]), "model": row["model"],
                "rel_type": row["rel_type"], "headings": list(row["headings"] or []), "pages": list(row["pages"] or []),
                "title": row["title"], "doi": row["doi"], "language": row["language"], "context_before": before,
                "chunk_text": " ".join((row["text"] or "").split()),
                "doc_study_regions": list(B.STUDY[row["doc_id"]][0]), "doc_kind": B.STUDY[row["doc_id"]][1],
            })
    ids = {p["sample_id"] for p in pairs}
    extra = [e for e in extra if e["sample_id"] not in ids]
    with (B.DATA / "pairs.jsonl").open("a", encoding="utf-8") as f:
        for e in extra:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    qtext = {q["query_id"]: q for q in B.QUERIES}
    with (B.ANN / "annotation_sheet.csv").open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p in extra:
            q = qtext[p["query_id"]]
            w.writerow([p["sample_id"], p["query_id"], q["raw_text"], reg.areas[q["target_region_id"]].name,
                        q["requested_quantifier"], q["requested_mode"], q["allow_analogies"], p["title"],
                        " / ".join(p["headings"]), ",".join(map(str, p["pages"])), p["context_before"], p["quote"],
                        p["src"], p["relation"], p["dst"], p["split"], "", "", "", "", "", "", "", "", ""])
    man = json.loads((B.DATA / "split_manifest.json").read_text("utf-8"))
    allp = pairs + extra
    man["n_pairs"] = len(allp)
    man["pairs_by_split"] = {s: sum(p["split"] == s for p in allp) for s in ("DEV", "TEST")}
    man["pairs_by_stratum"] = {s: sum(p["stratum"] == s for p in allp) for s in ("inn", "near", "out", "gen", "special", "inn2", "analog")}
    man["supplement"] = {"seed": SEED, "added": len(extra), "reason": "DIRECT<35 и нет кандидатов в аналоги после первой разметки; до прогона методов"}
    man["sha256"]["pairs.jsonl"] = B.sha_file(B.DATA / "pairs.jsonl")
    (B.DATA / "split_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    print("добавлено", len(extra), {s: sum(e["stratum"] == s for e in extra) for s in ("inn2", "analog")},
          man["pairs_by_split"])


if __name__ == "__main__":
    main()
