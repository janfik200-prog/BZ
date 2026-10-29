"""T6. Отсев фрагментов из подписей рисунков, легенд и таблиц до модели-строителя: оценка.

Классификатор и отсев — georag/graph/captions.py (без модели), в графе — третий отсев
первого этапа facts.run рядом с дублями и MIN_TERMS (GEORAG_SKIP_CAPTIONS=1). Здесь на
копии базы (georag_v14, только чтение):

1. Сетка: отсеиваемые виды (CAPTION+LEGEND; CAPTION+LEGEND+TABLE_ROW) × порог доли прозы.
   Для каждой — фрагментов отсеяно, фактов графа в них (всех и принятых ≥ 3 из 4),
   размеченные факты в них: gold_support из data/gold_annotations.csv и gold_correct из
   results/e_precision_sample.json, e2_precision_sample.json (разметка прошлых работ;
   новой разметки здесь нет).
2. Выбор на DEV по правилу, записанному до прогона TEST: из наборов, где на DEV не
   теряется ни одного верного размеченного факта (ENTAILED) и ни одного принятого факта
   графа, — наибольший порог (больше отсеянных фрагментов). Выбор должен совпасть с
   captions.SKIP_KINDS / MIN_PROSE_SHARE, иначе TEST не считается. TEST — один раз.
3. Справочно — по фактам: доля фактов графа, чья цитата — подпись/легенда/таблица
   (captions.quote_kind). Эти факты отсев фрагментов не убирает, если во фрагменте есть
   проза; число показывает, сколько мусора остаётся на долю другого способа.
4. captions_sample.json — выборка фактов DEV для слепой проверки человеком: поле
   gold_support пустое, вид скрыт (раскрывается в captions_decisions.jsonl по fact_id
   только после разметки). Агент её не размечал.

Пишет results/captions_eval.json и captions_decisions.jsonl (решение по каждому фрагменту).

    .venv/Scripts/python.exe research/v14/experiments/run_captions.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.graph import captions as C  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
V14 = ROOT / "research" / "v14"
DATA, RES = V14 / "data", V14 / "results"
EVAL, DECISIONS = RES / "captions_eval.json", RES / "captions_decisions.jsonl"
KIND_SETS = (frozenset({C.CAPTION, C.LEGEND}), frozenset({C.CAPTION, C.LEGEND, C.TABLE_ROW}))
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.5)
NONPROSE = (C.CAPTION, C.LEGEND, C.TABLE_ROW)


def name(kinds: frozenset[str]) -> str:
    return "+".join(k for k in NONPROSE if k in kinds)


def dropped(chars: dict[str, int], kinds: frozenset[str], threshold: float) -> bool:
    """То же правило, что captions.skip_reason, с другими видами и порогом."""
    total = sum(chars.values())
    return bool(total) and sum(v for k, v in chars.items() if k not in kinds) / total <= threshold


def main() -> None:
    if EVAL.exists() and "TEST" in json.loads(EVAL.read_text("utf-8")):
        sys.exit("TEST уже прогнан — один прогон (см. captions_eval.json)")
    with (DATA / "source_manifest.csv").open(encoding="utf-8") as fh:
        split = {r["source_id"]: r["split"] for r in csv.DictReader(fh)}
    with psycopg.connect(DSN) as conn:
        chunks = conn.execute("SELECT id, doc_id, text FROM chunks ORDER BY id").fetchall()
        facts = conn.execute("SELECT id, chunk_id, src, dst, quote, votes FROM facts").fetchall()
        try:  # в копии georag_v14 (снимок до первого этапа) столбца skipped нет
            skipped = dict(conn.execute("SELECT chunk_id, skipped FROM facts_pass WHERE skipped IS NOT NULL"))
        except psycopg.errors.UndefinedColumn:
            conn.rollback()
            skipped = {}
    texts = {cid: text for cid, _doc, text in chunks}
    chunk_split = {cid: split.get(doc, "?") for cid, doc, _ in chunks}
    by_chunk: dict[int, list[int]] = defaultdict(list)
    for _fid, cid, *_rest, votes in facts:
        by_chunk[cid].append(votes or 0)
    gold: dict[int, list[str]] = defaultdict(list)
    seen = set()
    with (DATA / "gold_annotations.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["fact_id"] not in seen:
                seen.add(r["fact_id"])
                gold[int(r["chunk_id"])].append(r["gold_support"])
    prec: dict[int, list[bool]] = defaultdict(list)
    for path in ("e_precision_sample.json", "e2_precision_sample.json"):
        for x in json.loads((RES / path).read_text("utf-8"))["labels"]:
            prec[x["chunk_id"]].append(bool(x["gold_correct"]))

    chars = {cid: C.kind_chars(text or "") for cid, text in texts.items()}

    def grid_row(sp: str, kinds: frozenset[str], thr: float) -> dict[str, Any]:
        drop = [cid for cid in chars if chunk_split[cid] == sp and dropped(chars[cid], kinds, thr)]
        g = Counter(v for cid in drop for v in gold[cid])
        p = Counter(v for cid in drop for v in prec[cid])
        n_chunks = sum(1 for cid in chars if chunk_split[cid] == sp)
        n_facts = sum(len(by_chunk[cid]) for cid in chars if chunk_split[cid] == sp)
        return {
            "kinds": name(kinds), "threshold": thr, "fragments": len(drop),
            "fragments_share": round(len(drop) / max(n_chunks, 1), 4),
            "fragments_already_skipped": sum(1 for cid in drop if cid in skipped),
            "fragments_without_facts": sum(1 for cid in drop if not by_chunk[cid]),
            "facts": sum(len(by_chunk[cid]) for cid in drop),
            "facts_share": round(sum(len(by_chunk[cid]) for cid in drop) / max(n_facts, 1), 4),
            "facts_accepted": sum(sum(v >= 3 for v in by_chunk[cid]) for cid in drop),
            "gold": dict(g), "gold_correct_lost": g.get("ENTAILED", 0),
            "precision_samples": {"correct": p.get(True, 0), "wrong": p.get(False, 0)},
        }  # fmt: skip

    totals = {
        sp: {
            "fragments": sum(1 for cid in chars if chunk_split[cid] == sp),
            "facts": sum(len(by_chunk[cid]) for cid in chars if chunk_split[cid] == sp),
            "gold_by_label": dict(Counter(v for cid in chars if chunk_split[cid] == sp for v in gold[cid])),
            "precision_labels": dict(Counter(str(v) for cid in chars if chunk_split[cid] == sp for v in prec[cid])),
        }
        for sp in ("DEV", "TEST")
    }  # fmt: skip
    dev_grid = [grid_row("DEV", k, t) for k in KIND_SETS for t in THRESHOLDS]
    passing = [r for r in dev_grid if r["gold_correct_lost"] == 0 and r["facts_accepted"] == 0]
    best = max(passing, key=lambda r: (r["fragments"], -len(r["kinds"])), default=None)
    # при равном числе фрагментов — меньший набор видов и меньший порог (первый по порядку)
    if best:
        best = next(r for r in passing if r["fragments"] == best["fragments"] and r["kinds"] == best["kinds"])
    choice = {
        "criterion": "на DEV не теряется ни один верный размеченный факт (ENTAILED) и ни один "
        "принятый факт графа; из таких — больше всего отсеянных фрагментов (при равенстве — "
        "меньше видов, меньший порог)",
        "kinds": best["kinds"].split("+") if best else [],
        "threshold": best["threshold"] if best else None,
    }
    code = {"kinds": sorted(C.SKIP_KINDS), "threshold": C.MIN_PROSE_SHARE}
    choice["matches_code"] = sorted(choice["kinds"]) == code["kinds"] and choice["threshold"] == code["threshold"]
    report: dict[str, Any] = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rules": C.CAPTIONS_RULES,
        "captions_sha256": hashlib.sha256((ROOT / "georag/graph/captions.py").read_bytes()).hexdigest(),
        "db": DSN.rsplit("/", 1)[-1],
        "totals": totals,
        "DEV": {"grid": dev_grid, "choice": choice, "code": code},
    }
    if not choice["matches_code"]:
        EVAL.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit(f"выбор DEV {choice} ≠ captions.py {code}: поправьте SKIP_KINDS/MIN_PROSE_SHARE")

    # TEST — один раз, на правиле из кода (= выбор DEV)
    test = grid_row("TEST", C.SKIP_KINDS, C.MIN_PROSE_SHARE)
    report["TEST"] = {"chosen": test, "grid_for_reference": [grid_row("TEST", k, t) for k in KIND_SETS for t in THRESHOLDS]}
    corpus = [cid for cid, text in texts.items() if C.skip_reason(text or "")]
    report["corpus"] = {
        "fragments": len(texts),
        "dropped": len(corpus),
        "dropped_share": round(len(corpus) / len(texts), 4),
        "dropped_without_facts": sum(1 for cid in corpus if not by_chunk[cid]),
        "dropped_already_skipped": sum(1 for cid in corpus if cid in skipped),
        "facts_in_dropped": sum(len(by_chunk[cid]) for cid in corpus),
        "facts_total": len(facts),
        "chars_share_by_kind": {
            k: round(sum(c[k] for c in chars.values()) / max(sum(sum(c.values()) for c in chars.values()), 1), 4)
            for k in C.KINDS
        },
    }  # fmt: skip
    # Справочно — вид цитаты каждого факта
    fact_kind = Counter()
    fact_kind_split: dict[str, Counter[str]] = defaultdict(Counter)
    for _fid, cid, src, dst, quote, votes in facts:
        kind = C.quote_kind(quote, texts.get(cid), src, dst)
        fact_kind[kind] += 1
        fact_kind_split[chunk_split.get(cid, "?")][kind] += 1
    report["facts_by_quote_kind"] = {
        "all": dict(fact_kind),
        "by_split": {k: dict(v) for k, v in fact_kind_split.items()},
        "nonprose_share": round(sum(fact_kind[k] for k in NONPROSE) / max(len(facts), 1), 4),
        "note": "сколько фактов графа выписано из подписей/легенд/таблиц по цитате; отсев "
        "фрагментов из них убирает только факты фрагментов без прозы",
    }
    EVAL.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with DECISIONS.open("w", encoding="utf-8", newline="\n") as fh:
        for cid, doc, text in chunks:
            reason = C.skip_reason(text or "")
            fh.write(json.dumps({
                "chunk_id": cid, "doc_id": doc, "split": chunk_split[cid],
                "decision": "DROP_CAPTIONS" if reason else "KEEP", "reason": reason or "ok",
                "chars": chars[cid], "facts": len(by_chunk[cid]), "rules": C.CAPTIONS_RULES,
            }, ensure_ascii=False) + "\n")  # fmt: skip
    print(json.dumps({k: report[k] for k in ("totals", "corpus", "facts_by_quote_kind")}, ensure_ascii=False, indent=1))
    print("DEV choice", choice)
    print("TEST", test)


if __name__ == "__main__":
    main()
