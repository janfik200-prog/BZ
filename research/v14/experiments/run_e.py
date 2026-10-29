"""WP02 / WP11, эксперимент E: первый этап без модели и модель-строитель.

E-offline (без модели, весь корпус, 4 272 фрагмента). Опорные связи — факты боевого
графа E0 (Qwen3:14b по всем фрагментам, 09–10.10.2026, принятые: ≥ 3 из 4 голосов).
Для порога k = 0…6 (терминов словаря во фрагменте) и почти-дублей (MinHash + проверка
защищённых отличий, prefilter.Prefilter): доля фрагментов к модели, доля токенов,
потеря опорных фактов (PrefilterLoss по E0) и потеря размеченных верных связей
(gold_support = ENTAILED) — отдельно DEV и TEST, RU и EN. k выбирается на DEV по
правилу, заданному до прогона: наибольший k с потерей опорных фактов DEV ≤ 5 %.
Абляция «без защищённых отличий». Журнал решений — results/prefilter_decisions.jsonl.

E-live (модель, выборка TEST): тот же сервер и модель (qwen3-14b, vLLM) по 100
фрагментам TEST: E0 — все фрагменты без списка предметов; E1 — только прошедшие
первый этап, со списком предметов (facts.messages). Токены и время — по usage сервера
(model_adapter), факты — facts.check. Сравнение фактов — по ключам концов.
E2 (меньшая модель) — по историческим прогонам georag_b14 / georag_b8v3, см. отчёт.

    .venv/Scripts/python.exe research/v14/experiments/run_e.py [--live N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.model_adapter import ModelAdapter  # noqa: E402
from georag.evidence.prefilter import Prefilter  # noqa: E402
from georag.graph import facts as F  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
V14 = ROOT / "research" / "v14"
DATA, RES = V14 / "data", V14 / "results"
KS = (0, 1, 2, 3, 4, 5, 6)
MAX_LOSS_DEV = 0.05


def load_corpus():
    split = {r["source_id"]: r["split"] for r in csv.DictReader((DATA / "source_manifest.csv").open(encoding="utf-8"))}
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, ord, text, n_tokens FROM chunks ORDER BY doc_id, ord")
        chunks = [dict(zip(("id", "doc_id", "ord", "text", "n_tokens"), r)) for r in cur.fetchall()]
        cur.execute("SELECT chunk_id, count(*) FILTER (WHERE coalesce(votes,0) >= 3), count(*) FROM facts GROUP BY chunk_id")
        facts = {c: (int(a), int(n)) for c, a, n in cur.fetchall()}
    for c in chunks:
        c["split"] = split[c["doc_id"]]
        c["ref_facts"], c["all_facts"] = facts.get(c["id"], (0, 0))
    return chunks


def offline(chunks, finder, reg) -> dict:
    gold = list(csv.DictReader((DATA / "gold_annotations.csv").open(encoding="utf-8")))
    gold_chunks = defaultdict(int)
    seen_facts = set()
    for g in gold:
        if g["gold_support"] == "ENTAILED" and g["fact_id"] not in seen_facts:
            seen_facts.add(g["fact_id"])
            gold_chunks[int(g["chunk_id"])] += 1
    # Термины и решение о дубле не зависят от k: считаем один раз, k применяем потом.
    pf = Prefilter(finder, reg, k=0, protect=True)
    pf_noprot = Prefilter(finder, reg, k=0, protect=False)
    decisions = []
    for c in chunks:
        d = pf.decide(c["id"], c["text"])
        d2 = pf_noprot.decide(c["id"], c["text"])
        decisions.append((c, d, d2))
    with (RES / "prefilter_decisions.jsonl").open("w", encoding="utf-8") as f:
        for c, d, d2 in decisions:
            row = d.to_json()
            row.update({"doc_id": c["doc_id"], "split": c["split"], "ref_facts": c["ref_facts"], "no_protect_decision": d2.decision})
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    rows = []
    for k in KS:
        for split in ("DEV", "TEST", "ALL"):
            for lang in ("all", "ru", "en", "mixed"):
                sel = [(c, d) for c, d, _ in decisions if (split == "ALL" or c["split"] == split) and (lang == "all" or d.lang == lang)]
                if not sel:
                    continue

                def kept(d):
                    return d.decision in ("KEEP", "REVIEW_NOT_DROP") and d.terms_found >= k or (d.decision == "DROP_SPARSE" and d.terms_found >= k)

                keep = [(c, d) for c, d in sel if d.decision != "DROP_DUPLICATE" and d.terms_found >= k]
                ref = sum(c["ref_facts"] for c, _ in sel)
                ref_kept = sum(c["ref_facts"] for c, _ in keep)
                gold_all = sum(gold_chunks.get(c["id"], 0) for c, _ in sel)
                gold_kept = sum(gold_chunks.get(c["id"], 0) for c, _ in keep)
                tok = sum(c["n_tokens"] for c, _ in sel)
                tok_kept = sum(c["n_tokens"] for c, _ in keep)
                rows.append({
                    "k": k, "split": split, "lang": lang, "n_chunks": len(sel), "chunks_to_model": len(keep),
                    "share_to_model": len(keep) / len(sel), "tokens_share": tok_kept / tok if tok else None,
                    "duplicates": sum(d.decision == "DROP_DUPLICATE" for _, d in sel),
                    "review_not_drop": sum(d.decision == "REVIEW_NOT_DROP" for _, d in sel),
                    "ref_facts": ref, "ref_facts_lost": ref - ref_kept, "prefilter_loss_ref": (ref - ref_kept) / ref if ref else None,
                    "gold_entailed": gold_all, "gold_lost": gold_all - gold_kept,
                    "prefilter_loss_gold": (gold_all - gold_kept) / gold_all if gold_all else None,
                })
    with (RES / "prefilter_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dev = {r["k"]: r for r in rows if r["split"] == "DEV" and r["lang"] == "all"}
    chosen = max(k for k in KS if (dev[k]["prefilter_loss_ref"] or 0) <= MAX_LOSS_DEV)
    test = {r["k"]: r for r in rows if r["split"] == "TEST" and r["lang"] == "all"}
    prot = {"REVIEW_NOT_DROP_with_protection": sum(d.decision == "REVIEW_NOT_DROP" for _, d, _ in decisions),
            "would_be_dropped_without_protection": sum(d.decision == "REVIEW_NOT_DROP" and d2.decision == "DROP_DUPLICATE" for _, d, d2 in decisions),
            "ref_facts_in_those": sum(c["ref_facts"] for c, d, d2 in decisions if d.decision == "REVIEW_NOT_DROP" and d2.decision == "DROP_DUPLICATE"),
            "duplicates_with_protection": sum(d.decision == "DROP_DUPLICATE" for _, d, _ in decisions),
            "duplicates_without_protection": sum(d2.decision == "DROP_DUPLICATE" for _, _, d2 in decisions),
            "ref_facts_in_duplicates": sum(c["ref_facts"] for c, d, _ in decisions if d.decision == "DROP_DUPLICATE")}
    # Аудиторская выборка отсеянных (k выбранный): 30 фрагментов — для ручной проверки.
    rng = random.Random(20261029)
    dropped = [(c, d) for c, d, _ in decisions if d.decision == "DROP_DUPLICATE" or d.terms_found < chosen]
    audit = rng.sample(dropped, min(30, len(dropped)))
    with (RES / "prefilter_audit_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "doc_id", "split", "lang", "decision", "terms_found", "ref_facts_E0", "text_head",
                    "auditor_has_geological_relation", "auditor_notes"])
        for c, d in audit:
            w.writerow([c["id"], c["doc_id"], c["split"], d.lang, d.decision if d.decision == "DROP_DUPLICATE" else "DROP_SPARSE",
                        d.terms_found, c["ref_facts"], " ".join(c["text"].split())[:600], "", ""])
    return {"chosen_k_on_dev": chosen, "rule": f"max k with DEV ref-loss <= {MAX_LOSS_DEV}",
            "dev_at_k": dev[chosen], "test_at_k": test[chosen], "test_k3": test[3], "protected_tokens": prot}


def live(chunks, finder, reg, k: int, n: int) -> dict:
    rng = random.Random(20261101)
    test = [c for c in chunks if c["split"] == "TEST"]
    sample = rng.sample(test, n)
    rules = F.load_rules()
    adapter = ModelAdapter(os.environ.get("GEORAG_LLM", "http://127.0.0.1:8100"), "qwen3-14b", log_path=RES / "llm_calls_e.jsonl")
    served = adapter.server_models()
    pf = Prefilter(finder, reg, k=k)
    plan = []
    for c in sorted(sample, key=lambda c: (c["doc_id"], c["ord"])):
        d = pf.decide(c["id"], c["text"])
        found = finder.find(c["text"])
        plan.append((c, d, found))

    def run(mode: str, c, found):
        system, user = F.messages(rules, c["text"], found.entities if mode == "E1" else None)
        r = adapter.chat_json(system, user, max_tokens=F.ANSWER_TOKENS, temperature=0.0, tag=f"{mode}:{c['id']}")
        checked = F.check(r.data, c["text"]) if r.ok else F.Checked(rejected=[f"model error: {r.error}"])
        return {"mode": mode, "chunk_id": c["id"], "doc_id": c["doc_id"], "ok": r.ok, "error": r.error,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": r.latency_s,
                "n_facts": len(checked.facts), "n_rejected": len(checked.rejected), "prompt_sha256": r.prompt_sha256,
                "facts": [[f.src, f.relation, f.dst, f.quote] for f in checked.facts], "model": r.model}

    jobs = [("E0", c, f) for c, d, f in plan] + [("E1", c, f) for c, d, f in plan if d.decision != "DROP_DUPLICATE" and d.terms_found >= k]
    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda j: run(*j), jobs))
    wall = time.monotonic() - t0
    with (RES / "raw_e_live.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(list)
    for r in out:
        by[r["mode"]].append(r)
    from georag.text import name_key

    def keys(rs):
        return {(r["chunk_id"], name_key(a), name_key(c)) for r in rs for a, _, c, _ in r["facts"]}

    e0k, e1k = keys(by["E0"]), keys(by["E1"])
    summ = {}
    for m, rs in by.items():
        summ[m] = {"calls": len(rs), "errors": sum(not r["ok"] for r in rs), "facts": sum(r["n_facts"] for r in rs),
                   "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs), "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs),
                   "latency_sum_s": round(sum(r["latency_s"] for r in rs), 1)}
    return {"n_chunks": n, "k": k, "served_models": served, "wall_s_both": round(wall, 1), "by_mode": summ,
            "e0_fact_keys": len(e0k), "e1_fact_keys": len(e1k), "e0_keys_recovered_by_e1": len(e0k & e1k),
            "e0_keys_in_dropped_chunks": len({x for x in e0k if x[0] not in {r['chunk_id'] for r in by['E1']}}),
            "note": "E1 = те же модель и правила, плюс список предметов в задании и отсев первого этапа"}


def historical_e2() -> dict:
    """E1(14B) против E2(8B) по историческим прогонам на одних 200 фрагментах (3871–4075)."""
    out = {}
    for db in ("georag_b14", "georag_b8v3"):
        dsn = DSN.rsplit("/", 1)[0] + "/" + db
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT model, count(*), count(*) FILTER (WHERE votes >= 3), count(DISTINCT chunk_id) FROM facts GROUP BY model")
            rows = cur.fetchall()
            cur.execute("SELECT min(done_at), max(done_at), count(*), count(*) FILTER (WHERE skipped IS NOT NULL) FROM facts_pass")
            t = cur.fetchone()
        out[db] = {"facts_by_model": [{"model": m, "facts": n, "votes_ge3": a, "chunks_with_facts": c} for m, n, a, c in rows],
                   "pass_started": str(t[0]), "pass_finished": str(t[1]), "chunks_passed": t[2], "skipped_first_stage": t[3],
                   "wall_minutes": round((t[1] - t[0]).total_seconds() / 60, 1)}
    out["caveat"] = ("исторические прогоны 27–28.10.2026; версия правил графа между ними могла отличаться "
                     "(коммиты 065123e, fe4673c); голоса ставила сама модель-строитель — сравнение ориентировочное")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=100)
    a = ap.parse_args()
    reg = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
    finder = Finder()
    chunks = load_corpus()
    res = {"offline": offline(chunks, finder, reg)}
    if a.live:
        res["live"] = live(chunks, finder, reg, res["offline"]["chosen_k_on_dev"], a.live)
    res["historical_e2"] = historical_e2()
    (RES / "e_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
