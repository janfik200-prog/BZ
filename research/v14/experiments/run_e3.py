"""T9, эксперимент E3: первый этап графа без списка предметов в задании.

В E (run_e.py) список предметов в задании модели снизил точность выписки (E0 22/33 →
E1 14/33) при росте числа фактов в 2,4 раза, а k = 4 на TEST потерял 5,4 % опорных
фактов (цель ≤ 5 %). Здесь отсев первого этапа оставлен, а список убран:

  E1′ — отсев k = 4 (как в E1), задание как в E0 (facts.messages без entities);
  E1″ — отсев k = 3, задание как в E0.

Модель, сервер, правила графа и температура — те же, что в E0/E1 (qwen3-14b, vLLM,
config/graph-rules.txt, t = 0). E0 и E1 не перезапускаются: их сырые выходы — из
raw_e_live.jsonl.

Этапы (каждый — отдельный запуск, по порядку):

  dev    — 100 фрагментов DEV (зерно DEV_SEED): E0, E1′, E1″ живьём; выбор варианта по
           критерию CHOICE_RULE ниже (записан до прогона DEV) → e3_summary.json["dev"].
  test   — те же 100 фрагментов TEST, что в run_e.py (зерно 20261101, сверка с
           raw_e_live.jsonl — несовпадение останавливает прогон): E1′ и E1″, один прогон.
  blind  — слепая выборка фактов на разметку: по ≥ 50 фактов на режим (E0, E1, E1′, E1″),
           для E0/E1 — 33 уже размеченных (e_precision_sample.json, e2_precision_sample.json)
           плюс добор; одинаковые факты (фрагмент, концы, связь, цитата) размечаются один раз;
           перемешано, режим скрыт. Лист — в файл (аргумент), ключ режимов — в другой файл.
  score  — сведение меток с режимами: точность (Wilson), затраты, «токены на верный факт»
           → e3_precision_sample.json и e3_summary.json["test"].

    .venv/Scripts/python.exe research/v14/experiments/run_e3.py dev|test
    .venv/Scripts/python.exe research/v14/experiments/run_e3.py blind SHEET.json KEY.json
    .venv/Scripts/python.exe research/v14/experiments/run_e3.py score LABELS.json KEY.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_e as E  # noqa: E402
from georag.evidence.contracts import sha256_text  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.model_adapter import ModelAdapter  # noqa: E402
from georag.evidence.prefilter import Prefilter  # noqa: E402
from georag.graph import facts as F  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402
from georag.text import name_key  # noqa: E402

RES = E.RES
HOST = os.environ.get("GEORAG_LLM", "http://127.0.0.1:8100")
MODEL = "qwen3-14b"
WORKERS = 4  # сервер общий с другими агентами: не больше 4 запросов сразу
N = 100
TEST_SEED = 20261101  # как в run_e.py live
DEV_SEED = 20261102
BOOT_SEED = 20261103
BOOT_N = 2000
BLIND_SEED = 20261104
PER_MODE = 50
ANNOTATOR = "claude-agent-T9"
MODES = {"E1p": 4, "E1pp": 3}  # E1′ и E1″: порог k, задание без списка предметов
LABEL = {"E0": "E0", "E1": "E1", "E1p": "E1′", "E1pp": "E1″"}

# Критерий выбора варианта — записан до прогона DEV (29.10.2026), не меняется после.
MAX_LOSS = 0.05
CHOICE_RULE = (
    "Кандидаты: E1′ (k=4) и E1″ (k=3), задание без списка предметов. Проходят оба условия "
    "на DEV: (1) верхняя граница 95 % интервала (бутстрэп по статьям, 2000 повторов) потери "
    "опорных фактов боевого графа на всём DEV (1433 фрагмента) ≤ 5 %; (2) на DEV-100 живьём "
    "доля ключей фактов E0 (фрагмент, концы), попавших в отсеянные режимом фрагменты, ≤ 5 %. "
    "Из прошедших — с наименьшей суммой токенов "
    "(задание + ответ) на DEV-100. Не прошёл никто — берётся вариант с меньшей потерей, "
    "с пометкой «критерий не выполнен». Точность на DEV не размечается: задание у E1′/E1″ то "
    "же, что у E0; точность проверяется один раз, на TEST, вслепую."
)
# Правило приёмки на TEST (записано до прогона TEST): режим не хуже E0, если сумма токенов
# на TEST-100 меньше, чем у E0, и точность «не ниже E0 в пределах интервала» — верхняя
# граница интервала Wilson режима ≥ точечной оценки E0. Отдельно — разность с E0 и
# 95 % интервал Ньюкомба (метод 10), чтобы было видно, насколько это слабое утверждение.
ACCEPT_RULE = "tokens(mode) < tokens(E0) и wilson_hi(mode) >= p(E0)"


# ---------------------------------------------------------------- статистика

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Разность p1 − p2 и 95 % интервал Ньюкомба (гибрид Wilson, метод 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def loss_bootstrap(rows: list[dict], k: int) -> dict:
    """Потеря опорных фактов при пороге k по prefilter_decisions.jsonl и интервал по статьям."""
    by_doc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        lost = r["decision"] == "DROP_DUPLICATE" or r["terms_found"] < k
        by_doc[r["doc_id"]][0] += r["ref_facts"]
        by_doc[r["doc_id"]][1] += r["ref_facts"] if lost else 0
    docs = list(by_doc.values())
    ref, lost = sum(d[0] for d in docs), sum(d[1] for d in docs)
    rng = random.Random(BOOT_SEED + k)
    stats = []
    for _ in range(BOOT_N):
        s = [docs[rng.randrange(len(docs))] for _ in docs]
        a = sum(x[0] for x in s)
        stats.append(sum(x[1] for x in s) / a if a else 0.0)
    stats.sort()
    return {"k": k, "n_chunks": len(rows), "n_docs": len(docs), "ref_facts": ref, "ref_lost": lost,
            "loss": lost / ref if ref else None,
            "ci95": [stats[int(0.025 * BOOT_N)], stats[int(0.975 * BOOT_N) - 1]]}


# ---------------------------------------------------------------- прогон

def kept(d, k: int) -> bool:
    return d.decision != "DROP_DUPLICATE" and d.terms_found >= k


def sample_split(chunks, split: str, seed: int):
    pool = [c for c in chunks if c["split"] == split]
    return sorted(random.Random(seed).sample(pool, N), key=lambda c: (c["doc_id"], c["ord"]))


def plan_for(sample, finder, reg, k: int) -> list[dict]:
    """Решение первого этапа по выборке — тем же проходом, что в run_e.py (дубли — внутри выборки)."""
    pf = Prefilter(finder, reg, k=k)
    return [c for c in sample if kept(pf.decide(c["id"], c["text"]), k)]


def run_jobs(jobs: list[tuple[str, dict]], split: str, rules: str) -> list[dict]:
    adapter = ModelAdapter(HOST, MODEL, log_path=RES / "llm_calls_e3.jsonl")

    def one(job):
        mode, c = job
        system, user = F.messages(rules, c["text"], None)  # без списка предметов — как E0
        r = adapter.chat_json(system, user, max_tokens=F.ANSWER_TOKENS, temperature=0.0, tag=f"{split}:{mode}:{c['id']}")
        checked = F.check(r.data, c["text"]) if r.ok else F.Checked(rejected=[f"model error: {r.error}"])
        return {"split": split, "mode": mode, "chunk_id": c["id"], "doc_id": c["doc_id"], "ok": r.ok, "error": r.error,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": r.latency_s,
                "n_facts": len(checked.facts), "n_rejected": len(checked.rejected), "prompt_sha256": r.prompt_sha256,
                "facts": [[f.src, f.relation, f.dst, f.quote] for f in checked.facts], "model": r.model}

    t0 = time.monotonic()
    with ThreadPoolExecutor(WORKERS) as pool:
        out = list(pool.map(one, jobs))
    print(f"{split}: {len(out)} вызовов за {time.monotonic() - t0:.0f} с", file=sys.stderr)
    return out


def save_raw(rows: list[dict], split: str) -> None:
    path = RES / "raw_e3_live.jsonl"
    old = [json.loads(x) for x in path.open(encoding="utf-8")] if path.exists() else []
    old = [r for r in old if r["split"] != split]
    with path.open("w", encoding="utf-8") as f:
        for r in old + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def keys(rs) -> set:
    return {(r["chunk_id"], name_key(a), name_key(b)) for r in rs for a, _, b, _ in r["facts"]}


def mode_stats(rs: list[dict]) -> dict:
    pt = sum(r["prompt_tokens"] or 0 for r in rs)
    ct = sum(r["completion_tokens"] or 0 for r in rs)
    facts = sum(r["n_facts"] for r in rs)
    return {"calls": len(rs), "errors": sum(not r["ok"] for r in rs), "facts": facts,
            "rejected_by_code": sum(r["n_rejected"] for r in rs), "prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct, "facts_per_call": facts / len(rs) if rs else None,
            "prompt_tokens_per_call": pt / len(rs) if rs else None,
            "latency_sum_s": round(sum(r["latency_s"] for r in rs), 1)}


def load_summary() -> dict:
    p = RES / "e3_summary.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def write_summary(s: dict) -> None:
    (RES / "e3_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), "utf-8")


def setup():
    reg = load(E.DATA / "regional_registry.csv", E.DATA / "registry_aliases.csv")
    return reg, Finder(), E.load_corpus(), F.load_rules()


def stage_dev() -> None:
    reg, finder, chunks, rules = setup()
    decisions = [json.loads(x) for x in (RES / "prefilter_decisions.jsonl").open(encoding="utf-8")]
    offline = {f"k{k}": loss_bootstrap([r for r in decisions if r["split"] == "DEV"], k) for k in MODES.values()}
    sample = sample_split(chunks, "DEV", DEV_SEED)
    plans = {m: plan_for(sample, finder, reg, k) for m, k in MODES.items()}
    jobs = [("E0", c) for c in sample] + [(m, c) for m, p in plans.items() for c in p]
    out = run_jobs(jobs, "DEV", rules)
    save_raw(out, "DEV")
    by = defaultdict(list)
    for r in out:
        by[r["mode"]].append(r)
    e0k = keys(by["E0"])
    live = {}
    for m in ("E0", *MODES):
        st = mode_stats(by[m])
        ids = {r["chunk_id"] for r in by[m]}
        mk = keys(by[m])
        st.update({"fact_keys": len(mk), "e0_keys_recovered": len(e0k & mk),
                   "e0_keys_in_dropped_chunks": len({x for x in e0k if x[0] not in ids}),
                   "e0_loss_in_dropped": len({x for x in e0k if x[0] not in ids}) / len(e0k) if e0k else None})
        if m != "E0":
            same = {r["chunk_id"]: r for r in by["E0"]}
            st["same_prompt_as_e0"] = sum(same[r["chunk_id"]]["prompt_sha256"] == r["prompt_sha256"] for r in by[m])
            st["same_facts_as_e0"] = sum(same[r["chunk_id"]]["facts"] == r["facts"] for r in by[m])
        live[m] = st
    passed = [m for m, k in MODES.items()
              if offline[f"k{k}"]["ci95"][1] <= MAX_LOSS and (live[m]["e0_loss_in_dropped"] or 0) <= MAX_LOSS]
    if passed:
        chosen, note = min(passed, key=lambda m: live[m]["total_tokens"]), "критерий выполнен"
    else:
        chosen = min(MODES, key=lambda m: offline[f"k{MODES[m]}"]["loss"])
        note = "критерий не выполнен ни одним вариантом"
    s = load_summary()
    s["rule"] = CHOICE_RULE
    s["accept_rule"] = ACCEPT_RULE
    s["dev"] = {"seed": DEV_SEED, "n_chunks": N, "chunk_ids": [c["id"] for c in sample], "model": MODEL, "host": HOST,
                "served_models": [m.get("root") for m in ModelAdapter(HOST, MODEL).server_models()],
                "offline_loss_full_dev": offline, "live": live, "passed": [LABEL[m] for m in passed],
                "chosen": LABEL[chosen], "chosen_mode": chosen, "chosen_k": MODES[chosen], "note": note}
    write_summary(s)
    print(json.dumps(s["dev"], ensure_ascii=False, indent=1, default=str))


def stage_test() -> None:
    s = load_summary()
    if "dev" not in s:
        raise SystemExit("сначала dev: выбор варианта — по DEV")
    reg, finder, chunks, rules = setup()
    prev = [json.loads(x) for x in (RES / "raw_e_live.jsonl").open(encoding="utf-8")]
    e0_ids = {r["chunk_id"] for r in prev if r["mode"] == "E0"}
    e1_ids = {r["chunk_id"] for r in prev if r["mode"] == "E1"}
    # Та же выборка, что run_e.py live: rng.sample по фрагментам TEST в порядке load_corpus.
    pool = [c for c in chunks if c["split"] == "TEST"]
    sample = sorted(random.Random(TEST_SEED).sample(pool, N), key=lambda c: (c["doc_id"], c["ord"]))
    if {c["id"] for c in sample} != e0_ids:
        raise SystemExit("выборка TEST не совпадает с E0 в raw_e_live.jsonl — прогон не сравним")
    plans = {m: plan_for(sample, finder, reg, k) for m, k in MODES.items()}
    if {c["id"] for c in plans["E1p"]} != e1_ids:
        raise SystemExit("первый этап (k=4) отобрал не те фрагменты, что в E1")
    # Задание E1′/E1″ должно совпасть с заданием E0 по тем же фрагментам (тот же хэш).
    e0_sha = {r["chunk_id"]: r["prompt_sha256"] for r in prev if r["mode"] == "E0"}
    sha_ok = sum(sha256_text("\n".join(F.messages(rules, c["text"], None))) == e0_sha[c["id"]] for c in sample)
    jobs = [(m, c) for m, p in plans.items() for c in p]
    out = run_jobs(jobs, "TEST", rules)
    save_raw(out, "TEST")
    decisions = [json.loads(x) for x in (RES / "prefilter_decisions.jsonl").open(encoding="utf-8")]
    s["test_run"] = {"seed": TEST_SEED, "n_chunks": N, "sample_matches_raw_e_live": True,
                     "e1p_chunks_match_e1": True, "prompt_sha_equal_to_e0": f"{sha_ok}/{N}",
                     "calls": {LABEL[m]: len(p) for m, p in plans.items()},
                     "offline_loss_full_test": {f"k{k}": loss_bootstrap([r for r in decisions if r["split"] == "TEST"], k)
                                                for k in MODES.values()},
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_summary(s)
    print(json.dumps(s["test_run"], ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- точность

def test_rows() -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in (RES / "raw_e_live.jsonl").open(encoding="utf-8"):
        r = json.loads(r)
        by[r["mode"]].append(r)
    for r in (RES / "raw_e3_live.jsonl").open(encoding="utf-8"):
        r = json.loads(r)
        if r["split"] == "TEST":
            by[r["mode"]].append(r)
    return by


def fkey(chunk_id, fact) -> str:
    return json.dumps([chunk_id, *fact], ensure_ascii=False)


def prior_labels() -> dict[str, bool]:
    """Прежние слепые метки (E, E2): одинаковый факт — одна метка."""
    out = {}
    for fn in ("e_precision_sample.json", "e2_precision_sample.json"):
        for lab in json.loads((RES / fn).read_text("utf-8"))["labels"]:
            if lab["mode"] in ("E0", "E1"):
                out[fkey(lab["chunk_id"], lab["fact"])] = lab["gold_correct"]
    return out


def stage_blind(sheet: Path, key: Path) -> None:
    by = test_rows()
    prior = prior_labels()
    rng = random.Random(BLIND_SEED)
    chosen: dict[str, list[str]] = {}
    for m in ("E0", "E1", "E1p", "E1pp"):
        pool = sorted({fkey(r["chunk_id"], f) for r in by[m] for f in r["facts"]})
        have = [x for x in pool if x in prior] if m in ("E0", "E1") else []
        rest = [x for x in pool if x not in set(have)]
        chosen[m] = have + rng.sample(rest, min(PER_MODE - len(have), len(rest)))
    items = sorted({x for v in chosen.values() for x in v if x not in prior})
    rng.shuffle(items)
    texts = {c["id"]: c["text"] for c in E.load_corpus() if c["id"] in {json.loads(x)[0] for x in items}}
    blind = []
    for i, x in enumerate(items):
        cid, src, rel, dst, quote = json.loads(x)
        blind.append({"bid": i, "fact": [src, rel, dst], "quote": quote, "chunk_text": texts[cid][:F.FRAGMENT_CHARS]})
    sheet.write_text(json.dumps(blind, ensure_ascii=False, indent=1), "utf-8")
    key.write_text(json.dumps({"items": items, "chosen": chosen}, ensure_ascii=False), "utf-8")
    print(f"на разметку {len(blind)} фактов (ещё {sum(x in prior for v in chosen.values() for x in v)} мест — прежние метки)")


def stage_score(labels_path: Path, key_path: Path) -> None:
    labels = json.loads(labels_path.read_text("utf-8"))  # {bid: true/false}
    key = json.loads(key_path.read_text("utf-8"))
    prior = prior_labels()
    by_item = {x: bool(labels[str(i)]) for i, x in enumerate(key["items"])}
    by = test_rows()
    out_labels, prec = [], {}
    for m, sel in key["chosen"].items():
        ok = 0
        for x in sel:
            new = x in by_item
            v = by_item[x] if new else prior[x]
            ok += v
            cid, *fact = json.loads(x)
            out_labels.append({"mode": LABEL[m], "chunk_id": cid, "fact": fact, "gold_correct": v,
                               "annotator": f"{ANNOTATOR} (не эксперт, режим скрыт)" if new else "claude-opus-5.5-annotator (blind to mode)",
                               "source": "e3" if new else "e/e2"})
        prec[m] = {"correct": ok, "n": len(sel), "precision": ok / len(sel), "ci95": list(wilson(ok, len(sel)))}
    e0 = prec["E0"]
    s = load_summary()
    table = {}
    for m in ("E0", "E1", "E1p", "E1pp"):
        st = mode_stats(by[m])
        p = prec[m]
        tok = st["total_tokens"]
        est = st["facts"] * p["precision"]
        lo, hi = p["ci95"]
        st.update({"precision": p, "est_correct_facts": est,
                   "tokens_per_correct_fact": tok / est if est else None,
                   "tokens_per_correct_fact_ci95": [tok / (st["facts"] * hi) if hi else None,
                                                    tok / (st["facts"] * lo) if lo else None],
                   "fact_keys": len(keys(by[m])), "e0_keys_recovered": len(keys(by["E0"]) & keys(by[m]))})
        if m != "E0":
            d = newcombe(p["correct"], p["n"], e0["correct"], e0["n"])
            st["precision_minus_e0"] = {"diff": d[0], "ci95": [d[1], d[2]]}
            st["accepted"] = bool(tok < mode_stats(by["E0"])["total_tokens"] and p["ci95"][1] >= e0["precision"])
        table[LABEL[m]] = st
    s["test"] = table
    s["blind"] = {"seed": BLIND_SEED, "per_mode": PER_MODE, "new_labels": len(by_item),
                  "reused_prior_labels": sum(x not in by_item for v in key["chosen"].values() for x in v),
                  "annotator": ANNOTATOR, "note": "не эксперт-геолог; одинаковый факт в разных режимах — одна метка"}
    write_summary(s)
    (RES / "e3_precision_sample.json").write_text(json.dumps({"labels": out_labels, "precision": {LABEL[m]: v for m, v in prec.items()}},
                                                             ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: {x: v[x] for x in ("calls", "total_tokens", "facts", "precision", "tokens_per_correct_fact",
                                            "tokens_per_correct_fact_ci95")} for k, v in table.items()}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dev", "test", "blind", "score"))
    ap.add_argument("files", nargs="*", type=Path)
    a = ap.parse_args()
    if a.stage == "dev":
        stage_dev()
    elif a.stage == "test":
        stage_test()
    elif a.stage == "blind":
        stage_blind(*a.files)
    else:
        stage_score(*a.files)


if __name__ == "__main__":
    main()
