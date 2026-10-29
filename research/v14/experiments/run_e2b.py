"""T15. Qwen3-8B со своими правилами: выбор на DEV, один прогон на TEST.

Варианты (модель одна — qwen3-8b на builder, порт 8101; первый этап k = 4):
  E2   — общие правила config/graph-rules.txt, список предметов в задании (как run_e2.py);
  E2r  — правила config/graph-rules.qwen3-8b.txt, список предметов;
  E2rn — правила qwen3-8b, без списка предметов.

Шаг dev: 100 фрагментов DEV (зерно 20261104) → сырые выходы и слепая выборка фактов
(results/e2b_dev_precision_sample.json, поле gold_correct ставит разметчик). Шаг choose:
критерий записан до прогона — наибольшая доля верных на DEV; при равенстве (разница
меньше 0,05) — меньше токенов на верный факт. Шаг test: выбранный вариант на тех же
фрагментах TEST, что E1/E2 (сверка с raw_e_live.jsonl), и слепая выборка.

    run_e2b.py dev | choose | test | score
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_e as E  # noqa: E402
from georag.evidence.georegistry import load  # noqa: E402
from georag.evidence.metrics import wilson  # noqa: E402
from georag.evidence.model_adapter import ModelAdapter  # noqa: E402
from georag.evidence.prefilter import Prefilter  # noqa: E402
from georag.graph import facts as F  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402

HOST = os.environ.get("GEORAG_BUILDER", "http://127.0.0.1:8101")
MODEL = "qwen3-8b"
RULES = {"E2": None, "E2r": F.CONFIG / "graph-rules.qwen3-8b.txt", "E2rn": F.CONFIG / "graph-rules.qwen3-8b.txt"}
LIST = {"E2": True, "E2r": True, "E2rn": False}
OUT = E.RES


def plan_for(split: str, seed: int, n: int, k: int = 4):
    reg = load(E.DATA / "regional_registry.csv", E.DATA / "registry_aliases.csv")
    finder = Finder()
    chunks = E.load_corpus()
    rng = random.Random(seed)
    pool = [c for c in chunks if c["split"] == split]
    sample = rng.sample(pool, n)
    pf = Prefilter(finder, reg, k=k)
    plan = []
    for c in sorted(sample, key=lambda c: (c["doc_id"], c["ord"])):
        d = pf.decide(c["id"], c["text"])
        if d.decision != "DROP_DUPLICATE" and d.terms_found >= k:
            plan.append((c, finder.find(c["text"])))
    return sample, plan


def run(modes: list[str], plan, tag: str) -> list[dict]:
    adapter = ModelAdapter(HOST, MODEL, log_path=OUT / f"llm_calls_e2b_{tag}.jsonl")
    rules = {m: F.load_rules(RULES[m]) for m in modes}

    def one(job):
        mode, c, found = job
        system, user = F.messages(rules[mode], c["text"], found.entities if LIST[mode] else None)
        r = adapter.chat_json(system, user, max_tokens=F.ANSWER_TOKENS, temperature=0.0, tag=f"{mode}:{c['id']}")
        checked = F.check(r.data, c["text"]) if r.ok else F.Checked(rejected=[f"model error: {r.error}"])
        return {"mode": mode, "split": tag, "chunk_id": c["id"], "doc_id": c["doc_id"], "ok": r.ok, "error": r.error,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": r.latency_s,
                "n_facts": len(checked.facts), "n_rejected": len(checked.rejected), "rejected": checked.rejected[:20],
                "facts": [[f.src, f.relation, f.dst, f.quote] for f in checked.facts], "model": r.model,
                "prompt_sha256": r.prompt_sha256}

    jobs = [(m, c, f) for m in modes for c, f in plan]
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, jobs))


def summary(rows: list[dict]) -> dict:
    out = {}
    for m in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == m]
        out[m] = {"calls": len(rs), "errors": sum(not r["ok"] for r in rs), "facts": sum(r["n_facts"] for r in rs),
                  "rejected_by_code": sum(r["n_rejected"] for r in rs),
                  "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
                  "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs)}
    return out


def blind_sample(rows: list[dict], per_mode: int, seed: int, path: Path) -> None:
    rng = random.Random(seed)
    items = []
    for m in sorted({r["mode"] for r in rows}):
        facts = [(m, r["chunk_id"], f) for r in rows if r["mode"] == m for f in r["facts"]]
        items += rng.sample(facts, min(per_mode, len(facts)))
    rng.shuffle(items)
    path.write_text(json.dumps([{"id": i, "mode": m, "chunk_id": c, "fact": f, "gold_correct": None}
                                for i, (m, c, f) in enumerate(items)], ensure_ascii=False, indent=0), "utf-8")


def score(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8"))
    out = {}
    for m in sorted({x["mode"] for x in data}):
        xs = [x for x in data if x["mode"] == m and x["gold_correct"] is not None]
        k = sum(bool(x["gold_correct"]) for x in xs)
        out[m] = {"correct": k, "n": len(xs), "precision": k / len(xs) if xs else None, "ci95": wilson(k, len(xs))}
    return out


def main() -> None:
    step = sys.argv[1]
    if step == "dev":
        _, plan = plan_for("DEV", 20261104, 100)
        t0 = time.monotonic()
        rows = run(list(RULES), plan, "dev")
        with (OUT / "raw_e2b_dev.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        blind_sample(rows, 25, 20261105, OUT / "e2b_dev_precision_sample.json")
        print(json.dumps({"wall_s": round(time.monotonic() - t0, 1), **summary(rows)}, ensure_ascii=False, indent=1))
    elif step == "choose":
        rows = [json.loads(x) for x in (OUT / "raw_e2b_dev.jsonl").open(encoding="utf-8")]
        s, p = summary(rows), score(OUT / "e2b_dev_precision_sample.json")
        for m in s:
            tok = s[m]["prompt_tokens"] + s[m]["completion_tokens"]
            est = s[m]["facts"] * (p[m]["precision"] or 0)
            s[m]["tokens_per_correct_fact"] = round(tok / est) if est else None
        best = max(p, key=lambda m: p[m]["precision"] or 0)
        close = [m for m in p if (p[best]["precision"] or 0) - (p[m]["precision"] or 0) < 0.05]
        chosen = min(close, key=lambda m: s[m]["tokens_per_correct_fact"] or 1e9)
        res = {"criterion": "max DEV precision; ties (<0.05) → fewer tokens per correct fact", "chosen": chosen,
               "dev_summary": s, "dev_precision": p}
        (OUT / "e2b_choice.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif step == "test":
        chosen = json.loads((OUT / "e2b_choice.json").read_text("utf-8"))["chosen"]
        sample, plan = plan_for("TEST", 20261101, 100)
        prev = [json.loads(x) for x in (OUT / "raw_e_live.jsonl").open(encoding="utf-8")]
        if {c["id"] for c in sample} != {r["chunk_id"] for r in prev if r["mode"] == "E0"}:
            raise SystemExit("выборка TEST не совпадает с E0/E1")
        rows = run([chosen], plan, "test")
        with (OUT / "raw_e2b_test.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Маскировка режима: к выбранному варианту подмешаны факты прежнего E2 (общие правила).
        old = [{**json.loads(x), "mode": "E2(old)"} for x in (OUT / "raw_e2_live.jsonl").open(encoding="utf-8")]
        blind_sample(rows + old, 30, 20261106, OUT / "e2b_test_precision_sample.json")
        print(json.dumps(summary(rows), ensure_ascii=False, indent=1))
    elif step == "score":
        res = {"dev": score(OUT / "e2b_dev_precision_sample.json"), "test": score(OUT / "e2b_test_precision_sample.json"),
               "choice": json.loads((OUT / "e2b_choice.json").read_text("utf-8"))["chosen"],
               "test_summary": summary([json.loads(x) for x in (OUT / "raw_e2b_test.jsonl").open(encoding="utf-8")])}
        (OUT / "e2b_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
