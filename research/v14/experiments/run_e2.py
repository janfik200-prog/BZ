"""WP11 E2: E1 с меньшей моделью-строителем — Qwen3-8B-AWQ (vLLM, порт 8101).

Та же выборка 100 фрагментов TEST, что у run_e.py live (то же зерно 20261101), тот же
первый этап (k = 4 и список предметов), те же правила графа и та же температура 0 —
меняется только модель. Выборка сверяется с raw_e_live.jsonl: несовпадение — остановка.
Выход: results/raw_e2_live.jsonl, llm_calls_e2.jsonl, раздел live_e2 в e_summary.json.

Перед запуском модель ответов остановлена, поднят builder (на 16 ГБ две модели не
помещаются) — по разрешению владельца системы, 29.10.2026.
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
from georag.evidence.model_adapter import ModelAdapter  # noqa: E402
from georag.evidence.prefilter import Prefilter  # noqa: E402
from georag.graph import facts as F  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402
from georag.text import name_key  # noqa: E402

HOST = os.environ.get("GEORAG_BUILDER", "http://127.0.0.1:8101")
MODEL = os.environ.get("GEORAG_BUILDER_MODEL", "qwen3-8b")


def main() -> None:
    summary = json.loads((E.RES / "e_summary.json").read_text("utf-8"))
    k, n = summary["live"]["k"], summary["live"]["n_chunks"]
    reg = load(E.DATA / "regional_registry.csv", E.DATA / "registry_aliases.csv")
    finder = Finder()
    chunks = E.load_corpus()
    rng = random.Random(20261101)
    test = [c for c in chunks if c["split"] == "TEST"]
    sample = rng.sample(test, n)
    prev = [json.loads(line) for line in (E.RES / "raw_e_live.jsonl").open(encoding="utf-8")]
    e0_ids = {r["chunk_id"] for r in prev if r["mode"] == "E0"}
    e1_ids = {r["chunk_id"] for r in prev if r["mode"] == "E1"}
    if {c["id"] for c in sample} != e0_ids:
        raise SystemExit("выборка не совпадает с E0/E1 — прогон не сравним")
    pf = Prefilter(finder, reg, k=k)
    plan = []
    for c in sorted(sample, key=lambda c: (c["doc_id"], c["ord"])):
        d = pf.decide(c["id"], c["text"])
        if d.decision != "DROP_DUPLICATE" and d.terms_found >= k:
            plan.append((c, finder.find(c["text"])))
    if {c["id"] for c, _ in plan} != e1_ids:
        raise SystemExit("первый этап отобрал не те фрагменты, что в E1")
    rules = F.load_rules()
    adapter = ModelAdapter(HOST, MODEL, log_path=E.RES / "llm_calls_e2.jsonl")
    served = adapter.server_models()

    def run(c, found):
        system, user = F.messages(rules, c["text"], found.entities)
        r = adapter.chat_json(system, user, max_tokens=F.ANSWER_TOKENS, temperature=0.0, tag=f"E2:{c['id']}")
        checked = F.check(r.data, c["text"]) if r.ok else F.Checked(rejected=[f"model error: {r.error}"])
        return {"mode": "E2", "chunk_id": c["id"], "doc_id": c["doc_id"], "ok": r.ok, "error": r.error,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": r.latency_s,
                "n_facts": len(checked.facts), "n_rejected": len(checked.rejected), "prompt_sha256": r.prompt_sha256,
                "facts": [[f.src, f.relation, f.dst, f.quote] for f in checked.facts], "model": r.model}

    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda j: run(*j), plan))
    wall = time.monotonic() - t0
    with (E.RES / "raw_e2_live.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def keys(rs):
        return {(r["chunk_id"], name_key(a), name_key(b)) for r in rs for a, _, b, _ in r["facts"]}

    e1 = [r for r in prev if r["mode"] == "E1"]
    e2k, e1k, e0k = keys(out), keys(e1), keys([r for r in prev if r["mode"] == "E0"])
    summary["live_e2"] = {
        "model": MODEL, "host": HOST, "served_models": served, "n_chunks_sample": n, "calls": len(out),
        "errors": sum(not r["ok"] for r in out), "facts": sum(r["n_facts"] for r in out),
        "rejected_by_code": sum(r["n_rejected"] for r in out),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in out),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in out),
        "latency_sum_s": round(sum(r["latency_s"] for r in out), 1), "wall_s": round(wall, 1),
        "e2_fact_keys": len(e2k), "shared_with_e1": len(e2k & e1k), "shared_with_e0": len(e2k & e0k),
        "e1_wall_note": "E0 и E1 шли одним пулом (wall_s_both); сравнение времени — по latency_sum_s",
        "e1_latency_sum_s": summary["live"]["by_mode"]["E1"]["latency_sum_s"],
    }
    (E.RES / "e_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(summary["live_e2"], ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
