"""WP04, отдельный эксперимент: независимая проверка поддержки одним вопросом модели.

confirm.py (четыре вопроса одной модели) отбрасывает много верных связей (полнота
0,61 на разметке). Здесь — другой инструмент той же серверной модели (Qwen3-14B-AWQ,
vLLM): один прямой вопрос по цитате с явной проверкой направления, отрицания и
предположения. Это не gold: метки разметки модели не показываются; модель — та, что
работает в системе, поэтому инструмент коррелирован с confirm.py, и это оговаривается.

Выход: results/raw_support_llm.jsonl (ответ и метка по каждому факту выборки),
results/llm_calls.jsonl (журнал вызовов: модель, хэш задания, токены, время).
Правило объединения с confirm.py выбирается в run_v_support.py только на DEV.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from georag.evidence.model_adapter import ModelAdapter  # noqa: E402

V14 = ROOT / "research" / "v14"
DATA, RES = V14 / "data", V14 / "results"
HOST = os.environ.get("GEORAG_LLM", "http://127.0.0.1:8100")
MODEL = os.environ.get("GEORAG_SUPPORT_MODEL", "qwen3-14b")

SYSTEM = "Ты проверяешь, что утверждает предложение из геологической статьи. Отвечай только JSON."
USER = """Предложение (дословно из статьи):
«{quote}»

Проверяемая связь: «{src}» — {rel} — «{dst}».

Ответь по одному этому предложению, без общих знаний:
- "verdict": "ENTAILED" — предложение прямо утверждает именно эту связь в этом направлении;
  "PARTIAL" — утверждает похожее, но слабее, сильнее, о части или с неточным концом;
  "NOT_STATED" — оба названы, но эта связь не утверждается (перечисление, другая связь, другие объекты);
  "CONTRADICTED" — предложение утверждает обратное;
- "direction_ok": true/false — направление связи совпадает со сказанным;
- "negated": true/false — в предложении эта связь отрицается («не выявлено», «absent»);
- "hedged": true/false — это предположение («вероятно», «may», «suggests»).

Верни JSON: {{"verdict": "...", "direction_ok": true, "negated": false, "hedged": false}}"""


def label(answer: dict | None) -> str:
    if not isinstance(answer, dict):
        return "UNKNOWN"
    v = str(answer.get("verdict", "")).upper()
    if v == "ENTAILED" and answer.get("direction_ok") is not False:
        return "ENTAILED"
    if v == "CONTRADICTED":
        return "CONTRADICTED"
    if v == "PARTIAL" or (v == "ENTAILED" and answer.get("direction_ok") is False):
        return "PARTIAL"
    return "UNKNOWN"


def main() -> None:
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open(encoding="utf-8")]
    facts = {}
    for p in pairs:
        facts.setdefault(p["fact_id"], p)
    adapter = ModelAdapter(HOST, MODEL, log_path=RES / "llm_calls.jsonl")
    served = [m.get("root") or m.get("id") for m in adapter.server_models()]

    def ask(p: dict) -> dict:
        r = adapter.chat_json(SYSTEM, USER.format(quote=p["quote"], src=p["src"], rel=p["relation"], dst=p["dst"]),
                              max_tokens=200, temperature=0.0, tag=f"support:{p['fact_id']}")
        return {"fact_id": p["fact_id"], "ok": r.ok, "error": r.error, "answer": r.data, "label": label(r.data),
                "model": r.model, "prompt_sha256": r.prompt_sha256, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "latency_s": r.latency_s, "server_models": served}

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(ask, facts.values()))
    with (RES / "raw_support_llm.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print(len(out), Counter(r["label"] for r in out), "errors", sum(not r["ok"] for r in out),
          "tokens", sum(r["prompt_tokens"] or 0 for r in out), sum(r["completion_tokens"] or 0 for r in out))


if __name__ == "__main__":
    main()
