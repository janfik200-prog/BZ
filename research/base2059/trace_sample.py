"""trace_audit.py на случайной выборке связей графа — для базы, где связей десятки тысяч.

Прослеживаемость, территория и проверка кодом считаются по N случайным связям графа
(store.all_facts), а не по всем: на 2 000 связях интервал доли ± 2 %. Сам trace_audit.py
не меняется — подменяется только чтение связей. База только читается.
Результат — research/base2059/results/ (summary.json, report.md, facts.csv).

    python research/base2059/trace_sample.py                 # 2 000 связей графа, без модели
    python research/base2059/trace_sample.py --n 3000
    python research/base2059/trace_sample.py --n 2000 --llm  # и четыре вопроса модели (Qwen3-8B, :8101)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "trace_audit"))
sys.path.insert(0, str(ROOT))

import trace_audit as T  # noqa: E402

SEED = 20261120


def main() -> None:
    argv = sys.argv[1:]
    n = 2000
    if "--n" in argv:
        i = argv.index("--n")
        n = int(argv[i + 1])
        del argv[i : i + 2]
    llm = "--llm" in argv
    if llm:
        argv.remove("--llm")
    full = T.load_facts

    def sampled(conn):
        graph = [f for f in full(conn) if f["in_graph"]]
        pick = random.Random(SEED).sample(graph, min(n, len(graph)))
        pick.sort(key=lambda f: f["id"])
        print(f"  выборка: {len(pick)} из {len(graph)} связей графа (seed {SEED})", flush=True)
        return pick

    T.load_facts = sampled
    T.RES = HERE / "results"
    sys.argv = [sys.argv[0], *([] if llm else ["--no-llm"]), *argv]
    T.main()


if __name__ == "__main__":
    main()
