"""Интервалы для опыта v18: бутстреп по фрагментам (связи одного фрагмента не независимы)."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
key = json.loads((HERE / "results" / f"sheet_{tag}_key.json").read_text(encoding="utf-8"))
ws = load_workbook(HERE / f"Разметка_сравнение_{tag}_проверено.xlsx").active
ok = {str(r[0]): int(r[7]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}
meta = json.loads((HERE / "results" / f"extraction_{tag}_meta.json").read_text(encoding="utf-8"))

per = defaultdict(lambda: {"A": [0, 0], "B": [0, 0]})  # фрагмент → вариант → [всего, верных]
for n, k in key.items():
    for m in k["modes"]:
        per[k["chunk_id"]][m][0] += 1
        per[k["chunk_id"]][m][1] += ok[n]
chunks = meta["chunks"]


def stats(sample):
    t = {m: [sum(per[c][m][0] for c in sample), sum(per[c][m][1] for c in sample)] for m in "AB"}
    pa = t["A"][1] / t["A"][0] if t["A"][0] else 0
    pb = t["B"][1] / t["B"][0] if t["B"][0] else 0
    return {"correct_A": t["A"][1], "correct_B": t["B"][1], "diff_correct": t["A"][1] - t["B"][1],
            "prec_A": pa, "prec_B": pb, "diff_prec": pa - pb,
            "ratio_correct": t["A"][1] / t["B"][1] if t["B"][1] else None}


point = stats(chunks)
rng = random.Random(18)
boot = [stats([rng.choice(chunks) for _ in chunks]) for _ in range(10000)]


def ci(name):
    v = sorted(b[name] for b in boot if b[name] is not None)
    return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]


wins = sum(per[c]["A"][1] > per[c]["B"][1] for c in chunks)
loss = sum(per[c]["A"][1] < per[c]["B"][1] for c in chunks)
out = {"point": {k: round(v, 3) for k, v in point.items()},
       "ci95": {k: ci(k) for k in ("diff_correct", "diff_prec", "prec_A", "prec_B", "ratio_correct")},
       "fragments_A_more_correct": wins, "fragments_B_more_correct": loss,
       "fragments_equal": len(chunks) - wins - loss}
# знаковый тест по фрагментам
from math import comb
n = wins + loss
p = sum(comb(n, i) for i in range(0, min(wins, loss) + 1)) / 2 ** n * 2
out["sign_test_p"] = round(min(p, 1.0), 4)
(HERE / "results" / f"stats_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
