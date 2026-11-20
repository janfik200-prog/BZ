"""Проверки к SHAP по размеченным связям: доли ошибок по признакам и разбор одной связи.

shap/checks.json:
  errors_by_feature — неверных связей с признаком и без (переворот кодом, оборот предположения,
                      голоса 4 из 4, тип «содержит» и «локализовано в», территория из названия, цитата
                      дословно, вошла по повтору пары) и виды ошибок у связей, перевёрнутых кодом;
  example           — разложение оценки доверия одной связи на вклады признаков (база + сумма = оценка):
                      неверная связь, у которой код изменил направление.

    python research/base2059/shap_checks.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import shap_links as SL  # noqa: E402


def main() -> None:
    import shap
    from openpyxl import load_workbook
    from sklearn.ensemble import RandomForestClassifier

    sys.stdout.reconfigure(encoding="utf-8")
    lab = SL.labels()
    F = SL.features([i for i, _ in lab])
    key = json.loads((HERE / "key.json").read_text("utf-8"))
    ws = load_workbook(SL.SHEET, read_only=True)["Связи"]
    kind = {key["facts"][r]: row[8] for r, row in enumerate(list(ws.iter_rows(values_only=True))[1:])}

    def rate(pred) -> dict:
        a = [v for i, v in lab if pred(F[i])]
        b = [v for i, v in lab if not pred(F[i])]
        return {"с признаком": [a.count(0), len(a)], "без": [b.count(0), len(b)]}

    checks = {
        "код изменил направление": rate(lambda f: f["тип перевёрнут кодом"]),
        "оборот предположения": rate(lambda f: f["оборот предположения"]),
        "голоса 4 из 4": rate(lambda f: f["голоса (0–4)"] == 4),
        "тип «содержит»": rate(lambda f: f["тип"] == "содержит"),
        "тип «локализовано в»": rate(lambda f: f["тип"] == "локализовано в"),
        "территория по названию статьи": rate(lambda f: f["территория"] == "название статьи"),
        "цитата найдена дословно": rate(lambda f: f["цитата дословно"]),
        "вошла по повтору пары": rate(lambda f: f["вошла по повтору пары"]),
        "голоса в разметке": dict(Counter(F[i]["голоса (0–4)"] for i, _ in lab)),
        "виды ошибок у перевёрнутых кодом": dict(Counter(kind[i] for i, v in lab if not v and F[i]["тип перевёрнут кодом"])),
        "виды ошибок у остальных": dict(Counter(kind[i] for i, v in lab if not v and not F[i]["тип перевёрнут кодом"])),
    }

    rows = [F[i] for i, _ in lab]
    y = np.array([v for _, v in lab])
    X, names = SL.matrix(rows)
    model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5, class_weight="balanced",
                                   random_state=SL.SEED).fit(X, y)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)[:, :, 1]
    base = float(ex.expected_value[1])
    k = next(j for j, (i, v) in enumerate(lab) if not v and F[i]["тип перевёрнут кодом"] and F[i]["тип"] == "входит в")
    order = np.argsort(-np.abs(sv[k]))
    example = {
        "связь (facts.id)": lab[k][0], "разметка": "неверна",
        "базовое значение": round(base, 3),
        "вклады": {names[j]: round(float(sv[k, j]), 3) for j in order[:6]},
        "прочие признаки": round(float(sv[k, order[6:]].sum()), 3),
        "оценка": round(base + float(sv[k].sum()), 3),
        "оценка модели": round(float(model.predict_proba(X[k:k + 1])[0, 1]), 3),
    }
    out = {"errors_by_feature": checks, "example": example}
    (HERE / "shap" / "checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
