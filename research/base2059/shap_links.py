"""Оценка доверия к связи и вклады признаков (SHAP) — проба на размеченных связях этой базы.

Метка — ручная разметка листа «Связи» (Разметка_база2059_проверено.xlsx: «Верно (1/0)»).
Признаки — то, что уже записано о связи или вычисляется по её цитате (база только читается):
голоса контрольных вопросов, вошла ли по повтору пары, сколько публикаций у пары, найдена ли
цитата дословно, откуда взята территория, тип связи, обороты предположения и отрицания,
язык цитаты, длина цитаты и имён, есть ли раздел.

Сначала — есть ли в признаках сигнал: повторная перекрёстная проверка (AUC) против той же
проверки на перемешанных метках. Затем SHAP (TreeExplainer) для случайного леса на всех
связях и устойчивость порядка признаков при повторных выборках.

    python research/base2059/shap_links.py  →  research/base2059/shap/
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "v16"))
sys.path.insert(0, str(ROOT / "research" / "trace_audit"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psycopg  # noqa: E402

import extraction_errors as XE  # noqa: E402
import trace_audit as T  # noqa: E402
from georag.evidence import gate  # noqa: E402
from georag.evidence.scope import resolve_scope  # noqa: E402
from georag.graph import store  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
SHEET = HERE / "Разметка_база2059_проверено.xlsx"
OUT = HERE / "shap"
SEED = 20261120
DERIV = {"EXPLICIT": "цитата", "ANAPHORA": "соседние предложения", "SECTION": "раздел", "TITLE": "название статьи"}


def labels() -> list[tuple[int, int]]:
    from openpyxl import load_workbook

    key = json.loads((HERE / "key.json").read_text("utf-8"))
    ws = load_workbook(SHEET, data_only=True)["Связи"]
    out = []
    for r in range(2, ws.max_row + 1):
        v = str(ws.cell(r, 8).value).strip()
        if v in ("0", "1"):
            out.append((key["facts"][r - 2], int(v)))
    return out


def features(ids: list[int]) -> dict[int, dict]:
    reg = gate.registry()
    with psycopg.connect(DSN + "?options=-c%20default_transaction_read_only%3Don") as c:
        rows = c.execute(
            """SELECT f.id, f.src, f.relation, f.dst, f.quote, f.votes, f.rel_type, f.rel_flip, f.src_key, f.dst_key,
                      ch.text, ch.headings, d.title
               FROM facts f JOIN chunks ch ON ch.id = f.chunk_id JOIN documents d ON d.doc_id = f.doc_id
               WHERE f.id = ANY(%s)""", (ids,)).fetchall()
        pairs = {(min(r[8], r[9]), max(r[8], r[9])): None for r in rows}
        for a, b in pairs:
            pairs[(a, b)] = c.execute(
                f"SELECT count(DISTINCT f.doc_id) FROM facts f WHERE {store.accepted(fresh=True)} "
                "AND least(f.src_key, f.dst_key) = %s AND greatest(f.src_key, f.dst_key) = %s", (a, b)).fetchone()[0]
    out = {}
    for fid, src, rel, dst, quote, votes, rtype, flip, sk, dk, text, heads, title in rows:
        quote = quote or ""
        sc = resolve_scope(quote, reg, chunk_text=T.ws(text), headings=heads or [], title=title or "")
        out[fid] = {
            "голоса (0–4)": votes or 0,
            "вошла по повтору пары": int((votes or 0) < store.MIN_VOTES),
            "публикаций у пары": pairs[(min(sk, dk), max(sk, dk))],
            "цитата дословно": int(T.locate(quote, text) == "дословно"),
            "территория": DERIV.get(sc.derivation, "не определена") if sc.resolved else "не определена",
            "тип": rtype or "без типа",
            "тип перевёрнут кодом": int(bool(flip)),
            "оборот предположения": int(XE.hedge_ext(rel, quote)),
            "отрицание в цитате": int(bool(XE.NEGATION.search(quote))),
            "цитата по-русски": int(bool(re.search("[а-яё]", quote, re.I))),
            "длина цитаты, знаков": len(quote),
            "слов в именах концов": len((src or "").split()) + len((dst or "").split()),
            "есть раздел": int(bool(heads)),
        }
        if CONTEXT:
            low = quote.lower()
            letters = sum(ch.isalpha() for ch in quote)
            out[fid].update({
                "цитата начинается с отсылки": int(bool(ANAPH_START.search(quote))),
                "отсылка внутри цитаты": int(bool(ANAPH_IN.search(quote))),
                "подпись рисунка или таблицы": int(bool(CAPTION.search(quote))),
                "номера строк или разрывы переноса": int(bool(LINE_NUM.search(quote) or HYPHEN_BREAK.search(quote))),
                "доля не-букв в цитате": round(1 - letters / max(1, len(quote)), 3),
                "оба конца в цитате дословно": int((src or "").lower() in low and (dst or "").lower() in low),
            })
    return out


# Признаки контекста и качества цитаты (--context): нужен ли смысл предыдущего предложения,
# не подпись ли это, не испорчен ли текст разбором PDF, названы ли концы так же, как в цитате.
CONTEXT = "--context" in sys.argv
ANAPH_START = re.compile(
    r"^\W*(?:this|these|those|it|its|they|their|such|here|there|the latter|the former|both|"
    r"он[аио]?|они|это|эти|этот|эта|здесь|там|данн\w*|последн\w*|так\w*)\b", re.I)
ANAPH_IN = re.compile(r"\b(?:this|these|those|such|the same|the latter|the former|данн\w+|указанн\w+|"
                      r"этот|эта|эти|этого|этой|этих|такие|такой)\s+\w+", re.I)
CAPTION = re.compile(r"\b(?:fig\.?|figure|table|рис\.?|рисунок|табл\.?|таблица)\s*\d", re.I)
LINE_NUM = re.compile(r"(?:^|\s)\d{2,4}\s+[A-ZА-ЯЁ]")
HYPHEN_BREAK = re.compile(r"\w- \w|\w -\w")


def matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    types = [t for t, n in Counter(r["тип"] for r in rows).most_common() if n >= 8]
    ters = sorted({r["территория"] for r in rows})
    names = [k for k in rows[0] if k not in ("тип", "территория")]
    names += [f"тип: {t}" for t in types] + ["тип: другой"] + [f"территория: {t}" for t in ters]
    X = []
    for r in rows:
        v = [r[k] for k in rows[0] if k not in ("тип", "территория")]
        v += [int(r["тип"] == t) for t in types] + [int(r["тип"] not in types)]
        v += [int(r["территория"] == t) for t in ters]
        X.append(v)
    return np.array(X, dtype=float), names


def cv_auc(model_fn, X, y, repeats=20, seed=SEED) -> list[float]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    out = []
    for r in range(repeats):
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            m = model_fn().fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, pred))
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lab = labels()
    feats = features([i for i, _ in lab])
    rows = [feats[i] for i, _ in lab]
    y = np.array([v for _, v in lab])
    X, names = matrix(rows)
    print(f"связей {len(y)}, верных {int(y.sum())}, ошибок {int((1 - y).sum())}, признаков {len(names)}", flush=True)
    if CONTEXT:
        out = OUT / "context"
        globals()["OUT"] = out
        # Простые доли ошибок по признакам контекста и голоса отдельно по языку цитаты.
        simple = {}
        for k in ("цитата по-русски", "цитата начинается с отсылки", "отсылка внутри цитаты",
                  "подпись рисунка или таблицы", "номера строк или разрывы переноса", "оба конца в цитате дословно"):
            a = [int(v) for r, v in zip(rows, y) if r[k]]
            b = [int(v) for r, v in zip(rows, y) if not r[k]]
            simple[k] = {"с признаком": [len(a) - sum(a), len(a)], "без": [len(b) - sum(b), len(b)]}
        for lang, flag in (("английские", 0), ("русские", 1)):
            sub = [(r, int(v)) for r, v in zip(rows, y) if r["цитата по-русски"] == flag]
            four = [v for r, v in sub if r["голоса (0–4)"] == 4]
            rest = [v for r, v in sub if r["голоса (0–4)"] < 4]
            simple[f"голоса, {lang} цитаты"] = {"4 из 4": [len(four) - sum(four), len(four)],
                                                "меньше 4": [len(rest) - sum(rest), len(rest)]}
        print(json.dumps(simple, ensure_ascii=False, indent=1), flush=True)

    def rf():
        return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5, class_weight="balanced",
                                      random_state=SEED)

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))

    rng = np.random.default_rng(SEED)
    res = {"links": len(y), "correct": int(y.sum()), "errors": int((1 - y).sum()), "features": names}
    if CONTEXT:
        res["errors_by_feature"] = simple
    for tag, fn in (("random_forest", rf), ("logistic", lr)):
        real = cv_auc(fn, X, y)
        null = [np.mean(cv_auc(fn, X, rng.permutation(y), repeats=2, seed=SEED + 100 + k)) for k in range(30)]
        res[tag] = {"cv_auc_mean": round(float(np.mean(real)), 3),
                    "cv_auc_range": [round(float(min(real)), 3), round(float(max(real)), 3)],
                    "shuffled_auc_mean": round(float(np.mean(null)), 3),
                    "shuffled_auc_95": round(float(np.percentile(null, 95)), 3),
                    "p_value": round((1 + sum(n >= np.mean(real) for n in null)) / (1 + len(null)), 3)}
        print(tag, res[tag], flush=True)

    # SHAP: лес на всех связях; класс 1 — «верна».
    model = rf().fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    mean_abs = np.abs(sv).mean(axis=0)
    direction = [float(np.corrcoef(X[:, j], sv[:, j])[0, 1]) if X[:, j].std() > 0 and sv[:, j].std() > 0 else 0.0
                 for j in range(len(names))]
    order = np.argsort(-mean_abs)

    # Устойчивость: место признака в порядке при 50 бутстрэп-выборках.
    ranks = {n: [] for n in names}
    for b in range(50):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        m = rf().fit(X[idx], y[idx])
        s = shap.TreeExplainer(m).shap_values(X)
        s = s[:, :, 1] if np.ndim(s) == 3 else (s[1] if isinstance(s, list) else s)
        ma = np.abs(s).mean(axis=0)
        for place, j in enumerate(np.argsort(-ma), 1):
            ranks[names[j]].append(place)
    res["shap"] = [{"feature": names[j], "mean_abs": round(float(mean_abs[j]), 4),
                    "direction": "выше значение → выше доверие" if direction[j] > 0.1 else
                                 "выше значение → ниже доверие" if direction[j] < -0.1 else "неоднозначно",
                    "rank_median": int(np.median(ranks[names[j]])),
                    "rank_90": [int(np.percentile(ranks[names[j]], 5)), int(np.percentile(ranks[names[j]], 95))],
                    "share_with_feature": round(float((X[:, j] > 0).mean()), 3)}
                   for j in order]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")

    plt.rcParams["font.family"] = "Times New Roman"
    shap.summary_plot(sv, X, feature_names=names, show=False, max_display=15, plot_size=(9, 6.5))
    plt.title("Вклад признаков в оценку «связь верна» (SHAP, случайный лес)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=160)
    print(json.dumps(res["shap"][:12], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
