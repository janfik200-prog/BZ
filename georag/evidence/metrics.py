"""WP12. Метрики по независимой разметке: FTR вместе с охватом, избирательный риск,
доверительные интервалы.

* FTR — ложных DIRECT среди пар, размеченных как чужие/несовместимые (ANALOG,
  OUT_OF_SCOPE): свой знаменатель, не доля всех фактов.
* DIRECT_coverage — вопросы, где есть хотя бы один верный DIRECT, среди вопросов, на
  которые по разметке есть прямой ответ. Без охвата FTR не интерпретируется: постоянный
  UNRESOLVED даёт FTR = 0 и охват 0.
* SelectiveRisk — ложные DIRECT среди всех выданных DIRECT.
* Wilson 95 % для долей; для разностей двух методов — бутстреп по семействам источников
  (кластер = семейство: пары из одной статьи не независимы).

Разметка только DEV/TEST (не SYNTHETIC); семейство не может быть в обоих разбиениях.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

VALID = {"DIRECT", "ANALOG", "OUT_OF_SCOPE", "UNRESOLVED"}
FOREIGN = {"ANALOG", "OUT_OF_SCOPE"}
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def check_gold(gold: list[dict[str, Any]]) -> None:
    """Разметка годится для оценки: DEV/TEST, без утечки семейств, статусы из словаря."""
    fam: dict[str, str] = {}
    ids = set()
    for g in gold:
        split = (g.get("split") or "").upper()
        if split not in ("DEV", "TEST"):
            raise ValueError(f"{g.get('sample_id')}: split {split!r} — только DEV/TEST, синтетика не оценивается")
        f = g["source_family"]
        if fam.setdefault(f, split) != split:
            raise ValueError(f"семейство {f} и в DEV, и в TEST — утечка")
        if g["gold_applicability"] not in VALID:
            raise ValueError(f"{g['sample_id']}: статус {g['gold_applicability']!r}")
        if g["sample_id"] in ids:
            raise ValueError(f"повтор sample_id {g['sample_id']}")
        ids.add(g["sample_id"])


@dataclass
class MetricsReport:
    method: str
    split: str
    n_pairs: int
    n_families: int
    n_foreign: int
    false_direct_foreign: int
    ftr: float | None
    ftr_ci: tuple[float | None, float | None]
    n_answerable: int
    covered: int
    direct_coverage: float | None
    coverage_ci: tuple[float | None, float | None]
    n_direct_pred: int
    n_false_direct: int
    selective_risk: float | None
    selective_risk_ci: tuple[float | None, float | None]
    n_gold_direct: int
    direct_recall_pairs: float | None
    unresolved_rate: float
    accuracy4: float

    def row(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def compute_metrics(raw_predictions: list[dict[str, Any]], gold: list[dict[str, Any]], method: str, split: str) -> MetricsReport:
    """raw_predictions: sample_id, predicted_status (по одному на пару для method).
    Отказывается считать, если метод покрыл не все пары разбиения."""
    check_gold(gold)
    g = {r["sample_id"]: r for r in gold if r["split"].upper() == split}
    p = {r["sample_id"]: r["predicted_status"] for r in raw_predictions if r["sample_id"] in g}
    if set(p) != set(g):
        raise ValueError(f"{method}: предсказаний {len(p)} из {len(g)} пар {split} — сравнение не считается")
    pairs = [(g[s], p[s]) for s in sorted(g)]
    foreign = [(x, y) for x, y in pairs if x["gold_applicability"] in FOREIGN]
    fd = sum(y == "DIRECT" for _, y in foreign)
    answerable = {x["query_id"] for x, _ in pairs if x["gold_applicability"] == "DIRECT"}
    covered = {x["query_id"] for x, y in pairs if x["gold_applicability"] == "DIRECT" and y == "DIRECT"}
    pred_direct = [(x, y) for x, y in pairs if y == "DIRECT"]
    false_direct = sum(x["gold_applicability"] != "DIRECT" for x, _ in pred_direct)
    gold_direct = [(x, y) for x, y in pairs if x["gold_applicability"] == "DIRECT"]
    return MetricsReport(
        method=method,
        split=split,
        n_pairs=len(pairs),
        n_families=len({x["source_family"] for x, _ in pairs}),
        n_foreign=len(foreign),
        false_direct_foreign=fd,
        ftr=fd / len(foreign) if foreign else None,
        ftr_ci=wilson(fd, len(foreign)),
        n_answerable=len(answerable),
        covered=len(covered),
        direct_coverage=len(covered) / len(answerable) if answerable else None,
        coverage_ci=wilson(len(covered), len(answerable)),
        n_direct_pred=len(pred_direct),
        n_false_direct=false_direct,
        selective_risk=false_direct / len(pred_direct) if pred_direct else None,
        selective_risk_ci=wilson(false_direct, len(pred_direct)),
        n_gold_direct=len(gold_direct),
        direct_recall_pairs=sum(y == "DIRECT" for _, y in gold_direct) / len(gold_direct) if gold_direct else None,
        unresolved_rate=sum(y == "UNRESOLVED" for _, y in pairs) / len(pairs),
        accuracy4=sum(x["gold_applicability"] == y for x, y in pairs) / len(pairs),
    )


def _stats(pairs: list[tuple[dict[str, Any], str]]) -> tuple[float | None, float | None]:
    foreign = [y for x, y in pairs if x["gold_applicability"] in FOREIGN]
    answerable = {x["query_id"] for x, _ in pairs if x["gold_applicability"] == "DIRECT"}
    covered = {x["query_id"] for x, y in pairs if x["gold_applicability"] == "DIRECT" and y == "DIRECT"}
    ftr = sum(y == "DIRECT" for y in foreign) / len(foreign) if foreign else None
    cov = len(covered) / len(answerable) if answerable else None
    return ftr, cov


def paired_bootstrap(
    preds_a: list[dict[str, Any]],
    preds_b: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    split: str,
    n_boot: int = 5000,
    seed: int = 20261029,
) -> dict[str, Any]:
    """Разность FTR и охвата (A − B) с 95 % перцентильным интервалом; ресэмплинг —
    семейства источников целиком (пары одной статьи вместе)."""
    g = {r["sample_id"]: r for r in gold if r["split"].upper() == split}
    pa = {r["sample_id"]: r["predicted_status"] for r in preds_a if r["sample_id"] in g}
    pb = {r["sample_id"]: r["predicted_status"] for r in preds_b if r["sample_id"] in g}
    by_fam: defaultdict[str, list[str]] = defaultdict(list)
    for s, r in g.items():
        by_fam[r["source_family"]].append(s)
    fams = sorted(by_fam)
    rng = random.Random(seed)

    def diff(sample: list[str]) -> tuple[float | None, float | None]:
        fa, ca = _stats([(g[s], pa[s]) for s in sample])
        fb, cb = _stats([(g[s], pb[s]) for s in sample])
        d_ftr = None if fa is None or fb is None else fa - fb
        d_cov = None if ca is None or cb is None else ca - cb
        return d_ftr, d_cov

    point = diff(sorted(g))
    ftrs, covs = [], []
    for _ in range(n_boot):
        sample = [s for f in (rng.choice(fams) for _ in fams) for s in by_fam[f]]
        a, b = diff(sample)
        if a is not None:
            ftrs.append(a)
        if b is not None:
            covs.append(b)

    def ci(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return (None, None)
        xs = sorted(xs)
        return (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))])

    return {
        "n_families": len(fams),
        "n_boot": n_boot,
        "delta_ftr": point[0],
        "delta_ftr_ci95": ci(ftrs),
        "delta_coverage": point[1],
        "delta_coverage_ci95": ci(covs),
        "note": "кластер — семейство источника; при < 10 семействах интервал ориентировочный",
    }


def confusion(raw_predictions: list[dict[str, Any]], gold: list[dict[str, Any]], split: str) -> dict[str, dict[str, int]]:
    g = {r["sample_id"]: r["gold_applicability"] for r in gold if r["split"].upper() == split}
    out: dict[str, dict[str, int]] = {a: {b: 0 for b in sorted(VALID)} for a in sorted(VALID)}
    for r in raw_predictions:
        if r["sample_id"] in g:
            out[g[r["sample_id"]]][r["predicted_status"]] += 1
    return out
