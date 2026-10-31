"""Оценка по размеченным листам (labeling_sheet.py): фильтр, типизация, отрицания, модальность.
Считается только по строкам, где автор поставил 1 или 0; пустая строка не считается размеченной.

Фильтр (исправлено в v17, задачи 1–2 ТЗ v17). Область фильтра — фрагменты, не удалённые как
дубликаты и не подписи (4 132). Выборка v16 бралась только из фрагментов длиной от 200
символов (рамка — 4 009), а доли переносились на всю область — так считал v16
(filter_recall_v16_original, сохранён для сравнения). Теперь:
  frame_ge200 — оценка для рамки: веса страт — численности групп терминов в рамке;
  domain_all  — оценка для всей области: рамка (стратифицированная выборка) + короткие
                фрагменты (123, размечены сплошь, лист «Короткие» Разметка_v17; ошибки
                выборки у этой страты нет). Пока короткие не размечены, для области даются
                только границы: все короткие без связей / все со связями.
Для порога t отсеиваются фрагменты с числом терминов < t. Интервалы — бутстрэп внутри страт
(10 000 повторов, зерно 1). Экономия токенов — доля токенов отсеянных фрагментов в той же
совокупности (по n_tokens; прямой замер для порога 3 на 100 фрагментах — 21,3 %).

    labeling_score.py [xlsx_v16] [xlsx_v17] → results/prefilter_evaluation.csv,
                                              results/labeling_scores.json
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = HERE / "results"
BUCKETS = ("0", "1", "2", "3", "4", "5+")
FIELDS = ["population", "threshold", "chunks", "filtered_chunks", "filtered_share", "token_saving",
          "relation_recall", "recall_ci_low", "recall_ci_high", "lost_relation_share",
          "relation_share_among_filtered", "missed_relations", "status"]


def _marks(ws, col: int) -> dict[int, int]:
    out = {}
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, col).value
        if str(v).strip() in ("0", "1"):
            out[r - 1] = int(str(v).strip())
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def _bucket(t: int) -> str:
    return "5+" if t >= 5 else str(t)


def _drop(t: int) -> list[str]:
    return [b for b in BUCKETS if b != "5+" and int(b) < t]


def _ci(v: list[float]) -> list[float]:
    v = sorted(v)
    return [round(v[250], 3), round(v[9750], 3)]


def prefilter(key: dict, wb, path_v17: str | None) -> dict:
    """Пороги 1–4 для рамки и для всей области фильтра; исходный расчёт v16 — отдельно."""
    from corpus_structure import SHORT, load

    chunks = {c["chunk_id"]: c for c in load()}
    domain = [c for c in chunks.values() if not c["duplicate"] and not c["caption"]]
    frame = [c for c in domain if c["length"] >= SHORT]
    short = [c for c in domain if c["length"] < SHORT]
    pop_f = {b: sum(_bucket(c["terms"]) == b for c in frame) for b in BUCKETS}
    s_n = {b: sum(_bucket(c["terms"]) == b for c in short) for b in BUCKETS}
    tok_f = {b: sum(c["n_tokens"] for c in frame if _bucket(c["terms"]) == b) for b in BUCKETS}
    tok_s = {b: sum(c["n_tokens"] for c in short if _bucket(c["terms"]) == b) for b in BUCKETS}
    assert sum(pop_f.values()) + len(short) == len(domain)

    ws = wb["Фильтр"]
    fm = _marks(ws, 4)
    labels: dict[str, list[int]] = defaultdict(list)
    covered: dict[str, list[int]] = defaultdict(list)  # фрагмент со связями: есть ли в нём связь графа
    examples, unlabelled = [], []
    for i, item in enumerate(key["filter"], 1):
        assert chunks[item["chunk_id"]]["length"] >= SHORT  # выборка v16 — только из рамки
        b = _bucket(item["terms"])
        if i not in fm:
            unlabelled.append({"row": i, "chunk_id": item["chunk_id"], "terms": item["terms"],
                               "note": ws.cell(i + 1, 5).value})
            continue
        labels[b].append(fm[i])
        if fm[i]:
            covered[b].append(int(item["facts_graph"] > 0))
            if item["terms"] < 4:
                examples.append({"sheet": "Фильтр", "row": i, "chunk_id": item["chunk_id"], "terms": item["terms"],
                                 "relations": ws.cell(i + 1, 5).value})
    assert all(labels[b] for b in BUCKETS), "пустая страта"

    short_lab: dict[int, int] = {}
    if path_v17 and Path(path_v17).exists():
        from openpyxl import load_workbook

        k17 = json.loads((RES / "labeling_key_v17.json").read_text("utf-8"))
        ws17 = load_workbook(path_v17, data_only=True)["Короткие"]
        for i, v in _marks(ws17, 4).items():
            item = k17["short"][i - 1]
            short_lab[item["chunk_id"]] = v
            if v and item["terms"] < 4:
                examples.append({"sheet": "Короткие", "row": i, "chunk_id": item["chunk_id"], "terms": item["terms"],
                                 "relations": ws17.cell(i + 1, 5).value})
    # Неразмеченный короткий фрагмент не считается ни 0, ни 1: тогда для области — только границы.
    short_done = len(short_lab) == len(short)
    s_rel = {b: sum(short_lab.get(c["chunk_id"], 0) for c in short if _bucket(c["terms"]) == b) for b in BUCKETS}

    def estimate(sh: dict[str, float], short_rel: dict[str, float] | None):
        rel = {b: pop_f[b] * sh[b] + (short_rel[b] if short_rel else 0.0) for b in BUCKETS}
        tot = sum(rel.values())
        return {t: 1 - sum(rel[b] for b in _drop(t)) / tot for t in (1, 2, 3, 4)}, rel

    share = {b: statistics.mean(labels[b]) for b in BUCKETS}
    rng = random.Random(1)
    boot_f, boot_d, cov = defaultdict(list), defaultdict(list), []
    for _ in range(10000):
        sh = {b: statistics.mean(rng.choice(labels[b]) for _ in labels[b]) for b in BUCKETS}
        rf, rel = estimate(sh, None)
        for t in rf:
            boot_f[t].append(rf[t])
        if short_done:
            rd, _ = estimate(sh, s_rel)
            for t in rd:
                boot_d[t].append(rd[t])
        cv = {b: statistics.mean(rng.choice(covered[b]) for _ in covered[b]) if covered[b] else 0.0 for b in BUCKETS}
        cov.append(sum(rel[b] * cv[b] for b in BUCKETS) / sum(rel.values()))

    rec_f, rel_f = estimate(share, None)
    rec_d, rel_d = estimate(share, s_rel) if short_done else (None, None)
    tot_f, tot_d = sum(tok_f.values()), sum(tok_f.values()) + sum(tok_s.values())
    rows = []
    for t in (1, 2, 3, 4):
        dr = _drop(t)
        nf = sum(pop_f[b] for b in dr)
        lost = sum(rel_f[b] for b in dr)
        rows.append({"population": "frame_ge200", "threshold": t, "chunks": len(frame), "filtered_chunks": nf,
                     "filtered_share": round(nf / len(frame), 3),
                     "token_saving": round(sum(tok_f[b] for b in dr) / tot_f, 3),
                     "relation_recall": round(rec_f[t], 3), "recall_ci_low": _ci(boot_f[t])[0],
                     "recall_ci_high": _ci(boot_f[t])[1], "lost_relation_share": round(1 - rec_f[t], 3),
                     "relation_share_among_filtered": round(lost / nf, 3), "missed_relations": round(lost),
                     "status": "оценка по стратифицированной выборке (149 размеченных из 150)"})
        nd = nf + sum(s_n[b] for b in dr)
        base = {"population": "domain_all", "threshold": t, "chunks": len(domain), "filtered_chunks": nd,
                "filtered_share": round(nd / len(domain), 3),
                "token_saving": round(sum(tok_f[b] + tok_s[b] for b in dr) / tot_d, 3)}
        if short_done:
            lost_d = sum(rel_d[b] for b in dr)
            rows.append({**base, "relation_recall": round(rec_d[t], 3), "recall_ci_low": _ci(boot_d[t])[0],
                         "recall_ci_high": _ci(boot_d[t])[1], "lost_relation_share": round(1 - rec_d[t], 3),
                         "relation_share_among_filtered": round(lost_d / nd, 3), "missed_relations": round(lost_d),
                         "status": "оценка: выборка рамки + сплошная разметка коротких"})
        else:
            lo = estimate(share, {b: 0.0 for b in BUCKETS})[0][t]
            hi = estimate(share, {b: float(s_n[b]) for b in BUCKETS})[0][t]
            rows.append({**base, "relation_recall": "", "recall_ci_low": round(min(lo, hi), 3),
                         "recall_ci_high": round(max(lo, hi), 3), "lost_relation_share": "",
                         "relation_share_among_filtered": "", "missed_relations": "",
                         "status": "только границы точечной оценки: короткие фрагменты не размечены"})

    # Расчёт v16 как есть: доли рамки, перенесённые на всю область (для сравнения).
    pop_d = {b: pop_f[b] + s_n[b] for b in BUCKETS}
    assert pop_d == {b: key["filter_population"].get(b, 0) for b in BUCKETS}
    rel16 = {b: pop_d[b] * share[b] for b in BUCKETS}
    v16 = {t: round(1 - sum(rel16[b] for b in _drop(t)) / sum(rel16.values()), 3) for t in (1, 2, 3, 4)}

    point_cov = {b: statistics.mean(covered[b]) if covered[b] else 0.0 for b in BUCKETS}
    return {
        "filter_sample": {"selected": len(key["filter"]), "labelled": sum(len(v) for v in labels.values()),
                          "unlabelled": unlabelled},
        "filter_population": {"domain": len(domain), "frame_ge200": len(frame), "short_lt200": len(short),
                              "short_labelled": len(short_lab), "short_with_relations": sum(short_lab.values()),
                              "short_complete": short_done},
        "filter_by_terms": {b: {"labelled": len(labels[b]), "with_relations": sum(labels[b]), "share": share[b],
                                "ci": wilson(sum(labels[b]), len(labels[b])), "population_frame": pop_f[b],
                                "short": s_n[b], "short_with_relations": s_rel[b] if short_lab else None}
                            for b in BUCKETS},
        "filter_thresholds": rows,
        "filter_recall_v16_original": v16,
        "filter_missed_examples": examples,
        "graph_coverage_of_relation_chunks": {
            "population": "frame_ge200",
            "point": round(sum(rel_f[b] * point_cov[b] for b in BUCKETS) / sum(rel_f.values()), 3),
            "ci": _ci(cov), "by_terms": {b: [sum(covered[b]), len(covered[b])] for b in BUCKETS}},
    }


def main(path: str, path_v17: str | None = None) -> None:
    from openpyxl import load_workbook

    key = json.loads((RES / "labeling_key.json").read_text("utf-8"))
    wb = load_workbook(path, data_only=True)
    out: dict = prefilter(key, wb, path_v17)

    for sheet, name, col in (("Типизация", "typing_wrongly_excluded", 7), ("Отрицания", "negation_correct", 7),
                             ("Модальность", "modality_hedge", 7)):
        m = _marks(wb[sheet], col)
        k, n = sum(m.values()), len(m)
        out[name] = {"selected": wb[sheet].max_row - 1, "labelled": n, "ones": k,
                     "share": round(k / n, 3) if n else None, "ci": wilson(k, n)}
    by_note = defaultdict(lambda: [0, 0])
    for i, v in _marks(wb["Типизация"], 7).items():
        note = key["typing"][i - 1]["type_note"] or ""
        g = "виды концов не подходят к типу" if note.startswith("виды не подходят") else note
        by_note[g][0] += v
        by_note[g][1] += 1
    out["typing_by_reason"] = dict(by_note)
    if out["typing_wrongly_excluded"]["share"] is not None:
        out["typing_wrongly_excluded"]["population"] = key["typing_population"]

    with (RES / "prefilter_evaluation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(out["filter_thresholds"])
    (RES / "labeling_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "Разметка_v16_проверено.xlsx"),
         sys.argv[2] if len(sys.argv) > 2 else str(HERE / "Разметка_v17_проверено.xlsx"))
