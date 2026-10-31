"""Задача 8 ТЗ v17: автоматические проверки комплекта и сравнение с опубликованным v16.

Проверки (каждая — assert с понятным сообщением; итог — results/checks.json):
  1. числа CSV и JSON согласованы (фильтр, поиск, воронка);
  2. неразмеченные строки не считаются размеченными (пусто ≠ 0/1), счёт сходится с листом;
  3. знаменатели: сумма по стратам = число размеченных; доли = k/n;
  4. нет пустых статистических групп;
  5. число связей сходится по стадиям (13 739 = граф + кандидаты + исключённые типизацией);
  6. исторические данные не изменились: отпечаток таблиц georag_v14 совпадает с
     сохранённым до запусков (results/db_fingerprint.json), повторный прогон v16-скриптов
     дал те же значения, что опубликованы в research/v16/results.
Сравнение «v16 → исправлено» — results/reproduction_comparison.csv.

    checks.py [--fingerprint] → results/checks.json, results/reproduction_comparison.csv
    --fingerprint — только записать отпечаток базы (до всех шагов).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V16 = HERE.parent / "v16" / "results"
RES = HERE / "results"
DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
TABLES = {  # таблица → столбцы, по которым считается отпечаток
    "facts": "id, chunk_id, doc_id, src, relation, dst, quote, votes, rel_type, rel_flip, typed_at, type_note",
    "facts_pass": "chunk_id, facts, rejected",
    "chunks": "id, doc_id, ord, md5(text)",
    "documents": "doc_id, title",
}


def fingerprint() -> dict[str, str]:
    import psycopg

    out = {}
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        for table, cols in TABLES.items():
            key = cols.split(",")[0]
            cur.execute(f"SELECT {cols} FROM {table} ORDER BY {key}")
            h = hashlib.sha256()
            n = 0
            for row in cur:
                h.update(repr(row).encode("utf-8"))
                n += 1
            out[table] = f"{n}:{h.hexdigest()[:16]}"
    return out


def load(path: Path):
    return json.loads(path.read_text("utf-8"))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    if "--fingerprint" in sys.argv:
        (RES / "db_fingerprint.json").write_text(json.dumps(fingerprint(), indent=1), "utf-8")
        print(fingerprint())
        return
    checks: dict[str, str] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = "ok" if ok else f"FAIL {detail}"

    # 6. Историческая база не изменилась.
    before = load(RES / "db_fingerprint.json")
    check("база georag_v14 не изменилась", fingerprint() == before, str(before))

    # 5. Воронка.
    audit = load(RES / "relation_pipeline_summary.json")
    check("13 739 = граф + кандидаты + исключённые",
          audit["stored"] == audit["graph"] + audit["candidates"] + audit["excluded_typing"])
    check("все проверки сходимости relation_audit", all(audit["check"].values()), str(audit["check"]))
    final = Counter(r["decision"] for r in rows(RES / "relation_pipeline_audit.csv") if r["stage"] == "final")
    check("CSV воронки = JSON", final["graph"] == audit["graph"] and final["candidate"] == audit["candidates"]
          and final["excluded_typing"] == audit["excluded_typing"] and final["rejected"] == audit["rejected_by_code"],
          str(final))
    un = load(RES / "untyped_53_summary.json")
    check("53 без типизации = граф − типизированные",
          un["untyped_in_graph"] == audit["graph"] - audit["typing"]["тип поставлен"])
    check("53 в CSV", len(rows(RES / "untyped_53_audit.csv")) == un["untyped_in_graph"])

    # 1–4. Разметка и фильтр.
    lab = load(RES / "labeling_scores.json")
    fs = lab["filter_sample"]
    check("фильтр: отобрано = размечено + неразмеченные", fs["selected"] == fs["labelled"] + len(fs["unlabelled"]))
    by = lab["filter_by_terms"]
    check("фильтр: сумма страт = размечено", sum(v["labelled"] for v in by.values()) == fs["labelled"])
    check("фильтр: нет пустых страт", all(v["labelled"] > 0 for v in by.values()))
    check("фильтр: доли = k/n", all(abs(v["share"] - v["with_relations"] / v["labelled"]) < 1e-9 for v in by.values()))
    pop = lab["filter_population"]
    check("фильтр: рамка + короткие = область", pop["frame_ge200"] + pop["short_lt200"] == pop["domain"])
    csv_rows = rows(RES / "prefilter_evaluation.csv")
    js = {(r["population"], str(r["threshold"])): r for r in lab["filter_thresholds"]}
    check("фильтр: CSV = JSON", len(csv_rows) == len(js) and all(
        str(js[(r["population"], r["threshold"])]["relation_recall"]) == r["relation_recall"]
        and str(js[(r["population"], r["threshold"])]["filtered_chunks"]) == r["filtered_chunks"] for r in csv_rows))
    check("фильтр: оценка области не выдаётся без разметки коротких",
          pop["short_complete"] or all(r["relation_recall"] == "" for r in csv_rows if r["population"] == "domain_all"))
    for name in ("typing_wrongly_excluded", "negation_correct", "modality_hedge"):
        x = lab[name]
        check(f"{name}: размечено ≤ отобрано, доля = k/n",
              x["labelled"] <= x["selected"] and abs(x["share"] - round(x["ones"] / x["labelled"], 3)) < 1e-9)

    # 1. Поиск: CSV по вопросам = сводка JSON.
    summ = load(RES / "retrieval_summary.json")["by_set"]["all"]
    agg = defaultdict(list)
    for r in rows(RES / "retrieval_results.csv"):
        agg[r["variant"]].append(int(r["article_top1"]))
    check("поиск: 173 вопроса в каждом варианте", all(len(v) == summ["n"] for v in agg.values()))
    check("поиск: CSV = JSON (статья Top-1)",
          all(abs(sum(v) / len(v) - summ[k]["first"]) < 0.0015 for k, v in agg.items()))
    pc = rows(RES / "territory_pairwise_cases.csv")
    ps = load(RES / "pairwise_summary.json")
    check("парный разбор: исходы сходятся с метриками",
          sum(int(r["b3_first"]) for r in pc) == sum(agg["B3"]) and sum(int(r["b2t_first"]) for r in pc) == sum(agg["B2t"])
          and sum(ps["outcomes"].values()) == summ["n"])

    # Статья v17: численность разметки и ключевые числа — те же, что в JSON.
    art = HERE.parent / "v14" / "article" / "Статья_ИПУЗ_2026_v17.docx"
    if art.exists():
        from docx import Document

        doc = Document(str(art))
        text = " ".join(p.text for p in doc.paragraphs) + " " + " ".join(
            c.text for t in doc.tables for r in t.rows for c in r.cells)
        main_pop = "domain_all" if pop["short_complete"] else "frame_ge200"
        f3 = next(r for r in lab["filter_thresholds"] if r["population"] == main_pop and r["threshold"] == 3)
        need = {
            "размечено из отобрано": f"{fs['labelled']} из {fs['selected']}",
            "потеря фильтра, порог 3": f"{100 * f3['lost_relation_share']:.1f} %".replace(".", ","),
            "типизированных связей": str(un["variants"]["B"]["relations"]),
            "уникальных типизированных троек": str(un["variants"]["B"]["unique_triples"]),
            "типизация напрасно": f"{lab['typing_wrongly_excluded']['ones']} из {lab['typing_wrongly_excluded']['labelled']}",
            "отрицания": f"{lab['negation_correct']['ones']}",
        }
        missing = [k for k, v in need.items() if v not in text]
        check("статья v17: числа совпадают с JSON", not missing, str(missing))
        check("статья v17: нет рабочих пометок", not any(m in text for m in ("TODO", "[N]", "[M]", "XXX", "??")))

    # Сравнение с опубликованным v16.
    comp = []

    def add(metric, v16, new, source, why):
        try:
            diff = round(float(new) - float(v16), 3)
        except (TypeError, ValueError):  # не число: пусто, «149/150», границы
            diff = ""
        comp.append([metric, v16, new, diff, source, why])

    a16 = load(V16 / "relation_pipeline_summary.json")
    for k in ("proposed", "rejected_by_code", "stored", "confirmed", "candidates", "excluded_typing", "graph",
              "graph_unique_relations"):
        add(f"воронка: {k}", a16[k], audit[k], "relation_pipeline_summary.json", "повторный запуск; совпадает"
            if a16[k] == audit[k] else "ПРОВЕРИТЬ")
    add("граф: только типизированные связи", "", un["variants"]["B"]["relations"], "untyped_53_summary.json",
        "новое: 53 связи приняты по повтору без типизации")
    add("граф: уникальные типизированные тройки", "", un["variants"]["B"]["unique_triples"], "untyped_53_summary.json",
        "новое: 3 783 включали 45 троек со словом статьи вместо типа")
    r16 = load(V16 / "retrieval_summary.json")["by_set"]["all"]
    for v in ("B0", "B1", "B2", "B2t", "B3", "B4", "B3+gf", "B3+gn"):
        for m in ("top5", "first", "frag", "mrr10"):
            add(f"поиск {v}: {m}", r16[v][m], summ[v][m], "retrieval_summary.json",
                "повторный запуск; совпадает" if r16[v][m] == summ[v][m] else "ПРОВЕРИТЬ")
    l16 = load(V16 / "labeling_scores.json")
    f16 = {int(t): r for t, *r in [(x[0], *x[1:]) for x in l16["filter_thresholds"]]}
    frame = {r["threshold"]: r for r in lab["filter_thresholds"] if r["population"] == "frame_ge200"}
    dom = {r["threshold"]: r for r in lab["filter_thresholds"] if r["population"] == "domain_all"}
    for t in (1, 2, 3, 4):
        old = l16["filter_recall_ci"][str(t)]
        add(f"фильтр порог {t}: полнота (рамка ≥200 знаков)", f16[t][3], frame[t]["relation_recall"],
            "labeling_scores.json", "веса страт — по рамке, из которой бралась выборка (v16: по всей области)")
        add(f"фильтр порог {t}: 95 % интервал, нижняя граница", old[0], frame[t]["recall_ci_low"],
            "labeling_scores.json", "то же")
        add(f"фильтр порог {t}: полнота (вся область)", "", dom[t]["relation_recall"] or
            f"{dom[t]['recall_ci_low']}–{dom[t]['recall_ci_high']} (границы)", "labeling_scores.json",
            dom[t]["status"])
        add(f"фильтр порог {t}: экономия токенов (рамка)", f16[t][2], frame[t]["token_saving"],
            "labeling_scores.json", "доля токенов рамки, а не всей области")
    add("фильтр: размечено / отобрано", "149/150", f"{fs['labelled']}/{fs['selected']}", "labeling_scores.json",
        "строка 36 оставлена автором пустой: повреждённая таблица; не считается")
    add("охват графа фрагментов со связями", l16["graph_coverage_of_relation_chunks"]["point"],
        lab["graph_coverage_of_relation_chunks"]["point"], "labeling_scores.json", "веса страт — по рамке")
    for k in ("typing_wrongly_excluded", "negation_correct", "modality_hedge"):
        add(f"{k}: доля", l16[k]["share"], lab[k]["share"], "labeling_scores.json",
            "совпадает" if l16[k]["share"] == lab[k]["share"] else "ПРОВЕРИТЬ")
    with (RES / "reproduction_comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "v16_value", "corrected_value", "difference", "source", "explanation"])
        w.writerows(comp)
    unexplained = [c for c in comp if c[5] == "ПРОВЕРИТЬ"]
    check("нет необъяснённых расхождений с v16", not unexplained, str(unexplained))
    (RES / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    if any(v != "ok" for v in checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
