"""Проверка ключевых понятий на новых вопросах (после настройки на search_eval.py).

20 вопросов записаны 30.10.2026 после того, как модуль georag/index/concepts.py и словарь
config/term-translations.yaml закоммичены (eb55114): 12 — о конкретных местах базы,
8 — о темах, которых не было среди первых 25. Способы (по 5 фрагментов, не больше 2 из
статьи):
  vector   — только по смыслу;
  hybrid   — объединение по смыслу и по словам, как в системе сейчас;
  concepts — то же, но кандидаты с ключевыми понятиями вопроса выше (GEORAG_CONCEPTS).
Выдачи сливаются в один перемешанный лист без указания способа.

    search_eval2.py sheet           — research/v15/Разметка_v15_проверка.xlsx
    search_eval2.py score <xlsx>    — results/search2_scores.json
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import search_eval as E  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.llm import LLM  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SHEET = HERE / "Разметка_v15_проверка.xlsx"
KEYFILE = RES / "search2_key.json"
SEED = 20261031
MODES = ("vector", "hybrid", "concepts")
POOL = 100

QUERIES = [
    "Какие разломы и зоны сдвига контролируют золотое оруденение в Нубийском щите Египта?",
    "Гидротермальные изменения на руднике Уругвай в районе Минас-ду-Камакуан",
    "Структура геофизических полей месторождения Сухой Лог",
    "Этапы гидротермально-метасоматических процессов золотого оруденения Среднего Урала",
    "Перспективы крупнообъёмных золоторудных месторождений Енисейского кряжа",
    "Золотое оруденение Чулковского участка Селигдарского рудного поля",
    "Прогноз медного оруденения в массиве Колдар",
    "Роль кольцевых структур гор Букантау в прогнозе золотого оруденения",
    "Рудно-магматические системы Восточного Донбасса",
    "Металлогеническая зональность Северного Вьетнама",
    "Россыпи золота и их коренные источники в Нахичеванской автономной республике",
    "Тектоническая эволюция уранового рудного поля Лудзин",
    "Геофизические признаки эпитермального золотого месторождения",
    "Выявление изменённых пород радарной съёмкой в тропических районах",
    "Прогнозное картирование при сильном дисбалансе обучающих данных",
    "Определение источника обломочного золота по составу примесей",
    "Золотое оруденение карлинского типа в терригенно-карбонатных толщах",
    "Редкоземельные элементы в оловоносных гранитах",
    "Разделение потенциальных полей на региональную и локальную составляющие",
    "Радиоактивность как поисковый признак редкометалльных месторождений",
]


def run_one(conn, emb, syn, llm, q: str) -> tuple[dict[str, list[int]], list[list[str]], dict[str, float]]:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    vec = emb.encode_one(q)
    t_vec = time.perf_counter() - t0
    out: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    vrows = X.vector_search(conn, vec, X.RRF_K)
    out["vector"] = [h.chunk_id for h in X.cap_per_doc([X._to_hit(r) for r in vrows], E.PER_DOC)[: E.K]]
    t["vector"] = t_vec + time.perf_counter() - t0
    alts = S.synonym_variants(q, syn)
    t0 = time.perf_counter()
    pool = X.hybrid_search(conn, emb, q, limit=POOL, candidates=50, max_per_doc=0, alternatives=alts, vector=vec)
    t["hybrid"] = t_vec + time.perf_counter() - t0
    out["hybrid"] = [h.chunk_id for h in X.cap_per_doc(pool, E.PER_DOC)[: E.K]]
    t0 = time.perf_counter()
    concepts = C.extract(llm, q)
    t["concepts_llm"] = time.perf_counter() - t0
    out["concepts"] = [h.chunk_id for h in X.cap_per_doc(C.rerank(pool, concepts), E.PER_DOC)[: E.K]]
    return out, concepts, t


def sheet() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    rng = random.Random(SEED)
    emb, syn = Embedder(), S.load()
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=120, temperature=0.0)
    runs, found, times = {}, {}, []
    with connect(E.DSN) as conn:
        for qi, q in enumerate(QUERIES, 1):
            runs[qi], found[qi], t = run_one(conn, emb, syn, llm, q)
            times.append(t)
            print(qi, q, "→", found[qi])
        pairs = sorted({(qi, c) for qi, o in runs.items() for ids in o.values() for c in ids})
        info = E._chunk_info(conn, sorted({c for _, c in pairs}))
    rng.shuffle(pairs)
    key = {"seed": SEED, "queries": QUERIES, "concepts": {str(k): v for k, v in found.items()},
           "runs": {str(k): v for k, v in runs.items()},
           "rows": [{"row": i + 2, "query": q, "chunk_id": c} for i, (q, c) in enumerate(pairs)],
           "latency_s": {m: round(statistics.median(t[m] for t in times[1:]), 3)
                         for m in ("vector", "hybrid", "concepts_llm")}}
    KEYFILE.write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Проверочная разметка v15: 20 новых вопросов. Заполняется только жёлтый столбец листа «Поиск».",
        "",
        "В строке — вопрос и найденный фрагмент статьи. Способ поиска не указан намеренно.",
        "«Полезен»: 1 — фрагмент содержит сведения, которые отвечают на вопрос или нужны для ответа;",
        "0 — не по вопросу (тот же термин, но о другом; другое место; общие слова; ссылки на литературу).",
        "Если вопрос о конкретном месте, а фрагмент о другом месте — 0.",
        "Оценивайте только текст фрагмента, не название статьи. Не уверены — оставьте пустым.",
        "Файл ключа (results/search2_key.json) до конца разметки не открывать.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Поиск")
    ws.append(["№", "Вопрос", "Статья", "Фрагмент", "Полезен (1/0)", "Комментарий"])
    for i, (q, c) in enumerate(pairs, 1):
        title, text = info[c]
        ws.append([i, QUERIES[q - 1], title, text, None, None])
    for col, w in zip("ABCDEF", (6, 34, 30, 110, 12, 30)):
        ws.column_dimensions[col].width = w
    fill = PatternFill("solid", fgColor="FFF2CC")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for r in range(1, ws.max_row + 1):
        ws.cell(r, 5).fill = fill
    ws.freeze_panes = "A2"
    wb.save(SHEET)
    print(f"{SHEET}: {len(pairs)} строк; задержки {key['latency_s']}")


def score(xlsx: str) -> None:
    from openpyxl import load_workbook

    key = json.loads(KEYFILE.read_text("utf-8"))
    ws = load_workbook(xlsx, data_only=True)["Поиск"]
    marks = E._marks(ws, 5)
    lab = {(r["query"], r["chunk_id"]): marks[r["row"]] for r in key["rows"] if r["row"] in marks}
    useful = {q: {c for (qq, c), v in lab.items() if qq == int(q) and v} for q in key["runs"]}
    places = set(range(1, 13))
    res: dict = {"rows_judged": len(lab), "rows_total": len(key["rows"]), "latency_s": key["latency_s"]}
    for part, qs in (("all", None), ("places", places), ("themes", set(range(13, 21)))):
        res[part] = {}
        for m in MODES:
            n = k = 0
            found = []
            for q, out in key["runs"].items():
                if qs and int(q) not in qs:
                    continue
                vals = [lab[(int(q), c)] for c in out[m] if (int(q), c) in lab]
                n += len(vals)
                k += sum(vals)
                if useful[q]:
                    found.append(len(useful[q] & set(out[m])) / len(useful[q]))
            res[part][m] = {"judged": n, "useful": k, "share_useful": round(k / n, 3) if n else None,
                            "ci95": E.wilson(k, n), "share_of_pooled_useful_found":
                            round(statistics.mean(found), 3) if found else None}
    (RES / "search2_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sheet() if sys.argv[1] == "sheet" else score(sys.argv[2])
