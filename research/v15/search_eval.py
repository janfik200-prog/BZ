"""Статья v15: сравнение способов поиска и точность фактов графа — листы для разметки автора.

Способы поиска (по 5 фрагментов на вопрос, не больше 2 из одной статьи — как в системе):
  vector  — только по смыслу (векторы BGE-M3);
  text    — только полнотекстовый поиск с вариантами слов и синонимами;
  hybrid  — оба списка, объединённые по местам (RRF), как в системе;
  graph   — то же плюс третий список: фрагменты, из которых извлечены близкие к вопросу
            факты графа.
Выдачи всех способов сливаются в один перемешанный лист без указания способа (ключ —
в results/search_key.json, разметчику не открывать). Факты графа — случайные 120 из
графа (зерно зафиксировано).

    search_eval.py sheet   — построить research/v15/Разметка_v15.xlsx
    search_eval.py score   — прочитать размеченный лист, results/search_scores.json
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from georag.graph import store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SHEET = HERE / "Разметка_v15.xlsx"
K, PER_DOC, CAND = 5, 2, 50
SEED = 20261030
N_FACTS = 120
MODES = ("vector", "text", "hybrid", "graph")

# Вопросы записаны до прогона, по темам корпуса (признаки прогноза, методы, типы оруденения).
QUERIES = [
    "Какие гидротермальные изменения сопровождают золоторудные месторождения?",
    "Как выделяют зоны гидротермальных изменений по снимкам ASTER?",
    "Отношения каналов Landsat 8 для картирования оксидов железа и глинистых минералов",
    "Как по аэрогамма-спектрометрии выделяют калиевые метасоматиты?",
    "Отношение калия к торию как признак гидротермальных изменений",
    "Связь линеаментов по аэромагнитным данным с оруденением",
    "Плотность линеаментов и перспективные на золото участки",
    "Структурный контроль орогенных месторождений золота",
    "Какие разломы контролируют размещение золотого оруденения?",
    "Метод Эйлера для оценки глубины магнитных источников",
    "Гравиразведка для оконтуривания железорудных тел",
    "Машинное обучение для прогнозного картирования рудоносности",
    "Как оценивают неопределённость в прогнозных картах машинного обучения?",
    "Выбор обучающих точек и отрицательных примеров для прогнозных моделей",
    "Минералы-индикаторы золото-кварцевого оруденения",
    "Березитизация и лиственитизация вмещающих пород",
    "Строение месторождения Сухой Лог и форма рудных тел",
    "Кольцевые структуры на космоснимках и золотое оруденение",
    "Состав примесей в россыпном золоте и поиск коренного источника",
    "Редкометалльная минерализация в гранитоидах",
    "Колчеданные месторождения Алтая: условия формирования",
    "Гиперспектральная съёмка для определения минералов изменённых пород",
    "Метод главных компонент для выделения изменённых пород",
    "Аналитический иерархический процесс при картировании перспективности",
    "Золото-урановое оруденение Балтийского щита",
]


def _row_hit(rows):
    return [X._to_hit(r) for r in rows]


def run_modes(conn, emb, syn, q: str) -> tuple[dict[str, list[int]], dict[str, float]]:
    t = {}
    t0 = time.perf_counter()
    vec = emb.encode_one(q)
    t_vec = time.perf_counter() - t0
    out: dict[str, list[int]] = {}

    t0 = time.perf_counter()
    vrows = X.vector_search(conn, vec, CAND)
    t["vector"] = t_vec + time.perf_counter() - t0
    vhits = {r[0]: X._to_hit(r) for r in vrows}
    out["vector"] = [h.chunk_id for h in X.cap_per_doc([vhits[r[0]] for r in vrows], PER_DOC)[:K]]

    alts = S.synonym_variants(q, syn)
    t0 = time.perf_counter()
    trows = X.text_search(conn, q, CAND, alternatives=alts)
    if not trows:
        trows = X.text_search(conn, q, CAND, any_word=True, alternatives=alts)
    t["text"] = time.perf_counter() - t0
    thits = {r[0]: X._to_hit(r) for r in trows}
    out["text"] = [h.chunk_id for h in X.cap_per_doc([thits[r[0]] for r in trows], PER_DOC)[:K]]

    t0 = time.perf_counter()
    hy = X.hybrid_search(conn, emb, q, limit=K, candidates=CAND, max_per_doc=PER_DOC, alternatives=alts, vector=vec)
    t["hybrid"] = t_vec + time.perf_counter() - t0
    out["hybrid"] = [h.chunk_id for h in hy]

    t0 = time.perf_counter()
    frows = [r for r in X.fact_search(conn, vec, CAND) if r[1] >= X.FACT_MIN_SIMILARITY]
    fact_ids = [r[0] for r in frows]
    hits = {**vhits, **thits}
    missing = [c for c in fact_ids if c not in hits]
    for r in X._chunks_by_id(conn, missing, vec) if missing else []:
        hits[r[0]] = X._to_hit(r)
    scores = X.rrf_merge([[r[0] for r in vrows], [r[0] for r in trows], fact_ids])
    ordered = sorted((hits[c] for c in scores if c in hits), key=lambda h: -scores[h.chunk_id])
    out["graph"] = [h.chunk_id for h in X.cap_per_doc(ordered, PER_DOC)[:K]]
    t["graph"] = t["hybrid"] + time.perf_counter() - t0
    return out, t


def _chunk_info(conn, ids: list[int]) -> dict[int, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, d.title, c.text FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
                    "WHERE c.id = ANY(%s)", (ids,))
        return {int(i): (t or "", x or "") for i, t, x in cur.fetchall()}


def sheet() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    RES.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    emb = Embedder()
    syn = S.load()
    runs, times, pool = {}, {m: [] for m in MODES}, []
    with connect(DSN) as conn:
        for qi, q in enumerate(QUERIES, 1):
            out, t = run_modes(conn, emb, syn, q)
            runs[qi] = out
            for m in MODES:
                times[m].append(t[m])
        pairs = sorted({(qi, c) for qi, out in runs.items() for ids in out.values() for c in ids})
        info = _chunk_info(conn, sorted({c for _, c in pairs}))
        facts = store.all_facts(conn)
    rng.shuffle(pairs)
    sample = rng.sample(facts, N_FACTS)

    key = {"seed": SEED, "k": K, "per_doc": PER_DOC, "queries": QUERIES,
           "runs": {str(q): o for q, o in runs.items()},
           "search_rows": [{"row": i + 2, "query": q, "chunk_id": c} for i, (q, c) in enumerate(pairs)],
           "fact_rows": [{"row": i + 2, "fact_id": f["id"], "votes": f["votes"]} for i, f in enumerate(sample)],
           "latency_s": {m: {"median": round(statistics.median(v), 3), "max": round(max(v), 3)} for m, v in times.items()}}
    (RES / "search_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")

    wb = Workbook()
    head = Font(bold=True)
    fill = PatternFill("solid", fgColor="FFF2CC")
    wrap = Alignment(wrap_text=True, vertical="top")
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Разметка для статьи v15. Два листа: «Поиск» и «Факты». Заполняются только жёлтые столбцы.",
        "",
        "ЛИСТ «ПОИСК». В строке — вопрос и найденный фрагмент статьи. Способ поиска не указан намеренно.",
        "Столбец «Полезен»: 1 — фрагмент содержит сведения, которые отвечают на вопрос или нужны для ответа;",
        "0 — не по вопросу (тот же термин, но о другом; общие слова; список литературы и т. п.).",
        "Оценивайте только текст фрагмента, не название статьи.",
        "",
        "ЛИСТ «ФАКТЫ». В строке — связь из графа знаний и предложение статьи, из которого она извлечена.",
        "Столбец «Верно»: 1 — предложение действительно утверждает эту связь между этими предметами",
        "(направление и смысл связи верные); 0 — не утверждает, искажено, предметы перепутаны или это не геология.",
        "Если связь в предложении высказана как предположение и в факте есть пометка «предположительно» — это 1.",
        "",
        "Пустую клетку можно оставить, если не уверены: такая строка в расчёт не войдёт.",
        "В столбце «Комментарий» можно пояснить спорные случаи.",
        "Файл с ключом (results/search_key.json) до конца разметки не открывать.",
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
    ws = wb.create_sheet("Факты")
    ws.append(["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Верно (1/0)", "Комментарий"])
    for i, f in enumerate(sample, 1):
        ws.append([i, f["src"], f["relation"], f["dst"], f["quote"], f["title"], None, None])
    for col, w in zip("ABCDEFGH", (6, 24, 18, 24, 90, 30, 11, 30)):
        ws.column_dimensions[col].width = w
    for ws in wb.worksheets[1:]:
        for cell in ws[1]:
            cell.font = head
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        mark = 5 if ws.title == "Поиск" else 7
        for r in range(1, ws.max_row + 1):
            ws.cell(r, mark).fill = fill
        ws.freeze_panes = "A2"
    wb.save(SHEET)
    print(f"{SHEET}: поиск {len(pairs)} строк, факты {len(sample)}; задержки {key['latency_s']}")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def _marks(ws, col: int) -> dict[int, int]:
    out = {}
    for r in range(2, ws.max_row + 1):
        v = ws.cell(r, col).value
        if v in (0, 1, "0", "1"):
            out[r] = int(v)
    return out


def score(path: Path = SHEET) -> None:
    from openpyxl import load_workbook

    key = json.loads((RES / "search_key.json").read_text("utf-8"))
    wb = load_workbook(path, data_only=True)
    rel = _marks(wb["Поиск"], 5)
    ok = _marks(wb["Факты"], 7)
    judged = {(r["query"], r["chunk_id"]): rel[r["row"]] for r in key["search_rows"] if r["row"] in rel}
    res: dict = {"search": {}, "facts": {}, "latency_s": key["latency_s"]}
    # Полезные фрагменты, найденные хоть одним способом, — знаменатель «доли найденного».
    useful = {q: {c for (qq, c), v in judged.items() if qq == int(q) and v} for q in key["runs"]}
    for m in MODES:
        p_at, found, n, k_rel, empty = [], [], 0, 0, 0
        for q, out in key["runs"].items():
            ids = out[m]
            marks = [judged[(int(q), c)] for c in ids if (int(q), c) in judged]
            if not ids:
                empty += 1
            n += len(marks)
            k_rel += sum(marks)
            if marks:
                p_at.append(sum(marks) / K)
            if useful[q]:
                found.append(len(useful[q] & set(ids)) / len(useful[q]))
        res["search"][m] = {"judged": n, "useful": k_rel, "share_useful": round(k_rel / n, 3) if n else None,
                            "ci95": wilson(k_rel, n), "precision_at_5_mean": round(statistics.mean(p_at), 3) if p_at else None,
                            "share_of_pooled_useful_found": round(statistics.mean(found), 3) if found else None,
                            "queries_without_results": empty}
    votes = {r["row"]: r["votes"] for r in key["fact_rows"]}
    k = sum(ok.values())
    res["facts"] = {"judged": len(ok), "correct": k, "precision": round(k / len(ok), 3) if ok else None,
                    "ci95": wilson(k, len(ok)), "unjudged": len(votes) - len(ok)}
    res["search_rows_judged"] = len(judged)
    res["search_rows_total"] = len(key["search_rows"])
    (RES / "search_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(Path(sys.argv[2]) if len(sys.argv) > 2 else SHEET)
    else:
        sheet()
