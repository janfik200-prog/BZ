"""Итоговая проверка для статьи: разметка автора.

25 новых вопросов (записаны 30.10.2026 до прогона; на них ничего не настраивалось):
13 — о местах базы, которые не встречались ни в прежних вопросах, ни в проверке сверки
мест (place_names_eval.py), 12 — о темах, которых не было среди прежних 45. Способы
(по 5 фрагментов, не больше 2 из статьи):
  vector  — только по смыслу;
  hybrid  — объединение по смыслу и по словам, как в системе сейчас;
  system  — разбор вопроса моделью, место → статьи о нём, ключевые понятия выше
            (GEORAG_CONCEPTS, georag/index/concepts.py).
В лист «Поиск» подмешаны 40 пар из прежней проверки, размеченной языковой моделью
(Codex), — чтобы посчитать согласие разметчиков. Лист «Факты» — те же 120 случайных связей
графа, что в первом листе. Способ и происхождение строк скрыты (ключ — results/final_key.json).

    final_eval.py sheet <xlsx проверки Codex>   — research/v15/Разметка_статья.xlsx
    final_eval.py score <xlsx автора> <xlsx первого листа Codex> <xlsx проверки Codex>
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import search_eval as E  # noqa: E402

from georag.graph import store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.llm import LLM  # noqa: E402

RES = HERE / "results"
KEYFILE = RES / "final_key.json"
SHEET = HERE / "Разметка_статья.xlsx"
SEED = 20261103
CONTROL = 40
MODES = ("vector", "hybrid", "system")
PLACES = set(range(1, 14))

QUERIES = [
    "Картирование минералов изменённых пород в районе Маунтин-Пасс",
    "Гиперспектральное картирование изменений в районе Куприт (Невада)",
    "Золотоносность супергруппы Рорайма в Гайане",
    "Литологические и структурные признаки золотого оруденения зеленокаменного пояса Менге",
    "Контроль медного оруденения в районе Китуи (Кения)",
    "Геологическое строение месторождения Арылах",
    "Прогноз золоторудных проявлений Баягского рудного поля",
    "Геологическое картирование золоторудного узла Олон-Овоот по космоснимкам",
    "История разведки и потенциал свинцово-цинкового района Центральной Арктики (Нунавут)",
    "Радиоактивность как поисковый признак месторождений Восточного Казахстана",
    "Золотоносность терригенно-карбонатных толщ Алмалыкского рудного района",
    "Строение Кировоградской и Звенигородско-Братской зон разломов Украинского щита",
    "Геофизические поиски эпитермального золота в бассейне Кастро",
    "Генеративно-состязательные сети в прогнозном картировании",
    "Инверсия гравиметрических данных ансамблевым фильтром Калмана",
    "Квантовые акселерометры для спутниковой гравиметрии",
    "Сейсмическая визуализация для оконтуривания рудных объектов",
    "Палеороссыпная модель золота в метаконгломератах",
    "Спектральные признаки железа и алюминия по снимкам Sentinel-2",
    "Гиперспектральные снимки PRISMA для картирования изменённых пород",
    "Геохимическая зональность эндогенного оруденения по вторичным ореолам",
    "Критерии эпитермального кислотно-сульфатного оруденения",
    "Тепловой поток и разуплотнение коры в рифтовых и плюмовых структурах",
    "Морфология и состав самородного золота",
    "Оценка ресурсов полезных ископаемых с помощью дистанционного зондирования и ГИС",
]


def run_one(conn, emb, syn, llm, q: str) -> tuple[dict, dict]:
    vec = emb.encode_one(q)
    vrows = X.vector_search(conn, vec, 50)
    out = {"vector": [h.chunk_id for h in X.cap_per_doc([X._to_hit(r) for r in vrows], E.PER_DOC)[: E.K]]}
    alts = S.synonym_variants(q, syn)
    pool = X.hybrid_search(conn, emb, q, limit=100, candidates=50, max_per_doc=0, alternatives=alts, vector=vec)
    out["hybrid"] = [h.chunk_id for h in X.cap_per_doc(pool, E.PER_DOC)[: E.K]]
    t0 = time.perf_counter()
    parsed = C.parse(llm, q)
    names = C.resolve(conn, emb, parsed.place) if parsed.place else []
    docs = C.place_docs(conn, emb, names) if names else []
    spool = X.hybrid_search(conn, emb, q, limit=100, candidates=50, max_per_doc=0, alternatives=alts,
                            vector=vec, doc_id=docs or None) if docs else pool
    out["system"] = [h.chunk_id for h in X.cap_per_doc(C.rerank(spool, parsed.concepts), E.PER_DOC)[: E.K]]
    info = {"place": parsed.place, "names": names, "docs": len(docs), "concepts": parsed.concepts,
            "seconds_extra": round(time.perf_counter() - t0, 2)}
    return out, info


def sheet(codex_check: str) -> None:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    rng = random.Random(SEED)
    emb, syn = Embedder(), S.load()
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=120, temperature=0.0)
    runs, infos = {}, {}
    with connect(E.DSN) as conn:
        for qi, q in enumerate(QUERIES, 1):
            runs[qi], infos[qi] = run_one(conn, emb, syn, llm, q)
            print(qi, q[:60], "| место:", infos[qi]["names"][:3], "статей", infos[qi]["docs"])
        rows = [("new", qi, QUERIES[qi - 1], c) for qi in runs for c in
                dict.fromkeys(c for m in MODES for c in runs[qi][m])]
        # Контроль согласия: пары прежней проверки, размеченные Codex.
        k2 = json.loads((RES / "search2_key.json").read_text("utf-8"))
        ws2 = load_workbook(codex_check, data_only=True)["Поиск"]
        marked = [r for r in k2["rows"] if ws2.cell(r["row"], 5).value in (0, 1, "0", "1")]
        rows += [("control", r["query"], k2["queries"][r["query"] - 1], r["chunk_id"])
                 for r in rng.sample(marked, CONTROL)]
        info = E._chunk_info(conn, sorted({r[3] for r in rows}))
        k1 = json.loads((RES / "search_key.json").read_text("utf-8"))
        fact_ids = [r["fact_id"] for r in k1["fact_rows"]]
        facts = {f["id"]: f for f in store.all_facts(conn) if f["id"] in set(fact_ids)}
    rng.shuffle(rows)
    key = {"seed": SEED, "queries": QUERIES, "runs": {str(k): v for k, v in runs.items()},
           "info": {str(k): v for k, v in infos.items()},
           "search_rows": [{"row": i + 2, "kind": k, "query": q, "chunk_id": c} for i, (k, q, _, c) in enumerate(rows)],
           "fact_rows": [{"row": i + 2, "fact_id": fid} for i, fid in enumerate(fact_ids)],
           "seconds_extra_median": statistics.median(v["seconds_extra"] for v in infos.values())}
    KEYFILE.write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")

    wb = Workbook()
    head, wrap = Font(bold=True), Alignment(wrap_text=True, vertical="top")
    fill = PatternFill("solid", fgColor="FFF2CC")
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Разметка для статьи. Два листа: «Поиск» и «Факты». Заполняются только жёлтые столбцы: 1 или 0.",
        "Не уверены — оставьте клетку пустой: строка в расчёт не войдёт. Спорные случаи — в «Комментарий».",
        "",
        "ЛИСТ «ПОИСК». В строке — вопрос и фрагмент статьи, найденный поиском. Каким способом — не указано.",
        "1 — фрагмент содержит сведения, которые отвечают на вопрос или нужны для ответа.",
        "0 — не по вопросу: тот же термин, но о другом; другое место или другой метод; общие слова;",
        "    только цель работы, ключевые слова или история объекта без сведений по вопросу; ссылки на литературу.",
        "Если вопрос о конкретном месте, а фрагмент о другом месте — 0. Если место из фрагмента не названо,",
        "но из названия статьи ясно, что речь о нём, — оценивайте по содержанию.",
        "",
        "ЛИСТ «ФАКТЫ». В строке — связь из графа знаний и предложение статьи, из которого она извлечена.",
        "1 — предложение действительно утверждает эту связь между этими предметами: смысл и направление верные.",
        "0 — не утверждает, искажено, направление обратное, предметы перепутаны или это не геология.",
        "Если в предложении связь высказана как предположение, а в факте есть «предположительно», — это 1;",
        "если оговорка в факте потеряна — 0.",
        "",
        "Файл ключа (research/v15/results/final_key.json) до конца разметки не открывать.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Поиск")
    ws.append(["№", "Вопрос", "Статья", "Фрагмент", "Полезен (1/0)", "Комментарий"])
    for i, (_, _, q, c) in enumerate(rows, 1):
        title, text = info[c]
        ws.append([i, q, title, text, None, None])
    for col, w in zip("ABCDEF", (6, 34, 30, 110, 12, 30)):
        ws.column_dimensions[col].width = w
    ws = wb.create_sheet("Факты")
    ws.append(["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Верно (1/0)", "Комментарий"])
    for i, fid in enumerate(fact_ids, 1):
        f = facts[fid]
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
    print(f"{SHEET}: поиск {len(rows)} строк (из них контроль {CONTROL}), факты {len(fact_ids)}")


def _paired(diffs: list[float]) -> list[float]:
    rng = random.Random(1)
    boot = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    return [round(statistics.mean(diffs), 3), round(boot[250], 3), round(boot[9750], 3)]


def score(author: str, codex_first: str, codex_check: str) -> None:
    from openpyxl import load_workbook

    key = json.loads(KEYFILE.read_text("utf-8"))
    wb = load_workbook(author, data_only=True)
    sm, fm = E._marks(wb["Поиск"], 5), E._marks(wb["Факты"], 7)
    lab = {(r["query"], r["chunk_id"]): sm[r["row"]] for r in key["search_rows"]
           if r["kind"] == "new" and r["row"] in sm}
    res: dict = {"search_rows": len(key["search_rows"]), "search_judged": len(sm), "facts_judged": len(fm),
                 "seconds_extra_median": key["seconds_extra_median"]}
    for part, qs in (("all", None), ("places", PLACES), ("themes", set(range(14, 26)))):
        res[part] = {}
        per_q: dict[str, dict[int, float]] = {m: {} for m in MODES}
        for m in MODES:
            vals = []
            for q, out in key["runs"].items():
                if qs and int(q) not in qs:
                    continue
                v = [lab[(int(q), c)] for c in out[m] if (int(q), c) in lab]
                vals += v
                if len(v) == len(out[m]) and v:
                    per_q[m][int(q)] = sum(v) / len(v)
            res[part][m] = {"judged": len(vals), "useful": sum(vals),
                            "share_useful": round(sum(vals) / len(vals), 3) if vals else None,
                            "ci95": E.wilson(sum(vals), len(vals))}
        common = sorted(set(per_q["system"]) & set(per_q["hybrid"]) & set(per_q["vector"]))
        if common:
            res[part]["system_minus_hybrid"] = _paired([per_q["system"][q] - per_q["hybrid"][q] for q in common])
            res[part]["system_minus_vector"] = _paired([per_q["system"][q] - per_q["vector"][q] for q in common])
            res[part]["questions"] = len(common)
    k = sum(fm.values())
    res["facts"] = {"judged": len(fm), "correct": k, "precision": round(k / len(fm), 3) if fm else None,
                    "ci95": E.wilson(k, len(fm))}
    # Согласие с Codex: контрольные пары поиска и те же 120 фактов.
    k2 = json.loads((RES / "search2_key.json").read_text("utf-8"))
    cm = E._marks(load_workbook(codex_check, data_only=True)["Поиск"], 5)
    codex2 = {(r["query"], r["chunk_id"]): cm[r["row"]] for r in k2["rows"] if r["row"] in cm}
    pairs = [(sm[r["row"]], codex2[(r["query"], r["chunk_id"])]) for r in key["search_rows"]
             if r["kind"] == "control" and r["row"] in sm and (r["query"], r["chunk_id"]) in codex2]
    cf = E._marks(load_workbook(codex_first, data_only=True)["Факты"], 7)
    fpairs = [(fm[r], cf[r]) for r in fm if r in cf]

    def agree(p):
        if not p:
            return None
        n = len(p)
        po = sum(a == b for a, b in p) / n
        pa, pb = sum(a for a, _ in p) / n, sum(b for _, b in p) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None}

    res["agreement_with_codex"] = {"search": agree(pairs), "facts": agree(fpairs)}
    (RES / "final_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        sheet(sys.argv[2])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
