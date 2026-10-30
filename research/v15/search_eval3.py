"""Правки 1–3 ключевых понятий (сверка места с базой, поиск в два слоя, длинные понятия по
главным словам) на обоих наборах вопросов, по уже сделанной разметке.

  set1 — 25 вопросов search_eval.py (на них понятия настраивались);
  set2 — 20 вопросов search_eval2.py (на них найдены ошибки, которые правят 1–3).
Новый способ «place»: разбор вопроса моделью → место есть и найдено в базе — поиск
(по смыслу и по словам) только в статьях о нём, иначе по всей базе → кандидаты с
понятиями выше → не больше 2 из статьи → до 5 (меньше, если статей о месте мало).
Фрагменты, которых нет в прежней разметке, — в лист Разметка_v15_доразметка.xlsx.

    search_eval3.py run <xlsx set1> <xlsx set2>
    search_eval3.py score <xlsx set1> <xlsx set2> <xlsx доразметки>
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import search_eval as E  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.llm import LLM  # noqa: E402

RES = HERE / "results"
KEYFILE = RES / "search3_key.json"
SHEET = HERE / "Разметка_v15_доразметка.xlsx"
SEED = 20261101
POOL = 100
BASE_MODE = {"set1": "hybrid", "set2": "hybrid"}
PREV_MODE = {"set1": None, "set2": "concepts"}


def _labels(xlsx: str, rows: list[dict]) -> dict[tuple[int, int], int]:
    from openpyxl import load_workbook

    ws = load_workbook(xlsx, data_only=True)["Поиск"]
    marks = E._marks(ws, 5)
    return {(r["query"], r["chunk_id"]): marks[r["row"]] for r in rows if r["row"] in marks}


def _sets() -> dict[str, dict]:
    k1 = json.loads((RES / "search_key.json").read_text("utf-8"))
    k2 = json.loads((RES / "search2_key.json").read_text("utf-8"))
    return {"set1": {"queries": k1["queries"], "runs": k1["runs"], "rows": k1["search_rows"]},
            "set2": {"queries": k2["queries"], "runs": k2["runs"], "rows": k2["rows"]}}


def run(x1: str, x2: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    sets = _sets()
    labels = {"set1": _labels(x1, sets["set1"]["rows"]), "set2": _labels(x2, sets["set2"]["rows"])}
    emb, syn = Embedder(), S.load()
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=120, temperature=0.0)
    key: dict = {"seed": SEED, "sets": {}}
    todo: list[tuple[str, int, int]] = []
    with connect(E.DSN) as conn:
        for name, data in sets.items():
            out = {}
            for qi, q in enumerate(data["queries"], 1):
                parsed = C.parse(llm, q)
                names = C.resolve(conn, emb, parsed.place) if parsed.place else []
                docs = C.place_docs(conn, emb, names) if names else []
                pool = X.hybrid_search(conn, emb, q, limit=POOL, candidates=50, max_per_doc=0,
                                       alternatives=S.synonym_variants(q, syn), doc_id=docs or None)
                top = [h.chunk_id for h in X.cap_per_doc(C.rerank(pool, parsed.concepts), E.PER_DOC)[: E.K]]
                out[str(qi)] = {"place": parsed.place, "names": names, "docs": len(docs),
                                "concepts": parsed.concepts, "top": top}
                todo += [(name, qi, c) for c in top if (qi, c) not in labels[name]]
                print(name, qi, q, "| место:", names, "статей", len(docs), "| новых", sum(
                    (qi, c) not in labels[name] for c in top))
            key["sets"][name] = out
        todo = sorted(set(todo))
        info = E._chunk_info(conn, sorted({c for _, _, c in todo}))
    random.Random(SEED).shuffle(todo)
    key["rows"] = [{"row": i + 2, "set": s, "query": q, "chunk_id": c} for i, (s, q, c) in enumerate(todo)]
    KEYFILE.write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Доразметка v15: фрагменты, которых не было в прежних листах. Правила те же.",
        "«Полезен»: 1 — фрагмент содержит сведения, которые отвечают на вопрос или нужны для ответа;",
        "0 — не по вопросу (тот же термин, но о другом; другое место; общие слова; ссылки на литературу).",
        "Если вопрос о конкретном месте, а фрагмент о другом месте — 0. Не уверены — оставьте пустым.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Поиск")
    ws.append(["№", "Вопрос", "Статья", "Фрагмент", "Полезен (1/0)", "Комментарий"])
    for i, (s, q, c) in enumerate(todo, 1):
        title, text = info[c]
        ws.append([i, sets[s]["queries"][q - 1], title, text, None, None])
    for col, w in zip("ABCDEF", (6, 34, 30, 110, 12, 30)):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for r in range(1, ws.max_row + 1):
        ws.cell(r, 5).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "A2"
    wb.save(SHEET)
    print(f"{SHEET}: {len(todo)} строк к доразметке")
    score(x1, x2, None)


def score(x1: str, x2: str, x3: str | None) -> None:
    from openpyxl import load_workbook

    sets = _sets()
    key = json.loads(KEYFILE.read_text("utf-8"))
    labels = {"set1": _labels(x1, sets["set1"]["rows"]), "set2": _labels(x2, sets["set2"]["rows"])}
    if x3:
        marks = E._marks(load_workbook(x3, data_only=True)["Поиск"], 5)
        for r in key["rows"]:
            if r["row"] in marks:
                labels[r["set"]][(r["query"], r["chunk_id"])] = marks[r["row"]]
    res: dict = {}
    for name, data in sets.items():
        res[name] = {}
        modes = {"hybrid": {q: o["hybrid"] for q, o in data["runs"].items()},
                 "place": {q: o["top"] for q, o in key["sets"][name].items()}}
        if name == "set2":
            modes["concepts"] = {q: o["concepts"] for q, o in data["runs"].items()}
        for m, runs in modes.items():
            vals = [labels[name].get((int(q), c)) for q, ids in runs.items() for c in ids]
            got = [v for v in vals if v is not None]
            res[name][m] = {"returned": len(vals), "judged": len(got), "useful": sum(got),
                            "share_useful": round(sum(got) / len(got), 3) if got else None,
                            "ci95": E.wilson(sum(got), len(got)), "unjudged": vals.count(None)}
        # Попарно по вопросам: место и прежний лучший способ.
        prev = "concepts" if name == "set2" else "hybrid"
        diffs = []
        for q, o in key["sets"][name].items():
            a = [labels[name].get((int(q), c)) for c in o["top"]]
            b = [labels[name].get((int(q), c)) for c in modes[prev][q]]
            if None not in a and None not in b:
                diffs.append(sum(a) / max(len(a), 1) - sum(b) / len(b))
        res[name]["vs_" + prev] = {"queries": len(diffs), "better": sum(d > 0 for d in diffs),
                                   "worse": sum(d < 0 for d in diffs), "same": sum(d == 0 for d in diffs)}
    (RES / "search3_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
