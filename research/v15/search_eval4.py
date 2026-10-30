"""Поиск с местом после фонетической сверки (Symphonym) — на обоих наборах вопросов.

Разбор вопросов (место и понятия) — тот же, что в search_eval3.py (results/search3_key.json),
модель заново не спрашивается: меняется только сверка места с базой (concepts.resolve) и
отбор статей о месте (place_docs). Разметка — все прежние листы; новые фрагменты — в лист
Разметка_v15_доразметка2.xlsx.

    search_eval4.py run <xlsx set1> <xlsx set2> <xlsx доразметки>
    search_eval4.py score <xlsx set1> <xlsx set2> <xlsx доразметки> <xlsx доразметки2>
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
import search_eval3 as S3  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index import search as X  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402

RES = HERE / "results"
KEY3 = RES / "search3_key.json"
KEYFILE = RES / "search4_key.json"
SHEET = HERE / "Разметка_v15_доразметка2.xlsx"
SEED = 20261102


def _all_labels(x1: str, x2: str, x3: str, x4: str | None = None) -> dict[str, dict]:
    from openpyxl import load_workbook

    sets = S3._sets()
    labels = {"set1": S3._labels(x1, sets["set1"]["rows"]), "set2": S3._labels(x2, sets["set2"]["rows"])}
    for path, keyfile in ((x3, KEY3), (x4, KEYFILE)):
        if not path:
            continue
        rows = json.loads(keyfile.read_text("utf-8"))["rows"]
        marks = E._marks(load_workbook(path, data_only=True)["Поиск"], 5)
        for r in rows:
            if r["row"] in marks:
                labels[r["set"]][(r["query"], r["chunk_id"])] = marks[r["row"]]
    return labels


def run(x1: str, x2: str, x3: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    sets = S3._sets()
    key3 = json.loads(KEY3.read_text("utf-8"))
    labels = _all_labels(x1, x2, x3)
    emb, syn = Embedder(), S.load()
    key: dict = {"seed": SEED, "sets": {}}
    todo: list[tuple[str, int, int]] = []
    with connect(E.DSN) as conn:
        for name, data in sets.items():
            out = {}
            for q, prev in key3["sets"][name].items():
                qi, qs = int(q), data["queries"][int(q) - 1]
                names = C.resolve(conn, emb, prev["place"]) if prev["place"] else []
                docs = C.place_docs(conn, emb, names) if names else []
                pool = X.hybrid_search(conn, emb, qs, limit=100, candidates=50, max_per_doc=0,
                                       alternatives=S.synonym_variants(qs, syn), doc_id=docs or None)
                top = [h.chunk_id for h in X.cap_per_doc(C.rerank(pool, prev["concepts"]), E.PER_DOC)[: E.K]]
                out[q] = {"names": names, "docs": len(docs), "top": top, "changed": top != prev["top"]}
                todo += [(name, qi, c) for c in top if (qi, c) not in labels[name]]
                if top != prev["top"]:
                    print(name, qi, qs[:60], "| место:", names[:3], "статей", len(docs), "(было",
                          prev["docs"], ") | новых", sum((qi, c) not in labels[name] for c in top))
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
        "Доразметка v15, вторая часть: фрагменты, которых не было в прежних листах. Правила те же.",
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
    print(f"{SHEET}: {len(todo)} строк к доразметке; изменились выдачи у "
          f"{sum(o['changed'] for d in key['sets'].values() for o in d.values())} вопросов")
    score(x1, x2, x3, None)


def score(x1: str, x2: str, x3: str, x4: str | None) -> None:
    sets = S3._sets()
    key3 = json.loads(KEY3.read_text("utf-8"))
    key = json.loads(KEYFILE.read_text("utf-8"))
    labels = _all_labels(x1, x2, x3, x4)
    res: dict = {}
    for name, data in sets.items():
        modes = {"hybrid": {q: o["hybrid"] for q, o in data["runs"].items()},
                 "place": {q: o["top"] for q, o in key3["sets"][name].items()},
                 "place+phonetic": {q: o["top"] for q, o in key["sets"][name].items()}}
        if name == "set2":
            modes["concepts"] = {q: o["concepts"] for q, o in data["runs"].items()}
        res[name] = {}
        for m, runs in modes.items():
            vals = [labels[name].get((int(q), c)) for q, ids in runs.items() for c in ids]
            got = [v for v in vals if v is not None]
            res[name][m] = {"judged": len(got), "useful": sum(got),
                            "share_useful": round(sum(got) / len(got), 3) if got else None,
                            "ci95": E.wilson(sum(got), len(got)), "unjudged": vals.count(None)}
        res[name]["changed_questions"] = [int(q) for q, o in key["sets"][name].items() if o["changed"]]
    (RES / "search4_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
