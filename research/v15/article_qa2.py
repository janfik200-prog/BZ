"""Проверка на новых вопросах: вариант «по смыслу + место» выбран по 89 вопросам article_qa.py,
его преимущество проверяется на других вопросах, составленных тем же способом по другим
фрагментам тех же статей (не исходный фрагмент первой проверки и не соседние с ним).

Варианты поиска — те же, что в ablation.py; понятия и место вопроса выделяются заново.

    article_qa2.py gen      — фрагменты и вопросы → results/article_qa2.json
    article_qa2.py run      — все варианты → results/article_qa2_runs.json, лист проверки
    article_qa2.py score [xlsx автора]  — results/article_qa2_scores.json
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

import ablation as A  # noqa: E402
import article_qa as Q  # noqa: E402
import search_eval as E  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402
from georag.index import concepts as C  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.llm import LLM  # noqa: E402

RES = HERE / "results"
QA2 = RES / "article_qa2.json"
RUNS2 = RES / "article_qa2_runs.json"
SHEET2 = HERE / "Разметка_вопросы_по_статьям_2.xlsx"
VARIANTS = ("hybrid", "hybrid+place", "hybrid+concepts", "hybrid+place+concepts",
            "vector", "vector+place", "vector+concepts", "vector+place+concepts")


def pick_fragments(conn) -> list[dict]:
    """По фрагменту на статью: больше всего терминов словаря, но не фрагмент первой проверки
    и не соседний с ним (±1 по порядку)."""
    used = {x["chunk_id"] for x in json.loads(Q.QA.read_text("utf-8"))}
    finder = Finder(use_spacy=False)
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, c.doc_id, c.ord, c.headings, c.text, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.doc_id, c.ord")
        rows = cur.fetchall()
    used_ord = {(doc, o) for cid, doc, o, *_ in rows if cid in used}
    best: dict[str, tuple[int, dict]] = {}
    for cid, doc, o, heads, text, title in rows:
        if cid in used or {(doc, o - 1), (doc, o), (doc, o + 1)} & used_ord:
            continue
        if Q.SKIP_HEAD.search(" ".join(heads or [])) or len(text or "") < 400:
            continue
        n = len({t.lower() for t in finder.find(text or "").terms})
        if n > best.get(doc, (-1, {}))[0]:
            best[doc] = (n, {"doc_id": doc, "chunk_id": cid, "title": title, "text": text, "terms": n})
    return [v for _, v in sorted(best.values(), key=lambda x: x[1]["doc_id"])]


def gen() -> None:
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=180, temperature=0.0)
    with connect(E.DSN) as conn:
        frags = pick_fragments(conn)
    out = []
    for i, f in enumerate(frags, 1):
        try:
            data = llm.chat_json(Q.SYSTEM, f["text"][:3000], temperature=0.0, max_tokens=300)
        except Exception as exc:  # noqa: BLE001
            data = {"question": None, "error": str(exc)}
        q = (data or {}).get("question")
        out.append({**{k: f[k] for k in ("doc_id", "chunk_id", "title", "terms")},
                    "question": q, "answer": (data or {}).get("answer")})
        print(i, f["doc_id"][:30], "|", q, flush=True)
    QA2.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"вопросов {sum(bool(x['question']) for x in out)} из {len(out)} статей")


def run() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    items = [x for x in json.loads(QA2.read_text("utf-8")) if x["question"]]
    emb, syn = Embedder(), S.load()
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=120, temperature=0.0)
    with connect(E.DSN) as conn:
        cur = conn.cursor()
        for x in items:
            t0 = time.perf_counter()
            parsed = C.parse(llm, x["question"])
            names = C.resolve(conn, emb, parsed.place) if parsed.place else []
            x["info"] = {"place": parsed.place, "names": names, "concepts": parsed.concepts,
                         "seconds_extra": round(time.perf_counter() - t0, 2)}
            x["runs"] = A.run_variants(conn, emb, syn, x)
            cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)",
                        (sorted({c for ids in x["runs"].values() for c in ids}),))
            doc = dict(cur.fetchall())
            x["docs"] = {v: [doc[c] for c in ids] for v, ids in x["runs"].items()}
            print(x["question"][:60], "| место:", names[:2], flush=True)
        texts = E._chunk_info(conn, [x["chunk_id"] for x in items])
    RUNS2.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Вторая проверка вопросов, составленных по статьям базы (другие фрагменты тех же статей).",
        "Заполняется один жёлтый столбец листа «Вопросы» — так же, как в первый раз.",
        "1 — вопрос осмысленный, геологический и фрагмент действительно на него отвечает.",
        "0 — вопрос бессмысленный, не о геологии, о самой статье, или фрагмент на него не отвечает.",
        "Не уверены — оставьте пустым. Такие вопросы в расчёт не войдут.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 110
    ws = wb.create_sheet("Вопросы")
    ws.append(["№", "Вопрос", "Ответ по фрагменту", "Фрагмент", "Статья", "Годится (1/0)", "Комментарий"])
    for i, x in enumerate(items, 1):
        ws.append([i, x["question"], x["answer"], texts[x["chunk_id"]][1], x["title"], None, None])
    for col, w in zip("ABCDEFG", (5, 40, 40, 90, 30, 11, 30)):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for r in range(1, ws.max_row + 1):
        ws.cell(r, 6).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "A2"
    wb.save(SHEET2)
    print(f"{SHEET2}: {len(items)} вопросов")
    score(None)


def _paired(d: list[int]) -> list[float]:
    rng = random.Random(1)
    boot = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(10000))
    return [round(statistics.mean(d), 3), round(boot[250], 3), round(boot[9750], 3)]


def score(xlsx: str | None) -> None:
    items = json.loads(RUNS2.read_text("utf-8"))
    keep = set(range(1, len(items) + 1))
    res: dict = {"questions_generated": len(items), "checked_by_author": bool(xlsx)}
    if xlsx:
        from openpyxl import load_workbook

        marks = E._marks(load_workbook(xlsx, data_only=True)["Вопросы"], 6)
        keep = {r - 1 for r, v in marks.items() if v == 1}
        res["questions_checked"] = len(marks)
    sel = [x for i, x in enumerate(items, 1) if i in keep]
    res["questions_used"] = len(sel)
    res["seconds_extra_median"] = statistics.median(x["info"]["seconds_extra"] for x in sel)
    for part, xs in (("all", sel), ("place", [x for x in sel if x["info"]["names"]]),
                     ("noplace", [x for x in sel if not x["info"]["names"]])):
        if not xs:
            continue
        res[part] = {"n": len(xs)}
        hits = {}
        for v in VARIANTS:
            hits[v] = {"top5": [int(x["doc_id"] in x["docs"][v]) for x in xs],
                       "first": [int(bool(x["docs"][v]) and x["docs"][v][0] == x["doc_id"]) for x in xs],
                       "frag": [int(x["chunk_id"] in x["runs"][v]) for x in xs]}
            res[part][v] = {m: round(statistics.mean(h), 3) for m, h in hits[v].items()}
        for a, b in (("vector+place", "vector"), ("vector+place", "hybrid+place+concepts"),
                     ("vector+concepts", "vector"), ("hybrid+place+concepts", "vector")):
            res[part][f"{a} - {b}"] = {m: _paired([p - q for p, q in zip(hits[a][m], hits[b][m])])
                                       for m in ("top5", "first", "frag")}
    (RES / "article_qa2_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"gen": gen, "run": run}.get(sys.argv[1], lambda: score(sys.argv[2] if len(sys.argv) > 2 else None))()
