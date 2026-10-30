"""Вопросы по статьям базы: у каждого вопроса есть ответ — фрагмент, по которому он составлен.

Из каждой статьи берётся один содержательный фрагмент (больше всего терминов геологического
словаря, без списков литературы и служебных разделов); локальная модель (qwen3-14b)
составляет по нему вопрос на русском, какой задал бы геолог, и краткий ответ по фрагменту.
Затем три способа поиска (как final_eval.py: по смыслу, объединение, система с местом и
понятиями) — и мера без ручной разметки: нашёл ли способ среди пяти первых исходную статью
и исходный фрагмент. Автор отмечает только, годится ли вопрос (лист «Вопросы»).

    article_qa.py gen      — фрагменты и вопросы → results/article_qa.json
    article_qa.py run      — поиск тремя способами → results/article_qa_runs.json, лист проверки
    article_qa.py score [xlsx автора]  — results/article_qa_scores.json
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import final_eval as F  # noqa: E402
import search_eval as E  # noqa: E402

from georag.graph import synonyms as S  # noqa: E402
from georag.graph.entities import Finder  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.llm import LLM  # noqa: E402

RES = HERE / "results"
QA = RES / "article_qa.json"
RUNS = RES / "article_qa_runs.json"
SHEET = HERE / "Разметка_вопросы_по_статьям.xlsx"
MODES = F.MODES
SKIP_HEAD = re.compile(r"referenc|литератур|acknowledg|благодар|funding|финансир|author|автор|"
                       r"conflict|data availab|supplement|abstract|аннотац", re.I)

SYSTEM = (
    "По фрагменту научной статьи составь один вопрос на русском языке, какой мог бы задать "
    "геолог, занятый прогнозом рудоносности, и на который этот фрагмент даёт ответ. Вопрос — "
    "о сути геологии: признаки, методы, объекты, условия, результаты; не о самой статье (без слов "
    "«в статье», «авторы», «в работе»). Если во фрагменте названо конкретное место (район, "
    "месторождение, рудное поле), включи его в вопрос по-русски. Не копируй фразы фрагмента, "
    "перефразируй; длина вопроса — 6–20 слов. Дай и краткий ответ по фрагменту, 1–2 предложения. "
    "Если во фрагменте нет геологических сведений, верни question: null.\n"
    'Ответ — только JSON: {"question": "...", "answer": "..."}'
)


def pick_fragments(conn) -> list[dict]:
    """По фрагменту на статью: больше всего разных терминов словаря, не служебный раздел."""
    finder = Finder(use_spacy=False)
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, c.doc_id, c.headings, c.text, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.doc_id, c.ord")
        rows = cur.fetchall()
    best: dict[str, tuple[int, dict]] = {}
    for cid, doc, heads, text, title in rows:
        if SKIP_HEAD.search(" ".join(heads or [])) or len(text or "") < 400:
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
            data = llm.chat_json(SYSTEM, f["text"][:3000], temperature=0.0, max_tokens=300)
        except Exception as exc:  # noqa: BLE001
            data = {"question": None, "error": str(exc)}
        q = (data or {}).get("question")
        out.append({**{k: f[k] for k in ("doc_id", "chunk_id", "title", "terms")},
                    "question": q, "answer": (data or {}).get("answer")})
        print(i, f["doc_id"][:30], "|", q)
    QA.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"вопросов {sum(bool(x['question']) for x in out)} из {len(out)} статей")


def run() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    items = [x for x in json.loads(QA.read_text("utf-8")) if x["question"]]
    emb, syn = Embedder(), S.load()
    llm = LLM(model="qwen3-14b", host="http://127.0.0.1:8100", timeout=120, temperature=0.0)
    with connect(E.DSN) as conn:
        cur = conn.cursor()
        for x in items:
            out, info = F.run_one(conn, emb, syn, llm, x["question"])
            cur.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)",
                        (sorted({c for ids in out.values() for c in ids}),))
            doc = dict(cur.fetchall())
            x["runs"] = out
            x["docs"] = {m: [doc[c] for c in ids] for m, ids in out.items()}
            x["info"] = info
            print(x["question"][:60], {m: (x["chunk_id"] in out[m], x["doc_id"] in x["docs"][m]) for m in MODES})
        texts = E._chunk_info(conn, [x["chunk_id"] for x in items])
    RUNS.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Проверка вопросов, составленных по статьям базы. Заполняется один жёлтый столбец листа «Вопросы».",
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
    wb.save(SHEET)
    print(f"{SHEET}: {len(items)} вопросов")
    score(None)


def _paired(d: list[float]) -> list[float]:
    rng = random.Random(1)
    boot = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(10000))
    return [round(statistics.mean(d), 3), round(boot[250], 3), round(boot[9750], 3)]


def score(xlsx: str | None) -> None:
    items = json.loads(RUNS.read_text("utf-8"))
    keep = set(range(1, len(items) + 1))
    res: dict = {"questions_generated": len(items)}
    if xlsx:
        from openpyxl import load_workbook

        marks = E._marks(load_workbook(xlsx, data_only=True)["Вопросы"], 6)
        keep = {r - 1 for r, v in marks.items() if v == 1}
        res["questions_checked"] = len(marks)
        res["questions_good"] = len(keep)
    sel = [x for i, x in enumerate(items, 1) if i in keep]
    res["questions_used"] = len(sel)
    for part, cond in (("all", lambda x: True), ("with_place", lambda x: bool(x["info"]["names"])),
                       ("without_place", lambda x: not x["info"]["names"])):
        xs = [x for x in sel if cond(x)]
        if not xs:
            continue
        res[part] = {"n": len(xs)}
        for m in MODES:
            doc_hit = [int(x["doc_id"] in x["docs"][m]) for x in xs]
            frag_hit = [int(x["chunk_id"] in x["runs"][m]) for x in xs]
            rr = [1 / (x["docs"][m].index(x["doc_id"]) + 1) if x["doc_id"] in x["docs"][m] else 0 for x in xs]
            res[part][m] = {"article_in_top5": round(statistics.mean(doc_hit), 3),
                            "article_ci95": E.wilson(sum(doc_hit), len(xs)),
                            "fragment_in_top5": round(statistics.mean(frag_hit), 3),
                            "fragment_ci95": E.wilson(sum(frag_hit), len(xs)),
                            "mrr_article": round(statistics.mean(rr), 3)}
        for other in ("hybrid", "vector"):
            res[part][f"system_minus_{other}_article"] = _paired(
                [int(x["doc_id"] in x["docs"]["system"]) - int(x["doc_id"] in x["docs"][other]) for x in xs])
    (RES / "article_qa_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"gen": gen, "run": run}.get(sys.argv[1], lambda: score(sys.argv[2] if len(sys.argv) > 2 else None))()
