"""Лист ручной разметки v16 (задачи 1–3 ТЗ): то, что нельзя оценить без человека.

Листы:
  Фильтр      — 150 фрагментов, по 25 из групп «терминов словаря 0, 1, 2, 3, 4, 5 и больше»
                (задача 3: пороги 1–4); число терминов и извлечённые связи на листе не
                показаны — разметка без знания решений фильтра и модели;
  Типизация   — 60 случайных связей из 3 585, исключённых типизацией (задача 1);
  Отрицания   — связи графа с отрицанием в цитате и без отрицания в связи (задача 2);
  Модальность — связи графа, которые пометило бы расширенное правило предположений
                (extraction_errors.HEDGE_EXTRA), а текущее — нет (задача 2).
Ключ (что скрыто от размечающего) — results/labeling_key.json; оценка после разметки —
labeling_score.py. Порядок строк перемешан, зерно 20261031.

    labeling_sheet.py → Разметка_v16.xlsx, results/labeling_key.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import psycopg  # noqa: E402

import extraction_errors as XE  # noqa: E402

from georag.graph import captions, dedup, store  # noqa: E402
from georag.graph.entities import Finder, normalize  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
RES = HERE / "results"
SHEET = HERE / "Разметка_v16.xlsx"
SEED = 20261031
BUCKETS = ("0", "1", "2", "3", "4", "5+")
PER_BUCKET = 25


def chunk_terms(conn) -> list[dict]:
    """Каждый фрагмент: число разных терминов словаря (как facts.run) и решение фильтра."""
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, c.doc_id, c.text, c.n_tokens, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.doc_id, c.ord")
        rows = cur.fetchall()
        cur.execute("SELECT chunk_id, count(*) FROM facts GROUP BY 1")
        stored = dict(cur.fetchall())
    graph = Counter(f["chunk_id"] for f in store.all_facts(conn))
    finder = Finder()
    index: dict[tuple[int, int], list[int]] = {}
    sigs: dict[int, list[int]] = {}
    out = []
    for cid, doc, text, ntok, title in rows:
        text = text or ""
        terms = len({normalize(t) for t in finder.find(text).terms})
        dup = False
        sig = dedup.signature(text)
        if sig is not None:
            keys = dedup.bands(sig)
            near = {o for b, k in enumerate(keys) for o in index.get((b, k), [])}
            dup = any(dedup.similarity(sig, sigs[o]) >= dedup.DUP_SIMILARITY for o in near)
            if not dup:
                sigs[cid] = sig
                for b, k in enumerate(keys):
                    index.setdefault((b, k), []).append(cid)
        out.append({"chunk_id": cid, "doc_id": doc, "title": title, "text": text, "n_tokens": ntok or 0,
                    "terms": terms, "duplicate": dup, "caption": bool(captions.skip_reason(text)),
                    "facts_stored": stored.get(cid, 0), "facts_graph": graph.get(cid, 0)})
    return out


def bucket(t: int) -> str:
    return "5+" if t >= 5 else str(t)


def main() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    rng = random.Random(SEED)
    conn = psycopg.connect(DSN, autocommit=True)
    chunks = chunk_terms(conn)
    pool = [c for c in chunks if not c["duplicate"] and not c["caption"] and len(c["text"]) >= 200]
    sample = []
    for b in BUCKETS:
        group = [c for c in pool if bucket(c["terms"]) == b]
        sample += rng.sample(group, min(PER_BUCKET, len(group)))
    rng.shuffle(sample)
    population = Counter(bucket(c["terms"]) for c in chunks if not c["duplicate"] and not c["caption"])
    tokens = Counter()
    for c in chunks:
        if not c["duplicate"] and not c["caption"]:
            tokens[bucket(c["terms"])] += c["n_tokens"]

    graph = store.all_facts(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT f.id, f.src, f.relation, f.dst, f.quote, d.title, f.type_note FROM facts f "
                    "JOIN documents d ON d.doc_id = f.doc_id WHERE f.typed_at IS NOT NULL AND f.rel_type IS NULL "
                    "ORDER BY f.id")
        excluded = cur.fetchall()
    typing = rng.sample(excluded, 60)
    neg = [f for f in graph if XE.NEGATION.search(f["quote"] or "") and not XE.NEG_REL.search(f["relation"])]
    hed = [f for f in graph if XE.hedge_ext(f["relation"], f["quote"] or "")
           and not XE.hedge_now(f["relation"], f["quote"] or "")]
    rng.shuffle(neg)
    rng.shuffle(hed)

    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Разметка v16. Заполняется жёлтый столбец каждого листа: 1, 0 или пусто (не уверен — не входит в расчёт).",
        "",
        "ФИЛЬТР. Фрагмент статьи. 1 — во фрагменте есть хотя бы одна геологическая связь, которую стоило бы",
        "внести в граф (что с чем связано: порода — минерал, оруденение — структура, процесс — признак и т. п.).",
        "0 — таких связей нет (методика, история изучения, ссылки, общие слова). В столбце «Какие связи» —",
        "по желанию кратко 1–2 связи. Число словарных терминов и решения системы намеренно не показаны.",
        "",
        "ТИПИЗАЦИЯ. Связь, которую система не включила в граф как «не о геологии» или с неподходящими видами.",
        "1 — связь геологическая и верно передаёт предложение (исключили напрасно); 0 — исключили правильно.",
        "",
        "ОТРИЦАНИЯ. В предложении есть отрицание («не», «нет», «not», «absent»…), в связи — нет.",
        "1 — связь передана верно (отрицание относится к другому); 0 — связь утверждает то, что в тексте отрицается.",
        "",
        "МОДАЛЬНОСТЬ. 1 — в предложении эта связь подана как предположение или интерпретация",
        "(«interpreted as», «proposed», «can», «предполагаемые»); 0 — как установленный факт.",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120

    def sheet(title, head, data, widths, mark_col):
        s = wb.create_sheet(title)
        s.append(head)
        for row in data:
            s.append(row)
        for col, w in zip("ABCDEFGHIJ", widths):
            s.column_dimensions[col].width = w
        for cell in s[1]:
            cell.font = Font(bold=True)
        for row in s.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for r in range(1, s.max_row + 1):
            s.cell(r, mark_col).fill = PatternFill("solid", fgColor="FFF2CC")
        s.freeze_panes = "A2"

    sheet("Фильтр", ["№", "Статья", "Фрагмент", "Есть геологическая связь (1/0)", "Какие связи (по желанию)"],
          [[i, c["title"], c["text"], None, None] for i, c in enumerate(sample, 1)], (5, 30, 100, 14, 40), 4)
    sheet("Типизация", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Исключена напрасно (1/0)"],
          [[i, f[1], f[2], f[3], f[4], f[5], None] for i, f in enumerate(typing, 1)], (5, 25, 18, 25, 70, 25, 14), 7)
    sheet("Отрицания", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Передана верно (1/0)"],
          [[i, f["src"], f["relation"], f["dst"], f["quote"], f["title"], None] for i, f in enumerate(neg, 1)],
          (5, 25, 18, 25, 70, 25, 14), 7)
    sheet("Модальность", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Предположение (1/0)"],
          [[i, f["src"], f["relation"], f["dst"], f["quote"], f["title"], None] for i, f in enumerate(hed, 1)],
          (5, 25, 18, 25, 70, 25, 14), 7)
    wb.save(SHEET)

    key = {"seed": SEED,
           "filter": [{k: c[k] for k in ("chunk_id", "doc_id", "terms", "n_tokens", "facts_stored", "facts_graph")}
                      for c in sample],
           "filter_population": dict(population), "filter_tokens": dict(tokens),
           "typing": [{"fact_id": f[0], "type_note": f[6]} for f in typing], "typing_population": len(excluded),
           "negation": [f["id"] for f in neg], "modality": [f["id"] for f in hed]}
    (RES / "labeling_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")
    print(SHEET, {"filter": len(sample), "typing": len(typing), "negation": len(neg), "modality": len(hed)},
          dict(population))


if __name__ == "__main__":
    main()
