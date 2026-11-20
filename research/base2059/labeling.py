"""Листы ручной разметки для базы из 2 059 статей — та же процедура, что в пилоте.

Пилот: лист «Факты» (120 случайных связей графа, research/v15) и листы v16
(research/v16/labeling_sheet.py: Фильтр, Типизация, Отрицания, Модальность). Здесь те же
правила отбора и то же зерно; отличия от пилота:
  Связи        — 200 случайных связей графа (в пилоте 120);
  Отрицания,
  Модальность  — в пилоте брались все подходящие связи графа (85 и 98), здесь граф в 13 раз
                 больше — случайные 100 из всех подходящих (сколько всего — в ключе).
База только читается. Порядок строк перемешан. Что скрыто от размечающего — key.json.

    python research/base2059/labeling.py  →  research/base2059/Разметка_база2059.xlsx, key.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "v16"))
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

import georag.graph.entities as ENT  # noqa: E402
from georag.text import name_key  # noqa: E402

# labeling_sheet.py (v16) берёт entities.normalize, которого в коде больше нет; число
# терминов ниже считается иначе (term_count), а заглушка нужна только для импорта.
if not hasattr(ENT, "normalize"):
    ENT.normalize = name_key

import extraction_errors as XE  # noqa: E402
import labeling_sheet as LS  # noqa: E402
from georag.graph import captions, dedup  # noqa: E402
from georag.graph import store  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
SHEET = HERE / "Разметка_база2059.xlsx"
KEY = HERE / "key.json"
SEED = LS.SEED
N_FACTS, N_TYPING, N_NEG, N_HEDGE = 200, 60, 100, 100


def term_count(finder: ENT.Finder, text: str) -> int:
    """Сколько разных терминов словаря во фрагменте — как Found.distinct_terms, по которому
    фильтр отсеивает фрагмент (facts.run), но без spaCy: места на число терминов не влияют,
    а spaCy на 114 тыс. фрагментов — час."""
    lowered = (text or "").lower().replace("ё", "е")
    return len({k for m in finder.pattern.finditer(lowered) if (k := finder._term_at(lowered, m))})


def chunk_terms(conn) -> list[dict]:
    """Как labeling_sheet.chunk_terms (v16): термины, почти-дубли, подписи — по всем фрагментам."""
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, c.doc_id, c.text, c.n_tokens, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.doc_id, c.ord")
        rows = cur.fetchall()
        cur.execute("SELECT chunk_id, count(*) FROM facts GROUP BY 1")
        stored = dict(cur.fetchall())
    graph = Counter(f["chunk_id"] for f in store.all_facts(conn))
    finder = ENT.Finder()
    index: dict[tuple[int, int], list[int]] = {}
    sigs: dict[int, list[int]] = {}
    out = []
    for cid, doc, text, ntok, title in rows:
        text = text or ""
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
                    "terms": term_count(finder, text), "duplicate": dup,
                    "caption": bool(captions.skip_reason(text)),
                    "facts_stored": stored.get(cid, 0), "facts_graph": graph.get(cid, 0)})
    return out


def shown(f: dict) -> tuple[str, str, str]:
    """Связь так, как она стоит в графе: тип из списка 14 (с направлением), иначе слова модели."""
    if f.get("rel_type"):
        a, b = (f["dst"], f["src"]) if f.get("rel_flip") else (f["src"], f["dst"])
        return a, f["rel_type"], b
    return f["src"], f["relation"], f["dst"]


def page(f: dict) -> str:
    return ", ".join(str(p) for p in f.get("pages") or [])


def main() -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    rng = random.Random(SEED)
    conn = psycopg.connect(DSN + "?options=-c%20default_transaction_read_only%3Don", autocommit=True)

    print("Фрагменты и термины…", flush=True)
    chunks = chunk_terms(conn)
    pool = [c for c in chunks if not c["duplicate"] and not c["caption"] and len(c["text"]) >= 200]
    filt = []
    for b in LS.BUCKETS:
        group = [c for c in pool if LS.bucket(c["terms"]) == b]
        filt += rng.sample(group, min(LS.PER_BUCKET, len(group)))
    rng.shuffle(filt)
    population = Counter(LS.bucket(c["terms"]) for c in chunks if not c["duplicate"] and not c["caption"])
    tokens: Counter[str] = Counter()
    for c in chunks:
        if not c["duplicate"] and not c["caption"]:
            tokens[LS.bucket(c["terms"])] += c["n_tokens"]

    print("Граф…", flush=True)
    graph = store.all_facts(conn)
    facts = rng.sample(graph, N_FACTS)
    with conn.cursor() as cur:
        cur.execute("SELECT f.id, f.src, f.relation, f.dst, f.quote, d.title, f.type_note FROM facts f "
                    "JOIN documents d ON d.doc_id = f.doc_id WHERE f.typed_at IS NOT NULL AND f.rel_type IS NULL "
                    "ORDER BY f.id")
        excluded = cur.fetchall()
    typing = rng.sample(excluded, N_TYPING)
    neg_all = [f for f in graph if XE.NEGATION.search(f["quote"] or "") and not XE.NEG_REL.search(f["relation"])]
    hed_all = [f for f in graph if XE.hedge_ext(f["relation"], f["quote"] or "")
               and not XE.hedge_now(f["relation"], f["quote"] or "")]
    neg = rng.sample(neg_all, min(N_NEG, len(neg_all)))
    hed = rng.sample(hed_all, min(N_HEDGE, len(hed_all)))

    wb = Workbook()
    ws = wb.active
    ws.title = "Как размечать"
    for line in [
        "Разметка базы из 2 059 статей. Заполняется жёлтый столбец каждого листа: 1, 0 или пусто",
        "(не уверен — не входит в расчёт). Правила те же, что в пилотной разметке.",
        "",
        "СВЯЗИ. Связь графа и предложение статьи, из которого она взята. 1 — связь верно передаёт предложение;",
        "0 — нет. Если 0, в столбце «Вид ошибки» одно из: оговорка (предположение автора подано как факт),",
        "тип (неверный тип связи), направление (перепутаны стороны), вне области (не о геологии), другое.",
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

    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    def clean(v):
        """Управляющие символы из разбора PDF Excel не принимает — убираются."""
        return ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v

    def sheet(title, head, data, widths, mark_cols):
        s = wb.create_sheet(title)
        s.append(head)
        for row in data:
            s.append([clean(v) for v in row])
        for col, w in zip("ABCDEFGHIJ", widths):
            s.column_dimensions[col].width = w
        for cell in s[1]:
            cell.font = Font(bold=True)
        for row in s.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for r in range(1, s.max_row + 1):
            for c in mark_cols:
                s.cell(r, c).fill = PatternFill("solid", fgColor="FFF2CC")
        s.freeze_panes = "A2"

    sheet("Связи", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Страница",
                    "Верно (1/0)", "Вид ошибки", "Комментарий"],
          [[i, *shown(f), f["quote"], f["title"], page(f), None, None, None] for i, f in enumerate(facts, 1)],
          (5, 25, 18, 25, 70, 30, 9, 11, 14, 30), (8, 9))
    sheet("Фильтр", ["№", "Статья", "Фрагмент", "Есть геологическая связь (1/0)", "Какие связи (по желанию)"],
          [[i, c["title"], c["text"], None, None] for i, c in enumerate(filt, 1)], (5, 30, 100, 14, 40), (4,))
    sheet("Типизация", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Исключена напрасно (1/0)"],
          [[i, f[1], f[2], f[3], f[4], f[5], None] for i, f in enumerate(typing, 1)],
          (5, 25, 18, 25, 70, 25, 14), (7,))
    sheet("Отрицания", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Передана верно (1/0)"],
          [[i, f["src"], f["relation"], f["dst"], f["quote"], f["title"], None] for i, f in enumerate(neg, 1)],
          (5, 25, 18, 25, 70, 25, 14), (7,))
    sheet("Модальность", ["№", "Предмет", "Связь", "Предмет", "Предложение статьи", "Статья", "Предположение (1/0)"],
          [[i, f["src"], f["relation"], f["dst"], f["quote"], f["title"], None] for i, f in enumerate(hed, 1)],
          (5, 25, 18, 25, 70, 25, 14), (7,))
    wb.save(SHEET)

    key = {"seed": SEED, "graph": len(graph),
           "facts": [f["id"] for f in facts],
           "filter": [{k: c[k] for k in ("chunk_id", "doc_id", "terms", "n_tokens", "facts_stored", "facts_graph")}
                      for c in filt],
           "filter_population": dict(population), "filter_tokens": dict(tokens),
           "typing": [{"fact_id": f[0], "type_note": f[6]} for f in typing], "typing_population": len(excluded),
           "negation": [f["id"] for f in neg], "negation_population": len(neg_all),
           "modality": [f["id"] for f in hed], "modality_population": len(hed_all)}
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), "utf-8")
    print(SHEET, {"связи": len(facts), "фильтр": len(filt), "типизация": f"{len(typing)} из {len(excluded)}",
                  "отрицания": f"{len(neg)} из {len(neg_all)}", "модальность": f"{len(hed)} из {len(hed_all)}"})


if __name__ == "__main__":
    main()
