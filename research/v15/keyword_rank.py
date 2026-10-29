"""Ключевые понятия вопроса по старшинству: место → главное искомое → остальное.

Предложение автора: модель сначала выделяет место (район, месторождение, структуру с
названием); если места нет — главное, что ищут (метод, показатель, минерал, тип
оруденения); затем дополнительные понятия. Фрагменты упорядочиваются по старшинству
совпадений: место, затем главное, затем число дополнительных, затем обычный объединённый
поиск (RRF). Сверка — на той же разметке, что keyword_gate.py.

    keyword_rank.py extract     — понятия по 25 вопросам → results/keywords_ranked.json
    keyword_rank.py check <xlsx>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import keyword_gate as G  # noqa: E402

SYSTEM = (
    "Ты помогаешь искать сведения в геологических статьях на русском и английском языках. "
    "Разбери вопрос по старшинству.\n"
    "1. «место» — район, месторождение, рудное поле, щит, пояс, бассейн или другая структура с "
    "собственным названием. Если в вопросе нет названия места — null.\n"
    "2. «главное» — самое важное, что ищут: метод, показатель или отношение, минерал, процесс, тип "
    "оруденения. Ровно одно понятие.\n"
    "3. «дополнительно» — ещё одно-два понятия, которые уточняют вопрос; может быть пусто.\n"
    "Общие слова («признак», «участок», «месторождение», «как выделяют», «связь») понятиями не "
    "бывают. Для каждого понятия дай написания: русское, английское и общепринятые сокращения "
    "(например «K/Th», «AHP», «ASTER»). Не придумывай сокращений, которых нет в литературе.\n"
    'Ответ — только JSON: {"место": ["написание", ...] или null, "главное": ["написание", ...], '
    '"дополнительно": [["написание", ...], ...]}'
)
FILE = G.RES / "keywords_ranked.json"


def extract() -> None:
    import requests

    out = {}
    for i, q in enumerate(G.KEY["queries"], 1):
        r = requests.post("http://127.0.0.1:8100/v1/chat/completions", timeout=120, json={
            "model": "qwen3-14b", "temperature": 0, "max_tokens": 600,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q + " /no_think"}]})
        text = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S)
        out[str(i)] = json.loads(text[text.index("{"): text.rindex("}") + 1])
        print(i, q, "→", out[str(i)])
    FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")


def matched(variants, words) -> bool:
    return bool(variants) and any(G.has(v, words) for v in variants)


def rank_key(c: dict, text: str) -> tuple[int, int, int]:
    words = [t.lower().replace("ё", "е") for t in G._WORD.findall(text)]
    return (int(matched(c.get("место"), words)), int(matched(c.get("главное"), words)),
            sum(matched(v, words) for v in c.get("дополнительно") or []))


def check(xlsx: str) -> None:
    from openpyxl import load_workbook

    from georag.graph import synonyms as S
    from georag.index import search as X
    from georag.index.db import connect
    from georag.index.embed import Embedder

    kw = json.loads(FILE.read_text("utf-8"))
    ws = load_workbook(xlsx, data_only=True)["Поиск"]
    lab = {(r["query"], r["chunk_id"]): int(ws.cell(r["row"], 5).value) for r in G.KEY["search_rows"]}
    emb, syn = Embedder(), S.load()
    res: dict = {}
    with connect(G.DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text FROM chunks")
        rows = cur.fetchall()
        text = {i: t or "" for i, _, t in rows}
        doc = {i: d for i, d, _ in rows}
        # Пары: доля полезных по совпадению главного понятия (и места, где оно есть).
        by = {}
        for (q, c), v in lab.items():
            k = rank_key(kw[str(q)], text[c])
            name = ("место да" if k[0] else "место нет") if kw[str(q)].get("место") else "без места"
            name += ", главное " + ("да" if k[1] else "нет")
            a = by.setdefault(name, [0, 0])
            a[0] += 1
            a[1] += v
        res["pairs"] = {k: {"pairs": a[0], "useful": a[1], "share": round(a[1] / a[0], 3)} for k, a in sorted(by.items())}
        runs = {"ranked": [], "ranked+graph": [], "main_required": []}
        for qi, q in enumerate(G.KEY["queries"], 1):
            alts = S.synonym_variants(q, syn)
            vec = emb.encode_one(q)
            v = [r[0] for r in X.vector_search(conn, vec, 50)]
            t = [r[0] for r in (X.text_search(conn, q, 50, alternatives=alts)
                                or X.text_search(conn, q, 50, any_word=True, alternatives=alts))]
            f = [r[0] for r in X.fact_search(conn, vec, 50) if r[1] >= X.FACT_MIN_SIMILARITY]
            c = kw[str(qi)]
            for name, lists in (("ranked", [v, t]), ("ranked+graph", [v, t, f]), ("main_required", [v, t])):
                sc = X.rrf_merge(lists)
                ids = list(sc)
                if name == "main_required":
                    # Место (если есть) и главное — обязательны; не набралось пяти — добираются остальные.
                    ok = [i for i in ids if rank_key(c, text[i])[1] and (rank_key(c, text[i])[0] or not c.get("место"))]
                    ids = sorted(ok, key=lambda i: -sc[i]) + sorted(set(ids) - set(ok), key=lambda i: -sc[i])
                else:
                    ids.sort(key=lambda i: (tuple(-x for x in rank_key(c, text[i])), -sc[i]))
                top, per = [], {}
                for i in ids:
                    if per.get(doc[i], 0) < 2:
                        top.append(i)
                        per[doc[i]] = per.get(doc[i], 0) + 1
                    if len(top) == 5:
                        break
                runs[name].append([(i, lab.get((qi, i))) for i in top])
        for name, rr in runs.items():
            m = [x for r in rr for _, x in r if x is not None]
            res[name] = {"labelled": len(m), "useful": sum(m), "share_useful_labelled": round(sum(m) / len(m), 3),
                         "unlabelled": sum(x is None for r in rr for _, x in r)}
    (G.RES / "keyword_rank.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    extract() if sys.argv[1] == "extract" else check(sys.argv[2])
