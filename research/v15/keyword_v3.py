"""Инструкция автора (30.10.2026): обязательные понятия, включая полезное ископаемое и
название места. Понятия → results/keywords_v3.json; сверка — порядок «доля найденных
понятий, затем объединённый поиск», как лучший вариант keyword_gate.

    keyword_v3.py extract | check <xlsx>
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

# Текст автора; «тип оруденения и полезное:название места» прочитано как перечень
# «тип оруденения, полезное ископаемое, название места».
SYSTEM = (
    "Ты помогаешь искать сведения в геологических статьях на русском и английском языках. "
    "Выдели от одного до трёх обязательных понятий: без каждого из них фрагмент не может отвечать на "
    "вопрос. Это метод, показатель, минерал или процесс, тип оруденения, полезное ископаемое, название "
    "места. Общие слова («признак», «участок», «месторождение») обязательными не бывают. Для каждого "
    "понятия дай написания: русское, английское, сокращения. "
    'Ответ — только JSON: {"must": [["написание", "написание", ...], ...]}'
)
FILE = G.RES / "keywords_v3.json"


def extract() -> None:
    import requests

    out = {}
    for i, q in enumerate(G.KEY["queries"], 1):
        r = requests.post("http://127.0.0.1:8100/v1/chat/completions", timeout=120, json={
            "model": "qwen3-14b", "temperature": 0, "max_tokens": 600, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q + " /no_think"}]})
        text = re.sub(r"<think>.*?</think>", "", r.json()["choices"][0]["message"]["content"], flags=re.S)
        out[str(i)] = json.loads(text[text.index("{"): text.rindex("}") + 1])["must"]
        print(i, q, "→", out[str(i)])
    FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")


def check(xlsx: str) -> None:
    from openpyxl import load_workbook

    from georag.graph import synonyms as S
    from georag.index import search as X
    from georag.index.db import connect
    from georag.index.embed import Embedder

    ws = load_workbook(xlsx, data_only=True)["Поиск"]
    lab = {(r["query"], r["chunk_id"]): int(ws.cell(r["row"], 5).value) for r in G.KEY["search_rows"]}
    emb, syn = Embedder(), S.load()
    res = {}
    with connect(G.DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text FROM chunks")
        rows = cur.fetchall()
        text = {i: t or "" for i, _, t in rows}
        doc = {i: d for i, d, _ in rows}
        cands = []
        for q in G.KEY["queries"]:
            alts = S.synonym_variants(q, syn)
            vec = emb.encode_one(q)
            v = [r[0] for r in X.vector_search(conn, vec, 50)]
            t = [r[0] for r in (X.text_search(conn, q, 50, alternatives=alts)
                                or X.text_search(conn, q, 50, any_word=True, alternatives=alts))]
            cands.append(X.rrf_merge([v, t]))
        for name, path in (("первая инструкция", G.RES / "keywords.json"), ("инструкция автора", FILE)):
            kw = json.loads(path.read_text("utf-8"))

            def share(qi: int, i: int) -> float:
                w = [x.lower().replace("ё", "е") for x in G._WORD.findall(text[i])]
                cs = kw[str(qi)]
                return sum(any(G.has(x, w) for x in c) for c in cs) / len(cs)

            none_useful = none_all = 0
            for (q, c), val in lab.items():
                if share(q, c) == 0:
                    none_all += 1
                    none_useful += val
            got = []
            for qi, sc in enumerate(cands, 1):
                top, per = [], {}
                for i in sorted(sc, key=lambda i: (-share(qi, i), -sc[i])):
                    if per.get(doc[i], 0) < 2:
                        top.append(i)
                        per[doc[i]] = per.get(doc[i], 0) + 1
                    if len(top) == 5:
                        break
                got += [lab.get((qi, i)) for i in top]
            m = [x for x in got if x is not None]
            res[name] = {"labelled": len(m), "useful": sum(m), "share_useful_labelled": round(sum(m) / len(m), 3),
                         "unlabelled": got.count(None), "pairs_without_concepts": none_all,
                         "useful_among_them": none_useful}
    (G.RES / "keyword_v3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    extract() if sys.argv[1] == "extract" else check(sys.argv[2])
