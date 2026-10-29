"""Проверка идеи «обязательные слова вопроса»: фрагмент допускается, только если в нём есть
ключевые понятия вопроса (метод, отношение, место, тип оруденения) — на любом языке.

Понятия выделяет локальная модель (qwen3-14b) один раз и записывает в
results/keywords.json до сверки с разметкой. Сверка — на уже размеченных парах
«вопрос — фрагмент» (Разметка_v15_размечено.xlsx): насколько допуск отделяет полезные
фрагменты от бесполезных, и что даёт поиск с таким допуском.

    keyword_gate.py extract            — понятия по 25 вопросам (модель, порт 8100)
    keyword_gate.py check <xlsx>       — сверка с разметкой → results/keyword_gate.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
KEY = json.loads((RES / "search_key.json").read_text("utf-8"))
DSN = "postgresql://georag:georag@127.0.0.1:5433/georag_v14"

SYSTEM = (
    "Ты помогаешь искать сведения в геологических статьях на русском и английском языках. "
    "Из вопроса выдели от одного до трёх обязательных понятий: без каждого из них фрагмент статьи "
    "не может отвечать на вопрос. Это метод, отношение или показатель, минерал или процесс, тип "
    "оруденения, название места или объекта. Общие слова («признак», «участок», «месторождение», "
    "«как выделяют») обязательными не бывают. Для каждого понятия дай написания: русское, "
    "английское, сокращения и обозначения (например «K/Th», «AHP», «ASTER»). "
    'Ответ — только JSON: {"must": [["написание", "написание", ...], ...]}'
)

_WORD = re.compile(r"[\w/]+", re.U)


def extract() -> None:
    import requests

    out = {}
    for i, q in enumerate(KEY["queries"], 1):
        r = requests.post("http://127.0.0.1:8100/v1/chat/completions", timeout=120, json={
            "model": "qwen3-14b", "temperature": 0, "max_tokens": 600,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q + " /no_think"}]})
        text = r.json()["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        out[str(i)] = json.loads(text[text.index("{"): text.rindex("}") + 1])["must"]
        print(i, q, "→", out[str(i)])
    (RES / "keywords.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")


def _stem(w: str) -> str:
    w = w.lower().replace("ё", "е")
    return w if len(w) <= 4 or "/" in w else w[: max(4, len(w) - 3)]


def has(variant: str, text_words: list[str]) -> bool:
    """Все слова написания есть во фрагменте (по началу слова: падеж и число не важны)."""
    if len(variant.strip()) < 3:  # модель придумывает «L», «F», «G» — они совпали бы с чем угодно
        return False
    words = [_stem(w) for w in _WORD.findall(variant)]
    return bool(words) and all(any(t.startswith(w) for t in text_words) for w in words)


def gate(concepts: list[list[str]], text: str) -> bool:
    tw = [t.lower().replace("ё", "е") for t in _WORD.findall(text)]
    return all(any(has(v, tw) for v in c) for c in concepts)


def check(xlsx: str) -> None:
    from openpyxl import load_workbook

    from georag.graph import synonyms as S
    from georag.index import search as X
    from georag.index.db import connect
    from georag.index.embed import Embedder

    kw = json.loads((RES / "keywords.json").read_text("utf-8"))
    ws = load_workbook(xlsx, data_only=True)["Поиск"]
    lab = {(r["query"], r["chunk_id"]): int(ws.cell(r["row"], 5).value) for r in KEY["search_rows"]}
    emb, syn = Embedder(), S.load()
    res = {"pairs": {}, "modes": {}}
    with connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text FROM chunks")
        text = {i: t or "" for i, _, t in cur.fetchall()}
        cur.execute("SELECT id, doc_id FROM chunks")
        doc = dict(cur.fetchall())
        # 1) Допуск на размеченных парах.
        tp = fp = fn = tn = 0
        for (q, c), v in lab.items():
            ok = gate(kw[str(q)], text[c])
            tp += ok and v
            fp += ok and not v
            fn += (not ok) and v
            tn += (not ok) and not v
        res["pairs"] = {"passed": tp + fp, "passed_useful": tp, "failed": fn + tn, "failed_useful": fn,
                        "share_useful_passed": round(tp / (tp + fp), 3), "share_useful_failed": round(fn / (fn + tn), 3),
                        "useful_kept": round(tp / (tp + fn), 3)}
        # 2) Поиск с допуском: кандидаты по смыслу и по словам (до 50), допуск, места RRF, 5 лучших.
        modes = {"vector+gate": [], "hybrid+gate": [], "hybrid+gate+graph": []}
        for qi, q in enumerate(KEY["queries"], 1):
            alts = S.synonym_variants(q, syn)
            vec = emb.encode_one(q)
            v_ids = [r[0] for r in X.vector_search(conn, vec, 50)]
            t_rows = X.text_search(conn, q, 50, alternatives=alts) or X.text_search(conn, q, 50, any_word=True, alternatives=alts)
            t_ids = [r[0] for r in t_rows]
            f_ids = [r[0] for r in X.fact_search(conn, vec, 50) if r[1] >= X.FACT_MIN_SIMILARITY]
            for name, lists in (("vector+gate", [v_ids]), ("hybrid+gate", [v_ids, t_ids]),
                                ("hybrid+gate+graph", [v_ids, t_ids, f_ids])):
                sc = X.rrf_merge([[c for c in ids if gate(kw[str(qi)], text[c])] for ids in lists])
                top, per = [], {}
                for c in sorted(sc, key=lambda c: -sc[c]):
                    if per.get(doc[c], 0) < 2:
                        top.append(c)
                        per[doc[c]] = per.get(doc[c], 0) + 1
                    if len(top) == 5:
                        break
                modes[name].append([(c, lab.get((qi, c))) for c in top])
        for name, runs in modes.items():
            marks = [v for run in runs for _, v in run if v is not None]
            res["modes"][name] = {"returned": sum(len(r) for r in runs), "labelled": len(marks),
                                  "useful": sum(marks), "share_useful_labelled": round(sum(marks) / len(marks), 3),
                                  "unlabelled": sum(v is None for run in runs for _, v in run),
                                  "queries_short": sum(len(r) < 5 for r in runs)}
        res["unlabelled_rows"] = sorted({(qi, c) for name, runs in modes.items() for qi, run in enumerate(runs, 1)
                                         for c, v in run if v is None})
    (RES / "keyword_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "unlabelled_rows"}, ensure_ascii=False, indent=1))
    print("неразмеченных пар:", len(res["unlabelled_rows"]))


if __name__ == "__main__":
    extract() if sys.argv[1] == "extract" else check(sys.argv[2])
