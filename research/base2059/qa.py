"""Вопросы по статьям для оценки поиска — процедура пилота (research/v15) на этой базе.

Пилот: по одному фрагменту из каждой из 114 статей (article_qa.py, набор 1) и по другому,
не соседнему фрагменту тех же статей (article_qa2.py, набор 2); вопрос и краткий ответ
составляет qwen3-14b (порт 8100), затем те же способы поиска. Здесь статей 2 059, поэтому
берутся случайные N статей (по умолчанию 120, зерно SEED); остальное — без изменений:
пилотные функции вызываются как есть, подменены только адрес базы (только чтение), набор
статей и папка результатов (research/base2059/qa — пилотные файлы v15 не трогаются).

    python research/base2059/qa.py gen1 | run1 | gen2 | run2      [--n 120]

После run1/run2 — листы «Вопросы» в qa/: автор ставит «Годится (1/0)».
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V15 = ROOT / "research" / "v15"
OUT = HERE / "qa"
SEED = 20261120

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
os.environ["GEORAG_EVAL_DSN"] = DSN + "?options=-c%20default_transaction_read_only%3Don"
sys.path.insert(0, str(V15))
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

import article_qa as Q  # noqa: E402
import article_qa2 as Q2  # noqa: E402
import search_eval as E  # noqa: E402

E.DSN = os.environ["GEORAG_EVAL_DSN"]
OUT.mkdir(exist_ok=True)
Q.RES = Q2.RES = OUT
Q.QA, Q.RUNS, Q.SHEET = OUT / "article_qa.json", OUT / "article_qa_runs.json", OUT / "Вопросы_набор1.xlsx"
Q2.QA2, Q2.RUNS2, Q2.SHEET2 = OUT / "article_qa2.json", OUT / "article_qa2_runs.json", OUT / "Вопросы_набор2.xlsx"


def sampled_docs(n: int) -> set[str]:
    with psycopg.connect(E.DSN) as c:
        docs = [r[0] for r in c.execute("SELECT doc_id FROM documents WHERE status = 'ok' ORDER BY doc_id")]
    return set(random.Random(SEED).sample(docs, n))


def only(pick, docs: set[str]):
    def wrapped(conn):
        return [f for f in pick(conn) if f["doc_id"] in docs]
    return wrapped


def main() -> None:
    args = sys.argv[1:]
    n = 120
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        del args[i : i + 2]
    step = args[0]
    docs = sampled_docs(n)
    print(f"статей в выборке: {len(docs)} (seed {SEED})", flush=True)
    Q.pick_fragments = only(Q.pick_fragments, docs)
    Q2.pick_fragments = only(Q2.pick_fragments, docs)
    {"gen1": Q.gen, "run1": Q.run, "gen2": Q2.gen, "run2": Q2.run}[step]()


if __name__ == "__main__":
    main()
