"""Воспроизведение результатов v17 одной командой.

    .venv/Scripts/python.exe -X utf8 research/v17/reproduce_results.py [--files-only]

Шаги и что им нужно — в docs/reproducibility_report.md. Все шаги только читают базу
georag_v14; в конце checks.py сверяет отпечаток базы с сохранённым (results/db_fingerprint.json)
и согласованность всех чисел. --files-only — только шаги, которым база не нужна (по
сохранённым results/*.json|csv, .cache/ и листам разметки).

Повторная типизация 53 связей (untyped_audit.py) берёт ответы модели из
.cache/untyped_retype.json; без кеша нужен vLLM с qwen3-14b на 127.0.0.1:8100.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parent / "v14" / "article"
DB = ["relation_audit.py", "extraction_errors.py", "retrieval_eval.py", "corpus_structure.py",
      "untyped_audit.py", "pairwise_cases.py"]
FILES = ["labeling_score.py", "figures.py", "reports.py",
         f"article_v17.py {ARTICLE / 'Статья_ИПУЗ_2026_v16.docx'} {ARTICLE / 'Статья_ИПУЗ_2026_v17.docx'}"]

steps = (FILES if "--files-only" in sys.argv else DB + FILES) + ["checks.py"]
for step in steps:
    script, *args = step.split(" ", 1)
    print("==", script, flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / script), *(args[0].split(" ") if args else [])],
                   check=True)
