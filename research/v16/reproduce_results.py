"""Воспроизведение результатов v16 одной командой (база georag_v14 должна быть запущена).

    .venv/Scripts/python.exe -X utf8 research/v16/reproduce_results.py [--with-sheet]

Шаги и их выходы — в docs/methodology.md. Все шаги только читают базу. Модель (vLLM) не
нужна: место и понятия вопросов взяты из сохранённых прогонов v15. --with-sheet — заново
собрать лист разметки (перезапишет Разметка_v16.xlsx — не запускать после начала разметки).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["relation_audit.py", "extraction_errors.py", "retrieval_eval.py"]

if "--with-sheet" in sys.argv:
    STEPS.append("labeling_sheet.py")
if (HERE / "Разметка_v16_проверено.xlsx").exists():
    STEPS.append("labeling_score.py Разметка_v16_проверено.xlsx")
    STEPS.append("figures.py")
for step in STEPS:
    script, *args = step.split()
    print("==", step, flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / script), *[str(HERE / a) for a in args]], check=True)
