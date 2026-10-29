# REPRODUCE — как пересчитать всё из сырых данных

Ветка `research/v14-applicability`. Windows, Python 3.13 (`.venv`), Docker: `georag-db`
(pgvector pg17, 127.0.0.1:5433), для E-live/P и проверки поддержки — `georag-llm`
(vLLM 0.30.0, `Qwen/Qwen3-14B-AWQ` как `qwen3-14b`, 127.0.0.1:8100).

## 0. Копия базы (боевую не трогать)

```bash
docker exec georag-db createdb -U georag georag_v14
docker exec georag-db pg_restore -U georag -d georag_v14 /tmp/georag_prod_20261029.dump
```

Дамп — `.local/backup/georag_prod_20261029.dump`, sha256 `08eba94a…098b02e` (BASELINE_MANIFEST.md).
Все скрипты читают `GEORAG_DSN` (по умолчанию `…/georag_v14`).

## 1. Тесты (без базы и модели)

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

Ожидается 713 passed, 14 xfailed; `tests/evidence` — 96 passed; матрица 40 сценариев →
`research/v14/qa/test_cases_report.csv` (40/40 PASS).

## 2. Данные (уже в репозитории; пересоздание детерминировано)

| Шаг | Команда | Выход |
|---|---|---|
| топонимы | `research/v14/experiments/mine_toponyms.py` | `data/toponym_candidates.csv` |
| справочник | `research/v14/data/build_registry.py` | `regional_registry.csv`, `registry_aliases.csv` |
| пары, семейства, разбиение | `experiments/build_dataset.py`, затем `build_supplement.py` | `pairs.jsonl`, `source_manifest.csv`, `source_families.csv`, `split_manifest.json`, `annotation/annotation_sheet.csv` |
| разметка → gold, заморозка | `experiments/freeze_gold.py` (из `annotation/labels_part*.csv`) | `data/gold_annotations.csv`, `results/FREEZE.json`, выборка для геолога |

`run_v.py` отказывается считать, если отпечаток любого файла не совпадает с `FREEZE.json`
(sha256 содержимого с CRLF→LF).

## 3. Эксперименты

| Эксперимент | Команда | Модель | Сырые данные | Метрики |
|---|---|---|---|---|
| V0–V3, абляции, кривые | `experiments/run_v.py` | нет | `results/raw_predictions.csv/.jsonl` | `metrics.csv/json`, `bootstrap.json`, `confusion.json`, `risk_coverage.csv`, `error_cases.csv` |
| качество инструментов, графики | `experiments/analyze_v.py` | нет | — | `support_eval.json/csv`, `frame_parse.csv`, `plots/*.png` |
| независимая поддержка | `experiments/run_support_llm.py` | qwen3-14b | `raw_support_llm.jsonl`, `llm_calls.jsonl` | — |
| правило поддержки (выбор на DEV) | `experiments/run_v_support.py` | нет | `raw_predictions_support.csv` | `support_rules.json` |
| первый этап, E0/E1, E2-история | `experiments/run_e.py --live 100` | qwen3-14b | `prefilter_decisions.jsonl`, `raw_e_live.jsonl`, `llm_calls_e.jsonl` | `prefilter_sweep.csv`, `e_summary.json`, `e_precision_sample.json` |
| E2 (Qwen3-8B): остановить `georag-llm`, `docker start georag-builder`, после — обратно | `experiments/run_e2.py` | qwen3-8b (порт 8101) | `raw_e2_live.jsonl`, `llm_calls_e2.jsonl` | `e_summary.json` (live_e2), `e2_precision_sample.json` |
| независимость, противоречия | `experiments/run_fusion.py` | нет | `conflicts.jsonl` | `independence_report.json` |
| аудит цитат | `python georag.py evidence audit` | нет | — | `provenance_audit.json` |
| досье A/B | `experiments/build_dossiers.py` | нет | `dossier/A`, `dossier/B` (JSON/CSV/MD) | `dossier/summary.json`, `switches.csv` |
| сквозные ответы P0–P2 | `experiments/run_p.py` | qwen3-14b + BGE-M3 | `raw_p.jsonl` | `p_metrics.json` |

Детерминированы (без модели): V, analyze, support-rules, fusion, audit, досье, E-offline.
С моделью (temperature 0): повтор даёт близкие, но не обязательно побайтно те же ответы vLLM.

## 4. Статья

```bash
.venv/Scripts/python.exe research/v14/article/make_figures.py
.venv/Scripts/python.exe research/v14/article/build_article.py
powershell -File research/v14/article/to_pdf.ps1 -Docx <docx> -Pdf <pdf>
```

Каждое число статьи — строка `article/SCIENCE_CLAIMS.csv` (файл и ключ результата).

## 5. Включить гейт в работающей системе (не включён по умолчанию)

```bash
set GEORAG_SCOPE_GATE=1
python georag.py web
```

Досье по вопросу: `python georag.py evidence dossier "вопрос" --topic structural --out <папка>`.
