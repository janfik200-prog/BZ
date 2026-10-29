# Пакет v14: контекстная применимость геологических фактов

Исполнение ТЗ «База геологических знаний v13 → воспроизводимый научный результат»
(`tz/01_FULL_RESEARCH_TZ.md`). Ветка `research/v14-applicability`; боевая база и `main` не менялись.

| Начать с | |
|---|---|
| `RESULTS.md` | итог: что доказано, что нет, все числа с файлами |
| `article/Статья_ИПУЗ_2026_база_знаний_v14.docx` (+ `.pdf`) | статья v14, числа — из `article/SCIENCE_CLAIMS.csv` |
| `docs/LIMITATIONS.md` | ограничения и блокеры |
| `REPRODUCE.md` | как пересчитать |

## Состав (по разделу 18 ТЗ)

| Gate | Где |
|---|---|
| G0 baseline | `docs/BASELINE_MANIFEST.md`, `docs/IMPLEMENTATION_STATUS.csv`, дамп `.local/backup/` |
| G1 extraction | `results/prefilter_sweep.csv`, `prefilter_decisions.jsonl`, `prefilter_audit_sample.csv`, `raw_e_live.jsonl`, `e_summary.json`, `e_precision_sample.json` |
| G2 source/context | `georag/evidence/` (contracts, provenance, georegistry, scope, support, applicability, fusion), `georag/evidence/schemas/`, `data/regional_registry.csv`, `results/provenance_audit.json` |
| G3 tests | `tests/evidence/` (96), `qa/test_cases_report.csv` (40/40), `qa/pytest_full.xml`, `qa/tests_summary.json` |
| G4 gold | `data/source_manifest.csv`, `data/gold_annotations.csv`, `data/split_manifest.json`, `annotation/annotation_guidelines.md`, `annotation/expert_review_sample.csv`, `annotation/annotator_agreement.csv` (пусто — нет эксперта), `results/FREEZE.json` |
| G5 science | `results/raw_predictions.csv`, `metrics.csv/json`, `bootstrap.json`, `confusion.json`, `risk_coverage.csv`, `support_rules.json`, `error_cases.csv/md`, `plots/` |
| G6 application | `dossier/A`, `dossier/B`, `dossier/switches.csv`, `dossier/demo_walkthrough.md`, `results/raw_p.jsonl`, `p_metrics.json` |
| G7 article | `article/` |
| G8 reproducibility | `REPRODUCE.md`, `experiments/`, `results/EXPERIMENT_MATRIX.csv`, `docs/DECISIONS.md` |
