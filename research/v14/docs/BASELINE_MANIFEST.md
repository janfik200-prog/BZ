# BASELINE_MANIFEST — нулевая точка (WP00), 29.10.2026

Снято **до** изменений кода; все эксперименты шли на копии базы, боевая база и `main`
не менялись.

## Репозиторий

| Что | Значение |
|---|---|
| Рабочая копия | `C:\Projects\BZ`, ветка до работы — `fix/parser`, дерево чистое (`git status`: nothing to commit) |
| HEAD до работы | `0dd267572f6e08669ccbc0830ebc6cfb426074f4` («Рамка формулы 0,05 действует в docling 2.140») |
| `main` / `origin/main` | `34f5360d433c8dc9786ccbbe3842c07fe0c0cc67` — совпадает с версией, просмотренной в ТЗ |
| Локальные коммиты сверх `main` | 8 (`fix/parser`): в т. ч. `065123e` и `fe4673c` — граф v2: MinHash/LSH (`graph/dedup.py`), словарь + spaCy (`graph/entities.py`), модель-строитель, буфер, Лейден |
| Другие ветки | `fix/graph`, `fix/test-plan-findings`, `secure-bz` (worktree `C:\Projects\BZ-secure`), `remotes/bzv/main` |
| Ветка этой работы | `research/v14-applicability` от `0dd2675`; откат — `git switch fix/parser` |
| Неотслеживаемое | `.research/` (исключён в `.git/info/exclude`: `runner.sh`, `setup.sh` — исследовательский раннер в контейнере, создан 29.10 10:48; в этой работе не использовался), `.local/` (статьи, логи, рукописи v9–v13) |

Вывод: MinHash + словарь + NER **есть в локальной ветке** (не в `main`); территориальный
`DIRECT/ANALOG/OUT_OF_SCOPE/UNRESOLVED` не было нигде — реализован в этой ветке.

## Окружение

| Компонент | Версия / состояние |
|---|---|
| ОС | Windows 11 Pro 10.0.26100 |
| Python | 3.13.2 (`.venv`) |
| PostgreSQL | 17.11 (Docker `pgvector/pgvector:pg17`), pgvector 0.8.6, порт 127.0.0.1:5433 |
| Модель ответов | vLLM `v0.30.0`, `Qwen/Qwen3-14B-AWQ` (ревизия кэша `31c69efc…`), имя `qwen3-14b`, `--max-model-len 12288`, `--kv-cache-dtype fp8` — **работает**, порт 8100 |
| Модель-строитель | контейнер `georag-builder`: `Qwen/Qwen3-8B-AWQ` (ревизия `4da05a8e…`), имя `qwen3-8b`, порт 8101 — **остановлен** (Exited 28.10 12:29); на 16 ГБ две модели сразу не помещаются |
| Видеокарта | NVIDIA GeForce RTX 5070 Ti, 16 ГБ (занято 14,6 ГБ моделью ответов) |
| Векторы | `BAAI/bge-m3` (локальный кэш), sentence-transformers 6.2.0, transformers 5.20.2 |
| Разбор PDF | docling 2.140.0 (версия разбора в базе не записана: `documents.parser = docling`) |
| NER | spaCy 3.8.18, `ru_core_news_sm`, `en_core_web_sm` |

## Корпус (боевая база `georag`, только чтение)

| | |
|---|---|
| Статьи | 114 (113 docling, 1 pymupdf4llm); RU/EN; 83 из OpenAlex, 31 из Crossref |
| Фрагменты | 4 272 (1 505 578 токенов BGE-M3) |
| Факты графа | 13 739 — Qwen3:14b (Ollama-эпоха), 09–10.10.2026, все фрагменты, без первого этапа (исторический E0) |
| Голоса confirm.py | у всех: 0 — 2 166, 1 — 1 112, 2 — 3 111, 3 — 2 503, 4 — 4 847 |
| Типы связей | у 7 350 фактов |
| Прочие исследовательские базы | `georag_b14` (14B + первый этап, 200 фрагментов, 27.10), `georag_b8v3` (8B, те же 200, 28.10), `georag_t*`, `georag_p_*` — прошлые опыты; не изменялись |

## Резервная копия и копия для экспериментов

* `pg_dump -Fc` боевой базы → `.local/backup/georag_prod_20261029.dump`,
  sha256 `08eba94a0e0c2a5911ffe6add404c4ab85180a7b9de0335c7ff74018e098b02e` (95,5 МБ).
* Из неё восстановлена `georag_v14` — все эксперименты читают её (`GEORAG_DSN`).

## Тесты до изменений

`python -m pytest tests -q` → **617 passed, 14 xfailed** (36,5 с). После работы →
см. `research/v14/qa/pytest_full.xml` и REPRODUCE.md (регрессий нет).
