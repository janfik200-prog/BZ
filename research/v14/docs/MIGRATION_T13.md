# Миграция T13: версия парсера и устаревшие факты

## Что добавляет

| Таблица | Колонка | Смысл |
|---|---|---|
| documents | parser_version text | «docling@2.140.0»: каким разбором статья в базе; NULL — не записано (статья загружена до T13) |
| documents | parsed_sha256 text | отпечаток разбора: sha256 нормализованных текстов фрагментов по порядку (= DocumentSnapshot.parsed_sha256) |
| documents | parsed_at timestamptz | когда отпечаток снят |
| chunks | parser_version text | каким разбором получен текст фрагмента |
| facts | stale_since timestamptz | с какого времени цитаты факта нет в тексте его фрагмента; NULL — факт не устарел |
| facts | stale_reason text | почему: `QUOTE_NOT_IN_PARSED_TEXT: переразбор docling@…` или `…: evidence audit` |

Пометка — колонки у факта, а не отдельная таблица: пометка у факта одна, уходит вместе с
фактом (каскад с фрагментом, новый проход модели пишет факты фрагмента заново), а граф
отсекает устаревшие тем же запросом без соединения. Факты не удаляются: устаревший факт
остаётся в базе среди кандидатов и ждёт, пока модель пройдёт изменённый фрагмент заново
(у изменённых фрагментов стирается `facts_pass`).

Файлы: `georag/index/migrations/t13_provenance.sql` (вперёд) и
`georag/index/migrations/t13_provenance_rollback.sql` (откат). Оба идемпотентны
(`ADD COLUMN IF NOT EXISTS`, `DROP COLUMN IF EXISTS`, `ALTER TABLE IF EXISTS facts`).

## Поведение кода

* `db.init_db` (команды `init`, `ingest`) применяет миграцию сама.
* `ingest`: версия парсера — из метаданных добычи (`parser_version`, пишет parse/pipeline →
  acquire); статья добыта до T13 — `docling@unrecorded`, установленная версия не
  подставляется. После `replace_chunks` — `record_parse`: версия у статьи и фрагментов,
  отпечаток разбора.
* `replace_chunks` с колонками T13: факты всех оставшихся фрагментов сверяются с новым
  текстом (нормализация как у `make_evidence`); цитаты нет — пометка, нашлась — пометка
  снята. Без колонок — как раньше (факты изменённых фрагментов стираются).
* Граф (`graph/store.py`: `all_facts`, `spellings`, `top_keys`) — устаревшие не идут в
  граф, только если колонка есть (`has_stale`).
* `snapshot_from_db` берёт версию из `documents.parser_version`, иначе `имя@unrecorded`.

## Применение (на копии сделано; на боевую — не применялось)

```
# 1. резервная копия
docker exec georag-db pg_dump -U georag -Fc georag -f /tmp/georag_before_t13.dump
# 2. миграция (или просто python georag.py init)
docker exec -i georag-db psql -U georag -d georag -v ON_ERROR_STOP=1 < georag/index/migrations/t13_provenance.sql
# 3. пометить устаревшие и записать отпечатки разбора
python georag.py evidence audit --mark-stale --out research/v14/results/stale_audit_prod.json
```

Шаг 3 пишет в базу (пометки, `parsed_sha256`); повторный запуск ничего нового не
помечает. Граф в памяти работающего сервера (`graph/api.py`) пересобирается по отпечатку
фактов, который пометку не учитывает — после шага 3 перезапустить `web`.

## Откат

```
docker exec -i georag-db psql -U georag -d georag -v ON_ERROR_STOP=1 < georag/index/migrations/t13_provenance_rollback.sql
```

Колонки и пометки уходят, факты и статьи остаются; код без колонок работает как до T13.
Важно: `init_db` применяет миграцию снова — после отката на базе, где ещё работает код
T13, колонки вернутся при следующем `init`/`ingest`.

## Проверка на копии georag_v14_t13

* миграция ×2, откат ×2, миграция — колонки 5 → 5 → 0 → 0 → 5 (до добавления
  `chunks.parser_version`);
* `evidence audit --mark-stale`: 13 739 фактов, 19 недействительных; помечено 14 — ровно
  14 `SPAN_LOST` из `provenance_audit.json`; 5 `OCR_BROKEN` не помечены (битый текст — не
  устаревание); 2 из 14 до пометки шли в граф (граф 3818 → 3816, кандидаты 6336 → 6338);
  второй прогон — 0 новых пометок; отпечаток записан 114 статьям, расхождений нет;
* переразбор (`reparse_demo` в `stale_audit_t13.json`, транзакция с откатом): тот же текст —
  0 пометок; цитата убрана из фрагмента — 2 факта помечены, ушли из графа; текст вернули —
  пометки сняты.
