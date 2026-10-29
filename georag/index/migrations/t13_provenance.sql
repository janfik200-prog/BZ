-- T13. Происхождение цитат при переразборе: версия парсера и отпечаток разбора у статьи,
-- пометка «устарел» у факта. Идемпотентна: можно применять сколько угодно раз
-- (init_db применяет её сама). Откат — t13_provenance_rollback.sql.

-- Статья: чем разобрана и какой получился текст. parsed_sha256 — как
-- DocumentSnapshot.parsed_sha256: sha256 нормализованных текстов фрагментов по порядку,
-- через перевод строки; parsed_at — когда отпечаток снят.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser_version text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_sha256 text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_at timestamptz;
-- Фрагмент: каким разбором получен его текст (пишется при загрузке, index/db.py record_parse).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parser_version text;

-- Факт: цитаты больше нет в тексте своего фрагмента (переразбор) — с какого времени и
-- почему. NULL — факт не устарел. Таблицы фактов может ещё не быть (граф не строили) —
-- тогда пропускается; init_db добавит колонки, когда она появится.
ALTER TABLE IF EXISTS facts ADD COLUMN IF NOT EXISTS stale_since timestamptz;
ALTER TABLE IF EXISTS facts ADD COLUMN IF NOT EXISTS stale_reason text;
