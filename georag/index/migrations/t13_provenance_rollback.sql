-- Откат T13 (t13_provenance.sql): колонки уходят вместе с записанным в них. Факты и
-- статьи остаются; пометки «устарел» теряются — код без колонок их не ищет, граф и
-- переиндексация работают как до T13. Идемпотентен.
ALTER TABLE IF EXISTS facts DROP COLUMN IF EXISTS stale_reason;
ALTER TABLE IF EXISTS facts DROP COLUMN IF EXISTS stale_since;
ALTER TABLE documents DROP COLUMN IF EXISTS parsed_at;
ALTER TABLE documents DROP COLUMN IF EXISTS parsed_sha256;
ALTER TABLE documents DROP COLUMN IF EXISTS parser_version;
ALTER TABLE chunks DROP COLUMN IF EXISTS parser_version;
