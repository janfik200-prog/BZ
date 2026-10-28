"""Хранилище графа: факты, которые Qwen3 выписала из фрагментов статей.

Факт — «от — связь — к» с дословной цитатой из фрагмента. Имена хранятся
как написала модель (плюс ключ для сравнения без падежей), а сводятся к
одному имени словарём синонимов при чтении: словарь можно править, не
прогоняя модель заново.

Таблицы:

* facts      — факты: фрагмент, статья, от, связь, к, цитата, модель;
* facts_pass — какие фрагменты модель уже прошла и что код отбросил.

Обе ссылаются на chunks с ON DELETE CASCADE: статью убрали из базы — её
факты ушли вместе с фрагментами. Переиндексация фрагменты не удаляет, а
обновляет на месте (index/db.py, replace_chunks): факты остаются, стираются
только у фрагментов с изменившимся текстом — их модель пройдёт заново.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id        bigserial PRIMARY KEY,
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id    text   NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    src       text   NOT NULL,
    relation  text   NOT NULL,
    dst       text   NOT NULL,
    src_key   text   NOT NULL,
    dst_key   text   NOT NULL,
    quote     text   NOT NULL,
    model     text,
    made_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (chunk_id, src_key, relation, dst_key)
);
CREATE INDEX IF NOT EXISTS facts_src_key ON facts (src_key);
CREATE INDEX IF NOT EXISTS facts_dst_key ON facts (dst_key);
CREATE INDEX IF NOT EXISTS facts_doc ON facts (doc_id);

-- Сколько из 4 проверок вопросами факт прошёл (confirm.py); NULL — ещё не проверяли.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS votes smallint;

CREATE TABLE IF NOT EXISTS facts_pass (
    chunk_id  bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model     text,
    facts     int NOT NULL DEFAULT 0,
    rejected  text[] NOT NULL DEFAULT '{}',
    done_at   timestamptz NOT NULL DEFAULT now()
);

-- Первый этап (entities.py, dedup.py): фрагмент отсеян до модели — почему (дубль,
-- мало предметов); NULL — модель его читала. Язык фрагмента и предметы, которые модель
-- назвала сверх списка первого этапа (кандидаты в словарь geo-terms.txt).
ALTER TABLE facts_pass ADD COLUMN IF NOT EXISTS skipped text;
ALTER TABLE facts_pass ADD COLUMN IF NOT EXISTS lang text;
ALTER TABLE facts_pass ADD COLUMN IF NOT EXISTS extra_names text[];

-- Отпечатки фрагментов (MinHash, dedup.py): почти-дубль модели не отдаётся. Полосы —
-- для поиска похожих индексом, подпись — для оценки сходства.
CREATE TABLE IF NOT EXISTS chunk_minhash (
    chunk_id  bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    sig       bigint[] NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_bands (
    chunk_id  bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    band      smallint NOT NULL,
    hash      bigint NOT NULL,
    PRIMARY KEY (chunk_id, band)
);
CREATE INDEX IF NOT EXISTS chunk_bands_hash ON chunk_bands (band, hash);
"""


def init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def todo(conn: Any, limit: int | None = None, retry_skipped: bool = False) -> list[tuple[Any, ...]]:
    """Фрагменты, которых модель ещё не видела: (chunk_id, doc_id, текст).
    retry_skipped — и отсеянные первым этапом (graph --no-filter)."""
    again = "OR p.skipped IS NOT NULL" if retry_skipped else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.id, c.doc_id, c.text FROM chunks c
            LEFT JOIN facts_pass p ON p.chunk_id = c.id
            WHERE p.chunk_id IS NULL {again}
            ORDER BY c.doc_id, c.ord
            LIMIT %s
            """,
            (limit,),
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def skip(conn: Any, chunk_id: int, reason: str, lang: str | None = None) -> None:
    """Фрагмент отсеян до модели (дубль, мало предметов): фактов нет, причина — в skipped.
    graph --no-filter пройдёт его заново."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts WHERE chunk_id = %s", (chunk_id,))
        cur.execute(
            """
            INSERT INTO facts_pass (chunk_id, model, facts, rejected, skipped, lang, extra_names)
            VALUES (%s, NULL, 0, '{}', %s, %s, NULL)
            ON CONFLICT (chunk_id) DO UPDATE SET model = NULL, facts = 0, rejected = '{}',
                skipped = EXCLUDED.skipped, lang = EXCLUDED.lang, extra_names = NULL,
                done_at = now()
            """,
            (chunk_id, reason, lang),
        )


def save(
    conn: Any,
    chunk_id: int,
    doc_id: str,
    facts: list[Any],
    rejected: list[str],
    model: str,
    votes: list[int] | None = None,
    lang: str | None = None,
    extra_names: list[str] | None = None,
) -> None:
    """Результат модели по фрагменту — целиком вместо прошлого.

    votes — подтверждения вопросами по каждому факту, если их уже поставили
    (facts.run с confirm); иначе пусто, и их потом поставит confirm.run.
    lang и extra_names — язык фрагмента и предметы сверх списка первого этапа."""
    marks: list[int | None] = list(votes) if votes is not None else [None] * len(facts)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts WHERE chunk_id = %s", (chunk_id,))
        if facts:
            cur.executemany(
                """
                INSERT INTO facts (chunk_id, doc_id, src, relation, dst, src_key, dst_key,
                                   quote, model, votes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                """,
                [
                    (
                        chunk_id,
                        doc_id,
                        f.src,
                        f.relation,
                        f.dst,
                        f.src_key,
                        f.dst_key,
                        f.quote,
                        model,
                        mark,
                    )
                    for f, mark in zip(facts, marks, strict=True)
                ],
            )
        cur.execute(
            """
            INSERT INTO facts_pass (chunk_id, model, facts, rejected, skipped, lang, extra_names)
            VALUES (%s, %s, %s, %s, NULL, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET model = EXCLUDED.model, facts = EXCLUDED.facts,
                rejected = EXCLUDED.rejected, skipped = NULL, lang = EXCLUDED.lang,
                extra_names = EXCLUDED.extra_names, done_at = now()
            """,
            (chunk_id, model, len(facts), list(rejected)[:40], lang, extra_names),
        )


def stored_facts(conn: Any) -> list[tuple[Any, ...]]:
    """Все факты для перепроверки кодом: (id, chunk_id, от, связь, к, цитата)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_id, src, relation, dst, quote FROM facts ORDER BY id")
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def drop_fact(conn: Any, fact_id: int, chunk_id: int, reason: str) -> None:
    """Убрать факт; причина — в список отброшенного по его фрагменту (graph --new)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts WHERE id = %s", (fact_id,))
        cur.execute(
            "UPDATE facts_pass SET facts = greatest(facts - 1, 0), "
            "rejected = array_append(rejected, %s) WHERE chunk_id = %s",
            (reason, chunk_id),
        )


def set_relation(conn: Any, fact_id: int, relation: str) -> bool:
    """Новая связь факта, если такого же факта с ней во фрагменте ещё нет."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE facts f SET relation = %s WHERE f.id = %s AND NOT EXISTS (
                SELECT 1 FROM facts g WHERE g.chunk_id = f.chunk_id AND g.src_key = f.src_key
                AND g.dst_key = f.dst_key AND g.relation = %s)
            """,
            (relation, fact_id, relation),
        )
        return bool(cur.rowcount)


def unconfirmed(conn: Any, limit: int | None = None) -> list[tuple[Any, ...]]:
    """Факты, которых ещё не проверяли вопросами: (id, от, связь, к, цитата)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, src, relation, dst, quote FROM facts WHERE votes IS NULL ORDER BY id LIMIT %s",
            (limit,),
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


def set_votes(conn: Any, fact_id: int, votes: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE facts SET votes = %s WHERE id = %s", (votes, fact_id))


def clear(conn: Any) -> None:
    """Забыть всё, что сделала модель, — перед проходом заново (правила поменялись)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM facts")
        cur.execute("DELETE FROM facts_pass")
        # Отпечатки — тоже: «первый» из дублей определяется порядком нового прохода.
        cur.execute("DELETE FROM chunk_bands")
        cur.execute("DELETE FROM chunk_minhash")


# Сколько из 4 проверок вопросами (confirm.py) должен пройти факт, чтобы попасть в граф.
MIN_VOTES = 3


def accepted(min_votes: int = MIN_VOTES, alias: str = "f") -> str:
    """Условие SQL «факт идёт в граф» (план графа v2, А5 — буфер редких связей).

    В граф — факт, подтверждённый вопросами (не меньше min_votes из 4), или факт, чьи
    концы связаны и в другой статье (в любую сторону, любой связью): повтор в двух
    статьях — уже не случайность. Остальное — кандидаты: в базе, на странице по
    переключателю, в граф перейдут сами, когда повторятся или подтвердятся.

    98,7 % пар в базе — из одной статьи (27.10.2026), поэтому порог только по числу
    статей опустошил бы граф; подтверждённая вопросами связь идёт и из одной. Концы
    сравниваются по ключу без падежей (src_key, dst_key), без словаря синонимов."""
    a = alias
    return (
        # coalesce: у непроверенного votes — NULL, и NOT (NULL OR …) выкинуло бы его и
        # из кандидатов.
        f"(coalesce({a}.votes, 0) >= {int(min_votes)} OR EXISTS (SELECT 1 FROM facts o "
        f"WHERE o.doc_id <> {a}.doc_id AND ((o.src_key = {a}.src_key AND o.dst_key = "
        f"{a}.dst_key) OR (o.src_key = {a}.dst_key AND o.dst_key = {a}.src_key))))"
    )


def all_facts(
    conn: Any,
    min_votes: int = MIN_VOTES,
    after: int = 0,
    upto: int | None = None,
    candidates: bool = False,
    keys: list[str] | None = None,
    like: list[str] | None = None,
    chunk_ids: list[int] | None = None,
    both_ends: bool = False,
) -> list[dict[str, Any]]:
    """Факты со статьёй: из них в памяти собирается граф (фактов — тысячи).

    В граф идут факты по условию accepted: подтверждённые вопросами или повторённые в
    другой статье; candidates=True — наоборот, буфер кандидатов (на странице — по
    переключателю). Не прошедшие остаются в базе.
    after/upto — только факты с номером в (after, upto]: дописать новые к графу.

    Подграф вместо всего графа (план графа v2, А6; api.around): keys — факты, у которых
    один конец (both_ends — оба) с таким ключом имени; like — ещё и ключи, содержащие эти
    слова («зоне Персияновского разлома» для «Персияновский разлом»); chunk_ids — факты
    этих фрагментов.

    Тип связи и виды концов (kinds.py) — если их уже поставили: факт, которому тип
    поставлен как «не о геологии» (typed_at есть, rel_type нет), в граф не идёт. Колонок
    типа может ещё не быть (база до 14.10.2026) — тогда граф как раньше, по словам статьи."""
    typed = (
        "f.rel_type, f.rel_flip, f.src_kind, f.dst_kind, f.typed_at"
        if has_types(conn)
        else "NULL, NULL, NULL, NULL, NULL"
    )
    where: list[str] = []
    params: list[Any] = []
    if keys is not None or like:
        patterns = [f"%{w}%" for w in like or []]
        ends = [f"(f.{end} = ANY(%s) OR f.{end} LIKE ANY(%s))" for end in ("src_key", "dst_key")]
        where.append(f"({ends[0]} {'AND' if both_ends else 'OR'} {ends[1]})")
        params += [list(keys or []), patterns, list(keys or []), patterns]
    if chunk_ids is not None:
        where.append("f.chunk_id = ANY(%s)")
        params.append(list(chunk_ids))
    extra = "".join(f" AND {w}" for w in where)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote,
                   d.title, d.year, d.url, d.doi, f.votes, {typed}, c.pages
            FROM facts f JOIN documents d ON d.doc_id = f.doc_id
                         JOIN chunks c ON c.id = f.chunk_id
            WHERE {"NOT " if candidates else ""}{accepted(min_votes)} AND f.id > %s
                  AND (%s::bigint IS NULL OR f.id <= %s){extra}
            ORDER BY f.id
            """,
            (after, upto, upto, *params),
        )
        return [
            {
                "id": r[0],
                "chunk_id": r[1],
                "doc_id": r[2],
                "src": r[3],
                "relation": r[4],
                "dst": r[5],
                "quote": r[6],
                "title": r[7] or r[2],
                "year": r[8],
                "url": r[9] or "",
                "doi": r[10],
                "votes": r[11] if len(r) > 11 else None,
                "rel_type": r[12] if len(r) > 12 else None,
                "rel_flip": bool(r[13]) if len(r) > 13 else False,
                "src_kind": r[14] if len(r) > 14 else None,
                "dst_kind": r[15] if len(r) > 15 else None,
                # Страницы фрагмента — последней колонкой: номера колонок выше не сдвигаются.
                "pages": list(r[17]) if len(r) > 17 and r[17] else [],
            }
            for r in cur.fetchall()
            if not (len(r) > 16 and r[16] is not None and r[12] is None)
        ]


def spellings(conn: Any, keys: list[str]) -> dict[str, Counter[str]]:
    """Как пишут имена с этими ключами — по всем фактам графа: подпись узла в подграфе
    (api.around) та же, что во всём графе, — самое частое написание."""
    out: dict[str, Counter[str]] = {}
    # Те же факты, что у всего графа (all_facts): без помеченных «не о геологии»; и тот же
    # порядок при равных частотах — по первому факту (Counter помнит порядок вставки).
    geo = " AND NOT (f.typed_at IS NOT NULL AND f.rel_type IS NULL)" if has_types(conn) else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT key, raw, count(*), min(id) FROM (
                SELECT f.id, f.src_key AS key, f.src AS raw FROM facts f
                WHERE f.src_key = ANY(%s) AND {accepted()}{geo}
                UNION ALL
                SELECT f.id, f.dst_key, f.dst FROM facts f
                WHERE f.dst_key = ANY(%s) AND {accepted()}{geo}
            ) x GROUP BY key, raw ORDER BY 4
            """,
            (keys, keys),
        )
        for key, raw, n, _first in cur.fetchall():
            out.setdefault(str(key), Counter())[str(raw)] += int(n)
    return out


def count_facts(conn: Any) -> int:
    """Сколько фактов в базе — решить, держать ли граф в памяти целиком (api.large)."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM facts")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def top_keys(conn: Any, limit: int) -> list[tuple[str, int, int]]:
    """Ключи имён, о которых больше всего статей: (ключ, статей, фактов). Для большого
    графа — что показать на карте и в списке датасетов вместо всего (api, А6)."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT key, count(DISTINCT doc_id), count(*) FROM (
                SELECT f.src_key AS key, f.doc_id FROM facts f WHERE {accepted()}
                UNION ALL
                SELECT f.dst_key, f.doc_id FROM facts f WHERE {accepted()}
            ) x GROUP BY key ORDER BY 2 DESC, 3 DESC, 1 LIMIT %s
            """,
            (limit,),
        )
        return [(str(k), int(d), int(n)) for k, d, n in cur.fetchall()]


def has_types(conn: Any) -> bool:
    """Есть ли у фактов колонки типа (kinds.py) — их добавляет graph types. У заглушки
    в проверках — нет."""
    from ..index.db import is_real

    if not is_real(conn):
        return False
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'facts' AND column_name = 'typed_at'"
        )
        return cur.fetchone() is not None


def summary(conn: Any) -> dict[str, Any]:
    """Сколько фрагментов прошла модель, сколько фактов, когда последний раз."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), coalesce(sum(facts), 0), "
            "coalesce(sum(cardinality(rejected)), 0), max(done_at) FROM facts_pass"
        )
        passed, facts, rejected, last = cur.fetchone() or (0, 0, 0, None)
        cur.execute("SELECT count(*) FROM chunks")
        chunks = (cur.fetchone() or (0,))[0]
        # Отсеянные первым этапом тоже «пройдены» (passed): модель их не читала. Колонки
        # skipped может ещё не быть — база до 27.10.2026, init её пока не добавил.
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'facts_pass' AND column_name = 'skipped'"
        )
        skipped: dict[str, int] = {}
        if cur.fetchone():
            cur.execute(
                "SELECT split_part(skipped, ':', 1), count(*) FROM facts_pass "
                "WHERE skipped IS NOT NULL GROUP BY 1"
            )
            skipped = {str(k): int(n) for k, n in cur.fetchall()}
    return {
        "passed": int(passed or 0),
        "facts": int(facts or 0),
        "rejected": int(rejected or 0),
        "chunks": int(chunks or 0),
        "skipped": skipped,
        "as_of": last.isoformat() if last is not None and hasattr(last, "isoformat") else last,
    }


def extra_names(conn: Any, top: int = 15) -> list[tuple[str, int]]:
    """Предметы, которые модель назвала сверх списка первого этапа, — от самых частых:
    кандидаты в config/geo-terms.txt."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT name, count(*) FROM facts_pass, unnest(extra_names) AS name "
            "GROUP BY name ORDER BY 2 DESC, 1 LIMIT %s",
            (top,),
        )
        return [(str(n), int(c)) for n, c in cur.fetchall()]


def reason_kind(reason: str) -> str:
    """Вид причины без имён: «в цитате нет «золото»» и «… «разломы»» — одна причина.
    Иначе первые 12 строк отчёта покрывали треть отказов, а остальное дробилось."""
    if reason.startswith("в цитате нет"):
        return "в цитате нет одного из концов"
    if reason.startswith("нет предложения"):
        return "нет предложения с таким номером"
    return reason


def rejected_reasons(conn: Any, examples: int = 2) -> list[tuple[str, int, list[str]]]:
    """Почему код отбрасывал ответы модели: вид причины, сколько раз и примеры
    («от» — связь — «к» с причиной), от самой частой."""
    reasons: Counter[str] = Counter()
    shown: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT unnest(rejected) FROM facts_pass")
        for (text,) in cur.fetchall():
            # «от» — связь — «к»: причина. Двоеточие бывает и в имени («CO3: pH > 7»),
            # поэтому причина — после закрывающей кавычки «к»; без кавычек — после
            # последнего двоеточия («ответ: не объект JSON»).
            head, quoted, tail = str(text).partition("»: ")
            if quoted:
                label, reason = head + "»", tail
            else:
                label, _, reason = str(text).rpartition(": ")
            kind = reason_kind(reason)
            reasons[kind] += 1
            seen = shown.setdefault(kind, [])
            if label and len(seen) < examples:
                seen.append(f"{label}: {reason}" if kind != reason else label)
    return [(kind, count, shown.get(kind, [])) for kind, count in reasons.most_common()]


def fragments(conn: Any, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Текст фрагментов — для ответа чат-бота по фактам."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.doc_id, c.ord, c.text, c.headings, c.pages, d.title, d.year, d.url,
                   d.authors, d.journal
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.id = ANY(%s)
            """,
            (list(chunk_ids),),
        )
        return {
            r[0]: {
                "chunk_id": r[0],
                "doc_id": r[1],
                "ord": r[2],
                "text": " ".join((r[3] or "").split()),
                "headings": list(r[4] or [])[-2:],
                "pages": list(r[5] or []),
                "title": r[6] or r[1],
                "year": r[7],
                "url": r[8] or "",
                "authors": list(r[9] or [])[:3],
                "journal": r[10],
            }
            for r in cur.fetchall()
        }
