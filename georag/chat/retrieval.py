"""Поиск с моделью: широкий набор → модель отбирает → ищет недостающее.

Модель читает всех кандидатов пачками, называет подходящие, говорит, чего не
хватает, и составляет новые запросы — до MAX_ROUNDS кругов. Молчит — отбор по
порогу близости (search.find_sources)."""

from __future__ import annotations

import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from ..index.search import FACT_MIN_SIMILARITY, FACT_SLOTS, cap_per_doc
from .plan import Plan, clean_queries
from .search import (
    PART_MIN,
    SIBLING_DOCS,
    add_neighbors,
    find_sources,
    fragment_key,
    is_close,
    merged_search,
    same_article,
    source_dict,
)
from .settings import Settings

# Поиск с моделью: широкий набор кандидатов, модель отбирает и ищет ещё.
WIDE = 16  # кандидатов на простой вопрос (и на запросы нового круга)
WIDE_PART = 8  # на каждую часть составного вопроса и на вопрос целиком
LOOSE_SIMILARITY = 0.35  # мягкий порог для кандидатов: отбирает уже модель
# Модель читает кандидатов пачками по JUDGE_MAX — все, а не первые сколько-то:
# раньше при трёх частях третьей доставалось 4 кандидата из 10, а найденное
# по вопросу целиком модель не видела вовсе. Текста — CANDIDATE_CHARS: фрагмент
# около 2000 знаков, и по первым 450 нужное отбрасывалось, если суть дальше.
JUDGE_MAX = 12
CANDIDATE_PER_DOC = 3  # кандидатов из одной статьи в наборе (пока есть другие)
TRANSLATED = 6  # кандидатов по вопросу в переводе (plan.translated) сверх основных
JUDGE_SLACK = 4  # короткий хвост пачки читается вместе с предыдущей
CANDIDATE_CHARS = 1000
MAX_ROUNDS = 3  # кругов поиска самое большее
EXTRA_QUERIES = 3  # новых запросов за круг
# Новый запрос, почти совпадающий по основам слов с уже сделанным, не ищется:
# он приносит те же фрагменты, а круг стоит вызова модели.
REPEAT_OVERLAP = 0.6
# Заметки (Settings.notes): сколько отобранных сверх TOP_K идут в ответ заметкой — 1–2
# фразы о том, что во фрагменте сказано по вопросу (около 60 токенов против 700 у
# фрагмента целиком), и параллельно сколько пачек модель читает одновременно.
NOTES = 24
NOTE_CHARS = 400
JUDGE_PARALLEL = 4

JUDGE_SYSTEM = (
    "Ты отбираешь фрагменты научных статей, из которых можно ответить на вопрос "
    "геолога, и решаешь, что ещё поискать в базе. Отвечай только JSON."
)

JUDGE_USER = """Вопрос: «{question}»
{parts}{taken}
Кандидаты из базы — нашёл поиск по словам и смыслу, среди них бывают случайные:
{candidates}

1. «подходят» — номера кандидатов, в которых есть сведения для ответа (хотя бы на одну
   часть вопроса), от самых полезных к менее полезным. Кандидат, где только похожие
   слова, а по сути речь о другом, — не бери.
2. «не_хватает» — каких сведений для полного ответа нет ни в одном подходящем
   фрагменте, коротко. Всего хватает — пустой список.
3. «запросы» — до {n} новых поисковых запросов, чтобы найти недостающее: 2–6 слов,
   терминами предметной области, по-русски или по-английски, не повторяя сделанные
   и не переставляя в них слова: {done}. Всего хватает или искать больше нечего —
   пустой список.
Сохраняй направление отношения из вопроса: «хлорит замещает минералы» не означает
«минералы замещают хлорит». Название объекта может находиться в соседнем абзаце.
{notes}
Верни JSON: {answer}"""

JUDGE_NOTES = """4. «заметки» — по каждому подходящему кандидату: что в нём сказано по вопросу, 1–2
   фразы словами фрагмента, с названиями, числами и местом (район, месторождение), без
   общих слов. Ключ — номер кандидата.
"""
ANSWER_PLAIN = '{{"подходят": [1, 4], "не_хватает": ["..."], "запросы": ["..."]}}'
ANSWER_NOTES = (
    '{{"подходят": [1, 4], "заметки": {{"1": "...", "4": "..."}}, "не_хватает": ["..."], '
    '"запросы": ["..."]}}'
)


@dataclass
class Retrieval:
    """Что нашёл поиск с моделью."""

    sources: list[dict[str, Any]]
    rounds: int = 1  # сколько кругов поиска было
    checked: int = 0  # сколько кандидатов модель прочла
    judged: bool = True  # отбирала модель (False — отбор по порогу, модель молчит)

    def info(self) -> dict[str, Any]:
        return {"rounds": self.rounds, "checked": self.checked, "judged": self.judged}


def by_fact(hit: Any) -> bool:
    """Фрагмент привёл близкий к запросу факт графа (index.search, facts=True)."""
    return (getattr(hit, "fact_similarity", None) or 0.0) >= FACT_MIN_SIMILARITY


def is_candidate(hit: Any) -> bool:
    """Годится ли в кандидаты: все слова запроса в тексте, похожесть от мягкого порога
    или его привёл близкий факт графа."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return strict or (hit.similarity or 0.0) >= LOOSE_SIMILARITY or by_fact(hit)


def keep_facts(hits: list[Any], limit: int) -> list[Any]:
    """Первые limit — но до FACT_SLOTS мест фрагментам, которые привели факты. В выдаче
    каждого запроса они в конце, сверх найденного, и при слиянии запросов уходили за
    limit."""
    extra = [h for h in hits[limit:] if by_fact(h)][:FACT_SLOTS]
    return hits[: limit - len(extra)] + extra


def candidates(
    conn: Any,
    embedder: Any,
    queries: list[str],
    cache: dict[str, Any],
    limit: int = WIDE,
    facts: bool = False,
) -> list[Any]:
    """Кандидаты для модели — не больше CANDIDATE_PER_DOC из одной статьи (лишние — в
    конец): одна подробная статья занимала 9 мест из 15, и модели не из чего было
    выбирать — ответ на составной вопрос шёл по одной статье. facts — с фрагментами,
    которые привели факты графа (keep_facts)."""
    found = merged_search(conn, embedder, queries, cache, facts)
    close = [h for h in found if is_candidate(h)]
    return keep_facts(cap_per_doc(close, CANDIDATE_PER_DOC), limit)


def judge_prompt(
    question: str,
    plan: Plan,
    batch: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    notes: bool = False,
) -> str:
    parts = ""
    if len(plan.parts) > 1:
        parts = (
            "Части вопроса:\n"
            + "\n".join(f"{i}) {p.question}" for i, p in enumerate(plan.parts, start=1))
            + "\n"
        )
    taken = ""
    if accepted:
        taken = (
            "\nУже отобрано раньше (снова не называй):\n"
            + "\n".join(
                f"- «{c['hit'].title[:90]}»: {' '.join(c['hit'].text.split())[:160]}"
                for c in accepted[:12]
            )
            + "\n"
        )
    lines = []
    for c in batch:
        h = c["hit"]
        year = f", {h.year}" if h.year else ""
        lines.append(
            f"[{c['id']}] «{(h.title or h.doc_id)[:100]}»{year}: "
            f"{' '.join(h.text.split())[:2000 if plan.narrow else CANDIDATE_CHARS]}"
        )
    shown = "\n\n".join(lines) or "— кандидатов нет: по запросам ничего близкого не нашлось."
    done = ", ".join(f"«{q}»" for q in plan.queries + plan.extra) or "—"
    return JUDGE_USER.format(
        question=question.strip(),
        parts=parts,
        taken=taken,
        candidates=shown,
        n=EXTRA_QUERIES,
        done=done,
        notes=JUDGE_NOTES if notes else "",
        answer=(ANSWER_NOTES if notes else ANSWER_PLAIN).format(),
    )


def judge_fragments(
    question: str,
    plan: Plan,
    batch: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any] | None:
    """Модель читает кандидатов: {"подходят": [id…], "не_хватает": […], "запросы": […]}.
    Не ответила — None: тогда отбор по порогу, как у обычного поиска."""
    try:
        data = settings.client().chat_json(
            JUDGE_SYSTEM,
            judge_prompt(question, plan, batch, accepted, settings.notes),
            temperature=0.1,
            timeout=(10, 180),
        )
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _query_words(query: str) -> set[str]:
    from ..text import name_key

    return set(name_key(query).split())


def is_repeat(query: str, done: list[str]) -> bool:
    """Тот же запрос, что уже был, — по основам слов, с перестановкой и падежами."""
    words = _query_words(query)
    if not words:
        return True
    for other in map(_query_words, done):
        if other and len(words & other) / len(words | other) >= REPEAT_OVERLAP:
            return True
    return False


def _ids(values: Any) -> list[int]:
    out = []
    for v in values if isinstance(values, list) else []:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def _select(
    accepted: list[dict[str, Any]],
    plan: Plan,
    settings: Settings,
    top_k: int,
    taken: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Отобранное моделью → фрагменты ответа: у каждой части свои места,
    не больше per_doc из статьи (пока есть другие), порядок — как решила модель."""
    per_doc: dict[str, int] = {}
    for s in taken:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    chosen: list[dict[str, Any]] = []

    def take(c: Any, strict: Any = True) -> bool:
        if c in chosen or len(chosen) >= top_k:
            return False
        doc = c["hit"].doc_id
        if strict and settings.per_doc and per_doc.get(doc, 0) >= settings.per_doc:
            return False
        per_doc[doc] = per_doc.get(doc, 0) + 1
        chosen.append(c)
        return True

    if len(plan.parts) > 1:
        quota = max(PART_MIN, (top_k - 2) // len(plan.parts))
        for i in range(1, len(plan.parts) + 1):
            got = 0
            for c in accepted:
                if got >= quota:
                    break
                if c["part"] == i and take(c):
                    got += 1
    for c in accepted:
        take(c)
    for c in accepted:  # статей мало — лишние куски тех же статей, в конце
        take(c, strict=False)
    if len(plan.parts) > 1:
        for i, part in enumerate(plan.parts, start=1):
            part.found = sum(c["part"] == i for c in chosen)
    elif plan.parts:
        plan.parts[0].found = len(chosen)
    return chosen


def retrieve(
    conn: Any,
    embedder: Any,
    question: str,
    plan: Plan,
    settings: Settings,
    judge: Any = judge_fragments,
    taken: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    max_rounds: int = MAX_ROUNDS,
    extra: list[Any] | None = None,
) -> Generator[Any, Any, Any]:
    """Поиск с моделью. Генератор: отдаёт события status, в конце возвращает Retrieval
    (`result = yield from retrieve(…)`).

    Модель читает всех кандидатов круга — пачками по JUDGE_MAX. Новый круг
    ищется, только если модель говорит, чего не хватает, и предлагает запросы,
    которых ещё не было; круг, не добавивший ни одного фрагмента, — последний.

    judge=None или молчащая модель — отбор по порогу близости (find_sources)."""
    taken = list(taken or [])
    if judge is None:  # без модели — сразу отбор по порогу
        sources = find_sources(
            conn, embedder, plan, settings, start=len(taken) + 1, taken=taken, limit=limit
        )
        return Retrieval(sources, rounds=1, checked=0, judged=False)
    top_k = limit or settings.top_k
    cache: dict[str, Any] = {}
    pool: list[dict[str, Any]] = []
    seen = {fragment_key(s) for s in taken}

    spare: list[tuple[Any, ...]] = []  # сверх CANDIDATE_PER_DOC из статьи — если других мало

    def add(hits: Any, part: Any, rnd: Any, capped: bool = True) -> list[dict[str, Any]]:
        new = []
        for h in hits:
            if fragment_key(h) in seen:
                continue
            if capped and sum(c["hit"].doc_id == h.doc_id for c in pool) >= CANDIDATE_PER_DOC:
                spare.append((h, part, rnd))
                continue
            seen.add(fragment_key(h))
            c = {"id": len(pool) + 1, "hit": h, "part": part, "round": rnd}
            pool.append(c)
            new.append(c)
        return new

    facts = settings.fact_search
    # Узкому вопросу (об одном факте) — прежний набор и без заметок: лишний материал
    # размывал ответ (вопросы из графа, 15.10.2026).
    narrow = plan.narrow
    if settings.notes and narrow:
        settings = replace(settings, notes=False)
    wide = settings.wide if settings.notes or narrow else WIDE
    wide_part = max(WIDE_PART, wide // 2) if settings.notes else WIDE_PART
    multi = len(plan.parts) > 1
    if multi:
        # Все запросы вопроса — одной пачкой, одновременно (search_all): по частям
        # они шли бы по три, а каждая пачка ждала бы предыдущую.
        everything = [q for p in plan.parts for q in (p.queries or [p.question])]
        merged_search(conn, embedder, everything + list(plan.queries), cache, facts)
    # Фрагменты о предмете вопроса из графа — первыми: порога близости у них нет,
    # они найдены не по похожести, а по фактам.
    add(extra or [], None if multi else 1, 1, capped=False)
    if multi:
        for i, part in enumerate(plan.parts, start=1):
            asked = part.queries or [part.question]
            add(candidates(conn, embedder, asked, cache, wide_part, facts), i, 1)
        add(candidates(conn, embedder, plan.queries, cache, wide_part, facts), None, 1)
    else:
        add(candidates(conn, embedder, plan.queries, cache, wide, facts), 1, 1)
    # Ограничение — на весь набор: поиски по частям вопроса складываются, и одна статья
    # набирала 7 мест из 16. Других статей мало — лишние куски той же добирают места.
    for h, part, rnd in spare[: max(0, wide - len(pool))]:
        add([h], part, rnd, capped=False)
    if narrow:
        # Вопрос об одном факте: из найденных статей — ещё куски, ближайшие к вопросу.
        docs = list(dict.fromkeys(c["hit"].doc_id for c in pool))[:SIBLING_DOCS]
        known = {c["hit"].chunk_id for c in pool}
        add(same_article(conn, embedder, question, docs, known), 1, 1, capped=False)
    # Вопрос в переводе на языки статей базы (plan.translated) — своя небольшая добавка,
    # после основных и после добора кусков той же статьи: нужное на китайском или японском
    # приходит, а мест основной выдачи не занимает (иначе выпадала аннотация с ответом).
    if plan.translated:
        translated = candidates(conn, embedder, plan.translated, cache, TRANSLATED, facts)
        add(translated, None if multi else 1, 1)

    fresh: list[dict[str, Any]] = list(pool)
    accepted: list[dict[str, Any]] = []
    rounds, checked, judged = 1, 0, False
    while True:
        before = len(accepted)
        batches = [fresh[i : i + JUDGE_MAX] for i in range(0, len(fresh), JUDGE_MAX)] or [[]]
        if len(batches) > 1 and len(batches[-1]) <= JUDGE_SLACK:
            batches[-2:] = [batches[-2] + batches[-1]]  # 12 + 1 — лишний вызов модели
        missing, proposed, silent, answered = [], [], [], False
        for batch, data in (
            yield from _judge_all(question, plan, batches, accepted, settings, judge)
        ):
            if data is None:  # модель не ответила на эту пачку
                silent += batch
                continue
            answered = True
            checked += len(batch)
            by_id = {c["id"]: c for c in batch}
            notes = data.get("заметки") if isinstance(data.get("заметки"), dict) else {}
            for i in _ids(data.get("подходят") or data.get("relevant")):
                if i in by_id and by_id[i] not in accepted:
                    accepted.append(by_id[i])
                    note = " ".join(str(notes.get(str(i)) or "").split())[:NOTE_CHARS]
                    if note:
                        by_id[i]["note"] = note
            missing += [
                " ".join(str(m).split()) for m in data.get("не_хватает") or [] if str(m).strip()
            ]
            proposed += clean_queries(data.get("запросы") or data.get("queries"), EXTRA_QUERIES)
        if not answered:
            break  # модель молчит — отбор уже сделанный
        judged = True
        if rounds == 1:
            # Модель ответила не на все пачки первого круга — из молчаливых
            # берётся близкое по порогу, как у обычного поиска.
            accepted += [c for c in silent if is_close(c["hit"])]
        if missing or rounds == 1:
            plan.missing = list(dict.fromkeys(missing))[:5]
        if rounds >= max_rounds:
            break
        if rounds > 1 and len(accepted) == before:
            break  # новый круг ничего не добавил
        if accepted and not missing:
            break  # всего хватает
        done = plan.queries + plan.extra
        queries: list[str] = []
        for q in proposed:
            if q.lower() not in {d.lower() for d in done + queries} and not is_repeat(q, done):
                queries.append(q)
        queries = queries[:EXTRA_QUERIES]
        if not queries:
            break
        plan.extra += queries
        rounds += 1
        yield {"type": "status", "text": "ищу недостающее: " + " · ".join(queries)}
        fresh = add(candidates(conn, embedder, queries, cache, wide, facts), None, rounds)
        if not fresh and not missing:
            break

    if not judged:  # модель не ответила — по порогу
        sources = find_sources(
            conn, embedder, plan, settings, start=len(taken) + 1, taken=taken, limit=limit
        )
        return Retrieval(sources, rounds=1, checked=0, judged=False)
    chosen = _select(accepted, plan, settings, top_k, taken)
    sources = []
    for n, c in enumerate(chosen, start=len(taken) + 1):
        s = source_dict(n, c["hit"], c["part"] if multi else None)
        if c["round"] > 1:
            s["round"] = c["round"]
        sources.append(s)
    # Остальное отобранное — заметками: модель ответа видит, что в них сказано, и
    # ссылается на них тем же номером. Текст фрагмента при источнике — для человека.
    if settings.notes:
        rest = [c for c in accepted if c not in chosen and c.get("note")][:NOTES]
        for c in rest:
            s = source_dict(len(taken) + len(sources) + 1, c["hit"], c["part"] if multi else None)
            s.update(kind="заметка", note=c["note"])
            sources.append(s)
    if conn is not None:
        add_neighbors(conn, sources)
    return Retrieval(sources, rounds=rounds, checked=checked, judged=True)


def _judge_all(
    question: str,
    plan: Plan,
    batches: list[list[dict[str, Any]]],
    accepted: list[dict[str, Any]],
    settings: Settings,
    judge: Any,
) -> Generator[Any, Any, list[tuple[list[dict[str, Any]], Any]]]:
    """Модель читает пачки кандидатов. С заметками — несколько пачек одновременно
    (vLLM считает их вместе): пачек втрое больше, а ждать — почти столько же. Без
    заметок — по одной, и каждая видит отобранное в предыдущих."""
    total = sum(len(b) for b in batches)
    if settings.notes and len(batches) > 1:
        yield {"type": "status", "text": f"модель читает найденное: {total} фрагментов…"}
        with ThreadPoolExecutor(min(JUDGE_PARALLEL, len(batches))) as pool:
            done = list(pool.map(lambda b: judge(question, plan, b, accepted, settings), batches))
        return list(zip(batches, done, strict=True))
    out = []
    for k, batch in enumerate(batches, start=1):
        pack = f" (пачка {k} из {len(batches)})" if len(batches) > 1 else ""
        yield {
            "type": "status",
            "text": (
                f"модель читает найденное: {len(batch)} фрагментов{pack}…"
                if batch
                else "в базе ничего близкого — модель подбирает другие запросы…"
            ),
        }
        data = judge(question, plan, batch, accepted, settings)
        out.append((batch, data))
        if data is not None:  # следующая пачка видит отобранное
            by_id = {c["id"]: c for c in batch}
            accepted = accepted + [
                by_id[i] for i in _ids(data.get("подходят") or data.get("relevant")) if i in by_id
            ]
    return out


def run(gen: Any) -> Any:
    """Прогнать генератор-поиск без событий и взять результат (для оценки и проверок)."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
