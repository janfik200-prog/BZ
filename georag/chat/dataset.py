"""Ответ из графа: датасет предмета вопроса и фрагменты о нём.

«Собрать всё» и вопросы о связях: поиск приносит восемь похожих кусков, а фактов
о предмете в базе может быть из тридцати статей — граф знает все, со счётом статей
и цитатами. Вопрос о названном предмете: фрагменты, из которых выписаны факты о
нём, — в кандидаты для модели."""

from __future__ import annotations

import re
from typing import Any

# «Собрать всё» о предмете вопроса: все факты о нём из графа и фрагменты к главным.
DATASET_EVIDENCE = 5
# Вопрос о названном предмете (не «собрать всё»): фрагменты, из которых граф
# выписал факты о нём, идут модели в кандидаты вместе с найденным поиском.
GRAPH_CANDIDATES = 6
DATASET_ITEMS = 40  # сколько фактов (×2) попадает в сводку
GROUP_ITEMS = 15  # сколько «с чем» в одной строке сводки (тип связи)


def _synonyms() -> Any:
    from ..graph import synonyms

    return synonyms.load()


def _stems(text: str) -> set[str]:
    """Основы значимых слов: первые 5 букв слов от 4 букв — падеж не мешает."""
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def resolve_entity(g: Any, hint: str | None, question: str) -> str | None:
    """О чём вопрос: по подсказке модели или по самому вопросу — среди сущностей графа."""
    from ..text import normalize, phrase_pattern

    if hint:
        name = g.resolve(hint)
        if name:
            return str(name)
    # По тексту вопроса: самое длинное имя (или синоним), что в нём встречается.
    # Если модель назвала предмет, а граф его не знает («Анабарский щит», о котором
    # статей нет), годится только имя с общим словом из её подсказки, иначе вместо
    # предмета подхватится общее слово вопроса («геологическое строение»).
    text = normalize(question)
    want = _stems(hint) if hint else set()
    hits = []
    for name in g.entities:
        spellings = [name, *g.syn.groups.get(name, [])]
        for spelled in spellings:
            if len(spelled) < 4 or (want and not want & _stems(spelled)):
                continue
            try:
                if phrase_pattern(spelled).search(text):
                    hits.append(name)
                    break
            except re.error:
                continue
    return max(hits, key=len) if hits else None


def question_subject(conn: Any, hint: str | None, question: str) -> str | None:
    """О чём вопрос: сущность графа (по подсказке модели или по тексту вопроса), иначе
    сама подсказка; None — не узнать."""
    try:
        from ..graph import api

        name = resolve_entity(api.graph(conn, _synonyms()), hint, question)
    except Exception:  # noqa: BLE001 — графа нет: только подсказка модели
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        name = None
    # Названо имя собственное («Билляхская зона разломов»), а граф знает только общее
    # («зоны разломов»): сбор читал 30 статей о разломах вообще и выдавал их за ответ о
    # Билляхской зоне (оценка 15.10.2026). Найденное должно содержать это имя.
    return named(name, hint, question) or hint or named(subject_in(question), None, question)


# «С какими породами связаны мигматиты?» — предмет после глагола связи. Модель при разборе
# предмет почти не называет (оставляет null из образца), а граф знает только «migmatites»:
# сбор не включался, ответ строился из 4 статей из 23 (замер 15.10.2026).
_LINK_VERB = re.compile(
    r"\b(?:связан\w*|ассоцииру\w*|сопровожда\w*|сопутству\w*|сочета\w*|сопряж\w*|"
    r"контролиру\w*)\s+(?:(?:с|со)\s+)?(?P<subject>[^?!.,;:()«»\"]+)",
    re.IGNORECASE,
)
_SUBJECT_END = re.compile(
    r"\s(?:в|во|на|по|при|для|у|из|от|среди|вблизи|около|и|или|если|когда)\s", re.IGNORECASE
)


def subject_in(question: str) -> str | None:
    """Предмет вопроса о связях — по тексту, без модели: «С чем связана аргиллизация?» →
    «аргиллизация»; не узнать (другой вопрос, длинная фраза) — None."""
    match = _LINK_VERB.search(question or "")
    # «Как связаны разломы и оруденение?» — о связи двух, не об одном предмете.
    if not match or re.match(r"\s*как\s+связ", question, re.IGNORECASE):
        return None
    subject = _SUBJECT_END.split(f" {match['subject'].strip()} ")[0].strip()
    words = subject.split()
    if not 1 <= len(words) <= 4 or len(subject) < 4 or subject.lower().startswith(("чем", "как")):
        return None
    return subject


def named(name: str | None, hint: str | None, question: str) -> str | None:
    """Сущность графа — если в ней есть имя собственное из подсказки или вопроса; иначе
    None: «Билляхская зона разломов» — не «зоны разломов» (граф знает только общее)."""
    need = proper_stems(hint or question)
    if name and need and not need <= {w.lower()[:5] for w in _WORDS.findall(name)}:
        return None
    return name


_WORDS = re.compile(r"[^\W\d_][\w-]*")
# Прилагательное от имени собственного: «Билляхской», «шахтаминский» (а «геологический» — нет).
_PROPER_ADJ = re.compile(r"(?<!че)(?:ск|цк)(?:ий|ая|ое|ие|ой|ого|ую|ом|их|ым|ими)$")


def proper_stems(text: str) -> set[str]:
    """Основы имён собственных в тексте: слово с большой буквы не в начале или
    прилагательное от имени («Билляхской зоне» → {«билля»})."""
    out = set()
    previous_end = 0
    for i, match in enumerate(_WORDS.finditer(text or "")):
        word = match.group()
        sentence_start = i == 0 or bool(re.search(r"[.!?\n]", text[previous_end : match.start()]))
        previous_end = match.end()
        if len(word) < 4:
            continue
        if (not sentence_start and word[0].isupper()) or _PROPER_ADJ.search(word.lower()):
            out.add(word.lower()[:5])
    return out


def collect_dataset(conn: Any, hint: str | None, question: str) -> dict[str, Any] | None:
    """Датасет для вопроса «собрать всё»: факты о сущности и фрагменты к главным.
    None — сущность не узнана, фактов нет (граф не строили) или про неё ничего нет."""
    if conn is None:
        return None
    from ..graph import api, store

    try:
        syn = _synonyms()
        g = api.graph(conn, syn)
        name = named(resolve_entity(g, hint, question), hint, question)
        if not name:
            return None
        data = api.dataset(conn, syn, name, g)
        if not data["facts"]:
            return None
        chunk_ids = []
        for row in data["facts"]:
            for cid in row["chunk_ids"][:1]:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
            if len(chunk_ids) >= DATASET_EVIDENCE:
                break
        frags = store.fragments(conn, chunk_ids)
        data["evidence"] = [frags[c] for c in chunk_ids if c in frags]
        data["pass"] = store.summary(conn)
        return data
    except Exception:  # noqa: BLE001 — нет фактов или словаря: обычный поиск
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def graph_candidates(conn: Any, hint: str | None, question: str) -> list[Any]:
    """Фрагменты, где сказано о предмете вопроса, — по графу, а не по похожести.

    Редкое название («Персияновский разлом») векторный поиск находит плохо:
    для него это просто похожие слова. Граф знает, из каких фрагментов о нём
    выписаны факты. Эти фрагменты — кандидаты: брать ли их, решает модель, как
    и для найденного поиском. Графа нет или предмет не узнан — пусто."""
    if conn is None or not hint:
        return []
    from ..graph import api, store
    from ..index.search import Hit

    try:
        syn = _synonyms()
        g = api.graph(conn, syn)
        name = named(resolve_entity(g, hint, question), hint, question)
        if not name:
            return []
        ids: list[int] = []
        for row in api.dataset(conn, syn, name, g)["facts"]:
            for cid in row["chunk_ids"]:
                if cid not in ids:
                    ids.append(cid)
            if len(ids) >= GRAPH_CANDIDATES:
                break
        ids = ids[:GRAPH_CANDIDATES]
        frags = store.fragments(conn, ids)
    except Exception:  # noqa: BLE001 — нет графа или словаря: обойдёмся поиском
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return []
    return [
        Hit(
            chunk_id=cid,
            doc_id=f["doc_id"],
            ord=f["ord"],
            text=f["text"],
            headings=f["headings"],
            pages=f["pages"],
            title=f["title"],
            year=f["year"],
            journal=f["journal"],
            url=f["url"],
            authors=f["authors"],
            found_by="граф",
        )
        for cid in ids
        if (f := frags.get(cid))
    ]


def _count(n: int) -> str:
    last2, last = n % 100, n % 10
    word = (
        "статья"
        if last == 1 and last2 != 11
        else "статьи" if 2 <= last <= 4 and not 12 <= last2 <= 14 else "статей"
    )
    return f"{n} {word}"


def dataset_summary(data: dict[str, Any]) -> str:
    """Факты о сущности одним текстом — «фрагмент» для модели."""
    includes = data.get("includes") or []
    # «Полный список» — только если модель прошла все фрагменты базы. Граф
    # строится ночью порциями, и новые статьи попадают в него не сразу.
    passed = data.get("pass") or {}
    done, total = passed.get("passed") or 0, passed.get("chunks") or 0
    scope = (
        "это полный список по графу"
        if not total or done >= total
        else f"это всё, что есть в графе, но модель прошла пока {done} из {total} фрагментов "
        "базы — в остальных может быть ещё"
    )
    lines = [
        f"Факты о «{data['name']}»"
        + (f" (вместе с тем, что в неё входит: {', '.join(includes)})" if includes else "")
        + " — выписаны моделью из статей базы и проверены по цитатам, а не найдены "
        f"поиском; {scope}. Статей: {data['documents']}."
    ]
    lines += _grouped(data["facts"][: DATASET_ITEMS * 2])
    more = len(data["facts"]) - DATASET_ITEMS * 2
    if more > 0:
        lines.append(f"- и ещё {more}")
    return "\n".join(lines)


def _grouped(rows: list[dict[str, Any]]) -> list[str]:
    """Строки сводки: по типу связи — «золото ассоциирует с: пирит (5 статей), …».

    Связи по типам (graph/kinds.py) идут строкой на тип, от самой весомой; факты без
    типа (граф до 14.10.2026, ещё не размеченные) — по одному, словами статьи."""
    from ..graph.kinds import HEDGE, TITLES

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    plain: list[dict[str, Any]] = []
    for row in rows:
        base = row["relation"].removeprefix(HEDGE + " ")
        if base in TITLES:
            # «ассоциирует с» в обе стороны — одно и то же: одной строкой.
            direction = "→" if TITLES[base][0] == TITLES[base][1] else row["direction"]
            groups.setdefault((row["about"], base, direction), []).append(row)
        else:
            plain.append(row)
    lines = []
    weight = {key: sum(r["documents"] for r in items) for key, items in groups.items()}
    for (about, base, direction), items in sorted(groups.items(), key=lambda kv: -weight[kv[0]]):
        title = TITLES[base][0 if direction == "→" else 1]
        named = ", ".join(
            f"{r['other']}"
            + (" — предположительно" if r["relation"].startswith(HEDGE) else "")
            + f" ({_count(r['documents'])})"
            for r in items[:GROUP_ITEMS]
        )
        rest = f" и ещё {len(items) - GROUP_ITEMS}" if len(items) > GROUP_ITEMS else ""
        lines.append(f"- {about} {title}: {named}{rest}")
    for row in plain:
        fact = (
            f"{row['about']} — {row['relation']} — {row['other']}"
            if row["direction"] == "→"
            else f"{row['other']} — {row['relation']} — {row['about']}"
        )
        lines.append(f"- {fact} ({_count(row['documents'])})")
    return lines


def dataset_sources(data: dict[str, Any], start: int = 1) -> list[dict[str, Any]]:
    """Сводка фактов — первым фрагментом, за ней фрагменты статей к главным фактам."""
    out = [
        {
            "n": start,
            "doc_id": f"датасет:{data['name']}",
            "ord": -1,
            "title": f"Факты из графа: {data['name']}",
            "year": None,
            "journal": None,
            "authors": [],
            "pages": [],
            "headings": [],
            "url": "",
            "text": dataset_summary(data),
            "similarity": None,
            "found_by": "датасет",
            "part": None,
            "kind": "датасет",
        }
    ]
    for e in data.get("evidence") or []:
        out.append(
            {
                "n": start + len(out),
                "doc_id": e["doc_id"],
                "ord": e["ord"],
                "title": e.get("title") or e["doc_id"],
                "year": e.get("year"),
                "journal": e.get("journal"),
                "authors": e.get("authors") or [],
                "pages": e.get("pages") or [],
                "headings": e.get("headings") or [],
                "url": e.get("url") or "",
                "text": e["text"],
                "similarity": None,
                "found_by": "датасет",
                "part": None,
            }
        )
    return out
