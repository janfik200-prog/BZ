"""WP07. Независимость источников и противоречия.

Две статьи — не два голоса, если это одна работа в двух местах (препринт и журнальная
версия, перепечатка): семейство источника определяется по почти-дублям фрагментов
(MinHash, тот же, что в graph/dedup.py) и по серии «Часть I / Часть II» одних авторов.
Вес факта — число независимых семейств, а не статей; это не вероятность истинности и
не обход слабой поддержки (support.py решает отдельно).

Противоречие регистрируется только для сравнимых высказываний: те же концы и связь, та
же территория (EXACT), та же стадия — и разная полярность. Разные районы или стадии —
DIFFERENT_SCOPE, не противоречие. Автоматически противоречие не разрешается: эксперт
видит обе цитаты (dossier.py).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..graph import dedup
from ..text import name_key, normalize

FAMILY_RULES = "families-v1"
DUP_CHUNK_SHARE = 0.3  # с какой доли почти-дублей фрагментов статьи — одна работа
_PART = re.compile(r"\b(?:часть|part)\s+[ivx\d]+\b\.?", re.I)


@dataclass(frozen=True)
class Family:
    family_id: str
    doc_ids: tuple[str, ...]
    kind: str  # SINGLE / DUPLICATE_PUBLICATION / SERIES
    evidence: str


TITLE_SAME_WORK = 0.6
_TOKEN = re.compile(r"[a-zа-я0-9]{3,}")


def _series_key(title: str) -> str:
    """Название до «Часть …» / «Part …»: общая часть серии."""
    m = _PART.search(title or "")
    return normalize((title or "")[: m.start()]).strip(" .:-–()") if m else ""


def _title_jaccard(a: str, b: str) -> float:
    wa, wb = set(_TOKEN.findall(normalize(a))), set(_TOKEN.findall(normalize(b)))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _authors_overlap(x: dict[str, Any], y: dict[str, Any]) -> float:
    def words(d: dict[str, Any]) -> set[str]:
        return {w for a in (d.get("authors") or [])[:4] for w in _TOKEN.findall(normalize(str(a)))}

    wx, wy = words(x), words(y)
    return len(wx & wy) / min(len(wx), len(wy)) if wx and wy else 0.0


def source_families(docs: list[dict[str, Any]], chunks: dict[str, list[str]]) -> list[Family]:
    """Семейства источников. docs: doc_id, title, authors; chunks: doc_id → тексты фрагментов."""
    sigs: dict[str, list[list[int]]] = {}
    for doc_id, texts in chunks.items():
        sigs[doc_id] = [s for t in texts if (s := dedup.signature(t)) is not None]
    parent = {d["doc_id"]: d["doc_id"] for d in docs}
    why: dict[tuple[str, str], str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str, reason: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        why[(a, b)] = reason

    # Полосы LSH по всем фрагментам: кандидаты — статьи с общей полосой.
    buckets: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for doc_id, ss in sigs.items():
        for s in ss:
            for band, h in enumerate(dedup.bands(s)):
                buckets[(band, h)].add(doc_id)
    pairs = {tuple(sorted(p)) for b in buckets.values() if 1 < len(b) < 20 for p in combinations(b, 2)}
    for a, b in sorted(pairs):
        sa, sb = sigs[a], sigs[b]
        if not sa or not sb:
            continue
        dup = sum(any(dedup.similarity(x, y) >= dedup.DUP_SIMILARITY for y in sb) for x in sa)
        share = dup / min(len(sa), len(sb))
        if share >= DUP_CHUNK_SHARE:
            union(a, b, f"DUPLICATE_PUBLICATION: {share:.0%} почти-дублей фрагментов")
    # Серия «Часть I / Часть II» и версии одной работы (препринт → журнал): одна команда
    # авторов (половина слов имён общие) и общее начало названия до «Часть» или почти
    # то же название (доля общих слов ≥ 0,6).
    for x, y in combinations(docs, 2):
        if _authors_overlap(x, y) < 0.5:
            continue
        tx, ty = x.get("title") or "", y.get("title") or ""
        if _PART.search(tx) and _PART.search(ty) and _series_key(tx) and _series_key(tx) == _series_key(ty):
            union(x["doc_id"], y["doc_id"], "SERIES: части одной работы одной команды авторов")
        elif _title_jaccard(tx, ty) >= TITLE_SAME_WORK:
            union(x["doc_id"], y["doc_id"], "SAME_WORK_VERSION: та же команда авторов, почти то же название")
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for d in docs:
        groups[find(d["doc_id"])].append(d["doc_id"])
    out = []
    for root, ids in sorted(groups.items()):
        ids = sorted(ids)
        reasons = sorted({r for (a, b), r in why.items() if a in ids and b in ids})
        kind = "SINGLE" if len(ids) == 1 else reasons[0].split(":")[0] if len({r.split(":")[0] for r in reasons}) == 1 else "MIXED"
        out.append(Family(f"FAM:{ids[0]}", tuple(ids), kind, "; ".join(reasons)))
    return out


def family_of(families: list[Family]) -> dict[str, str]:
    return {d: f.family_id for f in families for d in f.doc_ids}


def independent_support(claims: list[dict[str, Any]], families: dict[str, str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Сколько независимых семейств сообщают одну связь (src_key, relation, dst_key).
    Копии одной первичной работы — один голос (ТЗ WP07, приёмка). Дословно та же цитата
    в разных статьях (повтор текста одной группы авторов) — тоже один голос: статьи
    объединяются по цитате, даже если семейства разные."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    quote_group: dict[str, str] = {}
    for c in claims:
        key = (name_key(c["src"]), normalize(c["relation"]), name_key(c["dst"]))
        row = out.setdefault(key, {"docs": set(), "families": set(), "text_reuse": False})
        row["docs"].add(c["doc_id"])
        fam = families.get(c["doc_id"], c["doc_id"])
        q = normalize(c.get("quote") or "")
        if len(q) >= 60:
            first = quote_group.setdefault(q, fam)
            if first != fam:
                row["text_reuse"] = True
                fam = first
        row["families"].add(fam)
    for row in out.values():
        row["n_docs"] = len(row["docs"])
        row["n_independent"] = len(row["families"])
        row["independence_status"] = (
            "INDEPENDENT" if row["n_independent"] == row["n_docs"]
            else "TEXT_REUSE" if row["text_reuse"] else "DEPENDENT_COPIES"
        )
    return out


@dataclass(frozen=True)
class ConflictCheck:
    a: str
    b: str
    verdict: str  # CONFLICT / DIFFERENT_SCOPE / DIFFERENT_STAGE / CONSISTENT
    reason: str


def compare(a: dict[str, Any], b: dict[str, Any]) -> ConflictCheck | None:
    """Две записи: claim_id, src, relation, dst, polarity, region_ids, stage.
    None — несравнимы (другие концы или связь)."""
    same_ends = name_key(a["src"]) == name_key(b["src"]) and name_key(a["dst"]) == name_key(b["dst"])
    if not same_ends or normalize(a["relation"]).replace("предположительно ", "") != normalize(b["relation"]).replace("предположительно ", ""):
        return None
    ra, rb = set(a.get("region_ids") or ()), set(b.get("region_ids") or ())
    if not ra or not rb or ra != rb:
        return ConflictCheck(a["claim_id"], b["claim_id"], "DIFFERENT_SCOPE", "разные или неизвестные территории — не противоречие")
    if (a.get("stage") or None) != (b.get("stage") or None):
        return ConflictCheck(a["claim_id"], b["claim_id"], "DIFFERENT_STAGE", "разные стадии — не противоречие")
    if a.get("polarity") != b.get("polarity"):
        return ConflictCheck(a["claim_id"], b["claim_id"], "CONFLICT", "одна территория и стадия, разная полярность")
    return ConflictCheck(a["claim_id"], b["claim_id"], "CONSISTENT", "")


def conflicts(records: list[dict[str, Any]]) -> list[ConflictCheck]:
    """Все сравнимые пары; противоречия отдельно не «разрешаются»."""
    by_ends: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ends[(name_key(r["src"]), name_key(r["dst"]))].append(r)
    out = []
    for group in by_ends.values():
        for a, b in combinations(group, 2):
            c = compare(a, b)
            if c is not None:
                out.append(c)
    return out
