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


def _title_stem(title: str) -> str:
    return normalize(_PART.sub("", title or "")).strip(" .:-–")


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
    by_stem: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for d in docs:
        if _PART.search(d.get("title") or ""):
            first = (d.get("authors") or [""])[0]
            by_stem[(_title_stem(d["title"])[:60], normalize(first))].append(d["doc_id"])
    for ids in by_stem.values():
        for b in ids[1:]:
            union(ids[0], b, "SERIES: части одной работы одних авторов")
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for d in docs:
        groups[find(d["doc_id"])].append(d["doc_id"])
    out = []
    for root, ids in sorted(groups.items()):
        ids = sorted(ids)
        reasons = sorted({r for (a, b), r in why.items() if a in ids and b in ids})
        kind = "SINGLE" if len(ids) == 1 else ("SERIES" if all(r.startswith("SERIES") for r in reasons) else "DUPLICATE_PUBLICATION")
        out.append(Family(f"FAM:{ids[0]}", tuple(ids), kind, "; ".join(reasons)))
    return out


def family_of(families: list[Family]) -> dict[str, str]:
    return {d: f.family_id for f in families for d in f.doc_ids}


def independent_support(claims: list[dict[str, Any]], families: dict[str, str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Сколько независимых семейств сообщают одну связь (src_key, relation, dst_key).
    Копии одной первичной работы — один голос (ТЗ WP07, приёмка)."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in claims:
        key = (name_key(c["src"]), normalize(c["relation"]), name_key(c["dst"]))
        row = out.setdefault(key, {"docs": set(), "families": set()})
        row["docs"].add(c["doc_id"])
        row["families"].add(families.get(c["doc_id"], c["doc_id"]))
    for row in out.values():
        row["n_docs"] = len(row["docs"])
        row["n_independent"] = len(row["families"])
        row["independence_status"] = "INDEPENDENT" if row["n_independent"] == row["n_docs"] else "DEPENDENT_COPIES"
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
