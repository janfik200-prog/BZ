"""WP05. Область действия факта по тексту источника — с основанием каждого шага.

Порядок источников контекста (закрывает заглушку v13 «[порядок источников контекста]»):

1. EXPLICIT — территория названа в самой цитате: высший приоритет.
2. ANAPHORA — в цитате ссылка «this area / the study area / этот район …», а
   территория названа одним-двумя предложениями раньше в том же фрагменте; основание —
   то предложение (span). Без явной ссылки соседнее предложение не используется.
3. SECTION — заголовок раздела фрагмента; принимается, только если во фрагменте нет
   других, несовместимых с ним территорий.
4. TITLE — название статьи; только если оно конкретное (одна территория, не обзор) и
   фрагмент не говорит о других территориях. Общий или обзорный заголовок —
   REGION_TITLE_ONLY: основанием для DIRECT он не служит.
5. Иначе — UNKNOWN. Общие знания модели основанием не считаются.

resolve_scope ничего не достраивает: одно написание у двух id — ALIAS_AMBIGUOUS
(если другая территория той же цитаты не снимает неоднозначность вложенностью),
несколько несовместимых территорий в цитате — MIXED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text import split_sentences
from .georegistry import DISJOINT, AliasHit, GeoAreaRegistry

SCOPE_RULES = "scope-rules-v1"

EXPLICIT, ANAPHORA, SECTION, TITLE, UNKNOWN = "EXPLICIT", "ANAPHORA", "SECTION", "TITLE", "UNKNOWN"

ANAPHOR = re.compile(
    r"\b(?:this|that|the\s+same|the\s+study|the\s+studied|the\s+present|our)\s+"
    r"(?:area|region|district|zone|belt|field|ore\s+field|deposit|terrane|province|basin|"
    r"block|site|locality|territory|cluster|massif|prospect)s?\b|\bthe\s+study\s+area\b|"
    r"\b(?:study|research|investigated|survey)\s+area\b|\bin\s+the\s+area\b|\bhere\b|"
    r"(?<![а-яё])(?:этот|этого|этом|данн\w+|изучаем\w+|исследуем\w+|описываем\w+|"
    r"рассматриваем\w+)\s+(?:район\w*|территори\w+|площад\w+|участк\w*|рудн\w+\s+пол\w+|"
    r"месторождени\w+|зон\w+|пояс\w*|узл\w*|област\w+)|"
    r"(?<![а-яё])(?:район\w*\s+работ|в\s+его\s+пределах|в\s+её\s+пределах|в\s+их\s+пределах|"
    r"здесь)(?![а-яё])",
    re.I,
)
REVIEW_TITLE = re.compile(
    r"\breview\b|\boverview\b|\bsystematics\b|обзор|\bsynthesis\b|\(\s*\d+\s+years|"
    r"\bglobal\b|\bworld\b|deposits\s+of\s+\w+\s+and\s+\w+",
    re.I,
)
BACK_SENTENCES = 2


@dataclass(frozen=True)
class ScopeSpan:
    level: str  # EXPLICIT / ANAPHORA / SECTION / TITLE
    text: str
    alias: str
    area_ids: tuple[str, ...]


@dataclass
class GeoScopeAssessment:
    region_ids: tuple[str, ...] = ()
    derivation: str = UNKNOWN
    reason_codes: list[str] = field(default_factory=list)
    spans: list[ScopeSpan] = field(default_factory=list)
    ambiguous: bool = False
    mixed: bool = False
    title_general: bool = False
    context_conflict: bool = False
    rules: str = SCOPE_RULES

    @property
    def resolved(self) -> bool:
        return bool(self.region_ids) and not self.ambiguous


def _most_specific(reg: GeoAreaRegistry, ids: set[str]) -> set[str]:
    """Без предков других территорий набора: «Чулковский участок» и «Якутия» → участок."""
    return {a for a in ids if not any(a in reg.ancestors(b) for b in ids if b != a)}


def _ids_from_hits(reg: GeoAreaRegistry, hits: list[AliasHit]) -> tuple[set[str], bool]:
    """Территории по совпадениям; неоднозначное имя снимается, только если ровно один
    из его id вложен в другую названную рядом территорию (или содержит её)."""
    plain = {i for h in hits if not h.ambiguous for i in h.area_ids}
    ambiguous = False
    for h in hits:
        if not h.ambiguous:
            continue
        related = [
            i
            for i in h.area_ids
            if any(p in reg.ancestors(i) or i in reg.ancestors(p) for p in plain)
        ]
        if len(related) == 1:
            plain.add(related[0])
        else:
            ambiguous = True
    return plain, ambiguous


def _conflicts(reg: GeoAreaRegistry, base: set[str], text: str) -> bool:
    """Говорит ли текст о территории, несовместимой с base (не вложенной и не родительской)."""
    others, _ = _ids_from_hits(reg, reg.find(text))
    return any(reg.relation(o, b) == DISJOINT for o in others for b in base)


def title_is_general(reg: GeoAreaRegistry, title: str) -> bool:
    """Общий или обзорный заголовок: обзор, «25 лет», или несколько несовместимых территорий."""
    if REVIEW_TITLE.search(title or ""):
        return True
    ids, ambiguous = _ids_from_hits(reg, reg.find(title or ""))
    ids = _most_specific(reg, ids)
    if ambiguous:
        return True
    return any(reg.relation(a, b) == DISJOINT for a in ids for b in ids if a < b)


def resolve_scope(
    quote: str,
    registry: GeoAreaRegistry,
    *,
    chunk_text: str = "",
    headings: tuple[str, ...] | list[str] = (),
    title: str = "",
    mode: str = "full",
) -> GeoScopeAssessment:
    """Область действия факта по цитате и её контексту.

    mode="full" — порядок 1–5 из описания модуля (V3). mode="naive" — V2: цитата, иначе
    название статьи без проверки общности и конфликтов; без анафоры и раздела.
    mode="no_anaphora" — абляция V3 без соседнего предложения."""
    out = GeoScopeAssessment()
    hits = registry.find(quote)
    if hits:
        ids, ambiguous = _ids_from_hits(registry, hits)
        out.spans = [ScopeSpan(EXPLICIT, quote, h.alias, h.area_ids) for h in hits]
        if ambiguous and not ids:
            out.ambiguous = True
            out.derivation = EXPLICIT
            out.reason_codes.append("ALIAS_AMBIGUOUS")
            return out
        ids = _most_specific(registry, ids)
        out.region_ids = tuple(sorted(ids))
        out.derivation = EXPLICIT
        out.ambiguous = ambiguous
        if ambiguous:
            out.reason_codes.append("ALIAS_AMBIGUOUS")
        if any(registry.relation(a, b) == DISJOINT for a in ids for b in ids if a < b):
            out.mixed = True
            out.reason_codes.append("REGION_MIXED")
        return out

    if mode in ("full",) and ANAPHOR.search(quote) and chunk_text:
        before = _preceding(quote, chunk_text)
        for sentence in reversed(before[-BACK_SENTENCES:]):
            found = registry.find(sentence)
            ids, ambiguous = _ids_from_hits(registry, found)
            if ids and not ambiguous:
                ids = _most_specific(registry, ids)
                out.region_ids = tuple(sorted(ids))
                out.derivation = ANAPHORA
                out.spans = [ScopeSpan(ANAPHORA, sentence, h.alias, h.area_ids) for h in found]
                out.reason_codes.append("ANAPHORA_RESOLVED")
                if len(ids) > 1 and any(
                    registry.relation(a, b) == DISJOINT for a in ids for b in ids if a < b
                ):
                    out.mixed = True
                    out.reason_codes.append("REGION_MIXED")
                return out

    if mode == "full":
        for heading in reversed(list(headings or ())):
            found = registry.find(heading)
            ids, ambiguous = _ids_from_hits(registry, found)
            if ids and not ambiguous:
                ids = _most_specific(registry, ids)
                if chunk_text and _conflicts(registry, ids, chunk_text):
                    out.context_conflict = True
                    continue
                out.region_ids = tuple(sorted(ids))
                out.derivation = SECTION
                out.spans = [ScopeSpan(SECTION, heading, h.alias, h.area_ids) for h in found]
                out.reason_codes.append("REGION_FROM_SECTION")
                return out

    found = registry.find(title or "")
    ids, ambiguous = _ids_from_hits(registry, found)
    ids = _most_specific(registry, ids)
    if ids:
        spans = [ScopeSpan(TITLE, title, h.alias, h.area_ids) for h in found]
        if mode == "naive":
            out.region_ids = tuple(sorted(ids))
            out.derivation = TITLE
            out.spans = spans
            out.reason_codes.append("REGION_FROM_TITLE")
            return out
        general = ambiguous or title_is_general(registry, title)
        conflict = bool(chunk_text) and _conflicts(registry, ids, chunk_text)
        out.spans = spans
        out.title_general = general
        out.context_conflict = out.context_conflict or conflict
        if general or conflict:
            out.derivation = TITLE
            out.reason_codes.append("REGION_TITLE_ONLY")
            if conflict:
                out.reason_codes.append("REGION_CONTEXT_CONFLICT")
            return out
        out.region_ids = tuple(sorted(ids))
        out.derivation = TITLE
        out.reason_codes.append("REGION_FROM_TITLE")
        return out
    out.reason_codes.append("REGION_UNKNOWN")
    return out


def _preceding(quote: str, chunk_text: str) -> list[str]:
    """Предложения фрагмента до цитаты (по первому её предложению)."""
    sents = split_sentences(" ".join((chunk_text or "").split()))
    head = split_sentences(quote)[0] if quote else ""

    def norm(s: str) -> str:
        return " ".join(s.split())

    for i, s in enumerate(sents):
        if norm(s) == norm(head) or (head and norm(head)[:60] in norm(s)):
            return sents[:i]
    return []
