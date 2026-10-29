"""WP02. Первый этап без модели — с защищёнными отличиями и журналом решений.

MinHash (graph/dedup.py, 64 значения, 16 полос × 4) только предлагает кандидата в
почти-дубли. Пропустить фрагмент мимо модели можно лишь после пересчёта настоящей меры
Жаккара по пятёркам слов и сверки защищённых смысловых полей: отрицание, модальность,
территории, стороны света, числа с единицами, минералы и термины словаря. Отличие в
любом из них — REVIEW_NOT_DROP, даже при сходстве 0,95.

Решение по каждому фрагменту — строка prefilter_decisions.jsonl: KEEP, DROP_DUPLICATE
(с representative), DROP_SPARSE (мало терминов), REVIEW_NOT_DROP — с причиной. Отсеянное
не удаляется и проходится заново после пополнения словаря.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..graph import dedup
from ..text import normalize

PREFILTER_RULES = "prefilter-v14.1"
JACCARD_DROP = dedup.DUP_SIMILARITY

_NEG = re.compile(r"(?<![а-яa-z])(?:не|нет|ни|без|отсутству\w*|not|no|without|absent|lack\w*|none|never)(?![а-яa-z])")
_MODAL = re.compile(r"вероятн\w*|возможн\w*|предполож\w*|по-видимому|may|might|could|probabl\w*|likely|possibl\w*|suggest\w*")
_SIDE = re.compile(
    r"(?<![а-яa-z])(?:север\w*|юг\w*|запад\w*|восток\w*|восточн\w*|западн\w*|северн\w*|южн\w*|"
    r"north\w*|south\w*|east\w*|west\w*|ne|nw|se|sw)(?![а-яa-z])"
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|г/т|g/t|ppm|ppb|км|km|м|m|млн|ma|ga|°c|мм|mm)?")
_WORD = re.compile(r"[a-zа-я0-9]+")


def shingles(text: str) -> set[str]:
    words = _WORD.findall(normalize(text))
    return {" ".join(words[i : i + dedup.SHINGLE]) for i in range(max(len(words) - dedup.SHINGLE + 1, 0))}


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class Protected:
    negation: frozenset[str]
    modality: frozenset[str]
    sides: frozenset[str]
    numbers: frozenset[str]
    regions: frozenset[str]
    terms: frozenset[str]


def protected(text: str, registry: Any = None, finder: Any = None) -> Protected:
    norm = normalize(text)
    regions: frozenset[str] = frozenset()
    if registry is not None:
        regions = frozenset(i for h in registry.find(text) for i in h.area_ids)
    terms: frozenset[str] = frozenset()
    if finder is not None:
        terms = frozenset(normalize(t) for t in finder.find(text).terms)
    return Protected(
        negation=frozenset(m.group() for m in _NEG.finditer(norm)),
        modality=frozenset(m.group() for m in _MODAL.finditer(norm)),
        sides=frozenset(m.group()[:5] for m in _SIDE.finditer(norm)),
        numbers=frozenset(" ".join(m.group().split()) for m in _NUMBER.finditer(norm)),
        regions=regions,
        terms=terms,
    )


def protected_diff(a: str, b: str, registry: Any = None, finder: Any = None) -> list[str]:
    """Какие защищённые поля различаются у двух текстов."""
    pa, pb = protected(a, registry, finder), protected(b, registry, finder)
    out = []
    for name in ("negation", "modality", "sides", "numbers", "regions", "terms"):
        if getattr(pa, name) != getattr(pb, name):
            out.append(name)
    return out


@dataclass
class Decision:
    chunk_id: Any
    decision: str  # KEEP / DROP_DUPLICATE / DROP_SPARSE / REVIEW_NOT_DROP
    reason: str
    jaccard: float | None = None
    signature_similarity: float | None = None
    critical_differ: list[str] = field(default_factory=list)
    terms_found: int = 0
    threshold_k: int = 0
    representative: Any = None
    lang: str = ""

    def to_json(self) -> dict[str, Any]:
        return {**self.__dict__, "rules": PREFILTER_RULES}


class Prefilter:
    """Решение по фрагментам в одном проходе, в памяти (для аудита и экспериментов E1).
    В графе (facts.run) тот же порядок: дубль уже пройденного — потом мало терминов."""

    def __init__(self, finder: Any, registry: Any = None, k: int = 3, protect: bool = True) -> None:
        self.finder = finder
        self.registry = registry
        self.k = k
        self.protect = protect
        self.seen: dict[Any, tuple[list[int], str]] = {}
        self.buckets: dict[tuple[int, int], list[Any]] = {}

    def decide(self, chunk_id: Any, text: str) -> Decision:
        found = self.finder.find(text or "")
        n_terms = len({normalize(t) for t in found.terms})
        sig = dedup.signature(text)
        if sig is not None:
            keys = dedup.bands(sig)
            cands = {c for band, h in enumerate(keys) for c in self.buckets.get((band, h), [])}
            best = None
            for c in cands:
                score = dedup.similarity(sig, self.seen[c][0])
                if score >= dedup.DUP_SIMILARITY and (best is None or score > best[1]):
                    best = (c, score)
            if best is not None:
                other_text = self.seen[best[0]][1]
                jac = jaccard(text, other_text)
                diff = protected_diff(text, other_text, self.registry, self.finder) if self.protect else []
                if diff:
                    return Decision(chunk_id, "REVIEW_NOT_DROP", "PREFILTER_PROTECTED_TOKEN", jac, best[1], diff,
                                    n_terms, self.k, best[0], found.lang)
                if jac >= JACCARD_DROP or not self.protect:
                    return Decision(chunk_id, "DROP_DUPLICATE", "near-duplicate", jac, best[1], [], n_terms,
                                    self.k, best[0], found.lang)
                # подпись совпала, а настоящая мера Жаккара ниже порога — не дубль
            self.seen[chunk_id] = (sig, text)
            for band, h in enumerate(keys):
                self.buckets.setdefault((band, h), []).append(chunk_id)
        if n_terms < self.k:
            return Decision(chunk_id, "DROP_SPARSE", f"terms {n_terms} < k={self.k}", terms_found=n_terms,
                            threshold_k=self.k, lang=found.lang)
        return Decision(chunk_id, "KEEP", "ok", terms_found=n_terms, threshold_k=self.k, lang=found.lang)
