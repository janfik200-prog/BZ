"""WP04. Поддержка утверждения цитатой: ENTAILED / PARTIAL / CONTRADICTED / UNKNOWN /
INVALID_EVIDENCE — отдельно от применимости к вопросу.

Три уровня (ТЗ WP04), не смешиваются:

1. формальные проверки кода — цитата воспроизводится по снимку статьи (provenance),
   оба конца названы в ней, связь не сводится к положению «южнее …»;
2. confirm.py — четыре вопроса одной модели одним вызовом, 0–4 совпадения. Это один
   коррелированный инструмент, а не четыре независимых голоса: 3 из 4 — порог,
   подобранный на DEV, а не вероятность истинности;
3. итог — метка с причинами; человеческая разметка (gold) хранится отдельно и сюда не
   подмешивается.

Повтор той же связи во второй статье поддержку не повышает (I10, fusion.py): буфер
store.accepted тут не используется.
"""

from __future__ import annotations

import re

from ..text import mentions
from .contracts import EvidenceSpan, GeoClaim, SupportAssessment
from .provenance import DocumentSnapshot, verify_evidence

SUPPORT_RULES = "support-rules-v1"
DEFAULT_MIN_VOTES = 3

_POSITIONAL = re.compile(
    r"\b(?:south|north|east|west|southeast|southwest|northeast|northwest)(?:ern)?\s+of\b|"
    r"\b(?:adjacent|close|next)\s+to\b|\bnear(?:by)?\b|\bborder\w*\b|"
    r"(?<![а-яё])(?:южнее|севернее|западнее|восточнее|вблизи|рядом\s+с|к\s+(?:северу|югу|"
    r"западу|востоку|северо-западу|северо-востоку|юго-западу|юго-востоку)\s+от)(?![а-яё])",
    re.I,
)
_POSITIONAL_RELATION = re.compile(
    r"располож|находит|южнее|севернее|западнее|восточнее|граничит|примыкает|соседств|"
    r"located|lies|bordered|adjacent",
    re.I,
)


def assess_support(
    claim: GeoClaim,
    evidence: EvidenceSpan,
    *,
    document: DocumentSnapshot | None = None,
    votes: int | None = None,
    min_votes: int = DEFAULT_MIN_VOTES,
    check_provenance: bool = True,
    split: str = "UNSPLIT",
) -> SupportAssessment:
    """Метка поддержки с причинами. votes — число совпадений confirm.py (None — не
    проверяли: NOT_CHECKED, и это не «подтверждено»)."""
    reasons: list[str] = []
    quote_ok = True
    if check_provenance:
        check = verify_evidence(evidence, document)
        quote_ok = check.valid
        if not check.valid:
            return _out(
                claim, evidence, "INVALID_EVIDENCE", list(check.reason_codes), False, split, votes
            )
        reasons.append("SOURCE_SPAN_EXACT")
    quote = evidence.exact_quote
    if not (mentions(claim.subject, quote) and mentions(claim.object, quote)):
        return _out(
            claim, evidence, "UNKNOWN", [*reasons, "SUPPORT_ENDS_MISSING"], quote_ok, split, votes
        )
    if _POSITIONAL.search(quote) and not _POSITIONAL_RELATION.search(claim.predicate):
        reasons.append("RELATION_POSITIONAL_ONLY")
        return _out(claim, evidence, "PARTIAL", reasons, quote_ok, split, votes)
    if claim.polarity == "NEGATE":
        reasons.append("POLARITY_NEGATED_IN_SOURCE")
    if votes is None:
        return _out(claim, evidence, "UNKNOWN", [*reasons, "NOT_CHECKED"], quote_ok, split, votes)
    if votes >= min_votes:
        label = "ENTAILED"
        reasons.append(f"SUPPORT_VOTES_GE{min_votes}")
    elif votes > 0:
        label = "PARTIAL"
        reasons += ["SUPPORT_VOTES_LOW", "SUPPORT_NOT_ENTAILED"]
    else:
        label = "UNKNOWN"
        reasons.append("SUPPORT_NOT_ENTAILED")
    return _out(claim, evidence, label, reasons, quote_ok, split, votes)


def _out(
    claim: GeoClaim,
    evidence: EvidenceSpan,
    label: str,
    reasons: list[str],
    quote_ok: bool,
    split: str,
    votes: int | None,
) -> SupportAssessment:
    by_llm = (
        votes is not None
        and label in ("ENTAILED", "PARTIAL", "UNKNOWN")
        and "NOT_CHECKED" not in reasons
    )
    return SupportAssessment(
        support_id=f"S:{claim.claim_id}:{evidence.evidence_id}",
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        label=label,
        reason_codes=tuple(reasons) or ("NO_REASON",),
        assessor_type="LLM" if by_llm else "DETERMINISTIC",
        assessor_id=f"{SUPPORT_RULES}+confirm4q" if by_llm else SUPPORT_RULES,
        quote_verified=quote_ok,
        annotator_id=None,
        evaluation_split=split,
    )
