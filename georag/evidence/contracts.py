"""Контракты свидетельств (ТЗ v13, contracts/*.schema.json): EvidenceSpan, GeoClaim,
QueryFrame, SupportAssessment, DecisionRecord.

Каждый объект сериализуется в JSON по схеме и проверяется ею (validate). Схемы v13.1 —
как в пакете ТЗ, без изменений; v14 — та же схема с необязательными полями, которых в
v13.1 не было: стадия и масштаб в вопросе (случай C32 — разные стадии оруденения),
эпистемический статус и основание области действия в решении (раздельное хранение
применимости и модальности, ТЗ п. 9). Объект без новых полей пишется как v13.1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

SCHEMAS = Path(__file__).resolve().parent / "schemas"
SCHEMA_VERSION = "v13.1"
V14 = "v14"

POLARITIES = ("AFFIRM", "NEGATE", "UNCLEAR")
MODALITIES = ("ASSERTED", "HYPOTHESIS", "REPORTED", "UNKNOWN")
QUANTIFIERS = ("EXISTS", "LOCAL", "TYPICAL", "ALL", "UNSPECIFIED")
SUPPORT_LABELS = ("ENTAILED", "PARTIAL", "CONTRADICTED", "UNKNOWN", "INVALID_EVIDENCE")
STATUSES = ("DIRECT", "ANALOG", "OUT_OF_SCOPE", "UNRESOLVED")
DERIVATIONS = ("EXPLICIT", "ANAPHORA", "SECTION", "TITLE", "REGISTRY")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@cache
def schema(name: str, version: str = SCHEMA_VERSION) -> dict[str, Any]:
    """Схема по имени: evidence, claim, query, support, decision."""
    folder = SCHEMAS if version == SCHEMA_VERSION else SCHEMAS / version
    data: dict[str, Any] = json.loads((folder / f"{name}.schema.json").read_text("utf-8"))
    return data


def validate(name: str, data: dict[str, Any]) -> None:
    """Проверить объект по схеме его версии; ошибка — jsonschema.ValidationError."""
    from jsonschema import Draft202012Validator

    version = data.get("schema_version", SCHEMA_VERSION)
    Draft202012Validator(schema(name, version)).validate(data)


@dataclass(frozen=True)
class EvidenceSpan:
    """Неизменяемое доказательство: цитата по смещениям в нормализованном тексте фрагмента.

    Нормализация одна и зафиксирована (provenance.normalize): пробельные знаки подряд —
    один пробел, без пробелов по краям. Смещения — в нормализованном тексте фрагмента
    chunk_id; parsed_sha256 — отпечаток нормализованного текста всей статьи (фрагменты
    по порядку через перевод строки)."""

    evidence_id: str
    doc_id: str
    pdf_sha256: str
    parsed_sha256: str
    parser_version: str
    chunk_id: str
    page_start: int
    page_end: int
    offset_start: int
    offset_end: int
    exact_quote: str
    quote_sha256: str
    sentence_indices: tuple[int, ...]
    license_status: str = "UNKNOWN"

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["sentence_indices"] = list(self.sentence_indices)
        return {"schema_version": SCHEMA_VERSION, **out}


@dataclass(frozen=True)
class ScopeProvenance:
    field: str
    evidence_id: str
    derivation: str  # EXPLICIT / ANAPHORA / SECTION / TITLE / REGISTRY
    rule_or_model_id: str


@dataclass(frozen=True)
class Scope:
    """Область действия факта. Не названное в источнике — None, не достраивается."""

    region_id: str | None = None
    region_name: str | None = None
    region_role: str | None = None
    deposit_type: str | None = None
    geological_setting: str | None = None
    mineralization_stage: str | None = None
    spatial_scale: str | None = None
    geological_time: str | None = None
    temporal_scope: str | None = None
    parent_region_ids: tuple[str, ...] = ()
    scope_provenance: tuple[ScopeProvenance, ...] = ()

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["parent_region_ids"] = list(self.parent_region_ids)
        out["scope_provenance"] = [asdict(p) for p in self.scope_provenance]
        return out


@dataclass(frozen=True)
class GeoClaim:
    """Атомарный факт f = (s, r, o, e, g, k, m, p, z) — ТЗ, раздел 3."""

    claim_id: str
    subject: str
    predicate: str
    object: str
    evidence_ids: tuple[str, ...]
    subject_type: str = "UNKNOWN"
    object_type: str = "UNKNOWN"
    polarity: str = "AFFIRM"
    modality: str = "UNKNOWN"
    quantifier: str = "UNSPECIFIED"
    scope: Scope = field(default_factory=Scope)
    source_family_id: str | None = None
    model_id: str = "unknown"
    prompt_sha256: str = "0" * 64
    extraction_status: str = "CANDIDATE"
    # Не в схеме: откуда факт (id в facts) и голоса confirm.py — для baseline V0/V1.
    fact_id: int | None = None
    legacy_votes: int | None = None
    legacy_accepted: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "claim_id": self.claim_id,
            "subject": self.subject,
            "subject_type": self.subject_type or "UNKNOWN",
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type or "UNKNOWN",
            "polarity": self.polarity,
            "modality": self.modality,
            "quantifier": self.quantifier,
            "scope": self.scope.to_json(),
            "evidence_ids": list(self.evidence_ids),
            "source_family_id": self.source_family_id,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "extraction_status": self.extraction_status,
        }


@dataclass(frozen=True)
class QueryFrame:
    """Смысл вопроса, а не строка с названием района (ТЗ, раздел 3)."""

    query_id: str
    raw_text: str
    target_region_id: str | None = None
    target_region_name: str | None = None
    target_deposit_type: str | None = None
    required_predicate: str | None = None
    requested_polarity: str = "AFFIRM"
    requested_quantifier: str = "UNSPECIFIED"
    allow_analogies: bool = False
    requested_mode: str = "FACTUAL"  # FACTUAL / COMPARATIVE / RESEARCH_REVIEW
    target_stage: str | None = None  # v14
    target_scale: str | None = None  # v14

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        extra = {k: out.pop(k) for k in ("target_stage", "target_scale")}
        if any(v is not None for v in extra.values()):
            return {"schema_version": V14, **out, **extra}
        return {"schema_version": SCHEMA_VERSION, **out}


@dataclass(frozen=True)
class SupportAssessment:
    support_id: str
    claim_id: str
    evidence_id: str
    label: str
    reason_codes: tuple[str, ...]
    assessor_type: str = "DETERMINISTIC"  # DETERMINISTIC / LLM / HUMAN
    assessor_id: str = "support-rules-v1"
    quote_verified: bool = False
    annotator_id: str | None = None
    evaluation_split: str = "UNSPLIT"

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["reason_codes"] = list(self.reason_codes) or ["NO_REASON"]
        return {"schema_version": SCHEMA_VERSION, **out}


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    query_id: str
    claim_id: str
    support_id: str
    status: str
    reason_codes: tuple[str, ...]
    policy_version: str
    registry_version: str
    evidence_ids: tuple[str, ...]
    is_human_override: bool = False
    override_reason: str | None = None
    # v14: эпистемический статус отдельно от применимости (DIRECT + HYPOTHESIS — прямая
    # гипотеза о районе, а не доказанный факт) и основание области действия.
    modality: str | None = None
    polarity: str | None = None
    scope_derivation: str | None = None

    def to_json(self) -> dict[str, Any]:
        out = {
            "decision_id": self.decision_id,
            "query_id": self.query_id,
            "claim_id": self.claim_id,
            "support_id": self.support_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
            "registry_version": self.registry_version,
            "evidence_ids": list(self.evidence_ids),
            "is_human_override": self.is_human_override,
            "override_reason": self.override_reason,
        }
        extra = {
            "modality": self.modality,
            "polarity": self.polarity,
            "scope_derivation": self.scope_derivation,
        }
        if any(v is not None for v in extra.values()):
            return {"schema_version": V14, **out, **extra}
        return {"schema_version": SCHEMA_VERSION, **out}
