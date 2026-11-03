"""WP06. Применимость факта к вопросу: DIRECT / ANALOG / OUT_OF_SCOPE / UNRESOLVED.

decide_applicability — чистая функция без модели: тот же вопрос, факт, поддержка,
область действия и справочник дают то же решение, в любом порядке документов и при
любом тексте ответа модели. Статус говорит о применимости источника, а не об истинности
сказанного; модальность и полярность хранятся рядом (DIRECT + HYPOTHESIS — прямая
гипотеза о районе, а не доказанное наличие руды).

Инварианты (ТЗ, раздел 9), проверяются tests/evidence/test_applicability.py:
  I1  INVALID_EVIDENCE → не DIRECT;
  I2  PARTIAL / CONTRADICTED / UNKNOWN → не DIRECT;
  I3  известный чужой или несовместимый район → не DIRECT;
  I4  факт о части территории → не DIRECT для универсального свойства целой;
  I5  гипотеза не выдаётся за установленный факт;
  I6  отрицание не подтверждает положительный факт;
  I7  территория не установлена → не DIRECT;
  I8  область факта не берётся из вопроса (resolve_scope вопроса не видит);
  I9  решение воспроизводимо: evidence_ids, registry_version, policy_version, reasons;
  I10 снятие проверенного доказательства не повышает статус.

Варианты V0–V3 и абляции — PolicyConfig; основной метод — V3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from ..text import normalize
from .contracts import DecisionRecord, GeoClaim, QueryFrame, SupportAssessment
from .georegistry import CONTAINS, DISJOINT, EXACT, OVERLAP, PART_OF, UNKNOWN, GeoAreaRegistry
from .scope import GeoScopeAssessment

POLICY_VERSION = "applicability-v14.1"
_RANK = {EXACT: 0, PART_OF: 1, CONTAINS: 2, OVERLAP: 3, DISJOINT: 4, UNKNOWN: 5}


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "V3"
    use_support: bool = True  # False — признак старого графа (votes ≥ 3 или повтор)
    scope_mode: str = "full"  # none / string / naive / full
    use_deposit: bool = True
    use_quantifier: bool = True
    use_modality: bool = True
    use_polarity: bool = True
    strict_mixed: bool = True  # несколько несовместимых районов в цитате — UNRESOLVED
    use_temporal: bool = True


V0 = PolicyConfig(
    "V0",
    use_support=False,
    scope_mode="none",
    use_deposit=False,
    use_quantifier=False,
    use_modality=False,
    use_polarity=False,
    strict_mixed=False,
    use_temporal=False,
)
V1 = PolicyConfig(
    "V1",
    use_support=False,
    scope_mode="string",
    use_deposit=False,
    use_quantifier=False,
    use_modality=False,
    use_polarity=False,
    strict_mixed=False,
    use_temporal=False,
)
V2 = PolicyConfig(
    "V2",
    use_support=True,
    scope_mode="naive",
    use_deposit=True,
    use_quantifier=False,
    use_modality=False,
    use_polarity=False,
    strict_mixed=False,
    use_temporal=False,
)
V3 = PolicyConfig("V3")
VARIANTS = {"V0": V0, "V1": V1, "V2": V2, "V3": V3}
ABLATIONS = {
    "V3-registry": replace(V3, name="V3-registry", scope_mode="string"),
    "V3-quantifier": replace(V3, name="V3-quantifier", use_quantifier=False),
    "V3-modality_polarity": replace(
        V3, name="V3-modality_polarity", use_modality=False, use_polarity=False
    ),
    "V3-support": replace(V3, name="V3-support", use_support=False),
    "V3-anaphora": replace(V3, name="V3-anaphora"),  # scope считается в mode="no_anaphora"
    "V3-title_rules": replace(V3, name="V3-title_rules"),  # scope в mode="naive"
}


def _decision_id(query: QueryFrame, claim: GeoClaim, cfg: PolicyConfig, version: str) -> str:
    raw = f"{query.query_id}|{claim.claim_id}|{POLICY_VERSION}|{cfg.name}|{version}"
    return "D:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _name_match(query: QueryFrame, registry: GeoAreaRegistry, texts: list[str]) -> bool:
    """V1: название территории вопроса (и его синонимы из справочника) встречается в тексте —
    строкой, без вложенности и без проверки других районов."""
    names = {normalize(query.target_region_name or "")}
    if query.target_region_id:
        names |= {k for k, ids in registry.aliases.items() if query.target_region_id in ids}
    names.discard("")
    joined = normalize(" ".join(texts))
    return any(re.search(r"(?<![a-zа-я0-9])" + re.escape(n), joined) for n in names)


def _deposit_ok(query: QueryFrame, claim: GeoClaim) -> bool | None:
    """Совместим ли тип оруденения: None — не известен у факта или у вопроса."""
    if not query.target_deposit_type or not claim.scope.deposit_type:
        return None
    want = set(query.target_deposit_type.split(";"))
    have = set(claim.scope.deposit_type.split(";"))
    return bool(want & have)


def decide_applicability(
    query: QueryFrame,
    claim: GeoClaim,
    support: SupportAssessment,
    registry: GeoAreaRegistry,
    *,
    scope: GeoScopeAssessment | None = None,
    config: PolicyConfig = V3,
    name_texts: list[str] | None = None,
    conflicting: bool = False,
) -> DecisionRecord:
    """Решение по паре (вопрос, факт). scope — resolve_scope по цитате и её контексту (не по
    вопросу: I8); name_texts — тексты для V1 (цитата, заголовки, название статьи)."""
    reasons: list[str] = []
    status = "DIRECT"
    relation = None

    # --- поддержка источником (I1, I2, I10) ---------------------------------------------
    if config.use_support:
        if support.label == "INVALID_EVIDENCE":
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                ["SOURCE_HASH_INVALID", *support.reason_codes],
                scope,
            )
        if support.label != "ENTAILED":
            code = (
                "NOT_CHECKED" if "NOT_CHECKED" in support.reason_codes else "SUPPORT_NOT_ENTAILED"
            )
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [code, f"SUPPORT_{support.label}"],
                scope,
            )
        reasons.append("SUPPORT_ENTAILED")
    elif not claim.legacy_accepted:
        return _record(
            query,
            claim,
            support,
            registry,
            config,
            "UNRESOLVED",
            ["SUPPORT_NOT_ENTAILED", "LEGACY_CANDIDATE"],
            scope,
        )
    else:
        reasons.append("LEGACY_ACCEPTED")

    # --- территория -------------------------------------------------------------------
    target = query.target_region_id
    if config.scope_mode == "none" or target is None:
        reasons.append("REGION_NOT_CHECKED" if target is not None else "NO_TARGET_REGION")
    elif config.scope_mode == "string":
        if _name_match(query, registry, name_texts or []):
            reasons.append("REGION_NAME_MATCH")
        else:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "REGION_NAME_NOT_FOUND"],
                scope,
            )
    else:
        sc = scope or GeoScopeAssessment(reason_codes=["REGION_UNKNOWN"])
        reasons += [r for r in sc.reason_codes if r not in ("REGION_MIXED",)]
        if not sc.region_ids:
            if sc.ambiguous:
                reasons.append("ALIAS_AMBIGUOUS")
            if "REGION_TITLE_ONLY" not in reasons and "ALIAS_AMBIGUOUS" not in reasons:
                reasons.append("REGION_UNKNOWN")
            return _record(query, claim, support, registry, config, "UNRESOLVED", reasons, scope)
        rels = {r: registry.relation(r, target) for r in sc.region_ids}
        if config.strict_mixed and sc.mixed:
            kinds = set(rels.values())
            if kinds & {EXACT, PART_OF} and kinds - {EXACT, PART_OF, CONTAINS}:
                return _record(
                    query,
                    claim,
                    support,
                    registry,
                    config,
                    "UNRESOLVED",
                    [*reasons, "REGION_MIXED"],
                    scope,
                )
        relation = min(rels.values(), key=_RANK.__getitem__)
        if config.strict_mixed and sc.ambiguous:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "ALIAS_AMBIGUOUS"],
                scope,
            )
        if relation == EXACT:
            reasons.append("REGION_EXACT")
        elif relation == PART_OF:
            reasons.append("REGION_PART_OF")
        elif relation == CONTAINS:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "REGION_CONTAINS_TARGET", "SCOPE_UNIVERSALITY_MISMATCH"],
                scope,
            )
        elif relation == OVERLAP:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "REGION_OVERLAP"],
                scope,
            )
        elif relation == UNKNOWN:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "NO_VERIFIED_GEO_ID"],
                scope,
            )
        else:  # DISJOINT
            reasons.append("REGION_DISJOINT")
            if query.allow_analogies or query.requested_mode == "COMPARATIVE":
                # Аналогия — только при подтверждённой совместимости типа оруденения и общей
                # геологической обстановке из справочника; «тот же металл» — не основание.
                setting = any(registry.shared_settings(r, target) for r in sc.region_ids)
                if setting and _deposit_ok(query, claim) is True:
                    reasons.append("GEOLOGICAL_ANALOGY_GROUNDED")
                    status = "ANALOG"
                else:
                    reasons.append("ANALOGY_NOT_GROUNDED")
                    status = "OUT_OF_SCOPE"
            else:
                status = "OUT_OF_SCOPE"

    # --- тип оруденения ---------------------------------------------------------------------
    if config.use_deposit and status in ("DIRECT", "ANALOG") and _deposit_ok(query, claim) is False:
        return _record(
            query,
            claim,
            support,
            registry,
            config,
            "OUT_OF_SCOPE",
            [*reasons, "DEPOSIT_TYPE_MISMATCH"],
            scope,
        )
    if status != "DIRECT":
        return _record(query, claim, support, registry, config, status, reasons, scope)

    # --- полярность (I6), модальность (I5), квантор (I4), стадия ----------------------------
    if config.use_polarity:
        if claim.polarity == "NEGATE" and query.requested_polarity == "AFFIRM":
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "POLARITY_NEGATED"],
                scope,
            )
        if claim.polarity == "AFFIRM" and query.requested_polarity == "NEGATE":
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "POLARITY_MISMATCH"],
                scope,
            )
        if claim.polarity == "NEGATE":
            reasons.append("POLARITY_MATCH")
        if claim.polarity == "UNCLEAR":
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "POLARITY_UNCLEAR"],
                scope,
            )
    if config.use_modality and claim.modality == "HYPOTHESIS":
        reasons.append("MODALITY_HYPOTHESIS")
        if query.requested_mode != "RESEARCH_REVIEW":
            return _record(query, claim, support, registry, config, "UNRESOLVED", reasons, scope)
    if config.use_quantifier and query.requested_quantifier in ("TYPICAL", "ALL"):
        whole = relation in (None, EXACT) or config.scope_mode in ("none", "string")
        strong = {"TYPICAL": ("TYPICAL", "ALL"), "ALL": ("ALL",)}[query.requested_quantifier]
        if not whole or claim.quantifier not in strong:
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "SCOPE_UNIVERSALITY_MISMATCH"],
                scope,
            )
    if config.use_temporal and query.target_stage:
        if not claim.scope.mineralization_stage or normalize(
            claim.scope.mineralization_stage
        ) != normalize(query.target_stage):
            return _record(
                query,
                claim,
                support,
                registry,
                config,
                "UNRESOLVED",
                [*reasons, "TEMPORAL_SCOPE_MISMATCH"],
                scope,
            )
    if conflicting:
        return _record(
            query,
            claim,
            support,
            registry,
            config,
            "UNRESOLVED",
            [*reasons, "CONFLICTING_EVIDENCE"],
            scope,
        )
    return _record(query, claim, support, registry, config, "DIRECT", reasons, scope)


def _record(
    query: QueryFrame,
    claim: GeoClaim,
    support: SupportAssessment,
    registry: GeoAreaRegistry,
    cfg: PolicyConfig,
    status: str,
    reasons: list[str],
    scope: GeoScopeAssessment | None,
) -> DecisionRecord:
    seen: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    return DecisionRecord(
        decision_id=_decision_id(query, claim, cfg, registry.version),
        query_id=query.query_id,
        claim_id=claim.claim_id,
        support_id=support.support_id,
        status=status,
        reason_codes=tuple(seen) or ("NO_REASON",),
        policy_version=f"{POLICY_VERSION}:{cfg.name}",
        registry_version=registry.version,
        evidence_ids=tuple(claim.evidence_ids),
        modality=claim.modality,
        polarity=claim.polarity,
        scope_derivation=scope.derivation if scope is not None else None,
    )
