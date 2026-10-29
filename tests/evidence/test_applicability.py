"""Инварианты I1–I10 политики применимости (ТЗ, раздел 9) — на синтетике."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from evidence import synth

from georag.evidence import applicability as ap
from georag.evidence.contracts import validate
from georag.evidence.scope import resolve_scope

REG = synth.registry()


def decide(quote, *, q=None, c=None, s=None, cfg=ap.V3, **scope_kw):
    sc = resolve_scope(quote, REG, **scope_kw)
    return ap.decide_applicability(q or synth.query(), c or synth.claim(), s or synth.support(), REG, scope=sc, config=cfg,
                                   name_texts=[quote, scope_kw.get("title", "")])


def test_direct_same_region():
    d = decide("В районе Альфа золото приурочено к кварцевым жилам.")
    assert d.status == "DIRECT" and "REGION_EXACT" in d.reason_codes
    validate("decision", d.to_json())


def test_i1_invalid_evidence_never_direct():
    d = decide("В районе Альфа золото приурочено к кварцевым жилам.", s=synth.support("INVALID_EVIDENCE", ("SOURCE_HASH_INVALID",)))
    assert d.status != "DIRECT" and "SOURCE_HASH_INVALID" in d.reason_codes


@pytest.mark.parametrize("label", ["PARTIAL", "CONTRADICTED", "UNKNOWN"])
def test_i2_not_entailed_never_direct(label):
    d = decide("В районе Альфа золото приурочено к кварцевым жилам.", s=synth.support(label, ("SUPPORT_NOT_ENTAILED",)))
    assert d.status != "DIRECT" and "SUPPORT_NOT_ENTAILED" in d.reason_codes


def test_i3_foreign_region_never_direct():
    d = decide("В районе Бета золото приурочено к кварцевым жилам.")
    assert d.status == "OUT_OF_SCOPE" and "REGION_DISJOINT" in d.reason_codes


def test_i4_part_not_direct_for_universal():
    for quant in ("TYPICAL", "ALL"):
        d = decide("На участке Альфа-1 золото приурочено к кварцевым жилам.", q=synth.query(quantifier=quant))
        assert d.status == "UNRESOLVED" and "SCOPE_UNIVERSALITY_MISMATCH" in d.reason_codes
    assert decide("На участке Альфа-1 золото приурочено к кварцевым жилам.").status == "DIRECT"


def test_i5_hypothesis_not_fact():
    c = synth.claim(modality="HYPOTHESIS")
    d = decide("В районе Альфа золото, вероятно, приурочено к кварцевым жилам.", c=c)
    assert d.status == "UNRESOLVED" and "MODALITY_HYPOTHESIS" in d.reason_codes
    d = decide("В районе Альфа золото, вероятно, приурочено к кварцевым жилам.", c=c, q=synth.query(mode="RESEARCH_REVIEW"))
    assert d.status == "DIRECT" and d.modality == "HYPOTHESIS" and "MODALITY_HYPOTHESIS" in d.reason_codes


def test_i6_negation_not_positive():
    c = synth.claim(polarity="NEGATE")
    assert decide("В районе Альфа золото не выявлено в кварцевых жилах.", c=c).status == "UNRESOLVED"
    d = decide("В районе Альфа золото не выявлено в кварцевых жилах.", c=c, q=synth.query(polarity="NEGATE"))
    assert d.status == "DIRECT" and d.polarity == "NEGATE" and "POLARITY_MATCH" in d.reason_codes


def test_i7_unknown_region_not_direct():
    d = decide("Золото приурочено к кварцевым жилам.", title="Methods of gold exploration")
    assert d.status == "UNRESOLVED" and "REGION_UNKNOWN" in d.reason_codes


def test_i8_scope_not_inherited_from_query():
    quote = "Золото приурочено к кварцевым жилам."
    for region in ("S:A", "S:B", "S:A1"):
        assert decide(quote, q=synth.query(region=region)).status == "UNRESOLVED"


def test_i9_reproducible_record():
    a = decide("В районе Альфа золото приурочено к кварцевым жилам.")
    b = decide("В районе Альфа золото приурочено к кварцевым жилам.")
    assert a == b
    assert a.evidence_ids and a.registry_version == REG.version and a.policy_version.startswith(ap.POLICY_VERSION)


def test_i10_removing_evidence_never_upgrades():
    """Снятие проверенного доказательства (поддержка → UNKNOWN / INVALID) не повышает статус."""
    rank = {"DIRECT": 3, "ANALOG": 2, "OUT_OF_SCOPE": 1, "UNRESOLVED": 0}
    rng = random.Random(1)
    quotes = ["В районе Альфа золото приурочено к кварцевым жилам.", "В районе Бета золото приурочено к кварцевым жилам.",
              "Золото приурочено к кварцевым жилам.", "На участке Альфа-1 золото приурочено к кварцевым жилам."]
    for _ in range(300):
        q = synth.query(region=rng.choice(["S:A", "S:B", "S:A1"]), quantifier=rng.choice(["EXISTS", "TYPICAL", "ALL", "UNSPECIFIED"]),
                        analogies=rng.random() < 0.5)
        quote = rng.choice(quotes)
        full = decide(quote, q=q)
        for weaker in (synth.support("UNKNOWN", ("SUPPORT_NOT_ENTAILED",)), synth.support("INVALID_EVIDENCE", ("SOURCE_HASH_INVALID",))):
            assert rank[decide(quote, q=q, s=weaker).status] <= rank[full.status] or full.status != "DIRECT"
            assert decide(quote, q=q, s=weaker).status != "DIRECT"


def test_order_independence_property():
    """Решение не зависит от порядка решений по другим фактам (функция без состояния)."""
    rng = random.Random(7)
    items = [(f"В районе {n} золото приурочено к кварцевым жилам.", synth.query(region=r))
             for n in ("Альфа", "Бета", "Гамма") for r in ("S:A", "S:B")]
    first = {i: decide(qt, q=q).status for i, (qt, q) in enumerate(items)}
    for _ in range(5):
        order = list(range(len(items)))
        rng.shuffle(order)
        assert {i: decide(items[i][0], q=items[i][1]).status for i in order} == first


def test_analog_requires_grounding():
    q = synth.query(analogies=True, mode="COMPARATIVE")
    d = decide("В районе Бета золото приурочено к кварцевым жилам.", q=q)
    assert d.status == "ANALOG" and "GEOLOGICAL_ANALOGY_GROUNDED" in d.reason_codes
    d = decide("В районе Гамма золото приурочено к кварцевым жилам.", q=q)  # нет общей обстановки
    assert d.status == "OUT_OF_SCOPE" and "ANALOGY_NOT_GROUNDED" in d.reason_codes
    d = decide("В районе Бета золото приурочено к кварцевым жилам.", q=q, c=synth.claim(deposit=None))
    assert d.status == "OUT_OF_SCOPE"  # тип оруденения у факта не подтверждён


def test_variants_differ_as_designed():
    quote = "В районе Бета золото приурочено к кварцевым жилам."
    assert decide(quote, cfg=ap.V0).status == "DIRECT"  # старый граф: регион не смотрит
    assert decide(quote, cfg=ap.V1).status == "UNRESOLVED"  # имени «Альфа» в тексте нет
    assert decide("Золото района Бета, как и Альфа, приурочено к жилам.", cfg=ap.V1).status == "DIRECT"
    assert decide(quote, cfg=ap.V2).status == "OUT_OF_SCOPE"


def test_v0_respects_legacy_buffer():
    d = decide("В районе Альфа золото приурочено к кварцевым жилам.", cfg=ap.V0, c=synth.claim(votes=1, accepted=False))
    assert d.status == "UNRESOLVED" and "LEGACY_CANDIDATE" in d.reason_codes


def test_decision_schema_v13_compatible():
    d = decide("В районе Альфа золото приурочено к кварцевым жилам.")
    plain = replace(d, modality=None, polarity=None, scope_derivation=None).to_json()
    assert plain["schema_version"] == "v13.1"
    validate("decision", plain)
    validate("decision", d.to_json())
