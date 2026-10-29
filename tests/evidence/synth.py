"""Синтетика для проверок политики: вымышленные районы A, B, … — НЕ геологические данные.

Синтетические случаи проверяют свойства кода и никогда не идут в оценку метода
(metrics.check_gold отвергает split=SYNTHETIC)."""

from __future__ import annotations

from georag.evidence.contracts import GeoClaim, QueryFrame, Scope, SupportAssessment, sha256_text
from georag.evidence.georegistry import GeoArea, GeoAreaRegistry
from georag.evidence.provenance import ChunkSnapshot, DocumentSnapshot, make_evidence


def registry() -> GeoAreaRegistry:
    reg = GeoAreaRegistry(version="synthetic-registry")
    reg.add(GeoArea("S:ROOT", "Синтетический щит", "shield"), ["Synthetic Shield"])
    reg.add(GeoArea("S:A", "район Альфа", "district", ("S:ROOT",), deposit_types=("gold",), settings=("synth_shear_au",)),
            ["Alpha district", "Альфа"])
    reg.add(GeoArea("S:A:N", "Северная Альфа", "subdistrict", ("S:A",)), ["North Alpha"])
    reg.add(GeoArea("S:A:S", "Южная Альфа", "subdistrict", ("S:A",)), ["South Alpha"])
    reg.add(GeoArea("S:A1", "участок Альфа-1", "site", ("S:A",)), ["Alpha-1 site", "Альфа-1"])
    reg.add(GeoArea("S:A2", "участок Альфа-2", "site", ("S:A",)), ["Alpha-2 site", "Альфа-2"])
    reg.add(GeoArea("S:B", "район Бета", "district", ("S:ROOT",), deposit_types=("gold",), settings=("synth_shear_au",)),
            ["Beta district", "Бета"])
    reg.add(GeoArea("S:C", "район Гамма", "district", ("S:ROOT",), deposit_types=("gold",)), ["Gamma district", "Гамма"])
    # Одноимённые: «Дельта» — два разных места.
    reg.add(GeoArea("S:D1", "Дельта (север)", "deposit", ("S:A",)), ["Дельта", "Delta"])
    reg.add(GeoArea("S:D2", "Дельта (юг)", "deposit", ("S:B",)), ["Дельта", "Delta"])
    return reg


def doc(text: str, title: str = "Synthetic paper", pages: tuple[int, ...] = (3,), headings: tuple[str, ...] = ()) -> DocumentSnapshot:
    return DocumentSnapshot(
        doc_id="SYNTH-D1",
        pdf_sha256=sha256_text("synthetic pdf"),
        parser_version="SYNTHETIC",
        title=title,
        chunks={"1": ChunkSnapshot("1", text, pages, headings, 0)},
    )


def claim(subject="золото", predicate="приурочено к", obj="кварцевым жилам", *, polarity="AFFIRM",
          modality="ASSERTED", quantifier="EXISTS", deposit="gold", stage=None, votes=4, accepted=True,
          evidence_id="E:1") -> GeoClaim:
    return GeoClaim(
        claim_id="SYNTH-F1", subject=subject, predicate=predicate, object=obj, evidence_ids=(evidence_id,),
        polarity=polarity, modality=modality, quantifier=quantifier,
        scope=Scope(deposit_type=deposit, mineralization_stage=stage), model_id="synthetic",
        prompt_sha256=sha256_text("p"), extraction_status="FORMALLY_VALIDATED", fact_id=1,
        legacy_votes=votes, legacy_accepted=accepted,
    )


def support(label="ENTAILED", reasons=("SUPPORT_VOTES_GE3",)) -> SupportAssessment:
    return SupportAssessment("SYNTH-S1", "SYNTH-F1", "E:1", label, tuple(reasons), evaluation_split="SYNTHETIC")


def query(region="S:A", quantifier="EXISTS", mode="FACTUAL", polarity="AFFIRM", analogies=False,
          deposit="gold", stage=None) -> QueryFrame:
    reg = registry()
    return QueryFrame(
        query_id="SYNTH-Q1", raw_text="synthetic question", target_region_id=region,
        target_region_name=reg.areas[region].name if region else None, target_deposit_type=deposit,
        requested_polarity=polarity, requested_quantifier=quantifier, allow_analogies=analogies,
        requested_mode=mode, target_stage=stage,
    )


def evidence(text: str, quote: str | None = None, **kw):
    d = doc(text, **kw)
    return d, make_evidence(d, "1", quote or text)
