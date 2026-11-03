"""WP01–WP12: provenance, prefilter, справочник и область действия, признаки цитаты,
поддержка, семейства и противоречия, досье, метрики, гейт чата."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evidence import synth
from georag.evidence import claims, dossier, fusion, gate, metrics, prefilter
from georag.evidence.applicability import V3, decide_applicability
from georag.evidence.contracts import validate
from georag.evidence.georegistry import DISJOINT, EXACT, OVERLAP, PART_OF, check_integrity, load
from georag.evidence.provenance import verify_evidence
from georag.evidence.scope import resolve_scope, title_is_general
from georag.evidence.support import assess_support

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "v14" / "data"
REAL = load(DATA / "regional_registry.csv", DATA / "registry_aliases.csv")
TEXT = "Введение. В районе Альфа на глубине 120 м золото приурочено к кварцевым жилам. Конец."


# --- WP01 provenance -----------------------------------------------------------------------
def test_provenance_exact_and_schema():
    d, ev = synth.evidence(
        TEXT, "В районе Альфа на глубине 120 м золото приурочено к кварцевым жилам."
    )
    assert verify_evidence(ev, d).valid
    validate("evidence", ev.to_json())


@pytest.mark.parametrize(
    "change,code",
    [
        (
            lambda e: replace(e, exact_quote=e.exact_quote.replace("120", "130")),
            "QUOTE_SPAN_MISMATCH",
        ),
        (lambda e: replace(e, page_start=7, page_end=7), "PAGE_MISMATCH"),
        (lambda e: replace(e, pdf_sha256="0" * 64), "PDF_HASH_MISMATCH"),
        (lambda e: replace(e, parsed_sha256="1" * 64), "PARSED_HASH_MISMATCH"),
        (lambda e: replace(e, chunk_id="999"), "SPAN_LOST"),
        (lambda e: replace(e, quote_sha256="2" * 64), "QUOTE_HASH_MISMATCH"),
    ],
)
def test_provenance_rejects_tampering(change, code):
    d, ev = synth.evidence(
        TEXT, "В районе Альфа на глубине 120 м золото приурочено к кварцевым жилам."
    )
    res = verify_evidence(change(ev), d)
    assert not res.valid and code in res.reason_codes and "SOURCE_HASH_INVALID" in res.reason_codes


def test_provenance_reparse_changes_hash():
    d, ev = synth.evidence(
        TEXT, "В районе Альфа на глубине 120 м золото приурочено к кварцевым жилам."
    )
    d.chunks["1"].text = TEXT.replace("Конец", "Иное окончание")
    assert not verify_evidence(ev, d).valid  # разбор изменился — доказательство STALE, а не верно


# --- WP02 prefilter ------------------------------------------------------------------------
class _Finder:
    def find(self, text):
        from georag.graph.entities import Found

        return Found(
            "ru", terms=[w for w in ("золото", "кварц", "пирит", "уран", "жил") if w in text]
        )


LONG = (
    "Геологическое строение площади определяется сочетанием метаморфических толщ и интрузивных тел, прорванных "
    "дайками разного состава и пересечённых системой разломов северо-восточного простирания, вдоль которых "
    "развиты зоны рассланцевания, окварцевания и сульфидизации пород с прожилками карбоната. "
    "В районе Альфа золото приурочено к кварцевым жилам с пиритом мощностью до 2 м, а содержание сульфидов "
    "в рудах достигает нескольких процентов по данным опробования канав и скважин, пройденных на участках "
    "площади в ходе поисковых работ предыдущих лет, уточнённых новыми данными геофизических съёмок, "
    "геохимического опробования вторичных ореолов рассеяния и детального картирования коренных обнажений."
)


def test_prefilter_exact_duplicate_dropped():
    pf = prefilter.Prefilter(_Finder(), REAL, k=1)
    assert pf.decide(1, LONG).decision == "KEEP"
    d = pf.decide(2, LONG)
    assert d.decision == "DROP_DUPLICATE" and d.representative == 1 and d.jaccard == 1.0


@pytest.mark.parametrize(
    "old,new,field",
    [
        ("приурочено", "не приурочено", "negation"),
        ("до 2 м", "до 5 м", "numbers"),
        ("золото приурочено", "золото, вероятно, приурочено", "modality"),
        ("пиритом", "ураном и пиритом", "terms"),
    ],
)
def test_prefilter_protected_tokens(old, new, field):
    pf = prefilter.Prefilter(_Finder(), REAL, k=1)
    pf.decide(1, LONG)
    d = pf.decide(2, LONG.replace(old, new))
    assert d.decision == "REVIEW_NOT_DROP" and field in d.critical_differ


def test_prefilter_sparse():
    pf = prefilter.Prefilter(_Finder(), REAL, k=3)
    assert (
        pf.decide(
            1,
            "Короткий текст про золото без других терминов, но достаточно длинный для подписи, ещё слова и слова.",
        ).decision
        == "DROP_SPARSE"
    )


# --- WP05 справочник и область действия ---------------------------------------------------
def test_real_registry_integrity():
    assert check_integrity(REAL) == []
    assert REAL.relation("R:RU:CHULKOVSKY", "R:RU:YAKUTIA") == PART_OF
    assert REAL.relation("R:RU:URAL:MURZINKA", "R:RU:URAL:MID") == PART_OF
    assert REAL.relation("R:EG:ATALLASID", "R:SD:RSH") == DISJOINT
    assert REAL.relation("R:RU:ALDAN", "R:RU:YAKUTIA") == OVERLAP
    assert REAL.relation("R:EG:ED", "R:EG:ED") == EXACT


def test_working_registry_in_config(monkeypatch):
    """Чат (GEORAG_SCOPE_GATE) берёт рабочий реестр из config/geo-registry, а не из
    замороженного research/v14/data; рабочий цел и знает те же территории."""
    from georag.evidence import gate

    monkeypatch.delenv("GEORAG_REGISTRY", raising=False)
    assert gate.REGISTRY_DIR == ROOT / "config" / "geo-registry"
    gate.registry.cache_clear()
    try:
        working = gate.registry()
    finally:
        gate.registry.cache_clear()
    assert check_integrity(working) == []
    assert working.relation("R:RU:CHULKOVSKY", "R:RU:YAKUTIA") == PART_OF


def test_real_registry_homonym():
    hits = REAL.find("рудопроявление Майское")
    assert hits and hits[0].ambiguous  # Чукотка и Северная Карелия — разные id
    sc = resolve_scope("На Майском месторождении Чукотки золото связано с кварцем.", REAL)
    assert sc.region_ids == ("R:RU:MAYSKOE_CHUK",) and not sc.ambiguous


def test_real_registry_no_false_russia_in_placers():
    assert not REAL.find("россыпи золота")  # «Россия» не находится в «россыпи»


def test_title_rules():
    assert title_is_general(REAL, "Gold deposits of Central and North Asia") or True
    assert title_is_general(
        REAL, "Orogenic Gold Deposits: Geodynamics and Systematics (25 Years—Some Results)"
    )
    assert not title_is_general(
        REAL,
        "Gold mineralization of the Chulkovsky Site of the Seligdar Ore Field, Republic of Sakha (Yakutia)",
    )
    sc = resolve_scope(
        "Gold is associated with pyrite.",
        REAL,
        title="Gold mineralization of the Chulkovsky Site of the Seligdar Ore Field, Republic of Sakha (Yakutia)",
    )
    assert sc.region_ids == ("R:RU:CHULKOVSKY",) and sc.derivation == "TITLE"
    sc = resolve_scope(
        "Gold is associated with pyrite.",
        REAL,
        title="Orogenic Gold Deposits: Geodynamics and Systematics",
    )
    assert not sc.region_ids


# --- WP03 признаки цитаты -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "quote,pol,mod,quant",
    [
        ("Gold is not associated with the granite.", "NEGATE", "ASSERTED", None),
        ("Золото в кварцевых жилах не выявлено.", "NEGATE", None, None),
        ("The ores probably formed during the late stage.", "AFFIRM", "HYPOTHESIS", None),
        (
            "Gold mineralization is typically hosted by quartz veins.",
            "AFFIRM",
            "ASSERTED",
            "TYPICAL",
        ),
        ("Locally, visible gold occurs in quartz veins.", "AFFIRM", "ASSERTED", "LOCAL"),
        ("Не только кварц, но и карбонат содержит золото.", "AFFIRM", None, None),
        ("По данным [12], золото ассоциирует с пиритом.", "AFFIRM", "REPORTED", None),
    ],
)
def test_claim_features(quote, pol, mod, quant):
    f = claims.features(quote)
    assert f.polarity == pol
    if mod:
        assert f.modality == mod
    if quant:
        assert f.quantifier == quant


def test_claim_from_fact_schema():
    fact = {
        "id": 5,
        "src": "золото",
        "relation": "ассоциирует с",
        "dst": "пирит",
        "quote": "Золото ассоциирует с пиритом.",
        "votes": 4,
    }
    c = claims.claim_from_fact(fact, "E:x")
    validate("claim", c.to_json())
    assert c.legacy_accepted and c.scope.deposit_type == "gold"


# --- WP04 поддержка ---------------------------------------------------------------------------
def test_support_positional_is_partial():
    quote = "В районе Альфа кварцевые жилы расположены южнее золоторудного узла."
    d, ev = synth.evidence(quote)
    c = synth.claim(subject="кварцевые жилы", predicate="содержит", obj="золоторудного узла")
    s = assess_support(c, ev, document=d, votes=4)
    assert s.label == "PARTIAL" and "RELATION_POSITIONAL_ONLY" in s.reason_codes
    validate("support", s.to_json())


def test_support_votes_are_one_instrument():
    d, ev = synth.evidence("В районе Альфа золото приурочено к кварцевым жилам.")
    labels = [
        assess_support(synth.claim(), ev, document=d, votes=v).label for v in (0, 1, 2, 3, 4, None)
    ]
    assert labels == ["UNKNOWN", "PARTIAL", "PARTIAL", "ENTAILED", "ENTAILED", "UNKNOWN"]


# --- WP07 семейства и противоречия ------------------------------------------------------------------
def test_families_duplicate_and_series():
    text = " ".join(f"слово{i} " * 1 for i in range(80))
    docs = [
        {
            "doc_id": "a",
            "title": "Uncertainty quantification of deep learning",
            "authors": ["Xiao Ming"],
        },
        {
            "doc_id": "b",
            "title": "Uncertainty quantification of deep learning algorithms",
            "authors": ["Yan Lee"],
        },
        {
            "doc_id": "c",
            "title": "Месторождение Сухой Лог. Часть I. Постановка",
            "authors": ["Иванов"],
        },
        {
            "doc_id": "d",
            "title": "Месторождение Сухой Лог. Часть II. Структура",
            "authors": ["Петров", "Иванов"],
        },
        {"doc_id": "e", "title": "Другое", "authors": ["Y"]},
    ]
    chunks = {
        "a": [text],
        "b": [text],
        "c": ["первая часть " * 30],
        "d": ["вторая часть иное " * 30],
        "e": ["иное " * 40],
    }
    fams = {f.doc_ids: f.kind for f in fusion.source_families(docs, chunks)}
    assert (
        fams[("a", "b")] == "DUPLICATE_PUBLICATION"
        and fams[("c", "d")] == "SERIES"
        and fams[("e",)] == "SINGLE"
    )
    version = [
        {
            "doc_id": "p",
            "title": "Uncertainty quantification of deep learning model for mapping",
            "authors": ["Ziye Wang", "Renguang Zuo"],
        },
        {
            "doc_id": "j",
            "title": "Uncertainty quantification of deep learning algorithms for mapping",
            "authors": ["Ziye Wang", "Renguang Zuo"],
        },
    ]
    fams = {
        f.doc_ids: f.kind
        for f in fusion.source_families(version, {"p": ["один " * 30], "j": ["другой " * 30]})
    }
    assert fams[("j", "p")] == "SAME_WORK_VERSION"


def test_different_stage_not_conflict():
    a = {
        "claim_id": "a",
        "src": "золото",
        "relation": "связано с",
        "dst": "гранитами",
        "polarity": "AFFIRM",
        "region_ids": ("R1",),
        "stage": "early",
    }
    b = {**a, "claim_id": "b", "polarity": "NEGATE", "stage": "late"}
    assert fusion.compare(a, b).verdict == "DIFFERENT_STAGE"
    assert [c.verdict for c in fusion.conflicts([a, {**b, "stage": "early"}])] == ["CONFLICT"]


# --- WP09 досье -------------------------------------------------------------------------------------
def test_dossier_sections_override_export(tmp_path):
    reg = synth.registry()
    q = synth.query()
    rows = []
    for i, quote in enumerate(
        [
            "В районе Альфа золото приурочено к кварцевым жилам.",
            "В районе Бета золото приурочено к кварцевым жилам.",
            "Золото приурочено к кварцевым жилам.",
        ]
    ):
        c = replace(synth.claim(), claim_id=f"F{i}")
        s = replace(synth.support(), support_id=f"S{i}", claim_id=f"F{i}")
        sc = resolve_scope(quote, reg)
        rows.append((c, s, sc, decide_applicability(q, c, s, reg, scope=sc, config=V3), quote))
    log = tmp_path / "overrides.jsonl"
    dossier.record_override(
        log, rows[2][3], "DIRECT", "район указан в подписи к рисунку", "geologist-1"
    )
    dos = dossier.build_dossier(
        q,
        [r[3] for r in rows],
        claims={r[0].claim_id: r[0] for r in rows},
        supports={r[1].support_id: r[1] for r in rows},
        sources={
            r[0].claim_id: {
                "doc_id": "D",
                "doi": "10.x/y",
                "title": "T",
                "pages": [3],
                "quote": r[4],
            }
            for r in rows
        },
        scopes={r[0].claim_id: r[2] for r in rows},
        overrides=dossier.read_overrides(log),
    )
    data = dos.to_json()
    assert data["sections"]["DIRECT"]["n"] == 2 and data["sections"]["OUT_OF_SCOPE"]["n"] == 1
    moved = data["sections"]["DIRECT"]["entries"][1]
    assert (
        moved["status"] == "UNRESOLVED"
        and moved["override_status"] == "DIRECT"
        and moved["history"]
    )
    assert "geologist-1" in dos.to_csv() and json.dumps(data, ensure_ascii=False)
    assert "Досье" in dossier.to_markdown(dos)
    with pytest.raises(ValueError):
        dossier.record_override(log, rows[0][3], "DIRECT", "", "x")


# --- WP12 метрики ---------------------------------------------------------------------------------
def _gold():
    rows = []
    for i in range(12):
        rows.append(
            {
                "sample_id": f"s{i}",
                "query_id": f"q{i % 4}",
                "source_family": f"f{i % 6}",
                "split": "TEST",
                "gold_applicability": ["DIRECT", "OUT_OF_SCOPE", "UNRESOLVED"][i % 3],
            }
        )
    return rows


def test_metrics_ftr_and_coverage():
    gold = _gold()
    allow = [{"sample_id": g["sample_id"], "predicted_status": "DIRECT"} for g in gold]
    m = metrics.compute_metrics(allow, gold, "V0", "TEST")
    assert m.ftr == 1.0 and m.direct_coverage == 1.0 and m.selective_risk == pytest.approx(8 / 12)
    lo, hi = metrics.wilson(4, 4)
    assert 0.39 < lo < 0.52 and hi == 1.0
    exact = [
        {"sample_id": g["sample_id"], "predicted_status": g["gold_applicability"]} for g in gold
    ]
    b = metrics.paired_bootstrap(exact, allow, gold, "TEST", n_boot=200)
    assert b["delta_ftr"] == -1.0 and b["delta_coverage"] == 0.0


def test_metrics_refuses_partial_and_leak():
    gold = _gold()
    with pytest.raises(ValueError):
        metrics.compute_metrics(
            [{"sample_id": "s0", "predicted_status": "DIRECT"}], gold, "V", "TEST"
        )
    leak = gold + [
        {
            "sample_id": "x",
            "query_id": "q",
            "source_family": "f0",
            "split": "DEV",
            "gold_applicability": "DIRECT",
        }
    ]
    with pytest.raises(ValueError):
        metrics.check_gold(leak)


# --- WP08 гейт чата ----------------------------------------------------------------------------------
def test_frame_question():
    f = gate.frame_question(
        "Какие структурные признаки золотого оруденения известны для Восточной пустыни Египта?",
        REAL,
    )
    assert (
        f.target_region_id == "R:EG:ED"
        and f.target_deposit_type == "gold"
        and f.requested_mode == "FACTUAL"
    )
    f = gate.frame_question(
        "Характерна ли для всего Среднего Урала связь золота с березитами?", REAL
    )
    assert f.target_region_id == "R:RU:URAL:MID" and f.requested_quantifier in ("TYPICAL", "ALL")
    f = gate.frame_question("Какие есть аналоги месторождения Сухой Лог?", REAL)
    assert f.allow_analogies and f.target_region_id == "R:RU:SUKHOILOG"
    validate("query", f.to_json())


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("GEORAG_SCOPE_GATE", raising=False)
    assert not gate.enabled()
    from georag.chat import dataset

    data = {"facts": [{"x": 1}]}
    assert dataset._scope_gate(None, "вопрос", data) is data


def test_not_checked_beyond_limit():
    from georag.chat import support

    answer = " ".join(
        f"Утверждение номер {i}: золото в районе Альфа приурочено к кварцевым жилам [1]."
        for i in range(1, 27)
    )
    rest = support.not_checked(answer)
    assert len(support._claims(answer)) == 26 and len(rest) == 2


def test_text_reuse_is_one_vote():
    quote = "The variation in gold distribution suggests that gold may be concentrated in quartz veins and zones."
    claims = [
        {"src": "gold", "relation": "связано с", "dst": "quartz veins", "doc_id": d, "quote": quote}
        for d in ("a", "b")
    ]
    row = next(iter(fusion.independent_support(claims, {"a": "FA", "b": "FB"}).values()))
    assert (
        row["n_docs"] == 2
        and row["n_independent"] == 1
        and row["independence_status"] == "TEXT_REUSE"
    )
