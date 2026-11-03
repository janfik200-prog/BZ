"""WP03. Факт графа → типизированный GeoClaim: полярность, модальность, квантор, тип
оруденения — по цитате, правилами кода, без модели.

Связь выписывает модель (facts.py, без изменений); здесь — только то, что видно в
самой цитате. Чего в цитате нет, остаётся UNKNOWN/UNSPECIFIED/None и не
достраивается «по геологическому стереотипу» (ТЗ, раздел 3).

build_claims(фрагмент, модель) — та же выписка facts.py, но с журналом запуска
(model_adapter): модель, хэш задания, токены, время.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .contracts import GeoClaim, Scope, sha256_text

RULES_VERSION = "claim-features-v1"

# --- полярность ----------------------------------------------------------------------
# Отрицание, относящееся к утверждению: «не выявлено», «отсутствует», «not found».
# «Не только», «not only» — не отрицание.
_NOT_NEGATION = re.compile(r"\bне\s+только\b|\bnot\s+only\b|\bno\.\s*\d|\bнет\s+сомнени", re.I)
_NEGATION = re.compile(
    r"\bне\s+(?:был[аио]?\s+)?(?:выявл|обнаруж|установл|отмеч|наблюд|встреч|известн|связ|"
    r"характерн|проявл|зафиксир|содерж|прослеж|подтвержд|отмеча|выделя|участву|контролир)\w*"
    r"|\bотсутств\w*|\bне\s+имеет\b|\bне\s+имеют\b|\bнет\s+(?:данных|признаков|связи)\b"
    r"|\bни\s+одн\w+"
    r"|\b(?:was|were|is|are|has|have|had|been)\s+not\b|\bnot\s+(?:been\s+)?(?:found|observed|"
    r"detected|identified|associated|present|recorded|reported|related|known|seen|"
    r"controlled|hosted|linked|correlated)\b|\b(?:do|does|did)\s+not\b|\bno\s+(?:evidence|"
    r"gold|mineralization|significant|relationship|correlation|association|direct)\b"
    r"|\babsen(?:t|ce)\s+of\b|\black(?:s|ing)?\s+(?:of\s+)?(?:any\s+)?\w+|\bnone\s+of\b"
    r"|\bwithout\s+(?:any\s+)?(?:gold|mineraliz|sulphid|sulfid|alteration)\w*",
    re.I,
)

# --- модальность ----------------------------------------------------------------------
_HYPOTHESIS = re.compile(
    r"вероятн|по-видимому|предположительн|предполага|возможн|не\s+исключ|можно\s+предпол|"
    r"гипотез|по\s+мнению|видимо|можно\s+ожидать|perhaps|probabl|likely|possibl|presumabl|"
    r"\b(?:is|are|was|were)\s+(?:thought|believed|inferred|presumed|interpreted|considered)\b|"
    r"suggest|hypothes|propos(?:e|ed|es)\s+that|speculat|"
    r"\b(?:may|might|could)\s+(?:have|be|represent|indicate|reflect|serve|host|contain|"
    r"suggest|imply|point|record|relate|explain|correspond)\b|\bmight\b",
    re.I,
)
_REPORTED = re.compile(
    r"according\s+to|reported\s+by|\bafter\s+[A-Z][a-z]+\s*(?:et\s+al\.?)?\s*\(?\d{4}|"
    r"по\s+данным|по\s+сведениям|согласно|как\s+(?:показал|отмечал|установил)\w*|"
    r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\(\s*[A-ZА-ЯЁ][\w\-]+(?:\s+(?:et\s+al\.?|и\s+др\.?))?,?\s+\d{4}",
)

# --- квантор -------------------------------------------------------------------------
_ALL = re.compile(
    r"\b(?:all|every|each|throughout|entire|whole|everywhere|ubiquitous\w*)\b|"
    r"(?<![а-яё])(?:все|всех|всюду|повсеместн\w*|кажд\w+|всей|всего|всём|целиком)(?![а-яё])",
    re.I,
)
_TYPICAL = re.compile(
    r"typical\w*|characteri[sz]\w*|\bcommon(?:ly)?\b|\bgeneral(?:ly)?\b|\busual(?:ly)?\b|"
    r"predominant\w*|\bmostly\b|\bmainly\b|widespread|prevalent|"
    r"характерн\w*|типичн\w*|как\s+правило|обычно|преимущественн\w*|широко\s+распростран\w*|"
    r"главным\s+образом|в\s+основном|в\s+целом|наиболее\s+распростран\w*",
    re.I,
)
_LOCAL = re.compile(
    r"\blocal(?:ly)?\b|\bin\s+places\b|sporadic\w*|\bisolated\b|\brare(?:ly)?\b|\bsingle\b|"
    r"\bone\s+(?:sample|drill|hole|outcrop)|\bsome\s+(?:samples|veins|parts|areas)\b|"
    r"локальн\w*|местами|единичн\w*|\bредк\w*|в\s+(?:одной|отдельн\w+)\s+\w+|"
    r"на\s+(?:отдельн\w+\s+)?участк\w+|в\s+скважин\w+|в\s+пробе",
    re.I,
)
_EXISTS = re.compile(
    r"\b(?:occur\w*|found|identified|detected|observed|present|hosted|recorded|discovered|"
    r"exist\w*|reveal\w*|establish\w*|known|contain\w*|includ\w*)\b|"
    r"встреча\w*|выявл\w*|обнаруж\w*|установл\w*|отмеч\w*|присутств\w*|известн\w*|"
    r"наблюда\w*|проявл\w*|содерж\w*|локализ\w*",
    re.I,
)

# --- тип оруденения -------------------------------------------------------------------
COMMODITIES: dict[str, re.Pattern[str]] = {
    "gold": re.compile(r"\bgold\b|auriferous|\bau\b|золот\w*|\bau-|-au\b", re.I),
    "silver": re.compile(r"\bsilver\b|\bag\b|серебр\w*", re.I),
    "copper": re.compile(r"\bcopper\b|\bcu\b|медн\w*|\bмед[ьи]\b|cu-", re.I),
    "uranium": re.compile(r"uranium|\bu\b(?=[\s-]*(?:mineral|ore|deposit))|уран\w*", re.I),
    "iron": re.compile(
        r"iron[\s-]*ore|\bbif\b|banded\s+iron|железоруд\w*|железист\w+\s+кварцит", re.I
    ),
    "ree": re.compile(r"rare[\s-]*earth|\bree\b|рзэ|редкоземел\w*", re.I),
    "rare_metal": re.compile(
        r"rare[\s-]*metal|редкометал\w*|tantal\w*|niob\w*|lithium|литиев\w*", re.I
    ),
    "tin": re.compile(r"\btin\b|cassiterite|олов\w*|касситерит", re.I),
    "tungsten": re.compile(r"tungsten|wolfram\w*|scheelite|вольфрам\w*|шеелит", re.I),
    "base_metals": re.compile(
        r"\blead\b|\bzinc\b|\bpb\b|\bzn\b|polymetal\w*|полиметалл\w*|свинц\w*|цинк\w*|колчедан\w*|\bvms\b",
        re.I,
    ),
    "fluorite": re.compile(r"fluorite|флюорит\w*", re.I),
    "molybdenum": re.compile(
        r"molybden\w*|\bmo\b(?=[\s-]*(?:mineral|ore|deposit|porphyry))|молибден\w*", re.I
    ),
    "diamond": re.compile(r"diamond|кимберлит\w*|алмаз\w*", re.I),
    "lithium_brine": re.compile(r"\bbrine|рассол\w*", re.I),
}
_STAGE = re.compile(
    r"\b((?:early|late|main|first|second|third|pre-|post-|syn-)[\w-]*)\s+(?:stage|phase|event)\b|"
    r"\b(ранн\w+|поздн\w+|основн\w+|перв\w+|втор\w+|трет\w+)\s+(?:стади\w+|этап\w*|фаз\w+)",
    re.I,
)


@dataclass(frozen=True)
class Features:
    polarity: str
    modality: str
    quantifier: str
    deposit_types: tuple[str, ...]
    stage: str | None
    cues: tuple[str, ...]


def features(quote: str, relation: str = "") -> Features:
    """Полярность, модальность и квантор — по цитате и связи. Каждое решение — с меткой
    правила (cues): по ней видно, почему факт помечен."""
    text = " ".join((quote or "").split())
    cues: list[str] = []
    negated = bool(_NEGATION.search(text)) and not _NOT_NEGATION.search(text)
    if not negated and re.search(r"\bне\s|\bnot\b|\bno\b", relation or "", re.I):
        negated = True
    polarity = "NEGATE" if negated else "AFFIRM"
    if negated:
        cues.append("NEG_CUE")
    if _HYPOTHESIS.search(text) or (relation or "").startswith("предположительно"):
        modality = "HYPOTHESIS"
        cues.append("HEDGE_CUE")
    elif _REPORTED.search(text):
        modality = "REPORTED"
        cues.append("REPORTED_CUE")
    else:
        modality = "ASSERTED"
    if _ALL.search(text):
        quantifier = "ALL"
    elif _TYPICAL.search(text):
        quantifier = "TYPICAL"
    elif _LOCAL.search(text):
        quantifier = "LOCAL"
    elif _EXISTS.search(text):
        quantifier = "EXISTS"
    else:
        quantifier = "UNSPECIFIED"
    cues.append("Q_" + quantifier)
    deposits = tuple(name for name, rx in COMMODITIES.items() if rx.search(text))
    m = _STAGE.search(text)
    stage = (m.group(1) or m.group(2)).lower() if m else None
    return Features(polarity, modality, quantifier, deposits, stage, tuple(cues))


def claim_from_fact(
    fact: dict[str, Any],
    evidence_id: str,
    scope: Scope | None = None,
    min_votes: int = 3,
) -> GeoClaim:
    """Факт из facts (store.all_facts или строка выборки) → GeoClaim.

    Голоса confirm.py и признак «в графе» (accepted: ≥ min_votes или повтор в другой
    статье) сохраняются как есть — для baseline V0/V1; к поддержке (support.py) они
    идут как один коррелированный инструмент, а не четыре голоса."""
    feats = features(fact.get("quote", ""), fact.get("relation", ""))
    votes = fact.get("votes")
    accepted = fact.get("accepted")
    if accepted is None and votes is not None:
        accepted = int(votes) >= min_votes
    base_scope = scope or Scope()
    if base_scope.deposit_type is None and feats.deposit_types:
        base_scope = Scope(**{**base_scope.__dict__, "deposit_type": ";".join(feats.deposit_types)})
    if base_scope.mineralization_stage is None and feats.stage:
        base_scope = Scope(**{**base_scope.__dict__, "mineralization_stage": feats.stage})
    return GeoClaim(
        claim_id=f"F:{fact['id']}",
        subject=fact["src"],
        predicate=fact["relation"],
        object=fact["dst"],
        evidence_ids=(evidence_id,),
        subject_type=fact.get("src_kind") or "UNKNOWN",
        object_type=fact.get("dst_kind") or "UNKNOWN",
        polarity=feats.polarity,
        modality=feats.modality,
        quantifier=feats.quantifier,
        scope=base_scope,
        source_family_id=fact.get("source_family"),
        model_id=fact.get("model") or "unknown",
        prompt_sha256=fact.get("prompt_sha256") or sha256_text("legacy-prompt-unrecorded"),
        extraction_status="FORMALLY_VALIDATED",
        fact_id=int(fact["id"]),
        legacy_votes=None if votes is None else int(votes),
        legacy_accepted=None if accepted is None else bool(accepted),
    )


def build_claims(fragment: dict[str, Any], model_config: dict[str, Any]) -> list[GeoClaim]:
    """Выписка фрагмента моделью (facts.messages / facts.check, как в графе) → GeoClaim.

    fragment: chunk_id, doc_id, text, (entities). model_config: adapter (model_adapter.
    ModelAdapter), rules (текст правил). Невалидный JSON — пустой список, а ошибка — в
    журнале адаптера: она не подменяется «правильным» пустым ответом без следа."""
    from ..graph import facts

    adapter = model_config["adapter"]
    rules = model_config["rules"]
    system, user = facts.messages(rules, fragment["text"], fragment.get("entities"))
    result = adapter.chat_json(
        system, user, max_tokens=facts.ANSWER_TOKENS, tag=str(fragment["chunk_id"])
    )
    checked = facts.check(result.data, fragment["text"]) if result.ok else facts.Checked()
    out = []
    for i, f in enumerate(checked.facts):
        feats = features(f.quote, f.relation)
        out.append(
            GeoClaim(
                claim_id=f"X:{fragment['chunk_id']}:{i}",
                subject=f.src,
                predicate=f.relation,
                object=f.dst,
                evidence_ids=(f"E:{fragment['doc_id']}:{fragment['chunk_id']}:q{i}",),
                polarity=feats.polarity,
                modality=feats.modality,
                quantifier=feats.quantifier,
                model_id=adapter.model,
                prompt_sha256=sha256_text(system + "\n" + user),
                extraction_status="FORMALLY_VALIDATED",
            )
        )
    return out
