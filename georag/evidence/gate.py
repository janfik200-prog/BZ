"""WP08. Контекстный гейт в графе и чате: вопрос → QueryFrame, факт → решение.

Включается переменной GEORAG_SCOPE_GATE=1; без неё граф и чат работают как раньше
(режим без проверки сохраняется для парных испытаний P1/P2). Гейт ничего не пишет в
базу: решения считаются при ответе, досье — отдельно (dossier.py).

Где применяется:
* датасет предмета для ответа (chat/dataset.collect_dataset) — в сводку и фрагменты
  идут только факты DIRECT; ANALOG — отдельным списком «по другим районам», остальное —
  в data["scope_gate"]["excluded"] с причинами;
* фрагменты графа в кандидаты (chat/dataset.graph_candidates) — только из фактов DIRECT.
Если территория в вопросе не названа, гейт ничего не отсеивает (NO_TARGET_REGION):
применимость к району решается, только когда район спрошен.
"""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any

from ..paths import ROOT
from ..text import normalize
from .applicability import V3, PolicyConfig, decide_applicability
from .claims import COMMODITIES, claim_from_fact
from .contracts import QueryFrame
from .georegistry import GeoAreaRegistry, load
from .provenance import DocumentSnapshot, make_evidence, snapshot_from_db
from .scope import resolve_scope
from .support import assess_support

REGISTRY_DIR = ROOT / "research" / "v14" / "data"

_TYPICAL_Q = re.compile(r"характерн|типичн|в\s+целом|обычно|как\s+правило|typical|characteristic|in\s+general", re.I)
_ALL_Q = re.compile(r"везде|повсеместн|во\s+всех|всей\s+территори|каждом|throughout|\bevery\b|\ball\s+(?:parts|deposits|areas)", re.I)
_EXISTS_Q = re.compile(r"встречается\s+ли|есть\s+ли|известн\w+\s+ли|где-либо|хотя\s+бы|выявлен\w*\s+ли|\bany\b|\boccur", re.I)
_NEG_Q = re.compile(r"отсутств|не\s+выявлен|не\s+обнаружен|absence|not\s+found|\bnegative\b", re.I)
_HYP_Q = re.compile(r"гипотез|предполож|модел\w+\s+образован|hypothes|proposed\s+model", re.I)
_CMP_Q = re.compile(r"аналог|сравн|похож|analog|compar|similar", re.I)


def enabled() -> bool:
    return os.environ.get("GEORAG_SCOPE_GATE", "").strip() in ("1", "true", "yes")


@cache
def registry(path: str | None = None) -> GeoAreaRegistry:
    folder = Path(path or os.environ.get("GEORAG_REGISTRY", "") or REGISTRY_DIR)
    return load(folder / "regional_registry.csv", folder / "registry_aliases.csv")


def frame_question(question: str, reg: GeoAreaRegistry, query_id: str = "Q:chat") -> QueryFrame:
    """Смысл вопроса: территория (самая узкая из названных), тип оруденения, квантор,
    полярность, режим. Неоднозначное имя территории — без территории: гейт не гадает."""
    hits = reg.find(question)
    ids = {i for h in hits if not h.ambiguous for i in h.area_ids}
    ids = {a for a in ids if not any(a in reg.ancestors(b) for b in ids if b != a)}
    target = sorted(ids)[0] if len(ids) == 1 else None
    deposits = [name for name, rx in COMMODITIES.items() if rx.search(question)]
    quantifier = "ALL" if _ALL_Q.search(question) else "TYPICAL" if _TYPICAL_Q.search(question) else (
        "EXISTS" if _EXISTS_Q.search(question) else "UNSPECIFIED")
    mode = "RESEARCH_REVIEW" if _HYP_Q.search(question) else "COMPARATIVE" if _CMP_Q.search(question) else "FACTUAL"
    return QueryFrame(
        query_id=query_id,
        raw_text=question,
        target_region_id=target,
        target_region_name=reg.areas[target].name if target else None,
        target_deposit_type=";".join(deposits) or None,
        requested_polarity="NEGATE" if _NEG_Q.search(question) else "AFFIRM",
        requested_quantifier=quantifier,
        allow_analogies=mode == "COMPARATIVE",
        requested_mode=mode,
    )


class Context:
    """Кэш снимков статей и фрагментов на один вопрос."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.docs: dict[str, DocumentSnapshot | None] = {}

    def doc(self, doc_id: str) -> DocumentSnapshot | None:
        if doc_id not in self.docs:
            self.docs[doc_id] = snapshot_from_db(self.conn, doc_id)
        return self.docs[doc_id]

    def votes(self, chunk_id: int, quote: str) -> tuple[int | None, int | None]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, votes FROM facts WHERE chunk_id = %s AND quote = %s ORDER BY votes DESC NULLS LAST LIMIT 1",
                (chunk_id, quote),
            )
            row = cur.fetchone()
        return (int(row[0]), row[1]) if row else (None, None)


def decide_fact(
    ctx: Context,
    frame: QueryFrame,
    fact: dict[str, Any],
    reg: GeoAreaRegistry,
    config: PolicyConfig = V3,
) -> dict[str, Any]:
    """Решение по факту графа: fact — chunk_id, doc_id, quote, src, relation, dst (и id, votes)."""
    doc = ctx.doc(fact["doc_id"])
    if fact.get("votes") is None or fact.get("id") is None:
        fid, votes = ctx.votes(int(fact["chunk_id"]), fact["quote"])
        fact = {**fact, "id": fact.get("id") or fid or 0, "votes": votes}
    chunk = doc.chunks.get(str(fact["chunk_id"])) if doc else None
    evidence = make_evidence(doc, str(fact["chunk_id"]), fact["quote"]) if doc else None
    sc = resolve_scope(
        fact["quote"], reg,
        chunk_text=chunk.text if chunk else "",
        headings=chunk.headings if chunk else (),
        title=doc.title if doc else "",
    )
    claim = claim_from_fact(fact, evidence.evidence_id if evidence else f"E:missing:{fact['chunk_id']}")
    if evidence is None:
        from .contracts import SupportAssessment

        support = SupportAssessment(f"S:{claim.claim_id}", claim.claim_id, claim.evidence_ids[0], "INVALID_EVIDENCE",
                                    ("SOURCE_HASH_INVALID", "SPAN_LOST"))
    else:
        support = assess_support(claim, evidence, document=doc, votes=claim.legacy_votes)
    decision = decide_applicability(frame, claim, support, reg, scope=sc, config=config)
    return {"claim": claim, "support": support, "scope": sc, "decision": decision, "evidence": evidence}


def filter_dataset(conn: Any, question: str, data: dict[str, Any]) -> dict[str, Any]:
    """Датасет графа под вопрос о территории: в строках остаются цитаты DIRECT; строка без
    них уходит; ANALOG — в data["analogs"], остальное — в scope_gate.excluded."""
    reg = registry()
    frame = frame_question(question, reg)
    if frame.target_region_id is None:
        data["scope_gate"] = {"frame": frame.to_json(), "applied": False, "reason": "NO_TARGET_REGION"}
        return data
    ctx = Context(conn)
    kept, analogs, excluded = [], [], []
    for row in data.get("facts", []):
        direct_quotes = []
        for q in row.get("quotes", []):
            fact = {"chunk_id": q["chunk_id"], "doc_id": q["doc_id"], "quote": q["quote"],
                    "src": row["about"] if row["direction"] == "→" else row["other"],
                    "dst": row["other"] if row["direction"] == "→" else row["about"],
                    "relation": row["relation"]}
            out = decide_fact(ctx, frame, fact, reg)
            d = out["decision"]
            item = {**q, "status": d.status, "reason_codes": list(d.reason_codes), "decision_id": d.decision_id}
            if d.status == "DIRECT":
                direct_quotes.append(item)
            elif d.status == "ANALOG":
                analogs.append({**item, "relation": row["relation"], "about": row["about"], "other": row["other"]})
            else:
                excluded.append({**item, "relation": row["relation"], "about": row["about"], "other": row["other"]})
        if direct_quotes:
            kept.append({**row, "quotes": direct_quotes, "chunk_ids": [q["chunk_id"] for q in direct_quotes],
                         "documents": len({q["doc_id"] for q in direct_quotes})})
    data = {**data, "facts": kept, "analogs": analogs}
    data["scope_gate"] = {
        "frame": frame.to_json(), "applied": True, "direct_rows": len(kept), "analogs": len(analogs),
        "excluded": excluded, "policy": "V3",
    }
    return data


def mark_unchecked(claims: list[str], checked: int) -> list[dict[str, Any]]:
    """Статус проверки каждого утверждения ответа: первые checked — CHECKED, остальные —
    NOT_CHECKED (лимит chat/support.MAX_CLAIMS). Непроверенное не объявляется проверенным."""
    return [{"claim": c, "status": "CHECKED" if i < checked else "NOT_CHECKED"} for i, c in enumerate(claims)]


def normalized(text: str) -> str:
    return normalize(text)
