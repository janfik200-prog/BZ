"""WP09. Досье геолога: нормализованный вопрос, атомарные факты и четыре секции по
статусам — DIRECT, ANALOG, OUT_OF_SCOPE, UNRESOLVED.

У каждой записи — факт, модальность/полярность/квантор, статья/DOI/страница, дословная
цитата, поддержка, применимость с причинами, семейство источника, история правок.
Исправление специалиста — событие в журнале (append-only JSONL): исходный статус,
новый, причина, кто и когда; прошлое не перезаписывается, решение кода остаётся рядом.
Экспорт — JSON и CSV.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import DecisionRecord, GeoClaim, QueryFrame, SupportAssessment

SECTIONS = ("DIRECT", "ANALOG", "OUT_OF_SCOPE", "UNRESOLVED")
TITLES = {
    "DIRECT": "Прямые свидетельства о территории вопроса",
    "ANALOG": "Аналоги по другим районам (обоснованные)",
    "OUT_OF_SCOPE": "Другие территории / несовместимый контекст — не довод",
    "UNRESOLVED": "Не решено: территория, поддержка или квантор не установлены",
}
CSV_COLUMNS = [
    "section", "decision_id", "claim_id", "subject", "predicate", "object", "polarity", "modality",
    "quantifier", "doc_id", "doi", "title", "pages", "quote", "support", "status", "reason_codes",
    "scope_derivation", "region_ids", "source_family", "override_status", "override_reason", "override_by",
]


@dataclass
class EvidenceDossier:
    query: QueryFrame
    entries: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {s: [] for s in SECTIONS})
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query.to_json(),
            "meta": self.meta,
            "sections": {s: {"title": TITLES[s], "n": len(self.entries[s]), "entries": self.entries[s]} for s in SECTIONS},
        }

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in SECTIONS:
            for e in self.entries[s]:
                w.writerow({**e, "section": s, "reason_codes": ";".join(e["reason_codes"]),
                            "pages": ",".join(str(p) for p in e.get("pages") or []),
                            "region_ids": ";".join(e.get("region_ids") or [])})
        return buf.getvalue()


def build_dossier(
    query: QueryFrame,
    decisions: list[DecisionRecord],
    *,
    claims: dict[str, GeoClaim],
    supports: dict[str, SupportAssessment],
    sources: dict[str, dict[str, Any]],
    scopes: dict[str, Any] | None = None,
    overrides: list[dict[str, Any]] | None = None,
) -> EvidenceDossier:
    """sources: claim_id → doc_id, doi, title, pages, quote, source_family.
    overrides — события журнала; последнее по decision_id переносит запись в новую секцию,
    исходное решение кода остаётся в записи."""
    last: dict[str, dict[str, Any]] = {}
    history: dict[str, list[dict[str, Any]]] = {}
    for ev in overrides or []:
        last[ev["decision_id"]] = ev
        history.setdefault(ev["decision_id"], []).append(ev)
    dossier = EvidenceDossier(query, meta={
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy_versions": sorted({d.policy_version for d in decisions}),
        "registry_versions": sorted({d.registry_version for d in decisions}),
        "n_decisions": len(decisions),
        "note": "Статус — о применимости источника к вопросу, не об истинности сказанного.",
    })
    for d in decisions:
        c = claims[d.claim_id]
        src = sources.get(d.claim_id, {})
        sc = (scopes or {}).get(d.claim_id)
        ov = last.get(d.decision_id)
        section = ov["new_status"] if ov else d.status
        entry = {
            "decision_id": d.decision_id,
            "claim_id": d.claim_id,
            "subject": c.subject,
            "predicate": c.predicate,
            "object": c.object,
            "polarity": c.polarity,
            "modality": c.modality,
            "quantifier": c.quantifier,
            "doc_id": src.get("doc_id"),
            "doi": src.get("doi"),
            "title": src.get("title"),
            "pages": src.get("pages") or [],
            "quote": src.get("quote"),
            "evidence_ids": list(d.evidence_ids),
            "support": supports[d.support_id].label if d.support_id in supports else None,
            "status": d.status,
            "reason_codes": list(d.reason_codes),
            "scope_derivation": d.scope_derivation,
            "region_ids": list(getattr(sc, "region_ids", ()) or ()),
            "scope_spans": [s.__dict__ for s in getattr(sc, "spans", [])][:4],
            "source_family": src.get("source_family"),
            "policy_version": d.policy_version,
            "registry_version": d.registry_version,
            "override_status": ov["new_status"] if ov else None,
            "override_reason": ov["reason"] if ov else None,
            "override_by": ov["user"] if ov else None,
            "history": history.get(d.decision_id, []),
        }
        dossier.entries[section].append(entry)
    return dossier


def record_override(log_path: Path | str, decision: DecisionRecord, new_status: str, reason: str, user: str) -> dict[str, Any]:
    """Исправление специалиста — новой строкой журнала; решение кода не меняется."""
    if new_status not in SECTIONS:
        raise ValueError(f"статус {new_status!r} не из {SECTIONS}")
    if not reason.strip() or not user.strip():
        raise ValueError("нужны причина и кто исправил")
    event = {
        "decision_id": decision.decision_id,
        "claim_id": decision.claim_id,
        "query_id": decision.query_id,
        "original_status": decision.status,
        "new_status": new_status,
        "reason": reason,
        "user": user,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy_version": decision.policy_version,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def read_overrides(log_path: Path | str) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def to_markdown(dossier: EvidenceDossier, max_per_section: int = 12) -> str:
    """Читаемый отчёт: путь от вердикта до страницы статьи без логов модели."""
    q = dossier.query
    lines = [
        f"# Досье: {q.raw_text}",
        "",
        f"Территория вопроса: **{q.target_region_name}** (`{q.target_region_id}`); тип оруденения: "
        f"{q.target_deposit_type or '—'}; квантор: {q.requested_quantifier}; режим: {q.requested_mode}; "
        f"аналогии: {'разрешены' if q.allow_analogies else 'нет'}.",
        "",
        f"Политика: {', '.join(dossier.meta['policy_versions'])}; справочник: {', '.join(dossier.meta['registry_versions'])}.",
        "",
        "> " + dossier.meta["note"],
        "",
    ]
    for s in SECTIONS:
        items = dossier.entries[s]
        lines += [f"## {TITLES[s]} — {len(items)}", ""]
        for e in items[:max_per_section]:
            pages = ",".join(str(p) for p in e["pages"]) or "?"
            lines += [
                f"- **{e['subject']} — {e['predicate']} — {e['object']}** "
                f"({e['polarity']}, {e['modality']}, {e['quantifier']})",
                f"  - источник: {e['title']} — DOI {e['doi'] or '—'}, с. {pages}",
                f"  - цитата: «{e['quote']}»",
                f"  - поддержка: {e['support']}; решение: {e['status']} ← {', '.join(e['reason_codes'])}; "
                f"область: {', '.join(e['region_ids']) or '—'} ({e['scope_derivation'] or '—'})",
            ]
            if e["override_status"]:
                lines.append(f"  - исправлено: {e['override_status']} ({e['override_by']}: {e['override_reason']})")
        if len(items) > max_per_section:
            lines.append(f"- … ещё {len(items) - max_per_section} (полностью — в CSV/JSON)")
        lines.append("")
    return "\n".join(lines)
