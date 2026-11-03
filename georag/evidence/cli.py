"""Команда evidence: досье по вопросу, аудит цитат, разбор вопроса.

    python georag.py evidence frame "Какие признаки золота известны для Восточной пустыни Египта?"
    python georag.py evidence dossier "…вопрос…" --out research/v14/dossier/q1 [--topic REGEX] [--dsn …]
    python georag.py evidence override D:… DIRECT "причина" "кто" --log research/v14/dossier/overrides.jsonl
    python georag.py evidence audit [--out provenance_audit.json]
    python georag.py evidence audit --mark-stale --out research/v14/results/stale_audit_t13.json

Только чтение базы: досье и аудит пишутся в файлы, факты и статьи не меняются. Кроме
audit --mark-stale (T13): он применяет миграцию происхождения (index/migrations), помечает
устаревшими факты, чьей цитаты нет в тексте фрагмента, снимает пометку с тех, чья цитата
нашлась, и записывает статьям отпечаток разбора (documents.parsed_sha256).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..index import db as index_db
from ..index.db import connect
from . import dossier as D
from . import gate
from .applicability import V3
from .claims import COMMODITIES
from .provenance import make_evidence, snapshot_from_db, verify_evidence

STRUCTURAL = r"(структур|разлом|сдвиг|зон[аы]|жил|рассланц|складк|надвиг|shear|fault|structur|vein|fold|thrust|lineament|линеамент)"


def facts_for(conn: Any, frame: Any, topic: str | None) -> list[dict[str, Any]]:
    """Факты графа (принятые и кандидаты) о предмете вопроса: тип оруденения вопроса и, если
    задано, тема (регулярное выражение по цитате)."""
    commodity = (frame.target_deposit_type or "").split(";")[0]
    rx = COMMODITIES[commodity].pattern if commodity in COMMODITIES else "."
    with conn.cursor() as cur:
        cur.execute(
            "SELECT f.id, f.chunk_id, f.doc_id, f.src, f.relation, f.dst, f.quote, f.votes, f.model, d.doi, d.title, c.pages "
            "FROM facts f JOIN documents d ON d.doc_id = f.doc_id JOIN chunks c ON c.id = f.chunk_id "
            "WHERE (f.src || ' ' || f.dst || ' ' || f.quote) ~* %s AND (%s::text IS NULL OR f.quote ~* %s) ORDER BY f.id",
            (_pg(rx), topic, topic),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _pg(rx: str) -> str:
    """Шаблон Python → POSIX-шаблон PostgreSQL: \\b → \\y."""
    return rx.replace("\\b", "\\y")


def build(
    conn: Any, question: str, topic: str | None, query_id: str = "Q:cli"
) -> tuple[Any, list[dict[str, Any]]]:
    reg = gate.registry()
    frame = gate.frame_question(question, reg, query_id)
    ctx = gate.Context(conn)
    out = []
    for f in facts_for(conn, frame, topic):
        res = gate.decide_fact(ctx, frame, f, reg, config=V3)
        res["source"] = {
            "doc_id": f["doc_id"],
            "doi": f["doi"],
            "title": f["title"],
            "pages": list(f["pages"] or []),
            "quote": f["quote"],
            "source_family": None,
        }
        out.append(res)
    return frame, out


def write_dossier(
    frame: Any,
    results: list[dict[str, Any]],
    out_dir: Path,
    overrides: list[dict[str, Any]] | None = None,
) -> D.EvidenceDossier:
    dos = D.build_dossier(
        frame,
        [r["decision"] for r in results],
        claims={r["claim"].claim_id: r["claim"] for r in results},
        supports={r["support"].support_id: r["support"] for r in results},
        sources={r["claim"].claim_id: r["source"] for r in results},
        scopes={r["claim"].claim_id: r["scope"] for r in results},
        overrides=overrides,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dossier.json").write_text(
        json.dumps(dos.to_json(), ensure_ascii=False, indent=1), "utf-8"
    )
    (out_dir / "dossier.csv").write_text(dos.to_csv(), "utf-8")
    (out_dir / "dossier.md").write_text(D.to_markdown(dos), "utf-8")
    return dos


def audit_facts(conn: Any, mark_stale: bool = False) -> dict[str, Any]:
    """Аудит цитат всех фактов по снимкам статей из базы (provenance_audit.json).

    Цитаты нет в нормализованном тексте фрагмента (или фрагмента нет) — факт устарел: его
    фрагмент разобран заново, и гейт по нему доказательства не построит. Битый текст
    (OCR_BROKEN) и отпечатки PDF — другие причины недействительности, не устаревание.

    mark_stale — записать в базу (T13): миграция, пометки stale_since/stale_reason
    (новые — помечаются, нашедшиеся — снимаются, прежние остаются со своим временем),
    отпечаток разбора у статей, где его нет или он другой."""
    if mark_stale:
        index_db.migrate_provenance(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_id, doc_id, quote FROM facts ORDER BY id")
        rows = cur.fetchall()
        cur.execute("SELECT doc_id FROM documents ORDER BY doc_id")
        doc_ids = [str(r[0]) for r in cur.fetchall()]
    cache: dict[str, Any] = {}
    bad, stale, ok = [], [], 0
    for fid, cid, did, quote in rows:
        if did not in cache:
            cache[did] = snapshot_from_db(conn, did)
        doc = cache[did]
        ev = make_evidence(doc, str(cid), quote) if doc else None
        res = verify_evidence(ev, doc) if ev else None
        if res and res.valid:
            ok += 1
            continue
        bad.append(
            {
                "fact_id": fid,
                "doc_id": did,
                "chunk_id": cid,
                "reasons": (
                    list(res.reason_codes) if res else ["SPAN_LOST", "QUOTE_NOT_IN_PARSED_TEXT"]
                ),
                "quote": re.sub(r"\s+", " ", quote)[:200],
            }
        )
        if ev is None:
            stale.append(int(fid))
    report: dict[str, Any] = {
        "normalization": "collapse-whitespace-v1",
        "n_facts": len(rows),
        "valid": ok,
        "invalid": len(bad),
        "invalid_examples": bad[:200],
    }
    if not mark_stale:
        return report

    from ..graph import store

    lost = set(stale)
    reason = f"{index_db.QUOTE_LOST}: evidence audit"
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM facts WHERE stale_since IS NOT NULL")
        before = {int(r[0]) for r in cur.fetchall()}
        # Сколько устаревших шло в граф, пока пометки не было (accepted без fresh).
        cur.execute(
            f"SELECT count(*) FROM facts f WHERE f.id = ANY(%s) AND {store.accepted()}", (stale,)
        )
        in_graph = int((cur.fetchone() or (0,))[0])
        marked, cleared = sorted(lost - before), sorted(before - lost)
        if marked:
            cur.execute(
                "UPDATE facts SET stale_since = now(), stale_reason = %s WHERE id = ANY(%s)",
                (reason, marked),
            )
        if cleared:
            cur.execute(
                "UPDATE facts SET stale_since = NULL, stale_reason = NULL WHERE id = ANY(%s)",
                (cleared,),
            )
        cur.execute("SELECT doc_id, parsed_sha256 FROM documents")
        stored = {str(d): sha for d, sha in cur.fetchall()}
    # Отпечаток разбора у статьи: нет — записывается (база до T13); другой — текст статьи
    # менялся мимо загрузки, это и есть «невидимое расхождение»: оно в отчёте, факты
    # статьи только что сверены с её текущим текстом, и отпечаток — текущий.
    filled, differs, versions = [], [], {}
    for did in doc_ids:
        if did not in cache:
            cache[did] = snapshot_from_db(conn, did)
        doc = cache[did]
        versions[doc.parser_version] = versions.get(doc.parser_version, 0) + 1
        if stored.get(did) == doc.parsed_sha256:
            continue
        (differs if stored.get(did) else filled).append(did)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET parsed_sha256 = %s, parsed_at = now() WHERE doc_id = %s",
                (doc.parsed_sha256, did),
            )
    by_doc: dict[str, int] = {}
    for e in bad:
        if int(e["fact_id"]) in lost:
            by_doc[e["doc_id"]] = by_doc.get(e["doc_id"], 0) + 1
    report["stale"] = {
        "reason": reason,
        "n": len(stale),
        "fact_ids": stale,
        "by_doc": by_doc,
        "was_in_graph": in_graph,
        "marked_now": marked,
        "already_marked": sorted(before & lost),
        "cleared_now": cleared,
    }
    report["invalid_not_stale"] = [e for e in bad if int(e["fact_id"]) not in lost]
    report["documents"] = {
        "n": len(doc_ids),
        "parsed_sha256_filled": len(filled),
        "parsed_sha256_differed": differs,
        "parser_versions": versions,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python georag.py evidence")
    sub = p.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frame")
    fr.add_argument("question")
    ds = sub.add_parser("dossier")
    ds.add_argument("question")
    ds.add_argument("--out", default="research/v14/dossier/cli")
    ds.add_argument(
        "--topic",
        default=None,
        help="регулярное выражение по цитате; structural — структурные признаки",
    )
    ds.add_argument("--overrides", default=None)
    ov = sub.add_parser("override")
    ov.add_argument("dossier_json")
    ov.add_argument("decision_id")
    ov.add_argument("status")
    ov.add_argument("reason")
    ov.add_argument("user")
    ov.add_argument("--log", required=True)
    au = sub.add_parser("audit")
    au.add_argument("--out", default="research/v14/results/provenance_audit.json")
    au.add_argument(
        "--mark-stale",
        action="store_true",
        help="пометить в базе устаревшими факты, чьей цитаты нет в тексте фрагмента (T13)",
    )
    a = p.parse_args(argv)

    if a.cmd == "frame":
        print(
            json.dumps(
                gate.frame_question(a.question, gate.registry()).to_json(),
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0
    if a.cmd == "override":
        from .contracts import DecisionRecord

        data = json.loads(Path(a.dossier_json).read_text("utf-8"))
        for sec in data["sections"].values():
            for e in sec["entries"]:
                if e["decision_id"] == a.decision_id:
                    rec = DecisionRecord(
                        e["decision_id"],
                        data["query"]["query_id"],
                        e["claim_id"],
                        "",
                        e["status"],
                        tuple(e["reason_codes"]),
                        e["policy_version"],
                        e["registry_version"],
                        tuple(e["evidence_ids"]),
                    )
                    print(
                        json.dumps(
                            D.record_override(a.log, rec, a.status, a.reason, a.user),
                            ensure_ascii=False,
                        )
                    )
                    return 0
        print(f"нет решения {a.decision_id} в {a.dossier_json}", file=sys.stderr)
        return 1
    with connect() as conn:
        if a.cmd == "dossier":
            topic = STRUCTURAL if a.topic == "structural" else a.topic
            frame, results = build(conn, a.question, topic)
            overrides = D.read_overrides(a.overrides) if a.overrides else None
            dos = write_dossier(frame, results, Path(a.out), overrides)
            print(
                json.dumps(
                    {s: len(dos.entries[s]) for s in D.SECTIONS} | {"frame": frame.to_json()},
                    ensure_ascii=False,
                )
            )
            return 0
        if a.cmd == "audit":
            report = audit_facts(conn, mark_stale=a.mark_stale)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
            brief = {k: report[k] for k in ("n_facts", "valid", "invalid")}
            if "stale" in report:
                brief |= {
                    "stale": report["stale"]["n"],
                    "marked_now": len(report["stale"]["marked_now"]),
                    "cleared_now": len(report["stale"]["cleared_now"]),
                }
            print(json.dumps(brief, ensure_ascii=False))
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
