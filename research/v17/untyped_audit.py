"""Задача 3 ТЗ v17: связи графа, не прошедшие типизацию.

Ветка обхода — в коде системы:
  store.accepted   — в граф идёт связь с votes ≥ 3 ИЛИ связь, концы которой связаны и в
                     другой статье (повтор);
  kinds.todo       — тип ставится только связям с votes ≥ 3;
  store.all_facts  — из графа убираются связи, у которых typed_at есть, а rel_type нет.
Связь, принятая только по повтору, в typing не попадает, typed_at у неё NULL, и
all_facts её пропускает.

Варианты графа:
  A — текущий граф (все принятые связи);
  B — только прошедшие типизацию;
  C — B плюс те из 53, которым стандартная типизация (kinds.prompt/judge, та же модель
      через vLLM, temperature 0, пачки по 10) ставит тип. Ответ модели в базу не пишется.
Ручная оценка («содержательная геологическая связь») — столбец manual_label: его
заполняет человек по листу «Без типизации» в Разметка_v17.xlsx; модель её не заменяет.

    untyped_audit.py [--no-llm] → results/untyped_53_audit.csv, results/untyped_53_summary.json,
                                  .cache/untyped_retype.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import psycopg  # noqa: E402

from georag.graph import api, kinds, store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.text import name_key  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
RES = HERE / "results"
CACHE = HERE / ".cache" / "untyped_retype.json"
LABELS = HERE / "Разметка_v17_проверено.xlsx"
# Две версии одной работы: препринт EGUsphere и статья GMD (одни авторы, одна тема, почти
# одинаковые названия). Повтор между ними — не независимое подтверждение.
SAME_WORK = [{"10.5194_egusphere-2026-1293", "10.5194_gmd-19-8755-2026"}]


def retype(rows: list[tuple]) -> dict[int, dict]:
    """Стандартная типизация (kinds.prompt → модель → kinds.judge) без записи в базу."""
    if CACHE.exists():
        return {int(k): v for k, v in json.loads(CACHE.read_text("utf-8")).items()}
    from georag.llm import LLM

    llm = LLM()
    out: dict[int, dict] = {}
    for i in range(0, len(rows), kinds.BATCH):
        batch = [(r[0], r[1], r[2], r[3], r[4]) for r in rows[i:i + kinds.BATCH]]
        answer = llm.chat_json(kinds.SYSTEM, kinds.prompt(batch), temperature=0, max_tokens=kinds.ANSWER_TOKENS)
        answer = answer if isinstance(answer, dict) else {}
        for j, (fid, src, _rel, dst, _q) in enumerate(batch, start=1):
            relation, flip, sk, dk, note = kinds.judge(answer.get(str(j)), src, dst)
            out[fid] = {"rel_type": relation, "flip": flip, "src_kind": sk, "dst_kind": dk, "note": note,
                        "model": llm.model}
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out


def manual_labels() -> dict[int, int]:
    if not LABELS.exists():
        return {}
    from openpyxl import load_workbook

    ws = load_workbook(LABELS, data_only=True)["Без типизации"]
    out = {}
    for r in range(2, ws.max_row + 1):
        fid, v = ws.cell(r, 1).value, ws.cell(r, 8).value
        if fid is not None and str(v).strip() in ("0", "1"):
            out[int(fid)] = int(str(v).strip())
    return out


def main() -> None:
    conn = psycopg.connect(DSN, autocommit=True)
    graph = store.all_facts(conn)
    untyped = [f for f in graph if f["rel_type"] is None]
    ids = [f["id"] for f in untyped]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT f.id, f.src, f.relation, f.dst, f.quote, f.doc_id, f.chunk_id, f.votes, f.src_key, f.dst_key,
                   (SELECT array_agg(DISTINCT o.doc_id) FROM facts o WHERE o.doc_id <> f.doc_id AND
                      ((o.src_key = f.src_key AND o.dst_key = f.dst_key) OR
                       (o.src_key = f.dst_key AND o.dst_key = f.src_key))),
                   (SELECT array_agg(o.id ORDER BY o.id) FROM facts o WHERE o.doc_id <> f.doc_id AND
                      ((o.src_key = f.src_key AND o.dst_key = f.dst_key) OR
                       (o.src_key = f.dst_key AND o.dst_key = f.src_key)))
            FROM facts f WHERE f.id = ANY(%s) ORDER BY f.id""", (ids,))
        rows = cur.fetchall()
    gids = {f["id"] for f in graph}
    typed_by_id = {f["id"]: f for f in graph if f["rel_type"] is not None}
    llm = retype(rows) if "--no-llm" not in sys.argv else {}
    labels = manual_labels()

    syn = api._with_auto(conn, S.load())

    def node(name: str) -> str:
        k = name_key(name)
        return name_key(syn.names[k]) if k in syn.names else k

    untyped_ids = set(ids)
    out_rows = []
    for (fid, src, rel, dst, quote, doc, chunk, votes, _sk, _dk, docs, partners) in rows:
        partners = partners or []
        in_graph = [p for p in partners if p in gids]
        typed_partners = [p for p in in_graph if p in typed_by_id]
        names_bad = kinds.bad_name(src, "") or kinds.bad_name(dst, "")
        r = llm.get(fid, {})
        lab = labels.get(fid)
        same_work = bool(docs) and all({doc, o} in SAME_WORK for o in docs)
        only_untyped = bool(partners) and all(p in untyped_ids for p in partners)
        if r.get("rel_type"):
            action = ("исключить: тип поставлен, но по разметке связь неверна" if lab == 0 else
                      "включить с типом (вариант C)" if lab == 1 else
                      "включить с типом (вариант C) после ручной проверки")
        elif r:
            action = "исключить: " + (r.get("note") or "")
        else:
            action = "повторная типизация не выполнялась"
        out_rows.append({
            "relation_id": f"F{fid}", "source_id": chunk, "doc_id": doc, "src": src, "relation": rel, "dst": dst,
            "quote": " ".join((quote or "").split()),
            "original_status": f"в графе по повтору: вопросов {votes if votes is not None else '—'} из 4; "
                               f"концы связаны в других статьях: {len(docs or [])}",
            "votes": votes, "repeat_docs": len(docs or []), "repeat_relations": len(partners),
            "repeat_in_graph": len(in_graph), "repeat_typed": len(typed_partners),
            "repeat_types": ";".join(sorted({typed_by_id[p]["rel_type"] for p in typed_partners})),
            "repeat_partner_docs": ";".join(docs or []), "repeat_same_work_versions": int(same_work),
            "repeat_only_from_untyped": int(only_untyped),
            "name_rule": names_bad or "",
            "type_status": (f"не типизирована; повторно: {r['rel_type']}" if r.get("rel_type")
                            else "не типизирована; повторно: без типа" if r else "не типизирована"),
            "reason": r.get("note", "") if r and not r.get("rel_type") else "",
            "retype_kinds": f"{r.get('src_kind', '')}→{r.get('dst_kind', '')}" if r else "",
            "manual_label": "" if lab is None else lab, "proposed_action": action,
        })
    RES.mkdir(exist_ok=True)
    fields = ["relation_id", "source_id", "original_status", "type_status", "reason", "manual_label",
              "proposed_action", "doc_id", "src", "relation", "dst", "votes", "repeat_docs", "repeat_relations",
              "repeat_in_graph", "repeat_typed", "repeat_types", "repeat_partner_docs", "repeat_same_work_versions",
              "repeat_only_from_untyped", "name_rule", "retype_kinds", "quote"]
    with (RES / "untyped_53_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(out_rows)

    def triples(facts: list[dict], types: dict[int, tuple[str, bool]]) -> set:
        out = set()
        for f in facts:
            t, flip = types.get(f["id"], (f.get("rel_type"), f.get("rel_flip")))
            a, b = (node(f["dst"]), node(f["src"])) if flip else (node(f["src"]), node(f["dst"]))
            out.add((a, t or f["relation"], b))
        return out

    typed = [f for f in graph if f["rel_type"] is not None]
    c_add = {fid: (r["rel_type"], r["flip"]) for fid, r in llm.items() if r.get("rel_type")}
    c_facts = typed + [f for f in untyped if f["id"] in c_add]
    tA, tB, tC = triples(graph, {}), triples(typed, {}), triples(c_facts, c_add)
    summary = {
        "untyped_in_graph": len(untyped),
        "votes": dict(Counter(str(r[7]) for r in rows)),
        "repeat_docs": dict(Counter(len(r[10] or []) for r in rows)),
        "with_typed_partner_in_graph": sum(1 for x in out_rows if x["repeat_typed"]),
        "repeat_same_work_versions": sum(x["repeat_same_work_versions"] for x in out_rows),
        "repeat_only_from_untyped": sum(x["repeat_only_from_untyped"] for x in out_rows),
        "name_rule_hits": sum(1 for x in out_rows if x["name_rule"]),
        "variants": {
            "A": {"relations": len(graph), "unique_triples": len(tA),
                  "unique_typed_triples": len(triples(typed, {}))},
            "B": {"relations": len(typed), "unique_triples": len(tB), "removed": len(untyped),
                  "removed_triples": len(tA - tB)},
            "C": ({"relations": len(c_facts), "unique_triples": len(tC), "retyped": len(c_add),
                   "rejected": len(untyped) - len(c_add),
                   "rejected_reasons": dict(Counter((r.get("note") or "").split(":")[0]
                                                    for r in llm.values() if not r.get("rel_type"))),
                   "types": dict(Counter(r["rel_type"] for r in llm.values() if r.get("rel_type"))),
                   "model": next(iter(llm.values()))["model"]} if llm else None),
        },
        "manual": ({"labelled": len(labels), "geological": sum(labels.values()),
                    "geological_among_retyped": sum(labels.get(f, 0) for f in c_add),
                    "labelled_among_retyped": sum(1 for f in c_add if f in labels)} if labels else None),
    }
    (RES / "untyped_53_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
