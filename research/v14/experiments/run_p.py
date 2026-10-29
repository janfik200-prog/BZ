"""WP11 P: сквозные ответы чат-бота на вопросы о территориях (qwen3-14b, BGE-M3, база копии).

  P0 — гибридный поиск с моделью, без графа (prepare: graph_hits=None, datasets=None,
       collect выключен);
  P1 — как в боевой системе: граф и сбор включены, гейта нет;
  P2 — P1 + контекстный гейт (GEORAG_SCOPE_GATE=1: граф и фрагменты ответа);
  P3 (двухэтапное извлечение + гейт + досье) — в виде чата не оценивается: досье —
       отдельный артефакт (build_dossiers.py), см. отчёт.

Оценка без геолога и потому приблизительная — по статьям, на которые ссылается ответ:
у каждой статьи вручную задана область исследования (build_dataset.STUDY). Ссылка
«чужая», если область статьи несовместима с территорией вопроса (DISJOINT), «своя» —
EXACT/PART_OF, остальное — «неизвестно» (обзоры, методики). Это уровень статьи, а не
утверждения: точная оценка ответа — задача геолога (BLOCKED, LIMITATIONS.md).
Пишет results/raw_p.jsonl и results/p_metrics.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")

import build_dataset as B  # noqa: E402
from georag.chat.answer import answer, prepare  # noqa: E402
from georag.chat.settings import Settings  # noqa: E402
from georag.evidence.georegistry import CONTAINS, DISJOINT, EXACT, OVERLAP, PART_OF, load  # noqa: E402
from georag.index import db  # noqa: E402
from georag.index.embed import build_embedder  # noqa: E402

V14 = ROOT / "research" / "v14"
RES = V14 / "results"
QUESTIONS = ["Q01", "Q03", "Q06", "Q09", "Q10", "Q11", "Q15", "Q16", "Q18", "Q20", "Q21", "Q24", "Q26", "Q29", "Q31"]
EXTRA = [  # территории, о которых в корпусе статей нет: правильный ответ — «в базе нет»
    {"query_id": "X1", "raw_text": "Какие структурные признаки золотого оруденения установлены на Анабарском щите?", "target_region_id": "R:RU:SIBPLAT"},
    {"query_id": "X2", "raw_text": "Что известно о золотом оруденении Марокко?", "target_region_id": "R:MA"},
]
_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def run_one(conn, embedder, settings: Settings, question: str, mode: str) -> dict:
    os.environ["GEORAG_SCOPE_GATE"] = "1" if mode == "P2" else ""
    t0 = time.monotonic()
    if mode == "P0":
        prepared = None
        settings = replace(settings, collect=False)
        gen = prepare(conn, embedder, question, [], settings, graph_hits=None, datasets=None)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            prepared = stop.value
        events = answer(conn, embedder, question, settings=settings, prepared=prepared)
    else:
        events = answer(conn, embedder, question, settings=settings)
    text, sources, done, gate_info = "", [], {}, None
    for ev in events:
        if ev["type"] == "sources":
            sources = ev["sources"]
            gate_info = ev.get("scope_gate")
        elif ev["type"] == "text":
            text += ev["text"]
        elif ev["type"] == "revised":
            text = ev["text"]
        elif ev["type"] == "done":
            done = ev
            text = ev.get("text", text) if ev.get("mode") == "нет в базе" else text
            gate_info = gate_info or ev.get("scope_gate")
    cited = sorted({int(n) for g in _CITE.findall(text) for n in re.split(r"[,;]", g) if n.strip().isdigit()})
    by_n = {s["n"]: s for s in sources}
    return {"mode": mode, "question": question, "text": text, "cited": cited,
            "cited_docs": [by_n[n]["doc_id"] for n in cited if n in by_n],
            "n_sources": len(sources), "seconds": round(time.monotonic() - t0, 1), "done_mode": done.get("mode"),
            "not_checked": done.get("not_checked"), "scope_gate": gate_info}


def judge(reg, target: str, doc_ids: list[str]) -> dict:
    kinds = {"own": 0, "foreign": 0, "unknown": 0}
    for d in doc_ids:
        regions, kind = B.STUDY.get(d, ((), ""))
        rels = {reg.relation(r, target) for r in regions}
        if rels & {EXACT, PART_OF}:
            kinds["own"] += 1
        elif rels and rels <= {DISJOINT}:
            kinds["foreign"] += 1
        else:
            kinds["unknown"] += 1
    return kinds


def main() -> None:
    reg = load(B.DATA / "regional_registry.csv", B.DATA / "registry_aliases.csv")
    queries = {q["query_id"]: q for q in json.loads((B.DATA / "queries.json").read_text("utf-8"))}
    items = [queries[q] for q in QUESTIONS] + EXTRA
    settings = Settings()
    embedder = build_embedder(device=os.environ.get("GEORAG_EMBED_DEVICE", "cpu"))
    out = []
    with db.connect() as conn:
        for q in items:
            for mode in ("P0", "P1", "P2"):
                try:
                    r = run_one(conn, embedder, settings, q["raw_text"], mode)
                except Exception as exc:  # noqa: BLE001 — записать и идти дальше
                    r = {"mode": mode, "question": q["raw_text"], "error": f"{type(exc).__name__}: {exc}", "cited_docs": []}
                    conn.rollback()
                r["query_id"] = q["query_id"]
                r["target_region_id"] = q["target_region_id"]
                r["citations_by_doc_region"] = judge(reg, q["target_region_id"], r["cited_docs"])
                out.append(r)
                print(q["query_id"], mode, r.get("seconds"), r["citations_by_doc_region"], r.get("done_mode"), r.get("error", ""), flush=True)
                with (RES / "raw_p.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.environ["GEORAG_SCOPE_GATE"] = ""
    metrics = {}
    for mode in ("P0", "P1", "P2"):
        rows = [r for r in out if r["mode"] == mode and r["query_id"] in QUESTIONS]
        cites = sum(sum(r["citations_by_doc_region"].values()) for r in rows)
        foreign = sum(r["citations_by_doc_region"]["foreign"] for r in rows)
        own_q = sum(r["citations_by_doc_region"]["own"] > 0 for r in rows)
        any_foreign_q = sum(r["citations_by_doc_region"]["foreign"] > 0 for r in rows)
        extra = [r for r in out if r["mode"] == mode and r["query_id"] in ("X1", "X2")]
        metrics[mode] = {
            "n_questions": len(rows), "citations": cites, "foreign_citations": foreign,
            "foreign_citation_rate": foreign / cites if cites else None,
            "questions_with_own_citation": own_q, "questions_with_foreign_citation": any_foreign_q,
            "errors": sum("error" in r for r in rows), "mean_seconds": round(sum(r.get("seconds") or 0 for r in rows) / max(len(rows), 1), 1),
            "no_data_questions": [{"q": r["query_id"], "cited": r["citations_by_doc_region"], "mode": r.get("done_mode")} for r in extra],
        }
    (RES / "p_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
