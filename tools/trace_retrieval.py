"""Trace narrow factual questions against a live DB/model, plus oracle-context control.

Run from the repository: python tools/trace_retrieval.py --output .local/trace.json
Read-only DB; no collection cache is populated. Keyword checks are diagnostics,
not semantic grading. Inspect saved answers and evidence before accepting them.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georag.chat import answer as api  # noqa: E402
from georag.chat import retrieval, search
from georag.chat.messages import fit_budget  # noqa: E402
from georag.chat.settings import Settings  # noqa: E402
from georag.index import db  # noqa: E402
from georag.index.embed import Embedder  # noqa: E402
from georag.index.search import SELECT_FIELDS, _to_hit  # noqa: E402
from georag.text import locate_quote  # noqa: E402


def exhaust(generator):
    while True:
        try:
            event = next(generator)
            if event.get("type") == "status":
                print(event["text"], flush=True)
        except StopIteration as stop:
            return stop.value


def final(events, terms):
    text = ""
    for event in events:
        if event["type"] in ("general", "revised"):
            text = event.get("text", "") if event["type"] == "revised" else ""
        elif event["type"] == "token":
            text += event["text"]
    return {"text": text, "terms_present": {t: t in text.lower() for t in terms}, "events": events}


def run_case(conn, embedder, case):
    rows = conn.execute(
        f"SELECT {SELECT_FIELDS} FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
        "WHERE c.doc_id=%s AND c.ord=ANY(%s) ORDER BY c.ord",
        (case["doc_id"], case["context_ords"]),
    ).fetchall()
    evidence = [_to_hit(row) for row in rows]
    target = next(h for h in evidence if h.ord == case["evidence_ord"])
    if not locate_quote(case["quote"], target.text):
        raise ValueError(f"Gold quote no longer matches corpus: {case['id']}")
    trace = {
        "case": case,
        "target_chunk_id": target.chunk_id,
        "queries": [],
        "merged": [],
        "judge": [],
        "selection": [],
    }
    settings = Settings(collect=False, check_conflicts=False)
    original_search, original_merge = search.search_all, retrieval.merged_search
    original_select = retrieval._select

    def search_all(conn, emb, queries, facts):
        lists = original_search(conn, emb, queries, facts)
        trace["queries"].extend(
            {"query": q, "ids": [h.chunk_id for h in hits]}
            for q, hits in zip(queries, lists, strict=True)
        )
        return lists

    def merged(*args, **kwargs):
        hits = original_merge(*args, **kwargs)
        trace["merged"].append([h.chunk_id for h in hits])
        return hits

    def judge(question, plan, batch, accepted, settings):
        result = retrieval.judge_fragments(question, plan, batch, accepted, settings)
        trace["judge"].append(
            {
                "candidates": [{"id": c["id"], "chunk_id": c["hit"].chunk_id} for c in batch],
                "response": result,
            }
        )
        return result

    def select(accepted, *args):
        chosen = original_select(accepted, *args)
        trace["selection"].append(
            {
                "accepted": [c["hit"].chunk_id for c in accepted],
                "chosen": [c["hit"].chunk_id for c in chosen],
            }
        )
        return chosen

    with (
        patch.object(search, "search_all", search_all),
        patch.object(retrieval, "merged_search", merged),
        patch.object(retrieval, "_select", select),
    ):
        prepared = exhaust(
            api.prepare(conn, embedder, case["question"], [], settings, judge=judge, datasets=None)
        )
    trace["plan"] = asdict(prepared.plan)
    fitted, _ = fit_budget(case["question"], prepared.sources, [], settings, prepared.plan)
    trace["context"] = fitted
    trace["normal"] = final(
        list(api.answer(conn, embedder, case["question"], settings=settings, prepared=prepared)),
        case["answer_terms"],
    )
    oracle = api.Prepared(
        prepared.plan,
        [search.source_dict(n, hit) for n, hit in enumerate(evidence, 1)],
        retrieval.Retrieval([], judged=False),
    )
    trace["oracle"] = final(
        list(api.answer(conn, embedder, case["question"], settings=settings, prepared=oracle)),
        case["answer_terms"],
    )
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("tests/grounded_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    embedder = Embedder(device="cpu")
    output = []
    with db.connect() as conn:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        conn.commit()
        for case in json.loads(args.cases.read_text(encoding="utf-8")):
            print(case["id"], flush=True)
            output.append(run_case(conn, embedder, case))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
