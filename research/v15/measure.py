"""Статья v15, раздел «Экспериментальная оценка»: измерения без разметки.

Копия базы georag_v14 (снимок боевой от 29.10.2026), только чтение, кроме таблицы
кластеров (clusters.compute пишет их в копию). Результат — results/measure.json.

    .venv/Scripts/python.exe research/v15/measure.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from georag.graph import captions, dedup, store  # noqa: E402
from georag.graph import synonyms as S  # noqa: E402
from georag.graph.entities import MIN_TERMS, Finder, language  # noqa: E402
from georag.index.db import connect  # noqa: E402
from georag.text import name_key  # noqa: E402

DSN = os.environ.get("GEORAG_EVAL_DSN", "postgresql://georag:georag@127.0.0.1:5433/georag_v14")
OUT = Path(__file__).resolve().parent / "results"


def corpus(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), sum(pages), min(year), max(year) FROM documents")
        docs, pages, y0, y1 = cur.fetchone()
        cur.execute("SELECT parser, count(*) FROM documents GROUP BY 1")
        parsers = dict(cur.fetchall())
        cur.execute("SELECT n_tokens, has_table, text FROM chunks")
        rows = cur.fetchall()
    langs = Counter(language(t or "") for _, _, t in rows)
    toks = [n for n, _, _ in rows if n]
    return {"documents": docs, "pages": int(pages or 0), "years": [y0, y1], "parsers": parsers,
            "chunks": len(rows), "chunks_with_table": sum(bool(h) for _, h, _ in rows),
            "tokens_median": statistics.median(toks), "tokens_max": max(toks),
            "chunk_language": dict(langs)}


def prefilter(conn) -> dict:
    """Первый этап графа на всём корпусе, в том же порядке, что facts.run: повтор →
    мало терминов → подписи и легенды. Отпечатки — в памяти, база не меняется."""
    with conn.cursor() as cur:
        cur.execute("SELECT c.id, c.text FROM chunks c ORDER BY c.doc_id, c.ord")
        rows = cur.fetchall()
        cur.execute("SELECT chunk_id, count(*) FROM facts GROUP BY 1")
        facts_by_chunk = dict(cur.fetchall())
    in_graph = Counter(f["chunk_id"] for f in store.all_facts(conn))
    finder = Finder()
    index: dict[tuple[int, int], list[int]] = {}
    sigs: dict[int, list[int]] = {}
    out = Counter()
    lost = Counter()
    lost_graph = Counter()
    t0 = time.monotonic()
    for cid, text in rows:
        text = text or ""
        found = finder.find(text)
        kind = "to_model"
        sig = dedup.signature(text)
        if sig is not None:
            keys = dedup.bands(sig)
            near = {o for b, k in enumerate(keys) for o in index.get((b, k), [])}
            if any(dedup.similarity(sig, sigs[o]) >= dedup.DUP_SIMILARITY for o in near):
                kind = "duplicate"
            else:
                sigs[cid] = sig
                for b, k in enumerate(keys):
                    index.setdefault((b, k), []).append(cid)
        if kind == "to_model" and not found.enough(MIN_TERMS):
            kind = "sparse"
        if kind == "to_model" and captions.skip_reason(text):
            kind = "captions"
        out[kind] += 1
        lost[kind] += facts_by_chunk.get(cid, 0)
        lost_graph[kind] += in_graph.get(cid, 0)
    return {"chunks": len(rows), "by_decision": dict(out), "facts_in_dropped": dict(lost), "graph_facts_in_dropped": dict(lost_graph),
            "min_terms": MIN_TERMS, "dup_similarity": dedup.DUP_SIMILARITY,
            "seconds": round(time.monotonic() - t0, 1)}


def _syn(conn):
    from georag.graph import api
    return api._with_auto(conn, S.load())


def graph(conn) -> dict:
    syn = _syn(conn)
    acc = store.accepted()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT chunk_id), count(DISTINCT doc_id) FROM facts")
        total, chunks, docs = cur.fetchone()
        cur.execute(f"SELECT count(*) FROM facts f WHERE {acc}")
        accepted = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM facts f WHERE {acc} AND relation LIKE '%%предположительно%%'")
        hedged = cur.fetchone()[0]
        cur.execute(f"SELECT coalesce(rel_type, '—'), count(*) FROM facts f WHERE {acc} GROUP BY 1 ORDER BY 2 DESC")
        types = dict(cur.fetchall())
        cur.execute(f"""SELECT k, count(*) FROM (SELECT src_kind k FROM facts f WHERE {acc}
                        UNION ALL SELECT dst_kind FROM facts f WHERE {acc}) x GROUP BY 1 ORDER BY 2 DESC""")
        kinds = dict(cur.fetchall())
        cur.execute(f"SELECT src_key, dst_key, doc_id FROM facts f WHERE {acc}")
        edges = cur.fetchall()
        cur.execute("SELECT count(*), count(canon) FROM entity_names")
        names, auto = cur.fetchone()
        cur.execute("SELECT coalesce(sum(cardinality(rejected)), 0), coalesce(sum(facts), 0) FROM facts_pass")
        rejected, saved = cur.fetchone()
    def node(k):
        name = syn.names.get(k)
        return name_key(name) if name else k
    keys = {k for a, b, _ in edges for k in (a, b)}
    nodes = {node(k) for k in keys}
    pairs = Counter()
    for a, b, d in edges:
        x, y = sorted((node(a), node(b)))
        if x != y:
            pairs[(x, y)] += 1
    return {"facts_total": total, "fragments_with_facts": chunks, "documents_with_facts": docs,
            "accepted": accepted, "candidates": total - accepted, "hedged_accepted": hedged,
            "rejected_by_code": int(rejected), "rel_types": types, "kinds": kinds,
            "name_keys": len(keys), "nodes_after_synonyms": len(nodes), "edges": len(pairs),
            "entity_names": names, "auto_synonyms": auto, "synonym_dictionary": len(S.load().names), "with_auto": len(syn.names)}


def graph_only(conn) -> dict:
    """Сам граф (store.all_facts): объекты после синонимов, пары, повторы в статьях."""
    syn = _syn(conn)
    facts = store.all_facts(conn)

    def node(name: str) -> str:
        k = name_key(name)
        return name_key(syn.names[k]) if k in syn.names else k

    docs: dict[tuple[str, str], set[str]] = {}
    for f in facts:
        docs.setdefault(tuple(sorted((node(f["src"]), node(f["dst"])))), set()).add(f["doc_id"])
    return {"facts": len(facts), "documents": len({f["doc_id"] for f in facts}),
            "spellings": len({name_key(f[e]) for f in facts for e in ("src", "dst")}),
            "nodes": len({node(f[e]) for f in facts for e in ("src", "dst")}),
            "pairs": len(docs), "pairs_multi": sum(len(v) >= 2 for v in docs.values()),
            "hedged": sum("предположительно" in f["relation"] for f in facts)}


def clusters(conn) -> dict:
    from georag.graph import clusters as C
    stats = C.compute(conn, _syn(conn))
    conn.commit()
    return {k: v for k, v in stats.items() if not isinstance(v, (list, dict)) or k == "sizes"}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if sys.argv[1:] == ["graph"]:  # дописать только состав графа, без долгого отбора
        res = json.loads((OUT / "measure.json").read_text("utf-8"))
        with connect(DSN) as conn:
            res["graph_only"] = graph_only(conn)
        (OUT / "measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
        print(res["graph_only"])
        return
    res = {"dsn_db": DSN.rsplit("/", 1)[-1]}
    with connect(DSN) as conn:
        res["graph_only"] = graph_only(conn)
        res["corpus"] = corpus(conn)
        res["graph"] = graph(conn)
        res["graph"]["in_graph"] = len(store.all_facts(conn))
        res["clusters"] = clusters(conn)
        res["prefilter"] = prefilter(conn)
    (OUT / "measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
