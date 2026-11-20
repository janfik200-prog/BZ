"""Метрики коллекции и графа этой базы — то, что не считает trace_sample.py. База только читается.

results/collection.json — статьи, фрагменты, языки, статусы, каталоги, DOI или ссылка, годы;
                          территории статей по реестру config/geo-registry (по названию статьи).
results/graph_metrics.json — связи графа по моделям, подтверждённые вопросами и вошедшие по повтору,
                          типизация и связи, у которых код изменил направление, типы связей;
                          пары объектов в двух и более публикациях; проходы модели (facts_pass);
                          темы Лейдена на главной связной части (в памяти, в базу не пишется):
                          модулярность Q и NMI девяти запусков с другим зерном.

    python research/base2059/base_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from georag.evidence import gate  # noqa: E402
from georag.graph import api, clusters, store  # noqa: E402
from georag.graph.synonyms import load as load_synonyms  # noqa: E402
from georag.text import name_key  # noqa: E402

DSN = os.environ.get("GEORAG_DSN", "postgresql://georag:georag@127.0.0.1:5435/georag")
RES = HERE / "results"


def collection(c) -> dict:
    q = lambda s: c.execute(s).fetchall()  # noqa: E731
    reg = gate.registry()
    titles = q("SELECT doc_id, title FROM documents")
    per_area: Counter[str] = Counter()
    with_area = 0
    for _doc, title in titles:
        ids = {i for h in reg.find(title or "") if not h.ambiguous for i in h.area_ids}
        with_area += bool(ids)
        per_area.update(ids)
    return {
        "статей": q("SELECT count(*) FROM documents")[0][0],
        "статей с фрагментами": q("SELECT count(DISTINCT doc_id) FROM chunks")[0][0],
        "фрагментов": q("SELECT count(*) FROM chunks")[0][0],
        "фрагментов с вектором": q("SELECT count(embedding) FROM chunks")[0][0],
        "языки": dict(q("SELECT coalesce(language, 'не определён'), count(*) FROM documents GROUP BY 1 ORDER BY 2 DESC")),
        "статусы": dict(q("SELECT status, count(*) FROM documents GROUP BY 1 ORDER BY 2 DESC")),
        "каталоги": dict(q("SELECT source, count(*) FROM documents GROUP BY 1 ORDER BY 2 DESC")),
        "DOI или ссылка": q("SELECT count(*) FILTER (WHERE coalesce(doi, '') <> '' OR url <> '') FROM documents")[0][0],
        "годы": dict(zip(("от", "до", "указан у"), q("SELECT min(year), max(year), count(year) FROM documents")[0])),
        "территории по названию статьи": {
            "реестр, областей": len(reg.areas),
            "статей с территорией": with_area,
            "чаще всего": {reg.areas[i].name: k for i, k in per_area.most_common(25)},
        },
    }


def graph(c) -> dict:
    q = lambda s, *a: c.execute(s, *a).fetchall()  # noqa: E731
    g = store.all_facts(c)
    ids = [f["id"] for f in g]
    model = Counter(r[0] for r in q("SELECT model FROM facts WHERE id = ANY(%s)", (ids,)))
    voted = [f for f in g if (f["votes"] or 0) >= store.MIN_VOTES]
    keys = dict(q("SELECT id, least(src_key, dst_key) || '|' || greatest(src_key, dst_key) FROM facts "
                  "WHERE id = ANY(%s)", (ids,)))
    docs_of: dict[str, set] = defaultdict(set)
    for f in g:
        docs_of[keys[f["id"]]].add(f["doc_id"])
    multi = {k for k, v in docs_of.items() if len(v) >= 2}
    passes = q("SELECT model, count(*), sum(facts), sum(cardinality(rejected)), count(skipped) FROM facts_pass GROUP BY 1")
    return {
        "связей в таблице facts": q("SELECT count(*) FROM facts")[0][0],
        "связей в графе": len(g),
        "статей со связями графа": len({f["doc_id"] for f in g}),
        "связи графа по модели": dict(model),
        "подтверждены вопросами (≥ 3 из 4)": len(voted),
        "вошли по повтору пары": len(g) - len(voted),
        "с типом связи": sum(f["rel_type"] is not None for f in g),
        "код изменил направление при типизации": sum(f["rel_flip"] for f in g),
        "типы": dict(Counter(f["rel_type"] for f in g if f["rel_type"]).most_common()),
        "голоса всех выписанных связей": dict(q("SELECT votes, count(*) FROM facts GROUP BY 1 ORDER BY 1")),
        "исключены типизацией (тип не присвоен после типизации)":
            q("SELECT count(*) FROM facts WHERE typed_at IS NOT NULL AND rel_type IS NULL")[0][0],
        "пар объектов в графе": len(docs_of),
        "пар в двух и более публикациях": len(multi),
        "связей, чья пара в двух и более публикациях": sum(keys[f["id"]] in multi for f in g),
        "проходы модели по фрагментам": [
            {"модель": m, "фрагментов": n, "связей принято": a, "ответов отброшено": r, "отсеяно до модели": s}
            for m, n, a, r, s in passes],
    }


def leiden(c) -> dict:
    import igraph as ig
    import leidenalg as la

    syn = api._with_auto(c, load_synonyms(None))
    canon: dict[str, str] = {}

    def node(k: str) -> str:
        if k not in canon:
            name = syn.names.get(k)
            canon[k] = name_key(name) if name else k
        return canon[k]

    w: Counter[tuple[str, str]] = Counter()
    for a, b, docs in clusters._edges(c):
        u, v = node(a), node(b)
        if u != v:
            w[(min(u, v), max(u, v))] += docs
    names = sorted({x for p in w for x in p})
    idx = {nm: i for i, nm in enumerate(names)}
    G = ig.Graph(n=len(names), edges=[(idx[u], idx[v]) for u, v in w])
    G.es["weight"] = list(w.values())
    giant = G.connected_components().giant()
    base = la.find_partition(giant, la.ModularityVertexPartition, weights="weight", seed=clusters.SEED)
    nmi = [ig.compare_communities(base.membership,
                                  la.find_partition(giant, la.ModularityVertexPartition, weights="weight",
                                                    seed=clusters.SEED + s).membership, method="nmi")
           for s in range(1, 10)]
    return {"узлов (после синонимов)": G.vcount(), "рёбер": G.ecount(),
            "узлов в главной связной части": giant.vcount(), "зерно": clusters.SEED,
            "модулярность Q": round(giant.modularity(base.membership, weights="weight"), 3),
            "тем": len(set(base.membership)),
            "NMI девяти запусков: среднее, мин, макс": [round(sum(nmi) / 9, 3), round(min(nmi), 3), round(max(nmi), 3)]}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    RES.mkdir(exist_ok=True)
    with psycopg.connect(DSN + "?options=-c%20default_transaction_read_only%3Don") as c:
        col = collection(c)
        (RES / "collection.json").write_text(json.dumps(col, ensure_ascii=False, indent=1, default=str), "utf-8")
        print("коллекция:", col["статей"], "статей,", col["фрагментов"], "фрагментов", flush=True)
        gm = graph(c)
        print("граф:", gm["связей в графе"], "связей; темы Лейдена…", flush=True)
        gm["темы Лейдена"] = leiden(c)
    (RES / "graph_metrics.json").write_text(json.dumps(gm, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: v for k, v in gm.items() if k not in ("типы", "проходы модели по фрагментам")},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
