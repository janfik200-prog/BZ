"""Граф из фактов: сеть для страницы, датасеты, отчёт — всё с цитатами.

Факты хранятся как написала модель; здесь имена сводятся к одному:

* сначала словарь синонимов (config/synonyms.yaml) — «Donetsk basin» →
  «Донецкий бассейн»;
* чего в словаре нет — по ключу без падежей и числа: «Донецкого бассейна» и
  «Донецкий бассейн» — один узел, подписанный тем написанием, что чаще.

Датасет — у любой сущности: все факты о ней и о том, что в неё входит
(по фактам «входит в»), с цитатами и статьями. Его показывает карточка узла на
странице, отдаёт Телеграм-бот, по нему отвечает чат-бот на «что известно о …».
"""

from __future__ import annotations

import csv
import io
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..text import name_key, normalize
from . import store
from .synonyms import PART_OF, Synonyms

ENTITY = "сущность"
ARTICLE = "статья"
QUOTES_PER_LINK = 3


class RequestError(ValueError):
    def __init__(self, message: str, detail: str = "", known: list[str] | None = None):
        super().__init__(message)
        self.message, self.detail, self.known = message, detail, known or []

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.message}
        if self.detail:
            out["detail"] = self.detail
        if self.known:
            out["known"] = self.known[:50]
        return out


@dataclass
class Link:
    """Связь между двумя сущностями: все факты с этими концами и этой связью."""

    src: str
    relation: str
    dst: str
    facts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def documents(self) -> int:
        return len({f["doc_id"] for f in self.facts})

    def quotes(self, limit: int = QUOTES_PER_LINK) -> list[dict[str, Any]]:
        out, docs = [], set()
        for f in self.facts:  # сначала — из разных статей
            if f["doc_id"] in docs:
                continue
            docs.add(f["doc_id"])
            out.append(_quote(f))
            if len(out) >= limit:
                break
        return out


def _quote(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "quote": f["quote"],
        "doc_id": f["doc_id"],
        "title": f["title"],
        "year": f["year"],
        "url": f["url"] or (f"https://doi.org/{f['doi']}" if f.get("doi") else ""),
        "chunk_id": f["chunk_id"],
    }


class Graph:
    """Факты с именами, сведёнными словарём и ключом без падежей."""

    def __init__(self, rows: list[dict[str, Any]], syn: Synonyms):
        self.syn = syn
        spelled: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            for raw in (r["src"], r["dst"]):
                spelled[name_key(raw)][raw] += 1
        self.display = {k: c.most_common(1)[0][0] for k, c in spelled.items()}
        self.links: dict[tuple[str, str, str], Link] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.touching: dict[str, list[Link]] = defaultdict(list)  # сущность → её связи
        self.children: dict[str, list[str]] = defaultdict(list)  # целое → что входит
        for r in rows:
            src_raw, dst_raw = r["src"], r["dst"]
            # «щит включает зону» — это «зона входит в щит»: факт разворачивается.
            relation, flip = syn.relation_dir(r["relation"])
            if flip:
                src_raw, dst_raw = dst_raw, src_raw
            src, dst = self.canonical(src_raw), self.canonical(dst_raw)
            if src == dst:
                continue
            key = (src, relation, dst)
            if key not in self.links:
                self.links[key] = Link(src, relation, dst)
                self.touching[src].append(self.links[key])
                self.touching[dst].append(self.links[key])
                if relation == PART_OF:
                    self.children[dst].append(src)
            link = self.links[key]
            link.facts.append(r)
            for name, raw in ((src, src_raw), (dst, dst_raw)):
                e = self.entities.setdefault(
                    name,
                    {
                        "name": name,
                        "facts": 0,
                        "docs": set(),
                        "spellings": set(),
                        "in_dictionary": name in syn.groups,
                    },
                )
                e["facts"] += 1
                e["docs"].add(r["doc_id"])
                if normalize(raw) != normalize(name):
                    e["spellings"].add(raw)

        self._keys = {name: name_key(name).split() for name in self.entities}
        # Слова, которые где-то в графе пишутся со строчной: «high reflectance»,
        # «главная зона». Значит, «High» или «Главная» с заглавной — просто начало
        # имени, а не имя собственное.
        self._common = {
            w.lower()
            for name in self.entities
            for w in re.findall(r"[^\W\d_]+", name)
            if w[:1].islower()
        }

    def canonical(self, raw: str) -> str:
        return self.syn.name(raw) or self.display.get(name_key(raw)) or raw

    def resolve(self, raw: str) -> str | None:
        """Имя, как его написал человек, → сущность графа (или None).

        Такой сущности нет, но имя собственное стоит внутри других («зоне
        Персияновского разлома», «южнее Персияновского разлома…» для
        «Персияновский разлом») — возвращается само имя: датасет соберёт их."""
        if not raw:
            return None
        name = self.syn.name(raw)
        if name in self.entities:
            return name
        key = name_key(raw)
        for entity in self.entities:
            if name_key(entity) == key:
                return entity
        wanted = name or " ".join(raw.split())
        return wanted if self.related(wanted) else None

    def related(self, name: str) -> list[str]:
        """Сущности, в названии которых целиком стоит это имя собственное.

        Модель пишет «зоне Персияновского разлома», «дайки вдоль Персияновского
        разлома» — это всё о разломе, но отдельные узлы. Только для имён
        собственных: у «золото» таких «родственников» сотни, и они о другом."""
        if not self.proper(name):
            return []
        key = name_key(name).split()
        if not key:
            return []
        n = len(key)
        return [
            e
            for e, words in self._keys.items()
            if e != name
            and len(words) > n
            and any(words[i : i + n] == key for i in range(len(words) - n + 1))
        ]

    def proper(self, name: str) -> bool:
        return _proper(name, self._common)

    def parts(self, name: str) -> list[str]:
        """Само имя и всё, что в него входит по фактам «входит в», на любую глубину."""
        out, frontier = [name], [name]
        while frontier:
            for part in self.children.get(frontier.pop(), []):
                if part not in out:
                    out.append(part)
                    frontier.append(part)
        return out


def _proper(name: str, common: set[str] | frozenset[str] = frozenset()) -> bool:
    """Есть ли в имени имя собственное: слово с большой буквы, не аббревиатура и не
    слово, которое в других именах пишется со строчной («High», «Hydrothermal»)."""
    return any(
        w[:1].isupper() and not w.isupper() and w.lower() not in common
        for w in re.findall(r"[^\W\d_]+", name)
    )


_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _stamp(conn: Any) -> Any:
    """Отпечаток фактов: сколько их, последний номер и подтверждения. Только у настоящей базы."""
    try:
        import psycopg
    except ImportError:
        return None
    if not isinstance(conn, psycopg.Connection):
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), coalesce(max(id), 0), count(votes), coalesce(sum(votes), 0) FROM facts"
        )
        row = cur.fetchone()
        return tuple(row) if row else None


def graph(conn: Any, syn: Synonyms) -> Graph:
    """Граф из фактов. Собирается заново, только если факты или словарь поменялись:
    раньше его пересобирали из всех фактов на каждый запрос страницы и каждый
    вопрос чат-бота."""
    stamp = _stamp(conn)
    if stamp is None:
        return Graph(store.all_facts(conn), syn)
    key = (
        stamp,
        syn.sha
        or repr(
            (sorted(syn.names.items()), sorted(syn.relations.items()), sorted(syn.inverse.items()))
        ),
    )
    with _CACHE_LOCK:
        if _CACHE.get("key") == key:
            cached: Graph = _CACHE["graph"]
            return cached
    g = Graph(store.all_facts(conn), syn)
    with _CACHE_LOCK:
        _CACHE.update(key=key, graph=g)
    return g


# --------------------------------------------------------------------------- #
#  Сеть для страницы
# --------------------------------------------------------------------------- #
def network_response(conn: Any, syn: Synonyms, with_articles: bool = True) -> dict[str, Any]:
    """Узлы — сущности (и статьи), рёбра — связи с цитатами (и «из статьи»)."""
    g = graph(conn, syn)
    nodes, edges = [], []
    for e in g.entities.values():
        nodes.append(
            {
                "id": f"{ENTITY}:{e['name']}",
                "kind": ENTITY,
                "name": e["name"],
                "facts": e["facts"],
                "documents": len(e["docs"]),
                "spellings": sorted(e["spellings"])[:12],
                "in_dictionary": e["in_dictionary"],
            }
        )
    for link in g.links.values():
        edges.append(
            {
                "source": f"{ENTITY}:{link.src}",
                "target": f"{ENTITY}:{link.dst}",
                "type": "факт",
                "label": link.relation,
                "weight": link.documents,
                "fragments": len(link.facts),
                "quotes": link.quotes(),
            }
        )
    if with_articles:
        docs: dict[str, dict[str, Any]] = {}
        mentioned: dict[tuple[str, str], int] = Counter()
        for link in g.links.values():
            for f in link.facts:
                docs.setdefault(f["doc_id"], f)
                mentioned[(f["doc_id"], link.src)] += 1
                mentioned[(f["doc_id"], link.dst)] += 1
        for doc_id, f in docs.items():
            nodes.append(
                {
                    "id": f"{ARTICLE}:{doc_id}",
                    "kind": ARTICLE,
                    "name": f["title"],
                    "doc_id": doc_id,
                    "year": f["year"],
                    "url": f["url"],
                    "doi_url": f"https://doi.org/{f['doi']}" if f.get("doi") else None,
                }
            )
        for (doc_id, name), n in mentioned.items():
            edges.append(
                {
                    "source": f"{ARTICLE}:{doc_id}",
                    "target": f"{ENTITY}:{name}",
                    "type": "из статьи",
                    "weight": n,
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "pass": store.summary(conn),
        "dictionary": {"names": len(syn.groups), "sha": syn.sha},
    }


# --------------------------------------------------------------------------- #
#  Датасет — у любой сущности
# --------------------------------------------------------------------------- #
def dataset(conn: Any, syn: Synonyms, name: str, g: Graph | None = None) -> dict[str, Any]:
    """Все факты о сущности и о том, что в неё входит: с кем как связана, где сказано."""
    g = g or graph(conn, syn)
    names: list[str] = []
    for root in [name, *g.related(name)]:
        for part in g.parts(root):
            if part not in names:
                names.append(part)
    inside = set(names)
    rows: list[dict[str, Any]] = []
    docs: set[str] = set()
    seen: set[int] = set()
    for link in (lk for n in names for lk in g.touching.get(n, [])):
        if id(link) in seen:
            continue
        seen.add(id(link))
        if link.src in inside:
            about, direction, other = link.src, "→", link.dst
        elif link.dst in inside:
            about, direction, other = link.dst, "←", link.src
        else:
            continue
        if link.relation == PART_OF and link.src in inside and link.dst in inside:
            continue  # «зона входит в щит» — это состав, а не факт о нём
        docs |= {f["doc_id"] for f in link.facts}
        rows.append(
            {
                "about": about,
                "direction": direction,
                "relation": link.relation,
                "other": other,
                "documents": link.documents,
                "fragments": len(link.facts),
                "quotes": link.quotes(),
                "chunk_ids": [f["chunk_id"] for f in link.facts][:QUOTES_PER_LINK],
            }
        )
    rows.sort(key=lambda r: (-r["documents"], -r["fragments"], r["relation"], r["other"]))
    return {
        "name": name,
        "includes": names[1:],
        "documents": len(docs),
        "facts": rows,
        "in_dictionary": name in syn.groups,
    }


def datasets_response(conn: Any, syn: Synonyms, limit: int = 300) -> dict[str, Any]:
    """Датасеты не хранятся: собираются из фактов при каждом запросе — после нового
    прохода модели появляются новые сущности и новые факты у старых.
    Числа те же, что в карточке датасета: строк-фактов и статей за ними."""
    g = graph(conn, syn)
    rows = []
    for name in g.entities:
        d = dataset(conn, syn, name, g)
        if d["facts"] or d["includes"]:
            rows.append({"name": name, "facts": len(d["facts"]), "documents": d["documents"]})
    rows.sort(key=lambda r: (-r["documents"], -r["facts"], r["name"]))
    return {"datasets": rows[:limit], "total": len(rows), "pass": store.summary(conn)}


def dataset_response(conn: Any, syn: Synonyms, payload: dict[str, Any]) -> dict[str, Any]:
    raw = str(payload.get("name") or payload.get("territory") or "").strip()
    g = graph(conn, syn)
    known = [e["name"] for e in sorted(g.entities.values(), key=lambda e: -e["facts"])]
    if not raw:
        raise RequestError("не указано, чей датасет (name)", known=known)
    name = g.resolve(raw)
    if name is None:
        raise RequestError("такого в фактах из статей нет", raw, known)
    return dataset(conn, syn, name, g)


CSV_COLUMNS = [
    "о чём",
    "направление",
    "связь",
    "с чем",
    "статей",
    "фрагментов",
    "цитата",
    "статья",
    "ссылка",
]


def dataset_csv(data: dict[str, Any]) -> str:
    """Датасет таблицей: открывается в Excel (точка с запятой, UTF-8 с BOM)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for r in data["facts"]:
        q = r["quotes"][0] if r["quotes"] else {}
        title = f"{q.get('title', '')}{', ' + str(q['year']) if q.get('year') else ''}"
        writer.writerow(
            [
                r["about"],
                r["direction"],
                r["relation"],
                r["other"],
                r["documents"],
                r["fragments"],
                q.get("quote", ""),
                title,
                q.get("url", ""),
            ]
        )
    return "﻿" + buf.getvalue()


# --------------------------------------------------------------------------- #
#  Отчёт: что нашла модель и где ошибалась
# --------------------------------------------------------------------------- #
def report(conn: Any, syn: Synonyms, top: int = 40) -> dict[str, Any]:
    g = graph(conn, syn)
    entities = sorted(g.entities.values(), key=lambda e: (-len(e["docs"]), -e["facts"], e["name"]))
    relations: Counter[str] = Counter()
    for link in g.links.values():
        relations[link.relation] += len(link.facts)
    return {
        "pass": store.summary(conn),
        "entities": [
            (e["name"], len(e["docs"]), e["facts"], e["in_dictionary"], sorted(e["spellings"])[:4])
            for e in entities[:top]
        ],
        "entities_total": len(entities),
        "relations": relations.most_common(top),
        "rejected": store.rejected_reasons(conn).most_common(12),
    }


def entity_names(conn: Any, syn: Synonyms) -> list[str]:
    """Все сущности графа — для того, чтобы узнать их в вопросе чат-бота."""
    return list(graph(conn, syn).entities)
