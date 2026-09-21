"""Контракт с базой данных проекта: запрос по шаблону → фрагменты с цитатами.

Шаблон один, потому что спрашивает программа, а не человек, — у неё поля
должны быть постоянными:

    {"concept": "гидротермальные изменения",   обязательно; код из kb.concept
     "territory": "Билляхская зона",           необязательно
     "method": "ASTER",                        необязательно
     "limit": 5,                               1–100, по умолчанию 5
     "per_doc": 2,                             фрагментов из одной статьи, 0 — без ограничения
     "checked_only": false}                    только подтверждённое моделью

Ответ — фрагменты статей, каждый с дословной цитатой, DOI, страницей,
похожестью на определение понятия и пометкой, чем найден. Текста от себя база
знаний не пишет: формулирует паспорт агент на стороне базы данных, а
подтверждает человек в её панели.

Всё здесь — без HTTP, чтобы проверялось без сервера. Сервер только
разбирает запрос и отдаёт то, что вернули эти функции.
"""

from __future__ import annotations

from ..index import db
from . import store
from .vocabulary import Vocabulary

API_VERSION = 1
LIMIT_DEFAULT = 5
LIMIT_MAX = 100


class RequestError(ValueError):
    """Запрос не по шаблону. Отдаётся как 400 с перечнем допустимого."""

    def __init__(self, message: str, detail: str = "", known: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.known = known or []

    def payload(self) -> dict:
        out = {"api_version": API_VERSION, "error": self.message}
        if self.detail:
            out["detail"] = self.detail
        if self.known:
            out["known"] = self.known
        return out


class NotBuilt(RuntimeError):
    """Разметка ещё не построена — отвечать нечем. Отдаётся как 503."""


FIELDS = {"api_version", "concept", "territory", "method", "limit", "per_doc", "checked_only"}


def _int(payload: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = payload.get(name, default)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise RequestError(f"{name}: нужно целое число", f"пришло {raw!r}") from None
    if not lo <= value <= hi:
        raise RequestError(f"{name}: от {lo} до {hi}", f"пришло {value}")
    return value


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "да", "yes"}


def parse_request(vocab: Vocabulary, payload: dict) -> dict:
    """Запрос → нормализованный вид. Имена приводятся к каноническим из словаря."""
    if not isinstance(payload, dict):
        raise RequestError("запрос должен быть объектом JSON")
    unknown = sorted(set(payload) - FIELDS)
    if unknown:
        raise RequestError("неизвестные поля запроса", ", ".join(unknown), sorted(FIELDS))
    version = payload.get("api_version", API_VERSION)
    if str(version) != str(API_VERSION):
        raise RequestError(f"версия контракта {version!r} не поддерживается",
                           known=[str(API_VERSION)])

    name = str(payload.get("concept") or "").strip()
    if not name:
        raise RequestError("не указано понятие (concept)", known=list(vocab.concepts))
    concept = vocab.concept(name)
    if concept is None:
        raise RequestError("понятия нет в словаре базы знаний",
                           f"«{name}» — коды должны совпадать с kb.concept",
                           list(vocab.concepts))

    territory = method = None
    if payload.get("territory"):
        term = vocab.territory(str(payload["territory"]))
        if term is None:
            raise RequestError("территории нет в словаре", str(payload["territory"]),
                               list(vocab.territories))
        territory = term.name
    if payload.get("method"):
        term = vocab.method(str(payload["method"]))
        if term is None:
            raise RequestError("метода нет в словаре", str(payload["method"]),
                               list(vocab.methods))
        method = term.name

    return {
        "concept": concept.code,
        "territory": territory,
        "method": method,
        "limit": _int(payload, "limit", LIMIT_DEFAULT, 1, LIMIT_MAX),
        "per_doc": _int(payload, "per_doc", 2, 0, LIMIT_MAX),
        "checked_only": _bool(payload.get("checked_only", False)),
    }


def _header(conn, vocab: Vocabulary) -> dict:
    meta = store.meta(conn)
    if not meta.get("built_at"):
        raise NotBuilt("разметка не построена: на машине с базой знаний выполнить graph")
    info = db.stats(conn)
    head = {
        "api_version": API_VERSION,
        "as_of": meta["built_at"],
        "corpus": {"documents": info["documents"], "chunks": info["chunks"]},
    }
    if meta.get("vocabulary_sha") != vocab.sha:
        head["warning"] = ("словарь изменён после разметки — ответ построен по старому; "
                           "на машине с базой знаний выполнить graph")
    return head


def concept_response(conn, vocab: Vocabulary, request: dict) -> dict:
    """Ответ на запрос по шаблону. `request` — уже прошедший parse_request."""
    out = _header(conn, vocab)
    concept = vocab.concepts[request["concept"]]
    fragments, totals = store.evidence_for(
        conn, concept.code,
        territory=request["territory"], method=request["method"],
        checked_only=request["checked_only"],
        limit=request["limit"], per_doc=request["per_doc"],
    )
    for f in fragments:
        f["doi_url"] = f"https://doi.org/{f['doi']}" if f.get("doi") else None
    out.update(
        request=request,
        concept={"code": concept.code, "kind": concept.kind, "definition": concept.definition},
        total=totals["fragments"],          # фрагментов под запрос, до ограничений показа
        documents=totals["documents"],      # в скольких статьях
        fragments=fragments,
        relations=_relations(conn, vocab, concept.code),
    )
    return out


def _relations(conn, vocab: Vocabulary, code: str) -> list[dict]:
    """Выведенные связи понятия: с какими территориями и методами оно описано вместе.

    Основание у каждой — фрагменты, которые описывают понятие и в которых
    названа территория или метод. Число фрагментов и статей — рядом.
    """
    out = []
    for link in store.concept_links(conn, code):
        relation = "проявлено_на" if link["kind"] == "территория" else "измеряет"
        out.append({
            "type": relation,
            "label": vocab.label(relation),
            "concept": code,
            "target": link["name"],
            "target_kind": link["kind"],
            "fragments": link["fragments"],
            "documents": link["documents"],
        })
    return out


def concepts_response(conn, vocab: Vocabulary) -> dict:
    """Что есть в базе знаний: словарь и сколько доказательств у каждого понятия.

    По этому ответу сторона базы данных может сверить свой kb.concept со
    словарём базы знаний до первого настоящего запроса.
    """
    out = _header(conn, vocab)
    summary = store.concept_summary(conn)
    empty = {"fragments": 0, "confirmed": 0, "unchecked": 0, "rejected": 0, "documents": 0}
    out["concepts"] = [
        {"code": c.code, "kind": c.kind, "definition": c.definition,
         "synonyms": list(c.synonyms), **summary.get(c.code, empty),
         "relations": _relations(conn, vocab, c.code)}
        for c in vocab.concepts.values()
    ]
    out["territories"] = list(vocab.territories)
    out["methods"] = list(vocab.methods)
    out["relation_types"] = [
        {"type": r.name, "from": r.source, "to": r.target, "label": r.label}
        for r in vocab.relations.values()
    ]
    return out


def graph_response(conn, vocab: Vocabulary, checked_only: bool = False) -> dict:
    """Граф связей словаря: узлы — понятия, территории, методы; рёбра — выведенные связи.

    Узлы словаря показываются все, даже без статей: пустое место на картинке —
    тоже ответ («по этой территории про это понятие в корпусе ничего нет»).
    """
    out = _header(conn, vocab)
    data = store.graph_data(conn, checked_only)
    nodes = []
    for c in vocab.concepts.values():
        weight = data["concepts"].get(c.code, {"fragments": 0, "documents": 0})
        nodes.append({"id": f"понятие:{c.code}", "kind": "понятие", "name": c.code,
                      "group": c.kind, **weight})
    for pool, kind in ((vocab.territories, "территория"), (vocab.methods, "метод")):
        for name in pool:
            nodes.append({"id": f"{kind}:{name}", "kind": kind, "name": name,
                          "documents": data["terms"].get((kind, name), 0)})
    edges = []
    for e in data["edges"]:
        if e["concept"] not in vocab.concepts:
            continue
        relation = "проявлено_на" if e["kind"] == "территория" else "измеряет"
        edges.append({
            "concept": f"понятие:{e['concept']}",
            "term": f"{e['kind']}:{e['name']}",
            "type": relation,
            "label": vocab.label(relation),
            "fragments": e["fragments"],
            "documents": e["documents"],
        })
    known = {n["id"] for n in nodes}
    out["checked_only"] = checked_only
    out["nodes"] = nodes
    out["edges"] = [e for e in edges if e["term"] in known]
    return out


def network_response(conn, vocab: Vocabulary, checked_only: bool = False,
                     max_entities: int = 200) -> dict:
    """Сеть знаний: узлы шести видов и рёбра «описывает» / «упоминает».

    Узел — {id, kind, name, …}; ребро — {source, target, type, weight}.
    Вес ребра — сколько фрагментов статьи за ним стоит.
    """
    out = _header(conn, vocab)
    data = store.network_data(conn, checked_only, max_entities)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}

    for d in data["documents"]:
        nodes[f"статья:{d['doc_id']}"] = {
            "id": f"статья:{d['doc_id']}", "kind": "статья", "name": d["title"] or d["doc_id"],
            "doc_id": d["doc_id"], "year": d["year"], "url": d["url"],
            "doi_url": f"https://doi.org/{d['doi']}" if d.get("doi") else None,
        }
    for c in vocab.concepts.values():
        nodes[f"понятие:{c.code}"] = {"id": f"понятие:{c.code}", "kind": "понятие",
                                      "name": c.code, "group": c.kind}

    def edge(doc_id: str, target: str, kind: str, weight: int, **extra) -> None:
        source = f"статья:{doc_id}"
        if source not in nodes or target not in nodes:
            return
        known = edges.get((source, target))
        if known:                       # одно и то же найдено и словарём, и правилами
            known["weight"] = max(known["weight"], weight)
            return
        edges[(source, target)] = {"source": source, "target": target, "type": kind,
                                   "weight": weight, **extra}

    for e in data["described"]:
        edge(e["doc_id"], f"понятие:{e['concept']}", "описывает", e["fragments"],
             confirmed=e["confirmed"])
    for t in data["tagged"]:
        node_id = f"{t['kind']}:{t['name']}"
        nodes.setdefault(node_id, {"id": node_id, "kind": t["kind"], "name": t["name"]})
        edge(t["doc_id"], node_id, "упоминает", t["fragments"])
    for m in data["mentioned"]:
        # «Анабарский щит», найденный правилами, — тот же узел, что территория
        # словаря: иначе на картинке два одинаковых узла рядом.
        same = vocab.territory(m["name"]) or vocab.method(m["name"])
        if same is not None:
            node_id = f"{same.kind}:{same.name}"
            nodes.setdefault(node_id, {"id": node_id, "kind": same.kind, "name": same.name})
        else:
            node_id = f"название:{m['key']}"
            nodes.setdefault(node_id, {"id": node_id, "kind": m["kind"], "name": m["name"]})
        edge(m["doc_id"], node_id, "упоминает", m["fragments"])

    edges = list(edges.values())
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    for node_id, node in nodes.items():
        node["degree"] = degree.get(node_id, 0)

    out["checked_only"] = checked_only
    out["nodes"] = list(nodes.values())
    out["edges"] = edges
    return out
