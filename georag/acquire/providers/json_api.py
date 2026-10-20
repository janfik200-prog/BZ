"""Источник, описанный в каталоге, а не в коде.

Этот провайдер существует, чтобы добавление источника не требовало ни строчки Python.
В sources.yaml описывается, куда делать запрос и где в ответе лежат поля, — и всё.

Как читаются пути в map:

    title                     item["title"]
    externalIds.DOI           item["externalIds"]["DOI"]
    authors[].name            [a["name"] for a in item["authors"]]
    openAccessPdf.url|link    первый непустой из двух путей
    title                     если по пути оказался список, берётся первый элемент

Понимает и адрес с подстановкой: если в url есть {query}, запрос уходит в путь,
а не в параметры — так устроены некоторые API.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

from ..models import Candidate

FIELDS = (
    "external_id",
    "title",
    "abstract",
    "authors",
    "year",
    "journal",
    "doi",
    "language",
    "landing_page_url",
    "pdf_url",
)


def _dig(node: Any, parts: list[str]) -> Any:
    if node is None or not parts:
        return node
    part, rest = parts[0], parts[1:]

    if part.endswith("[]"):
        key = part[:-2]
        seq = node.get(key) if isinstance(node, dict) else node
        if not isinstance(seq, list):
            return None
        values = [_dig(element, rest) for element in seq]
        return [v for v in values if v not in (None, "", [])]

    if isinstance(node, list):  # по пути оказался список — берём первый элемент
        return _dig(node[0], parts) if node else None
    if not isinstance(node, dict):
        return None
    return _dig(node.get(part), rest)


def pick(item: Any, spec: str | None) -> Any:
    """Значение по пути. Несколько путей через | — берётся первый непустой."""
    for path in (spec or "").split("|"):
        path = path.strip()
        if not path:
            continue
        value = _dig(item, path.split("."))
        if value not in (None, "", []):
            return value
    return None


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def _year(value: Any) -> int | None:
    text = _text(value)
    digits = "".join(ch for ch in text[:4] if ch.isdigit())
    return int(digits) if len(digits) == 4 else None


def _strip_doi(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    for prefix in ("https://doi.org/", "http://dx.doi.org/", "doi:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


class JsonApiProvider:
    """Любой поисковый API, отдающий JSON, описанный параметрами из каталога."""

    def __init__(self, name: str = "json_api", params: dict[str, Any] | None = None):
        params = params or {}
        if not params.get("url"):
            raise ValueError(f"источнику «{name}» нужен параметр url")

        self.name = name
        self.url = params["url"]
        self.query_param = params.get("query_param", "q")
        self.limit_param = params.get("limit_param")
        self.extra_params = dict(params.get("extra_params") or {})
        self.headers = dict(params.get("headers") or {})
        self.items_path = params.get("items_path", "")
        self.map = dict(params.get("map") or {})
        self.timeout = int(params.get("timeout") or 30)
        self.pause_sec = float(params.get("pause_sec") or 0.3)

        unknown = set(self.map) - set(FIELDS)
        if unknown:
            raise ValueError(
                f"источник «{name}»: в map неизвестные поля {sorted(unknown)}; "
                f"допустимы {list(FIELDS)}"
            )

    def search(self, query: str, limit: int = 25) -> list[Candidate]:
        import requests

        url = self.url
        params = dict(self.extra_params)
        if "{query}" in url:
            url = url.replace("{query}", quote(query))
        else:
            params[self.query_param] = query
        if self.limit_param:
            params[self.limit_param] = limit

        response = requests.get(
            url,
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": "georag/0.1 (knowledge base builder)", **self.headers},
        )
        if self.pause_sec:
            time.sleep(self.pause_sec)
        response.raise_for_status()

        payload = response.json()
        items = pick(payload, self.items_path) if self.items_path else payload
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return []

        return [self._to_candidate(item, query) for item in items][:limit]

    def _to_candidate(self, item: dict[str, Any], query: str) -> Candidate:
        authors = pick(item, self.map.get("authors"))
        if not isinstance(authors, list):
            authors = [authors] if authors else []

        pdf = pick(item, self.map.get("pdf_url"))
        pdf_urls = [_text(u) for u in pdf] if isinstance(pdf, list) else [_text(pdf)]
        pdf_urls = [u for u in pdf_urls if u.startswith("http")]

        doi = _strip_doi(pick(item, self.map.get("doi")))
        external_id = _text(pick(item, self.map.get("external_id"))) or doi or ""

        return Candidate(
            source=self.name,
            external_id=external_id,
            title=_text(pick(item, self.map.get("title"))) or "(без названия)",
            abstract=_text(pick(item, self.map.get("abstract"))),
            authors=[_text(a) for a in authors][:10],
            year=_year(pick(item, self.map.get("year"))),
            journal=_text(pick(item, self.map.get("journal"))) or None,
            doi=doi,
            language=_text(pick(item, self.map.get("language"))) or None,
            landing_page_url=_text(pick(item, self.map.get("landing_page_url"))) or None,
            pdf_urls=pdf_urls,
            is_oa=bool(pdf_urls),
            query=query,
        )
