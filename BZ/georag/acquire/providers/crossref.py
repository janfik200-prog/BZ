"""Поиск через Crossref — в том числе по конкретному издателю.

Зачем так, а не напрямую по сайту: у MDPI собственный поиск закрыт в robots.txt
(`Disallow: /search*`), официального поискового API у издателя нет, но все его статьи
зарегистрированы в Crossref (MDPI AG — участник 1968). Crossref отдаёт поиск по
словам, DOI и, что важно, прямые ссылки на полный текст в поле link. Скачивание
самого PDF с mdpi.com robots.txt не запрещает — запрещён только поиск.

Провайдер общий: сменив member в каталоге, тем же кодом ищем у любого другого
издателя, а убрав его — по всему Crossref.

Две особенности, которые ломают наивный код:

* аннотация приходит куском JATS XML (<jats:p>…</jats:p>), а не текстом — теги
  надо снимать, иначе LLM фильтрует статьи по разметке;
* в поле link лежат ссылки для text-mining, и у MDPI это обычная страница статьи
  с суффиксом /pdf. Если подходящей ссылки нет, она достраивается из адреса статьи.
"""

from __future__ import annotations

import re
import time

from ..models import Candidate

BASE = "https://api.crossref.org/works"

SELECT = ",".join(
    [
        "DOI",
        "title",
        "abstract",
        "author",
        "issued",
        "container-title",
        "link",
        "license",
        "publisher",
        "type",
        "URL",
    ]
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def abstract_from_jats(raw: str | None) -> str:
    """Вытащить текст аннотации из JATS-фрагмента Crossref."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&#x2010;", "-")
    )
    text = _WS_RE.sub(" ", text).strip()
    # Crossref часто оставляет слово Abstract в начале — оно не несёт смысла.
    return re.sub(r"^abstract[:\s]+", "", text, flags=re.IGNORECASE)


def pdf_urls_from_work(work: dict) -> list[str]:
    """Все известные адреса полного текста, по порядку предпочтения."""
    links = work.get("link") or []
    urls = [
        link.get("URL")
        for link in links
        if (link.get("content-type") or "").lower() == "application/pdf" and link.get("URL")
    ]

    # У MDPI полный текст лежит по адресу статьи с суффиксом /pdf.
    for link in links:
        url = link.get("URL") or ""
        if "mdpi.com" in url and "/pdf" not in url:
            urls.append(url.rstrip("/") + "/pdf")

    # Прочие ссылки оставляем в конце: вдруг отдадут файл, хотя тип указан иначе.
    urls.extend(link.get("URL") for link in links if link.get("URL"))
    return list(dict.fromkeys(u for u in urls if u))



def _year(work: dict) -> int | None:
    parts = ((work.get("issued") or {}).get("date-parts") or [[]])[0]
    return int(parts[0]) if parts and parts[0] else None


def _first(values, default=None):
    if isinstance(values, list):
        return values[0] if values else default
    return values or default


class CrossrefProvider:
    def __init__(self, name: str = "crossref", params: dict | None = None):
        self.name = name
        params = params or {}
        self.mailto = (params.get("mailto") or "").strip()
        self.member = params.get("member")           # 1968 — MDPI AG
        self.min_year = params.get("min_year")
        self.types = params.get("types") or ["journal-article"]
        self.only_with_fulltext = bool(params.get("only_with_fulltext", False))
        self.per_page = int(params.get("per_page") or 50)
        self.timeout = int(params.get("timeout") or 30)
        self.pause_sec = float(params.get("pause_sec") or 0.3)

    def _filters(self) -> str:
        parts = []
        if self.member:
            parts.append(f"member:{int(self.member)}")
        for kind in self.types:
            parts.append(f"type:{kind}")
        if self.min_year:
            parts.append(f"from-pub-date:{int(self.min_year)}-01-01")
        if self.only_with_fulltext:
            parts.append("has-full-text:true")
        return ",".join(parts)

    def _params(self, query: str, limit: int) -> dict:
        params = {
            "query.bibliographic": query,
            "select": SELECT,
            "rows": min(max(limit, 1), 100),
            "sort": "relevance",
            "order": "desc",
        }
        filters = self._filters()
        if filters:
            params["filter"] = filters
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def search(self, query: str, limit: int = 25) -> list[Candidate]:
        import requests

        params = self._params(query, limit)
        response = requests.get(
            BASE, params=params, timeout=self.timeout, headers={"User-Agent": self._user_agent()}
        )

        # Crossref отвергает запрос целиком, если в select попало поле, которого он
        # не отдаёт, — и отвечает 400 без объяснений. Повторяем без select: ответ
        # толще, зато приходит. Так один неудачный список полей не рушит источник.
        if response.status_code == 400 and "select" in params:
            params.pop("select")
            response = requests.get(
                BASE, params=params, timeout=self.timeout, headers={"User-Agent": self._user_agent()}
            )

        if self.pause_sec:
            time.sleep(self.pause_sec)
        response.raise_for_status()
        items = ((response.json() or {}).get("message") or {}).get("items") or []
        return [self._to_candidate(item, query) for item in items][:limit]

    def _user_agent(self) -> str:
        base = "georag/0.1 (knowledge base builder)"
        return f"{base} mailto:{self.mailto}" if self.mailto else base

    def _to_candidate(self, work: dict, query: str) -> Candidate:
        doi = work.get("DOI")
        licenses = work.get("license") or []
        urls = pdf_urls_from_work(work)
        return Candidate(
            source=self.name,
            external_id=doi or "",
            title=(_first(work.get("title")) or "").strip() or "(без названия)",
            abstract=abstract_from_jats(work.get("abstract")),
            authors=[
                " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
                for a in (work.get("author") or [])
                if a.get("family") or a.get("given")
            ][:10],
            year=_year(work),
            journal=_first(work.get("container-title")),
            doi=doi,
            language=work.get("language"),  # приходит не всегда, это нормально
            landing_page_url=work.get("URL") or (f"https://doi.org/{doi}" if doi else None),
            pdf_urls=urls,
            is_oa=bool(licenses),
            oa_status="crossref-license" if licenses else None,
            license=_first([lic.get("URL") for lic in licenses if lic.get("URL")]),
            query=query,
        )
