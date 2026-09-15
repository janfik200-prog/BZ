"""Поиск через OpenAlex.

Почему он: официальный открытый API без ключа и без проверок на робота, и при этом
индексирует русскую периодику с DOI — «Отечественная геология», «Руды и металлы»,
«Региональная геология и металлогения» там есть. Ключ не обязателен, но с ним дневной
лимит выше; e-mail в mailto переводит запросы в polite pool с более ровным временем ответа.

Две вещи, на которых обычно спотыкаются:

1. Аннотация приходит не текстом, а инвертированным индексом
   ({"слово": [позиции]}) — её надо собирать обратно, иначе фильтрация по аннотациям
   работает вслепую.
2. Наличие pdf_url не значит, что по ссылке лежит PDF: часто это страница статьи.
   Проверка делается уже при скачивании, по сигнатуре файла.
"""

from __future__ import annotations

import time

from ..models import Candidate

BASE = "https://api.openalex.org/works"

# Просим только нужные поля: ответ меньше, разбор быстрее.
SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_year",
        "language",
        "type",
        "authorships",
        "abstract_inverted_index",
        "open_access",
        "best_oa_location",
        "primary_location",
        "locations",  # все копии полного текста: репозитории, архивы, зеркала
    ]
)


def abstract_from_inverted_index(index: dict | None) -> str:
    """Собрать текст аннотации из инвертированного индекса OpenAlex."""
    if not index:
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in index.items():
        for pos in positions or []:
            slots.append((int(pos), word))
    slots.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in slots)


def _strip_prefix(value: str | None, prefix: str) -> str | None:
    if not value:
        return None
    return value[len(prefix):] if value.startswith(prefix) else value


class OpenAlexProvider:
    def __init__(self, name: str = "openalex", params: dict | None = None):
        self.name = name
        params = params or {}
        self.mailto = (params.get("mailto") or "").strip()
        self.api_key = (params.get("api_key") or "").strip()
        self.per_page = int(params.get("per_page") or 50)
        self.min_year = params.get("min_year")
        self.only_oa = bool(params.get("only_oa", False))
        self.types = params.get("types") or ["article", "review", "preprint"]
        self.timeout = int(params.get("timeout") or 30)
        self.pause_sec = float(params.get("pause_sec") or 0.3)

    # -- внутреннее -------------------------------------------------------- #
    def _filters(self) -> str:
        parts = []
        if self.min_year:
            parts.append(f"from_publication_date:{int(self.min_year)}-01-01")
        if self.only_oa:
            parts.append("is_oa:true")
        if self.types:
            parts.append("type:" + "|".join(self.types))  # | внутри фильтра — это ИЛИ
        return ",".join(parts)  # запятая между фильтрами — это И

    def _params(self, query: str, limit: int) -> dict:
        params = {
            "search": query,
            "select": SELECT,
            "per-page": min(max(limit, 1), 200),
        }
        filters = self._filters()
        if filters:
            params["filter"] = filters
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    # -- публичное --------------------------------------------------------- #
    def search(self, query: str, limit: int = 25) -> list[Candidate]:
        import requests

        def ask(text: str):
            return requests.get(
                BASE,
                params=self._params(text, limit),
                timeout=self.timeout,
                headers={"User-Agent": self._user_agent()},
            )

        response = ask(query)

        # Кавычки вокруг словосочетания OpenAlex принимает не всегда и отвечает
        # ошибкой на весь запрос. Кавычки нужны Crossref и MDPI — там без них
        # «ore cluster» ловит звёздные скопления, — поэтому не убираем их из
        # запроса, а повторяем здесь без них. Источник перестал молчать.
        if response.status_code >= 400 and '"' in query:
            response = ask(query.replace('"', " "))

        if self.pause_sec:
            time.sleep(self.pause_sec)  # вежливость: не упираемся в лимит запросов
        response.raise_for_status()
        payload = response.json()

        return [self._to_candidate(item, query) for item in (payload.get("results") or [])][:limit]

    def _user_agent(self) -> str:
        base = "georag/0.1 (knowledge base builder)"
        return f"{base} mailto:{self.mailto}" if self.mailto else base

    def _to_candidate(self, item: dict, query: str) -> Candidate:
        best_oa = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}
        oa = item.get("open_access") or {}

        # Собираем все известные адреса полного текста по порядку предпочтения:
        # сперва лучшая открытая копия, потом версия издателя, потом остальные
        # репозитории. Один заблокированный адрес не должен стоить нам статьи.
        candidates_urls = [
            best_oa.get("pdf_url"),
            oa.get("oa_url"),
            primary.get("pdf_url"),
            *[loc.get("pdf_url") for loc in (item.get("locations") or [])],
        ]
        urls = list(dict.fromkeys(u for u in candidates_urls if u))
        landing = (
            best_oa.get("landing_page_url")
            or primary.get("landing_page_url")
            or item.get("doi")
        )
        source_block = best_oa.get("source") or primary.get("source") or {}

        return Candidate(
            source=self.name,
            external_id=(item.get("id") or "").rsplit("/", 1)[-1],
            title=(item.get("title") or "").strip() or "(без названия)",
            abstract=abstract_from_inverted_index(item.get("abstract_inverted_index")),
            authors=[
                (a.get("author") or {}).get("display_name", "")
                for a in (item.get("authorships") or [])
                if (a.get("author") or {}).get("display_name")
            ][:10],
            year=item.get("publication_year"),
            journal=source_block.get("display_name"),
            doi=_strip_prefix(item.get("doi"), "https://doi.org/"),
            language=item.get("language"),
            landing_page_url=landing,
            pdf_urls=urls,
            is_oa=bool(oa.get("is_oa")),
            oa_status=oa.get("oa_status"),
            license=best_oa.get("license") or primary.get("license"),
            query=query,
        )
