"""Реестр провайдеров поиска.

Добавить источник = положить рядом модуль с классом, у которого есть search(),
и зарегистрировать его здесь. Каталог sources.yaml ссылается на него по имени.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ..models import Candidate
from ..sources import SourceConfig


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[Candidate]:
        """Найти статьи по запросу. Полного текста может не быть — это нормально."""


def build_providers(sources: list[SourceConfig]) -> tuple[list[SearchProvider], list[str]]:
    """Поднимаем провайдеров для источников, которые можно опрашивать автоматически.

    Возвращаем также список пояснений по тем, что пропущены, — чтобы в отчёте было
    видно, почему источник не участвовал.
    """
    from .crossref import CrossrefProvider
    from .json_api import JsonApiProvider
    from .openalex import OpenAlexProvider

    registry: dict[str, Callable[[str, dict[str, Any]], SearchProvider]] = {
        "openalex": OpenAlexProvider,
        "crossref": CrossrefProvider,
        "json_api": JsonApiProvider,  # источник, описанный в каталоге, без кода
    }

    providers: list[SearchProvider] = []
    skipped: list[str] = []

    for src in sources:
        if not src.enabled:
            skipped.append(f"{src.name}: выключен в каталоге")
            continue
        if src.mode != "api":
            skipped.append(f"{src.name}: способ «{src.mode}» — автоматический поиск не делаем")
            continue
        factory = registry.get(src.provider)
        if factory is None:
            skipped.append(
                f"{src.name}: провайдер «{src.provider}» неизвестен; " f"есть {sorted(registry)}"
            )
            continue
        try:
            providers.append(factory(src.name, src.params))
        except Exception as exc:  # noqa: BLE001 — ошибка в каталоге, не в коде
            skipped.append(f"{src.name}: {exc}")

    return providers, skipped
