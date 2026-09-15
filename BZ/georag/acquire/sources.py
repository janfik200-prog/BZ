"""Каталог источников: читаем sources.yaml, код не правим.

Три способа доступа, и разница между ними принципиальная:

  api    — официальный программный интерфейс. Можно дёргать по расписанию.
  html   — документ забирается по прямой ссылке; поиска по сайту нет.
  manual — сайту нужен человек: форма поиска, проверка на робота. В автоматику не идёт.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

API = "api"
HTML = "html"
MANUAL = "manual"


@dataclass
class SourceConfig:
    name: str
    mode: str = API
    provider: str = ""
    enabled: bool = False
    params: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def automatable(self) -> bool:
        return self.enabled and self.mode == API and bool(self.provider)


def _load_raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "для sources.yaml нужен pyyaml (pip install pyyaml) "
                "или положите каталог в sources.json"
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_sources(path: Path) -> list[SourceConfig]:
    if not path.exists():
        raise FileNotFoundError(f"каталог источников не найден: {path}")

    raw = _load_raw(path)
    items = raw.get("sources") if isinstance(raw, dict) else raw
    if not items:
        raise ValueError(f"в каталоге {path} нет ни одного источника")

    sources: list[SourceConfig] = []
    for item in items:
        sources.append(
            SourceConfig(
                name=item["name"],
                mode=item.get("mode", API),
                provider=item.get("provider", ""),
                enabled=bool(item.get("enabled", False)),
                params=item.get("params") or {},
                notes=(item.get("notes") or "").strip(),
            )
        )
    return sources
