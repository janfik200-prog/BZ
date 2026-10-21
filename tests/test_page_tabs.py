"""Вкладки страницы (georag/web/index.html) — в одну строку и на узком экране: не влезли —
ряд листается вбок, а не переносит «Чат-бот» и «Граф связей» по слову на строку.
Снимок шириной 390 px с эмуляцией телефона проверялся вручную (headless Chrome)."""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "georag" / "web" / "index.html"


def rule(css: str, selector: str) -> str:
    """Тело первого правила с этим селектором."""
    found = re.search(rf"(?m)^\s*{re.escape(selector)}\s*\{{([^}}]*)\}}", css)
    assert found, selector
    return found.group(1)


def test_tabs_stay_in_one_row() -> None:
    css = PAGE.read_text(encoding="utf-8")
    tab = rule(css, ".tab")
    assert "white-space: nowrap" in tab and "flex: none" in tab
    tabs = rule(css, ".tabs")
    assert "overflow-x: auto" in tabs and "scrollbar-width: none" in tabs
    assert "display: none" in rule(css, ".tabs::-webkit-scrollbar")
    assert re.search(r"@media \(max-width: 420px\) \{ \.tab \{ padding: 8px 10px; \} \}", css)
