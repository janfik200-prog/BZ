"""Доля мусора (georag/parse/validate.garbage_ratio): статья на любом письме — не мусор,
битый шрифт — мусор."""

from __future__ import annotations

import pytest

from georag.parse.validate import garbage_ratio

CLEAN = [
    "Линеаменты выделены по снимкам Landsat 8, угол падения 45° ± 5°.",
    "断层形成于海西期。后者在地表有较大的弯曲变形，断面倾角小。",
    "日本列島は植生が多く,浸食作用が卓越し,地質学的に変動帯である。",
    "Tự động tách chiết các yếu tố dạng tuyến từ ảnh viễn thám.",
    "การสกัดแนวเส้นจากข้อมูลการรับรู้ระยะไกล",
    "გად იშიფრებიან აერო- და კოსმოსურ სურათებზე",
    "Intégration des données de télédétection — géologie structurale.",
]

BROKEN = [
    pytest.param("(cid:34)(cid:12)(cid:90)(cid:7) (cid:55)(cid:3)(cid:44)", id="cid"),
    pytest.param(" текст", id="private-use"),
    pytest.param("Ð»Ð¸Ð½ÐµÐ°Ð¼ÐµÐ½Ñ‚Ñ‹", id="utf8-as-latin1"),
    pytest.param("■□■□▪▫■□ ◆◇◆◇ �����", id="symbols-and-replacement"),
]


@pytest.mark.parametrize("text", CLEAN)
def test_any_script_is_not_garbage(text: str) -> None:
    assert garbage_ratio(text) < 0.05


@pytest.mark.parametrize("text", BROKEN)
def test_broken_fonts_are_garbage(text: str) -> None:
    assert garbage_ratio(text) > 0.3
