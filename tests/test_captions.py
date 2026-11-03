"""Подписи рисунков, легенды и строки таблиц (graph/captions.py) и третий отсев первого
этапа графа (facts.run, GEORAG_SKIP_CAPTIONS)."""

from __future__ import annotations

from typing import Any

import pytest

from georag.graph import captions as C
from georag.graph import facts as F

TABLE = (
    "Stage 3, SH max = 154. Stage 3, Sh min = 64. Stage 3, Regime = NF. "
    "Stage 4, n = 26. Stage 4, Regime = TF"
)
LEGEND = (
    "Геологическое строение проявления Звонкое: 1 - четвертичные отложения; 2 - граниты "
    "второй фазы; 3 - разломы; 4 - кварцевые жилы"
)


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Fig. 4 Geological map of the Koldar massif", C.CAPTION),
        ("Рис. 2. Схема геологического строения Олимпиаднинского месторождения", C.CAPTION),
        ("Figure 10: Multi-Directional Hill Shade Maps generated from a DEM", C.CAPTION),
        ("Table 2 Comparative spectral capabilities of ASTER and Landsat-9", C.CAPTION),
        ("Figure 5 shows that chlorite is mapped in red.", C.PROSE),
        ("Table 3 displays the proposed paragenetic sequence.", C.PROSE),
        ("На рис. 3 приведено расположение вулканов.", C.PROSE),
        (LEGEND, C.LEGEND),
        ("Inclusions in gold Gn - galena; Mol - molybdenite; Py - pyrite; Qz - quartz.", C.LEGEND),
        ("Stage 3, Regime = NF.", C.TABLE_ROW),
        ("Коллизионная, Основные факторы = Активизация разломов.", C.TABLE_ROW),
        ("Чулковское = Линейные зоны развития золото-сульфидно-кварцевых руд.", C.TABLE_ROW),
        ("12.5 13.1 0.44 7.2 18.0 3.3 1.1", C.TABLE_ROW),
        ("Gold occurs in quartz veins hosted by schists of the belt.", C.PROSE),
        ("Запасы оценены в 2 - 6 г/т при мощности 1 - 3 м.", C.PROSE),
        # формула в прозе — не ячейка таблицы
        ("Минерализация, по данным включений, формировалась при Т = 270-300 °С.", C.PROSE),
        ("", C.PROSE),
    ],
)
def test_classify(text: str, kind: str) -> None:
    assert C.classify(text) == kind


def test_caption_line_glued_to_prose_by_splitter() -> None:
    """Подпись без точки split_sentences склеивает со следующим абзацем; вид — по строке
    Docling и по куску, где названы концы факта."""
    text = (
        "Fig. 21 Iron ore prospectivity map of the study area\nLineament density controls iron ore."
    )
    quote = (
        "Fig. 21 Iron ore prospectivity map of the study area Lineament density controls iron ore."
    )
    assert C.quote_kind(quote, text, "lineament density", "iron ore") == C.PROSE
    assert C.quote_kind(quote, text, "prospectivity map", "study area") == C.CAPTION


def test_sentence_inside_caption_line_takes_line_kind() -> None:
    text = "Gold is hosted by quartz veins.\nFig. 4 BSE images of uraninite. Note the radial shape of uranophane."
    kinds = C.sentence_kinds(text)
    assert kinds == [C.PROSE, C.CAPTION, C.CAPTION]


def test_legend_inside_prose_paragraph_does_not_spread() -> None:
    """Перечень «1 - …, 2 - …» в абзаце прозы — только своё предложение."""
    text = "Вулканы образуют цепь. Их расположение: 1 - Голубинский, 2 - Синебугровский."
    assert C.sentence_kinds(text) == [C.PROSE, C.LEGEND]


def test_formula_paragraph_is_not_table_line() -> None:
    text = (
        "Here, the threshold is set high, so that R = 0.45 and n = 26 for the model. Gold is rare."
    )
    assert C.sentence_kinds(text) == [C.PROSE, C.PROSE]


def test_skip_reason() -> None:
    captions_only = "Fig. 4 Geological map of the area\n" + LEGEND
    assert C.skip_reason(captions_only) is not None
    assert "легенды" in C.skip_reason(captions_only)
    # таблицы не отсеиваются: в них бывают верные факты
    assert C.skip_reason(TABLE) is None
    assert C.skip_reason("Gold occurs in quartz veins hosted by schists of the belt.") is None
    assert C.skip_reason("") is None
    # проза больше порога — фрагмент идёт к модели
    prose = "Gold occurs in quartz veins hosted by schists of the Tura belt near the fault zone."
    assert C.skip_reason(prose + "\nFig. 4 Map") is None


# --------------------------------------------------------------------------- #
#  Отсев в facts.run
# --------------------------------------------------------------------------- #
class _Found:
    entities = ["gold"]
    terms = ["gold", "quartz", "schist"]
    lang = "en"

    def enough(self, _need: int) -> bool:
        return True


class _Finder:
    def find(self, _text: str) -> _Found:
        return _Found()


class _LLM:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        self.calls += 1
        return {"факты": []}


class _Conn:
    def commit(self) -> None:
        pass


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from georag.graph import dedup, store

    log: dict[str, Any] = {"skipped": {}, "saved": []}
    rows = [
        (1, "d", "Fig. 4 Geological map of the area\n" + LEGEND),
        (2, "d", "Gold occurs in quartz veins."),
    ]
    monkeypatch.setattr(store, "todo", lambda *_a, **_k: rows)
    monkeypatch.setattr(
        store, "skip", lambda _c, cid, reason, _lang=None: log["skipped"].__setitem__(cid, reason)
    )
    monkeypatch.setattr(store, "save", lambda _c, cid, *_a, **_k: log["saved"].append(cid))
    monkeypatch.setattr(dedup, "find_duplicate", lambda *_a: None)
    monkeypatch.setattr(F, "load_rules", lambda *_a: "правила")
    return log


def test_run_skips_caption_fragment_when_enabled(pipeline: dict[str, Any]) -> None:
    llm = _LLM()
    stats = F.run(_Conn(), llm, finder=_Finder(), captions=True, log=lambda *_: None)
    assert stats["captions"] == 1
    assert list(pipeline["skipped"]) == [1] and "подписи" in pipeline["skipped"][1]
    assert pipeline["saved"] == [2] and llm.calls == 1


def test_run_unchanged_by_default(
    pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEORAG_SKIP_CAPTIONS", raising=False)
    llm = _LLM()
    stats = F.run(_Conn(), llm, finder=_Finder(), log=lambda *_: None)
    assert stats["captions"] == 0 and not pipeline["skipped"]
    assert pipeline["saved"] == [1, 2] and llm.calls == 2


def test_run_flag_from_environment(
    pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEORAG_SKIP_CAPTIONS", "1")
    stats = F.run(_Conn(), _LLM(), finder=_Finder(), log=lambda *_: None)
    assert stats["captions"] == 1


def test_no_filter_reprocesses(pipeline: dict[str, Any]) -> None:
    """graph --no-filter: отсев выключен, отсеянные раньше проходятся заново."""
    stats = F.run(
        _Conn(), _LLM(), finder=_Finder(), filtering=False, captions=True, log=lambda *_: None
    )
    assert stats["captions"] == 0 and pipeline["saved"] == [1, 2]
