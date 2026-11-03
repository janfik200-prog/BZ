"""Подписи рисунков, легенды и строки таблиц — первый этап графа, без модели (T6).

Часть мусорных фактов графа — не из прозы статьи: из подписи («Fig. 4 Back-scattered
electron images… h radial shape of uranophane» → «uranophane — содержит — radial
shape»), из легенды карты («Gn - galena; Mol - molybdenite» → «Gn — является признаком
— galena») и из строки таблицы («Stage 4, Regime = TF»). Здесь — классификатор куска
текста на четыре вида: CAPTION, LEGEND, TABLE_ROW, PROSE, — и отсев фрагмента, где прозы
почти нет (skip_reason). Отсев стоит в facts.run рядом с отсевом дублей и фрагментов,
где мало терминов (entities.MIN_TERMS): отсеянный фрагмент пишется в facts_pass.skipped
с причиной, graph --no-filter пройдёт его заново.

Как таблица и подпись попадают в chunks.text. Фрагмент — это элементы разбора Docling
через перевод строки: абзац, подпись, пункт списка, таблица — каждый своей строкой.
Таблицу HybridChunker пишет тройками «строка, столбец = значение.» подряд («Stage 3,
SH max = 154. Stage 3, Regime = NF.»), пустая ячейка — «= .». Подпись стоит отдельной
строкой и часто без точки в конце — тогда split_sentences склеивает её со следующим
абзацем в одно «предложение» («Fig. 21 Iron ore prospectivity map … The result from the
FR shows that…»). Поэтому вид определяется по строке Docling, а не только по
предложению; quote_kind для факта смотрит ещё, в каком куске цитаты названы оба конца.

Признаки:
* CAPTION — строка начинается меткой «Fig./Figure/Рис./Рисунок/Table/Таблица N», за
  которой не глагол («Figure 5 shows…», «Table 3 displays…» — проза, ссылка на рисунок).
  Метка в конце строки («…as can be seen Figure 13.») подписью не считается: в корпусе
  все такие строки — абзацы прозы;
* LEGEND — перечень «1 - …; 2 - …» (не меньше двух номеров, среди них соседние) или
  расшифровка сокращений «Gn - galena; Mol - molybdenite» (не меньше трёх);
* TABLE_ROW — тройки Docling «A, B = C» (по строке — две и больше; по куску — «шапка,
  столбец = текст» в начале), ряд «|», большая доля чисел.

Оценка на копии базы — research/v14/experiments/run_captions.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from ..text import mentions, split_sentences
from .facts import FRAGMENT_CHARS  # столько фрагмента видит модель — и отсев смотрит столько же

CAPTIONS_RULES = "captions-v14.t6"

CAPTION, LEGEND, TABLE_ROW, PROSE = "CAPTION", "LEGEND", "TABLE_ROW", "PROSE"
KINDS = (CAPTION, LEGEND, TABLE_ROW, PROSE)

# Фрагмент отсеивается, если на прозу и таблицы приходится не больше MIN_PROSE_SHARE
# знаков его предложений, а остальное — подписи и легенды. Выбор — на DEV копии базы
# (run_captions.py, results/captions_eval.json): с таблицами в отсев уходят верные факты
# (русские таблицы-сводки: «Рудопроявление Чулковское = Геохимические аномалии золота…»,
# 5 из 96 верных размеченных фактов DEV), без таблиц до порога 0,3 — ни одного.
SKIP_KINDS = frozenset({CAPTION, LEGEND})
MIN_PROSE_SHARE = 0.3

_LABEL = (
    r"(?:fig(?:ure)?s?\.?|рис(?:унок|унке|\.)?|табл(?:ица|\.)?|tables?|tab\.|"
    r"photos?|plates?|фото)"
)
_NUM = r"(?:S?\d+(?:[.,]\d+)*[a-zа-я]?|[IVX]+\b)"
_CAPTION_START = re.compile(rf"^[\W_]{{0,3}}{_LABEL}\s*{_NUM}", re.IGNORECASE)
# За меткой глагол — это проза, которая ссылается на рисунок: «Figure 5 shows that…».
_PROSE_AFTER_LABEL = re.compile(
    r"^[\s.:)]*(?:(?:and|и|,)\s*(?:fig\w*\.?|tables?|рис\.?|табл\.?)?\s*[\d,\s–-]*[a-zа-я]?\s*)?"
    r"(?:a\s+and\s+b\s+)?(?:\(\s*[a-zа-я]\s*\)\s*)?"
    r"(?:shows?|showing|presents?|illustrates?|displays?|depicts?|demonstrates?|summari[sz]es?|"
    r"compares?|reveals?|indicates?|highlights?|gives?|lists?|provides?|contains?|reports?|"
    r"outlines?|describes?|represents?|exhibits?|confirms?|suggests?|is|are|was|were|has|have|"
    r"also|clearly|further|below|above|shown|presented|"
    r"показ\w*|приведен\w*|представлен\w*|демонстрир\w*|иллюстрир\w*|отража\w*|содерж\w*|"
    r"дает|даёт|свидетельств\w*|видно|изображен\w*|хорошо|наглядно|также|можно)\b",
    re.IGNORECASE,
)
# Номер легенды: «1 - …», «12 – …», «1, 2 - …», «1-4 - …»; после тире — буква.
_ENUM = re.compile(r"(?:^|(?<=[\s;:,.(]))(\d{1,2})(?:\s*[-–,]\s*\d{1,2})?\s*[—–-]\s*(?=[^\W\d_])")
# Расшифровка сокращения: «Gn - galena», «Py – пирит».
_ABBR = re.compile(r"(?:^|(?<=[\s;:,(]))[A-ZА-ЯЁ][a-zа-яё]{0,3}\s+[—–-]\s+[a-zа-яё]{3,}")
# Тройка Docling «строка, столбец = значение».
_EQ = re.compile(r"\s=\s|\s=\s*$|\s=\.")
_TRIPLE_HEAD = re.compile(r"^(?P<head>[^=]{1,250}?,\s[^=]{1,250}?)\s=(?:\s|\.|$)\s*(?P<value>\S?)")
# Шапка таблицы — строка и столбец; длиннее — это предложение прозы с «=» внутри.
MAX_HEAD_WORDS = 30


def _formula(triple: re.Match[str]) -> bool:
    """«…при Т = 270-300 °С формировалась…» — формула в прозе, а не ячейка таблицы:
    перед «=» обозначение в 1–2 знака, после — число."""
    last = triple.group("head").split()[-1]
    return len(last.strip("()")) <= 2 and triple.group("value")[:1].isdigit()


_SHORT_HEAD = re.compile(r"^(?P<head>[^=]{1,90}?)\s=\s+(?P<value>\S)")
_PIPES = re.compile(r"\|")
_TOKEN = re.compile(r"\S+")
_NUMERIC = re.compile(r"^[\W_]*[-+±~<>]?\d[\d.,:/×x%°–\-]*[\W_]*$")


@dataclass(frozen=True)
class Verdict:
    kind: str
    rule: str


def _caption(line: str) -> str | None:
    match = _CAPTION_START.match(line)
    if match:
        rest = line[match.end() :]
        if not _PROSE_AFTER_LABEL.match(rest):
            return "caption:label-start"
    return None


def _legend(text: str) -> str | None:
    numbers = [int(m.group(1)) for m in _ENUM.finditer(text)]
    if len(numbers) >= 2 and any(b == a + 1 for a, b in pairwise(numbers)):
        return "legend:numbered"
    if len(_ABBR.findall(text)) >= 3:
        return "legend:abbreviations"
    return None


def _table(text: str, line_level: bool = False) -> str | None:
    if len(_PIPES.findall(text)) >= 3:
        return "table:pipes"
    eqs = len(_EQ.findall(text))
    if line_level and eqs >= 2:
        segments = [s for s in re.split(r"(?<=\.)\s+", text) if s.strip()]
        with_eq = sum(1 for s in segments if _EQ.search(s))
        if with_eq >= 2 and with_eq * 2 >= len(segments):
            return "table:triples"
    if eqs and not line_level:  # по строке — только тройки: абзац прозы с формулой не таблица
        triple = _TRIPLE_HEAD.match(text)
        if triple and len(triple.group("head").split()) <= MAX_HEAD_WORDS and not _formula(triple):
            return "table:row-col-value"
        short = _SHORT_HEAD.match(text)
        if short and len(short.group("head").split()) <= 8 and not short.group("value").isdigit():
            return "table:head-value"
    tokens = _TOKEN.findall(text)
    if len(tokens) >= 6 and sum(1 for t in tokens if _NUMERIC.match(t)) / len(tokens) >= 0.6:
        return "table:numbers"
    return None


def classify_verdict(text: str, line_level: bool = False) -> Verdict:
    """Вид куска текста и правило, которое сработало. line_level — кусок целиком
    строка Docling (элемент разбора): тогда тройки таблицы считаются по всей строке."""
    text = " ".join((text or "").split())
    if not text:
        return Verdict(PROSE, "empty")
    for kind, rule in (
        (CAPTION, _caption(text)),
        (TABLE_ROW, _table(text, line_level)),
        (LEGEND, _legend(text)),
    ):
        if rule:
            return Verdict(kind, rule)
    return Verdict(PROSE, "prose")


def classify(text: str) -> str:
    """Вид предложения или цитаты без контекста фрагмента: CAPTION / LEGEND /
    TABLE_ROW / PROSE."""
    return classify_verdict(text).kind


@dataclass(frozen=True)
class Piece:
    """Кусок цитаты внутри одной строки фрагмента и его вид."""

    text: str
    kind: str
    rule: str
    line_kind: str


def _line_spans(text: str) -> tuple[str, list[tuple[int, int, Verdict]]]:
    """Фрагмент → текст со схлопнутыми пробелами (как у split_sentences) и строки
    Docling в нём: (начало, конец, вид строки)."""
    spans: list[tuple[int, int, Verdict]] = []
    parts: list[str] = []
    pos = 0
    for raw in (text or "").split("\n"):
        line = " ".join(raw.split())
        if not line:
            continue
        spans.append((pos, pos + len(line), classify_verdict(line, line_level=True)))
        parts.append(line)
        pos += len(line) + 1
    return " ".join(parts), spans


def _combine(piece: str, line: Verdict) -> Piece:
    own = classify_verdict(piece)
    if own.kind != PROSE:
        return Piece(piece, own.kind, own.rule, line.kind)
    # Предложение внутри подписи или таблицы («…a uraninite, b coffinite. Note the…»)
    # без своей метки — вид строки. Легенда по строке — не переносится: в абзаце прозы
    # бывает перечень «1 - Голубинский, 2 - …», остальные предложения абзаца — проза.
    if line.kind in (CAPTION, TABLE_ROW):
        return Piece(piece, line.kind, "line:" + line.rule, line.kind)
    return Piece(piece, PROSE, own.rule, line.kind)


def pieces(quote: str, text: str | None = None) -> list[Piece]:
    """Цитата → куски по строкам фрагмента (и по предложениям внутри строки) с видом.
    Без фрагмента или цитаты нет во фрагменте — по предложениям самой цитаты."""
    flat = " ".join((quote or "").split())
    if not flat:
        return []
    if text:
        joined, spans = _line_spans(text)
        start = joined.find(flat)
        if start >= 0:
            end = start + len(flat)
            out: list[Piece] = []
            for s, e, verdict in spans:
                if e <= start or s >= end:
                    continue
                chunk = joined[max(s, start) : min(e, end)]
                for sentence in split_sentences(chunk) or [chunk]:
                    out.append(_combine(sentence, verdict))
            if out:
                return out
    return [_combine(s, Verdict(PROSE, "no-line")) for s in split_sentences(flat) or [flat]]


_ORDER = {CAPTION: 0, LEGEND: 1, TABLE_ROW: 2}


def quote_verdict(
    quote: str, text: str | None = None, src: str | None = None, dst: str | None = None
) -> Verdict:
    """Вид цитаты факта. Смотрятся куски, где названы оба конца (нет таких — хотя бы
    один, нет и таких — все). Есть среди них проза — PROSE: факт мог быть сказан в
    тексте статьи. Иначе — самый частый вид (при равенстве: подпись, легенда, таблица)."""
    found = pieces(quote, text)
    if not found:
        return Verdict(PROSE, "empty")
    candidates = found
    if src and dst:
        both = [p for p in found if mentions(src, p.text) and mentions(dst, p.text)]
        either = [p for p in found if mentions(src, p.text) or mentions(dst, p.text)]
        candidates = both or either or found
    prose = [p for p in candidates if p.kind == PROSE]
    if prose:
        return Verdict(PROSE, prose[0].rule)
    counts: dict[str, int] = {}
    for p in candidates:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    kind = min(counts, key=lambda k: (-counts[k], _ORDER.get(k, 9)))
    rule = next(p.rule for p in candidates if p.kind == kind)
    return Verdict(kind, rule)


def quote_kind(
    quote: str, text: str | None = None, src: str | None = None, dst: str | None = None
) -> str:
    """Вид цитаты факта: CAPTION / LEGEND / TABLE_ROW / PROSE (см. quote_verdict)."""
    return quote_verdict(quote, text, src, dst).kind


def sentence_kinds(text: str) -> list[str]:
    """Вид каждого предложения фрагмента — тех же и в том же порядке, что split_sentences
    (graph/facts.sentences отдаёт их модели под номерами)."""
    return [quote_kind(s, text) for s in split_sentences(text)]


def kind_chars(text: str) -> dict[str, int]:
    """Знаков в предложениях каждого вида — по той части фрагмента, что видит модель."""
    text = (text or "")[:FRAGMENT_CHARS]
    out = dict.fromkeys(KINDS, 0)
    for sentence, kind in zip(split_sentences(text), sentence_kinds(text), strict=True):
        out[kind] += len(sentence)
    return out


def skip_reason(text: str) -> str | None:
    """Причина не отдавать фрагмент модели — прозы в нём почти нет, только подписи,
    легенды и таблицы; None — фрагмент идёт к модели."""
    chars = kind_chars(text)
    total = sum(chars.values())
    if not total:
        return None
    if sum(v for k, v in chars.items() if k not in SKIP_KINDS) / total > MIN_PROSE_SHARE:
        return None
    parts = [
        f"{NAMES[k]} {round(100 * chars[k] / total)} %"
        for k in KINDS
        if chars[k] and k in SKIP_KINDS
    ]
    return "подписи, легенды и таблицы: " + ", ".join(parts)


NAMES = {CAPTION: "подписи", LEGEND: "легенды", TABLE_ROW: "таблицы", PROSE: "проза"}
