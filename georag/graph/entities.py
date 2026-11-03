"""Первый этап графа: предметы фрагмента — на процессоре, без большой модели
(план графа v2, А2–А3).

Модель читала каждый фрагмент целиком и искала в нём всё сама; из двух третей
фрагментов она не выписывала ничего. Теперь до неё:

1. Язык фрагмента — по доле кириллицы и латиницы (language): русский, английский
   или смесь (тогда — обе модели spaCy).
2. Термины геологии — словарь config/geo-terms.txt, в любом падеже и числе
   (text.keyword_pattern). Общий NER их не знает: «пирит», «зона смятия» для него не
   сущности, — поэтому словарь главный.
3. Места и объекты — spaCy (ru_core_news_sm, en_core_web_sm): месторождения,
   районы, реки с названием, которых нет в словаре. spaCy не установлен — только
   словарь, с предупреждением.

Фрагмент, где терминов меньше MIN_TERMS, модели не отдаётся. Порог подобран на
4 272 фрагментах базы (27.10.2026), где известно, какие дали факты: при 3 терминах
к модели идёт 71 % фрагментов, из фрагментов с фактами — 94,8 % (96,4 % фактов);
при 4 — 63 % и 91,7 % (94 % фактов). «Пустые» фрагменты — тоже геология (медиана —
3 термина против 11), так что резче без потерь не отсеять. Найденное уходит модели
списком (facts.messages): она только определяет связи между ними.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import env
from ..paths import CONFIG
from ..text import keyword_pattern, normalize
from .facts import FRAGMENT_CHARS

TERMS_PATH = CONFIG / "geo-terms.txt"
MIN_TERMS = 3
# Список предметов в задании модели (facts.messages). По умолчанию — да, как было.
# GEORAG_GRAPH_ENTITY_LIST=0 — отсев первого этапа остаётся, а задание модели — как без
# первого этапа. Зачем: в E (research/v14) список снизил точность выписки (22/33 → 14/33);
# проверка варианта без списка — E3 (run_e3.py, T9).
ENTITY_LIST_ENV = "GEORAG_GRAPH_ENTITY_LIST"


def entity_list_enabled() -> bool:
    """Идёт ли список предметов первого этапа модели (флаг ENTITY_LIST_ENV, по умолчанию да)."""
    return env.flag(ENTITY_LIST_ENV, default=True)


MAX_ENTITIES = 40  # сколько предметов уходит модели списком
SPACY_MODELS = {"ru": "ru_core_news_sm", "en": "en_core_web_sm"}
# Метки spaCy, которые бывают предметами геологии: места, объекты, организации-
# месторождения («Kibali»), изделия («Landsat»). Людей, даты и числа — нет.
PLACE_LABELS = {"LOC", "GPE", "FAC", "ORG", "PRODUCT", "NORP"}
SPACY_CHARS = FRAGMENT_CHARS  # дальше модель всё равно не читает
MIXED_SHARE = 0.3  # доля букв меньшего алфавита, с которой фрагмент — смесь языков

_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)


def language(text: str) -> str:
    """«ru», «en» или «mixed» — по буквам фрагмента."""
    cyr, lat = len(_CYR.findall(text or "")), len(_LAT.findall(text or ""))
    if not cyr and not lat:
        return "en"
    if min(cyr, lat) / (cyr + lat) >= MIXED_SHARE:
        return "mixed"
    return "ru" if cyr > lat else "en"


def load_terms(path: Path | str | None = None) -> list[str]:
    """Термины из файла: строка — термин, # — комментарий."""
    text = Path(path or TERMS_PATH).read_text(encoding="utf-8-sig")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@dataclass
class Found:
    """Что первый этап нашёл во фрагменте."""

    lang: str
    terms: list[str] = field(default_factory=list)  # термины словаря, как в тексте
    names: list[str] = field(default_factory=list)  # места и объекты от spaCy

    @property
    def entities(self) -> list[str]:
        """Предметы списком для модели: без повторов, не больше MAX_ENTITIES."""
        out: list[str] = []
        seen: set[str] = set()
        for name in [*self.terms, *self.names]:
            key = normalize(name)
            if key and key not in seen:
                seen.add(key)
                out.append(name)
        return out[:MAX_ENTITIES]

    def enough(self, min_terms: int = MIN_TERMS) -> bool:
        return len({normalize(t) for t in self.terms}) >= min_terms


class Finder:
    """Ищет предметы во фрагменте. Словарь и модели spaCy грузятся один раз."""

    def __init__(self, terms_path: Path | str | None = None, use_spacy: bool = True) -> None:
        # Один большой шаблон: каждый термин — в любой форме, с начала слова. Длинные —
        # первыми: иначе «quartz» забирал бы начало «quartz veins».
        terms = sorted(load_terms(terms_path), key=len, reverse=True)
        self.pattern = re.compile("|".join(f"(?:{keyword_pattern(t).pattern})" for t in terms))
        self.use_spacy = use_spacy
        self._nlp: dict[str, Any] = {}

    def _spacy(self, lang: str) -> Any:
        if not self.use_spacy:
            return None
        if lang not in self._nlp:
            try:
                import spacy

                self._nlp[lang] = spacy.load(SPACY_MODELS[lang])
            except (ImportError, OSError) as exc:
                print(
                    f"spaCy ({SPACY_MODELS[lang]}) не загрузился: {exc} — предметы ищутся "
                    "только по словарю. Поставить: ЗАПУСК.md, «Граф».",
                    file=sys.stderr,
                )
                self._nlp[lang] = None
        return self._nlp[lang]

    def find(self, text: str) -> Found:
        text = text or ""
        found = Found(language(text))
        # Термины — по тексту в нижнем регистре (той же длины, что исходный), а в
        # список — как написано в статье.
        lowered = text.lower().replace("ё", "е")
        same = len(lowered) == len(text)  # «İ» в нижнем регистре — два знака
        found.terms = [
            (text if same else lowered)[m.start() : m.end()] for m in self.pattern.finditer(lowered)
        ]
        for lang in ("ru", "en") if found.lang == "mixed" else (found.lang,):
            nlp = self._spacy(lang)
            if nlp is None:
                continue
            for ent in nlp(text[:SPACY_CHARS]).ents:
                label = ent.label_.upper()
                if label in PLACE_LABELS and len(ent.text) >= 3:
                    found.names.append(ent.text.strip())
        return found
