"""Детерминированный отбор без модели.

Нужен для полной автоматики. Если Ollama не поднята или ответила мусором, статьи
не должны вставать в очередь «нужен человек» — иначе ночной прогон бесполезен.
Тогда решение принимает эта эвристика: считает, сколько терминов темы встретилось
в названии и аннотации, и по доле совпадений отбирает статьи.

Морфология берётся из валидации (та же обработка склонений и беглой гласной, из-за
которой «рудный узел» находится в «рудного узла»). Плюс небольшой глоссарий:
по русской теме нужно искать и англоязычные работы, а без модели перевести запрос
больше нечем. Глоссарий пополняется — это обычный словарь внизу файла.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..validate import _entity_pattern, _normalize
from .models import Candidate

# Слова, которые ничего не добавляют к поиску.
STOPWORDS = {
    "и", "в", "на", "по", "с", "для", "из", "при", "о", "об", "от", "до", "как",
    "это", "их", "его", "или", "а", "но", "не", "the", "a", "of", "in", "on",
    "for", "and", "to", "with", "by", "from",
}

# Русско-английский глоссарий предметной области. Сначала ищутся фразы, потом слова.
# Значение — либо один термин, либо список: первый уходит в поисковый запрос,
# остальные считаются синонимами при отборе. Без синонимов статья вроде
# «Relationship of Geochemical Blocks and Ore Districts» проходит мимо темы.
GLOSSARY: dict[str, str | list[str]] = {
    # объекты и процессы
    "рудный узел": ["ore cluster", "ore district", "orefield", "metallogenic node"],
    "рудное поле": ["ore field", "orefield"],
    "рудный район": ["ore district", "mining district"],
    "оруденение": ["mineralization", "ore mineralization"],
    "минерализация": "mineralization",
    "месторождение": "deposit",
    "россыпь": "placer",
    "щит": "shield",
    "золото": "gold",
    "медь": "copper",
    "алмаз": "diamond",
    "кимберлит": "kimberlite",
    "окварцевание": "silicification",
    # методы
    "прогноз": ["prospectivity", "forecast", "prediction"],
    "прогнозирование": ["prospectivity mapping", "predictive mapping"],
    "выделение": ["delineation", "identification", "targeting"],
    "поиски": "exploration",
    "разведка": "exploration",
    "линеамент": "lineament",
    "дешифрирование": "interpretation",
    "дистанционное зондирование": "remote sensing",
    "космический снимок": "satellite imagery",
    "геохимия": "geochemistry",
    "геофизика": "geophysics",
    "магниторазведка": "magnetic survey",
    "электроразведка": "electrical survey",
    "кокригинг": "cokriging",
    "металлогения": "metallogeny",
    "перспективность": "prospectivity",
    "закономерности размещения": "spatial distribution",
    # географические названия. Без них тема «Анабарский щит металлогения»
    # переводилась в «shield metallogeny» — щиты всего мира вместо нашего.
    # Собственные имена нельзя выбрасывать: часто они и есть суть темы.
    "анабарский": "Anabar",
    "анабар": "Anabar",
    "оленёкский": "Olenek",
    "оленекский": "Olenek",
    "куонамский": "Kuonamka",
    "якутия": "Yakutia",
    "якутский": "Yakutian",
    "сибирская платформа": "Siberian Platform",
    "сибирский кратон": "Siberian Craton",
    "таймыр": "Taimyr",
    "енисейский кряж": "Yenisei Ridge",
}


# Общенаучные слова. Они есть в любой статье любой науки, поэтому совпадение по ним
# почти ничего не доказывает — вес снижен. Без этого тема «геофизические методы
# выделение перспективных площадей» принимала палеогеографию и философию: три слова
# из пяти совпали, а про руду в статье ни слова.
GENERIC = {
    "анализ", "методы", "метод", "исследование", "изучение", "оценка", "применение",
    "использование", "развитие", "особенности", "результаты", "данные", "модель",
    "моделирование", "подход", "площадей", "площади", "территории", "район",
    "analysis", "method", "methods", "study", "research", "assessment", "application",
    "results", "data", "model", "modeling", "approach", "area", "areas", "region",
}
GENERIC_WEIGHT = 0.35
DOMAIN_WEIGHT = 1.0
UNKNOWN_WEIGHT = 0.7

# Доменный фильтр: о чём вообще эта база. Статья, где нет ни одного такого слова
# ни в названии, ни в аннотации, к рудной тематике отношения не имеет, сколько бы
# общих слов темы в ней ни совпало. Проверено на живом прогоне: так отсеиваются
# «Прогноз развития маркетинговой парадигмы» и K-Means по инфракрасным спектрам.
DOMAIN_PATTERNS = [
    re.compile(p)
    for p in (
        r"руд[аыоуе]?\w*", r"оруденен\w*", r"минерализац\w*", r"месторожден\w*",
        r"металлоген\w*", r"рудоносн\w*", r"кимберлит\w*", r"золот\w*", r"медно\w*",
        r"алмаз\w*", r"россып\w*", r"полезных ископаемых", r"минераг\w*",
        r"\bore\b", r"\bores\b", r"mineraliz\w*", r"mineralis\w*", r"\bmineral\w*",
        r"metallogen\w*", r"prospectivity", r"orefield\w*", r"kimberlit\w*",
        r"\bdeposit\w*", r"\bgold\b", r"\bcopper\b", r"\bdiamond\w*", r"placer\w*",
        r"exploration target\w*", r"greenstone belt",
    )
]


def has_domain_term(text: str) -> bool:
    """Есть ли в тексте хоть одно слово рудной тематики."""
    lowered = _normalize(text)
    return any(p.search(lowered) for p in DOMAIN_PATTERNS)


def _english(value) -> list[str]:
    """Синонимы термина. Первый — основной, он попадает в поисковый запрос."""
    return [value] if isinstance(value, str) else list(value)


@dataclass
class TopicTerm:
    """Термин темы вместе с англоязычным соответствием, если оно известно."""

    source: str
    patterns: list[re.Pattern] = field(default_factory=list)
    weight: float = UNKNOWN_WEIGHT

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


def translate_topic(topic: str) -> str | None:
    """Грубый перевод темы по глоссарию. None, если переводить нечего.

    Составные термины берутся в кавычки. Без этого поисковик ищет слова по
    отдельности, и запрос `ore cluster` по русскому «рудный узел» приводит
    звёздные скопления в Орионе — проверено на живом прогоне.
    """
    text = _normalize(topic)

    # Фразы заменяем маркерами, чтобы они дальше не распались на слова.
    marks: dict[str, str] = {}
    for phrase in sorted(GLOSSARY, key=len, reverse=True):
        if " " not in phrase:
            continue
        pattern = _entity_pattern(phrase)
        if pattern.search(text):
            mark = f"\x00{len(marks)}\x00"
            marks[mark] = _english(GLOSSARY[phrase])[0]
            text = pattern.sub(f" {mark} ", text)

    terms: list[str] = []
    translated_any = False
    for word in text.split():
        if word in marks:
            terms.append(marks[word])
            translated_any = True
            continue
        if re.search(r"[a-z]", word):  # уже по-английски — оставляем как есть
            terms.append(word)
            continue
        for src, dst in GLOSSARY.items():
            if " " not in src and _entity_pattern(src).fullmatch(word):
                terms.append(_english(dst)[0])
                translated_any = True
                break
        # Непереведённые русские слова в англоязычный запрос не берём: смесь
        # «prospectivity gold анабарском» хуже короткого чистого запроса.

    terms = list(dict.fromkeys(terms))
    if not translated_any or len(terms) < 2:
        return None
    return " ".join(f'"{t}"' if " " in t else t for t in terms)


def topic_terms(topic: str) -> list[TopicTerm]:
    """Разбить тему на термины; у каждого — русский шаблон и, если есть, английский."""
    terms: list[TopicTerm] = []
    text = _normalize(topic)

    # Фразы из глоссария считаем одним термином.
    for phrase in sorted(GLOSSARY, key=len, reverse=True):
        if " " not in phrase:
            continue
        pattern = _entity_pattern(phrase)
        if pattern.search(text):
            terms.append(
                TopicTerm(
                    source=phrase,
                    patterns=[pattern]
                    + [re.compile(re.escape(e)) for e in _english(GLOSSARY[phrase])],
                    weight=DOMAIN_WEIGHT,
                )
            )
            text = pattern.sub(" ", text)

    for word in text.split():
        if len(word) < 3 or word in STOPWORDS:
            continue
        patterns = [_entity_pattern(word)]
        english = GLOSSARY.get(word)
        if not english:
            for src, dst in GLOSSARY.items():
                if " " not in src and _entity_pattern(src).fullmatch(word):
                    english = dst
                    break
        if english:
            patterns += [re.compile(re.escape(e)) for e in _english(english)]
        if word in GENERIC:
            weight = GENERIC_WEIGHT
        elif english:
            weight = DOMAIN_WEIGHT
        else:
            weight = UNKNOWN_WEIGHT
        terms.append(TopicTerm(source=word, patterns=patterns, weight=weight))

    return terms


@dataclass
class HeuristicFilter:
    """Отбор по доле совпавших терминов темы. Интерфейс как у OllamaLLM."""

    model: str = "эвристика"
    threshold: float = 0.5
    # Аннотацию в Crossref депонируют далеко не все издатели. Когда её нет, судить
    # приходится по одному названию — доказательств меньше, поэтому и порог ниже,
    # иначе половина статей по теме отсеется просто из-за пустого поля.
    threshold_title_only: float = 0.34
    title_bonus: float = 0.15
    require_domain: bool = True   # без единого слова рудной тематики статья не проходит
    # Совпадение по рудной терминологии — само по себе довод. Термины темы совпадают
    # не всегда буквально («metallogenic zoning» против «metallogenic node»), и без
    # этой прибавки статьи, которые мы потом разбирали руками, отсеивались в ноль.
    domain_bonus: float = 0.35

    def queries(self, topic: str, n: int = 5) -> list[str]:
        variants = [topic.strip()]
        english = translate_topic(topic)
        if english:
            variants.append(english)
        # Тема без служебных слов — иногда находит больше, чем целая фраза.
        trimmed = " ".join(w for w in _normalize(topic).split() if w not in STOPWORDS)
        if trimmed and trimmed not in {_normalize(v) for v in variants}:
            variants.append(trimmed)
        return list(dict.fromkeys(v for v in variants if v))[:n]

    def filter_batch(self, topic: str, batch: list[Candidate]) -> None:
        terms = topic_terms(topic)
        if not terms:  # тема из одних служебных слов — принимаем всё, решать нечем
            for cand in batch:
                cand.relevant = True
                cand.reason = "эвристика: тема слишком общая для отбора"
            return

        total = sum(t.weight for t in terms) or 1.0

        for cand in batch:
            title = _normalize(cand.title)
            full = f"{cand.title} {cand.abstract}"

            # Сначала доменный фильтр: нет ни одного слова про руду — дальше не смотрим.
            if self.require_domain and not has_domain_term(full):
                cand.relevant = False
                cand.reason = "эвристика: ни одного термина рудной тематики"
                continue

            normalized = _normalize(full)
            matched = [t for t in terms if t.matches(normalized)]
            in_title = sum(t.weight for t in terms if t.matches(title))

            score = (
                sum(t.weight for t in matched) / total
                + self.title_bonus * (in_title / total)
                + (self.domain_bonus if self.require_domain else 0.0)
            )
            has_abstract = bool(cand.abstract.strip())
            threshold = self.threshold if has_abstract else self.threshold_title_only
            cand.relevant = score >= threshold
            cand.reason = (
                f"эвристика: {len(matched)} из {len(terms)} терминов, вес {score:.2f}"
                + (f" ({', '.join(t.source for t in matched[:4])})" if matched else "")
                + ("" if has_abstract else ", только по названию")
            )
