"""Словарь базы знаний: понятия, территории, методы и типы связей.

Живёт в `vocabulary.yaml` в корне проекта и правится руками. Код отсюда только
читает его, проверяет и превращает синонимы в шаблоны поиска по тексту.

Понятия — копия справочника kb.concept из базы данных проекта. Коды обязаны
совпадать: по коду ответ базы знаний ложится в bridge.feature_concept на той
стороне, и расхождение в одной букве тихо выбрасывает связь из датасета.

Почему синонимы пишутся в любой форме, а не шаблонами. Словарь правит человек,
который не пишет регулярных выражений. Окончания русских слов код снимает сам:
«окварцевание» находит «окварцевания» и «окварцеванием», «Анабарский щит» —
«Анабарского щита». Длинные сложные слова при этом не цепляются: «разлом» не
найдёт «разломообразование» — хвост после основы ограничен длиной окончания.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "vocabulary.yaml"

CONCEPT_KINDS = ("рудоконтролирующий фактор", "условие наблюдения")
TERRITORY = "территория"
METHOD = "метод"

# Какие связи код умеет строить. Словарь может их переименовать для показа,
# но не может завести новую: её сперва надо научить строить.
SUPPORTED_RELATIONS: dict[tuple[str, str], str] = {
    ("фрагмент", "понятие"): "описывает",
    ("понятие", "территория"): "проявлено_на",
    ("метод", "понятие"): "измеряет",
}
DEFAULT_LABELS = {
    "описывает": "описывает",
    "проявлено_на": "проявлено на",
    "измеряет": "изучается методом",
}


class VocabularyError(ValueError):
    """Словарь заполнен так, что по нему нельзя работать."""


# --------------------------------------------------------------------------- #
#  Синоним → шаблон поиска
# --------------------------------------------------------------------------- #
# Окончания, которые снимаются с русского слова, чтобы получить основу.
# Длинные идут первыми: «-иями» раньше «-ями», иначе останется лишняя «и».
_RU_ENDINGS = sorted(
    [
        "иями", "ями", "ами", "ией", "иях", "ием", "ого", "его", "ому", "ему",
        "ыми", "ими", "ов", "ев", "ей", "ой", "ий", "ый", "ая", "яя", "ое", "ее",
        "ые", "ие", "ых", "их", "ую", "юю", "ом", "ем", "ам", "ям", "ах", "ях",
        "ия", "ии", "ию", "ья", "ье", "ьи", "ью",
        "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
    ],
    key=len,
    reverse=True,
)
_MIN_STEM = 3
_RU_TAIL = r"[а-яё]{0,4}"
_EN_TAIL = r"[a-z]{0,3}"
_JOIN = r"[\s\-–]+"
_ACRONYM = re.compile(r"[A-ZА-ЯЁ0-9]{2,6}")


def normalize(text: str) -> str:
    """Для сравнения имён: без регистра, «ё» как «е», пробелы схлопнуты."""
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _ru_stem(word: str) -> str:
    for ending in _RU_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= _MIN_STEM:
            return word[: -len(ending)]
    return word


def _word_pattern(word: str) -> str:
    if _ACRONYM.fullmatch(word):
        # Аббревиатура ищется как есть и с учётом регистра: «DEM» — да, «dem» — нет.
        return f"(?-i:{re.escape(word)})"
    low = word.lower().replace("ё", "е")
    if re.search(r"[a-z]", low):
        stem = low[:-1] if len(low) > 4 and low[-1] in "es" else low
        return re.escape(stem) + _EN_TAIL
    stem = _ru_stem(low)
    # В статьях «ё» пишут через раз: «обнажённость» и «обнаженность» — одно.
    return re.escape(stem).replace("е", "[её]") + _RU_TAIL


def term_pattern(phrase: str) -> str:
    """Шаблон фразы: слова подряд, у каждого снято окончание."""
    words = [w for w in re.split(r"[\s\-–]+", phrase.strip()) if w]
    if not words:
        raise VocabularyError("пустой синоним")
    body = _JOIN.join(_word_pattern(w) for w in words)
    return rf"(?<!\w){body}(?!\w)"


@dataclass(frozen=True)
class Matcher:
    """Набор синонимов одного имени, собранный в один шаблон."""

    name: str
    synonyms: tuple[str, ...]
    pattern: re.Pattern

    def found(self, text: str) -> list[str]:
        """Какие именно синонимы нашлись в тексте — для пояснения, почему."""
        if not self.pattern.search(text):
            return []
        return [s for s in self.synonyms
                if re.search(term_pattern(s), text, re.IGNORECASE | re.UNICODE)]


def _matcher(name: str, synonyms: list[str]) -> Matcher:
    seen: dict[str, str] = {}
    for item in [name, *synonyms]:
        item = re.sub(r"\s+", " ", str(item)).strip()
        if item and normalize(item) not in seen:
            seen[normalize(item)] = item
    unique = tuple(seen.values())
    combined = "|".join(f"(?:{term_pattern(s)})" for s in unique)
    return Matcher(name, unique, re.compile(combined, re.IGNORECASE | re.UNICODE))


# --------------------------------------------------------------------------- #
#  Словарь
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Concept:
    code: str
    kind: str
    definition: str
    matcher: Matcher

    @property
    def synonyms(self) -> tuple[str, ...]:
        return self.matcher.synonyms


@dataclass(frozen=True)
class Term:
    name: str
    kind: str          # территория | метод
    matcher: Matcher


@dataclass(frozen=True)
class RelationType:
    name: str
    source: str
    target: str
    label: str


@dataclass
class Vocabulary:
    concepts: dict[str, Concept]
    territories: dict[str, Term]
    methods: dict[str, Term]
    relations: dict[str, RelationType]
    sha: str = ""
    warnings: list[str] = field(default_factory=list)

    # -- поиск по имени, как его может прислать другая сторона ------------ #
    def concept(self, name: str) -> Concept | None:
        """Точное совпадение кода; без него — без регистра и с «е» вместо «ё»."""
        if name in self.concepts:
            return self.concepts[name]
        wanted = normalize(name)
        return next((c for c in self.concepts.values() if normalize(c.code) == wanted), None)

    def _term(self, pool: dict[str, Term], name: str) -> Term | None:
        wanted = normalize(name)
        for term in pool.values():
            if wanted in {normalize(s) for s in term.matcher.synonyms}:
                return term
        return None

    def territory(self, name: str) -> Term | None:
        return self._term(self.territories, name)

    def method(self, name: str) -> Term | None:
        return self._term(self.methods, name)

    def label(self, relation: str) -> str:
        rel = self.relations.get(relation)
        return rel.label if rel else DEFAULT_LABELS.get(relation, relation)

    def tags(self, text: str) -> list[tuple[str, str]]:
        """Территории и методы, названные в тексте: (вид, каноническое имя)."""
        out = []
        for pool in (self.territories, self.methods):
            for term in pool.values():
                if term.matcher.pattern.search(text):
                    out.append((term.kind, term.name))
        return out


def _as_list(value, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(v, (str, int, float)) for v in value):
        raise VocabularyError(f"{where}: нужен список слов, а стоит {value!r}")
    return [str(v) for v in value]


def parse(data: dict, sha: str = "") -> Vocabulary:
    """Словарь из уже прочитанного YAML. Ошибки — с указанием, где именно."""
    if not isinstance(data, dict):
        raise VocabularyError("файл словаря пуст или это не словарь YAML")
    unknown = set(data) - {"понятия", "территории", "методы", "связи"}
    if unknown:
        raise VocabularyError(f"неизвестные разделы: {', '.join(sorted(unknown))}")

    raw_concepts = data.get("понятия")
    if not isinstance(raw_concepts, dict) or not raw_concepts:
        raise VocabularyError("раздел «понятия» пуст — без понятий размечать нечего")

    concepts: dict[str, Concept] = {}
    seen_norm: dict[str, str] = {}
    for code, body in raw_concepts.items():
        code = str(code).strip()
        where = f"понятие «{code}»"
        if not isinstance(body, dict):
            raise VocabularyError(f"{where}: нужны поля род, определение, синонимы")
        kind = str(body.get("род") or "").strip()
        if kind not in CONCEPT_KINDS:
            raise VocabularyError(
                f"{where}: род «{kind}» — допустимы: {', '.join(CONCEPT_KINDS)}")
        definition = re.sub(r"\s+", " ", str(body.get("определение") or "")).strip()
        if len(definition) < 20:
            raise VocabularyError(
                f"{where}: нет определения — по нему ищутся похожие фрагменты")
        if normalize(code) in seen_norm:
            raise VocabularyError(
                f"{where}: совпадает с «{seen_norm[normalize(code)]}» с точностью до регистра или «ё»")
        seen_norm[normalize(code)] = code
        synonyms = _as_list(body.get("синонимы"), f"{where}, синонимы")
        concepts[code] = Concept(code, kind, definition, _matcher(code, synonyms))

    def terms(section: str, kind: str) -> dict[str, Term]:
        raw = data.get(section) or {}
        if not isinstance(raw, dict):
            raise VocabularyError(f"раздел «{section}»: нужен список «имя: [синонимы]»")
        return {
            str(name).strip(): Term(str(name).strip(), kind,
                                    _matcher(str(name).strip(),
                                             _as_list(syn, f"{section}, «{name}»")))
            for name, syn in raw.items()
        }

    territories = terms("территории", TERRITORY)
    methods = terms("методы", METHOD)

    relations: dict[str, RelationType] = {}
    raw_rel = data.get("связи") or {}
    if not isinstance(raw_rel, dict):
        raise VocabularyError("раздел «связи»: нужен список «имя: {от, к, подпись}»")
    for name, body in raw_rel.items():
        body = body or {}
        pair = (str(body.get("от") or ""), str(body.get("к") or ""))
        if pair not in SUPPORTED_RELATIONS:
            known = "; ".join(f"{a} → {b}" for a, b in SUPPORTED_RELATIONS)
            raise VocabularyError(
                f"связь «{name}»: {pair[0]} → {pair[1]} код строить не умеет. Умеет: {known}")
        canonical = SUPPORTED_RELATIONS[pair]
        relations[canonical] = RelationType(
            canonical, pair[0], pair[1],
            str(body.get("подпись") or DEFAULT_LABELS[canonical]).strip())

    vocab = Vocabulary(concepts, territories, methods, relations, sha)

    # Один синоним у двух понятий — не ошибка, но фрагмент уйдёт в оба: пусть видно.
    owners: dict[str, list[str]] = {}
    for concept in concepts.values():
        for syn in concept.synonyms:
            owners.setdefault(normalize(syn), []).append(concept.code)
    for syn, codes in owners.items():
        if len(set(codes)) > 1:
            vocab.warnings.append(f"синоним «{syn}» есть у нескольких понятий: {', '.join(codes)}")
    return vocab


def load(path: str | Path | None = None) -> Vocabulary:
    import yaml

    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        raise VocabularyError(f"нет файла словаря: {path}")
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw.decode("utf-8-sig"))
    except yaml.YAMLError as exc:
        raise VocabularyError(f"{path.name}: YAML не читается — {exc}") from exc
    return parse(data, sha=hashlib.sha256(raw).hexdigest()[:16])
