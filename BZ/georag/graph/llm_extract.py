"""Извлечение сущностей и связей моделью — поверх того, что нашли правила.

Что модель умеет, а правила нет:

* имя в любом порядке слов — «трубка Удачная», «месторождение Сухой Лог»;
* понимание, что «Тырныаузский рудный узел» и «Тырныаузское рудное поле» —
  про одно место (правила разводят их по разным опорным словам);
* и главное — **тип связи**. Правила знают только «упомянуты вместе».
  Модель говорит, что месторождение находится в пределах узла, а метод
  применён к площади. Вот это и есть граф, а не соседство по тексту.

Что модель не умеет: отвечать одинаково дважды. Поэтому правила остаются
костяком, а модель дополняет. Ключи считаются той же функцией, что и у правил,
поэтому найденное обоими способами схлопывается само.

Модель отвечает строго по тексту куска. Ей запрещено достраивать по своим
знаниям: в графе не должно быть месторождений, которых в статье нет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import COMMODITY, METHOD, OBJECT, Entity, _key

KINDS = {OBJECT, COMMODITY, METHOD}

SYSTEM = (
    "Ты размечаешь геологические статьи. Отвечай только JSON, без пояснений. "
    "Пиши только то, что есть в тексте; ничего не добавляй по своим знаниям."
)

USER = """Ниже фрагмент геологической статьи.

Выпиши из него:
1) названия — геологические объекты (рудные узлы, поля, районы, месторождения,
   рудопроявления, трубки, площади, зоны, массивы, щиты), полезные ископаемые
   и методы работ;
2) связи между ними — только те, что прямо следуют из текста.

Правила:
- имя пиши в именительном падеже, как принято в литературе;
- вид: «объект», «ископаемое» или «метод»;
- тип связи — короткий глагол или предлог из текста: «находится в»,
  «приурочено к», «применён к», «содержит», «входит в»;
- в связях используй те же имена, что выписал выше;
- ничего не выдумывай: если связь не написана в тексте, её нет.

Текст:
\"\"\"{text}\"\"\"

Верни JSON:
{{"entities": [{{"name": "...", "kind": "объект"}}],
  "relations": [{{"from": "...", "to": "...", "type": "..."}}]}}"""


@dataclass(frozen=True)
class Relation:
    from_key: str
    to_key: str
    type: str


def _clean_name(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .,;:«»\"'")


def parse_response(payload: dict) -> tuple[list[Entity], list[Relation]]:
    """Ответ модели → сущности и связи. Мусор молча отбрасывается."""
    entities: dict[str, Entity] = {}
    by_name: dict[str, str] = {}

    for item in (payload.get("entities") or []):
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name"))
        kind = _clean_name(item.get("kind")).lower()
        if len(name) < 3 or kind not in KINDS:
            continue
        key = _key(name, kind)
        entities.setdefault(key, Entity(key, name, kind))
        by_name[name.lower()] = key

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in (payload.get("relations") or []):
        if not isinstance(item, dict):
            continue
        source = by_name.get(_clean_name(item.get("from")).lower())
        target = by_name.get(_clean_name(item.get("to")).lower())
        kind = _clean_name(item.get("type")).lower()[:40]
        # Связь только между тем, что модель сама же и выписала: иначе она
        # приводит объекты, которых в тексте нет, и граф начинает врать.
        if not source or not target or source == target or not kind:
            continue
        mark = (source, target, kind)
        if mark not in seen:
            seen.add(mark)
            relations.append(Relation(source, target, kind))

    return list(entities.values()), relations


def extract_with_llm(llm, text: str) -> tuple[list[Entity], list[Relation]]:
    """Разметить кусок моделью. При любой ошибке — пусто, костяк уже собран правилами."""
    try:
        return parse_response(llm.chat_json(SYSTEM, USER.format(text=text[:4000])))
    except Exception:  # noqa: BLE001 — модель молчит или ответила не тем
        return [], []
