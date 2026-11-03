"""WP05. Справочник территорий: вложенность с основанием, синонимы с источником.

Территория — строка regional_registry.csv: id, каноническое имя, уровень, родители,
типы оруденения и геологическая обстановка (для обоснованных аналогий), основание
каждой связи включения (цитата статьи корпуса или административное деление) и статус
проверки. Синонимы — registry_aliases.csv: написание → id, язык, откуда взято.

Правила, которые справочник не нарушает (ТЗ WP05):

* включение r1 ⊂ r2 — только по записи справочника, не по сходству названий;
* одно написание у двух id — ALIAS_AMBIGUOUS, а не выбор «похожего»;
* две известные территории без включения — DISJOINT, если не записано, что они
  перекрываются (overlaps); перекрытие — OVERLAP, по нему DIRECT не выдаётся;
* id не выдумываются: свои внутренние (R:…), official_id — только с источником.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..text import normalize

EXACT = "EXACT"
PART_OF = "PART_OF"  # территория факта внутри территории вопроса
CONTAINS = "CONTAINS"  # территория факта шире территории вопроса
DISJOINT = "DISJOINT"
OVERLAP = "OVERLAP"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GeoArea:
    area_id: str
    name: str
    level: str
    parent_ids: tuple[str, ...] = ()
    country: str = ""
    deposit_types: tuple[str, ...] = ()
    settings: tuple[str, ...] = ()
    overlaps: tuple[str, ...] = ()
    basis: str = ""
    basis_doc_id: str = ""
    basis_quote: str = ""
    review_status: str = "CURATED_UNREVIEWED"


@dataclass(frozen=True)
class AliasHit:
    alias: str
    area_ids: tuple[str, ...]
    start: int
    end: int

    @property
    def ambiguous(self) -> bool:
        return len(self.area_ids) > 1


@dataclass
class GeoAreaRegistry:
    areas: dict[str, GeoArea] = field(default_factory=dict)
    aliases: dict[str, set[str]] = field(default_factory=dict)  # normalize(alias) → ids
    version: str = "empty"
    _patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list, repr=False)

    # --- сборка -------------------------------------------------------------------
    def add(self, area: GeoArea, aliases: Iterable[str] = ()) -> None:
        self.areas[area.area_id] = area
        for alias in (area.name, *aliases):
            self.add_alias(alias, area.area_id)

    def add_alias(self, alias: str, area_id: str) -> None:
        key = normalize(alias)
        if len(key) < 3:
            return
        self.aliases.setdefault(key, set()).add(area_id)
        self._patterns = []

    def _compiled(self) -> list[tuple[str, re.Pattern[str]]]:
        if not self._patterns:
            # Длинные первыми: «Central Eastern Desert» не распадается на «Eastern Desert».
            keys = sorted(self.aliases, key=len, reverse=True)
            self._patterns = [(k, _alias_pattern(k)) for k in keys]
        return self._patterns

    # --- поиск в тексте -------------------------------------------------------------
    def find(self, text: str) -> list[AliasHit]:
        """Территории, названные в тексте. Совпадение внутри более длинного («Egypt» в
        «Eastern Desert of Egypt») не считается отдельным; частично перекрытые остаются оба
        («Central Eastern Desert of Egypt» — и CED, и Восточная пустыня Египта)."""
        norm = normalize(text)
        found: list[AliasHit] = []
        for key, pattern in self._compiled():
            for m in pattern.finditer(norm):
                found.append(AliasHit(key, tuple(sorted(self.aliases[key])), m.start(), m.end()))
        hits = [
            h
            for h in found
            if not any(
                o is not h
                and o.start <= h.start
                and h.end <= o.end
                and (o.end - o.start) > (h.end - h.start)
                for o in found
            )
        ]
        unique = {(h.start, h.end, h.area_ids): h for h in hits}
        return sorted(unique.values(), key=lambda h: (h.start, h.end))

    def resolve_name(self, name: str | None) -> tuple[str, ...]:
        """id по имени целиком (вопрос: «Средний Урал»); нет — пусто."""
        if not name:
            return ()
        key = normalize(name)
        if key in self.aliases:
            return tuple(sorted(self.aliases[key]))
        hits = self.find(name)
        return hits[0].area_ids if len(hits) == 1 else ()

    # --- отношения -------------------------------------------------------------------
    def ancestors(self, area_id: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.areas[area_id].parent_ids) if area_id in self.areas else []
        while stack:
            a = stack.pop()
            if a in out or a not in self.areas:
                continue
            out.add(a)
            stack.extend(self.areas[a].parent_ids)
        return out

    def chain(self, area_id: str) -> list[str]:
        """Территория и все её предки — для основания решения."""
        return [area_id, *sorted(self.ancestors(area_id))]

    def overlapping(self, a: str, b: str) -> bool:
        """Записано ли перекрытие a и b или их предков (Восточная пустыня Египта и
        Аравийско-Нубийский щит — геологическая и географическая единицы)."""
        sa, sb = {a, *self.ancestors(a)}, {b, *self.ancestors(b)}
        for x in sa:
            area = self.areas.get(x)
            if area and set(area.overlaps) & sb:
                return True
        for x in sb:
            area = self.areas.get(x)
            if area and set(area.overlaps) & sa:
                return True
        return False

    def relation(self, claim_area: str, target_area: str) -> str:
        """Как территория факта относится к территории вопроса."""
        if claim_area not in self.areas or target_area not in self.areas:
            return UNKNOWN
        if claim_area == target_area:
            return EXACT
        if target_area in self.ancestors(claim_area):
            return PART_OF
        if claim_area in self.ancestors(target_area):
            return CONTAINS
        if self.overlapping(claim_area, target_area):
            return OVERLAP
        return DISJOINT

    def shared_settings(self, a: str, b: str) -> set[str]:
        """Общая геологическая обстановка (с учётом предков) — основание аналогии."""

        def settings(x: str) -> set[str]:
            out: set[str] = set()
            for y in (x, *self.ancestors(x)):
                if y in self.areas:
                    out |= set(self.areas[y].settings)
            return out

        return settings(a) & settings(b)

    def deposit_types(self, area_id: str) -> set[str]:
        area = self.areas.get(area_id)
        return set(area.deposit_types) if area else set()


# Падежные окончания русского имени собственного — закрытый список: «Россия» не
# находится в «россыпи», «Урал» — в «уральских» (прилагательное — отдельный синоним).
_RU_ENDINGS = (
    "ями|ами|ыми|ими|ого|его|ому|ему|ией|ией|ая|яя|ое|ее|ой|ей|ом|ем|ий|ый|ую|юю|ых|их|ым|им|"
    "ах|ях|ов|ев|ия|ии|ию|а|я|ы|и|у|ю|е|ь"
)


def _ru_word(word: str) -> str:
    from ..text import ru_stem

    stem = ru_stem(word, 3) if len(word) > 4 else word
    stems = {stem, word}
    return (
        "(?:"
        + "|".join(re.escape(x) for x in sorted(stems, key=len, reverse=True))
        + f")(?:{_RU_ENDINGS})?"
    )


def _alias_pattern(key: str) -> re.Pattern[str]:
    """Имя в любой форме, целыми словами. Латинское — с окончанием прилагательного или
    множественного числа («Egypt» → «Egyptian», «Ural» → «Urals»); русское — с падежным
    окончанием из закрытого списка."""
    words = [w for w in re.split(r"[\s\-]+", key) if w]
    if re.search(r"[а-я]", key):
        body = r"[\s\-]+".join(
            _ru_word(w) if re.search(r"[а-я]", w) else re.escape(w) for w in words
        )
        return re.compile(r"(?<![а-яa-z0-9])" + body + r"(?![а-яa-z0-9])")
    body = r"[\s\-]+".join(re.escape(w) for w in words)
    return re.compile(r"(?<![a-zа-я0-9])" + body + r"(?:s|es|ian|ean|an|n|ic)?(?![a-zа-я0-9])")


def _split(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split(";") if v.strip())


def load(areas_csv: Path | str, aliases_csv: Path | str | None = None) -> GeoAreaRegistry:
    """Справочник из CSV. Версия — отпечаток обоих файлов: решение воспроизводится по ней."""
    areas_path = Path(areas_csv)
    digest = hashlib.sha256(areas_path.read_bytes())
    reg = GeoAreaRegistry()
    with areas_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            area = GeoArea(
                area_id=row["area_id"].strip(),
                name=row["canonical_name"].strip(),
                level=row.get("level", "").strip(),
                parent_ids=_split(row.get("parent_ids", "")),
                country=row.get("country", "").strip(),
                deposit_types=_split(row.get("deposit_types", "")),
                settings=_split(row.get("settings", "")),
                overlaps=_split(row.get("overlaps", "")),
                basis=row.get("basis", "").strip(),
                basis_doc_id=row.get("basis_doc_id", "").strip(),
                basis_quote=row.get("basis_quote", "").strip(),
                review_status=row.get("review_status", "").strip() or "CURATED_UNREVIEWED",
            )
            reg.add(area)
    if aliases_csv is not None:
        alias_path = Path(aliases_csv)
        digest.update(alias_path.read_bytes())
        with alias_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["area_id"].strip() in reg.areas:
                    reg.add_alias(row["alias"], row["area_id"].strip())
    reg.version = "registry-" + digest.hexdigest()[:12]
    return reg


def check_integrity(reg: GeoAreaRegistry) -> list[str]:
    """Ошибки справочника: родитель не из справочника, цикл, связь без основания."""
    problems = []
    for area in reg.areas.values():
        for p in area.parent_ids:
            if p not in reg.areas:
                problems.append(f"{area.area_id}: родителя {p} нет в справочнике")
        if area.area_id in reg.ancestors(area.area_id):
            problems.append(f"{area.area_id}: цикл вложенности")
        if area.parent_ids and not area.basis:
            problems.append(f"{area.area_id}: включение без основания")
    return problems
