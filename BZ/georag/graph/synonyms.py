"""Словарь синонимов: разные написания одного и того же → одно имя.

Файл `config/synonyms.yaml`, правится Блокнотом (сохранять в UTF-8):

    имена:
      Донецкий бассейн: [Donetsk basin, Донбасс, Donbass]
      метод главных компонент: [PCA, principal component analysis]
    связи:
      приурочено к: [приурочен к, связано с разломами, associated with]
    обратные:
      входит в: [включает, состоит из, contains]

Первое — каноническое имя, в списке — как ещё пишут. Падеж и число не важны:
«Донецкого бассейна» и «Donetsk basins» узнаются сами (см. text.name_key).
Чего в словаре нет, в граф идёт как есть — как написала модель.

«обратные» — связи, которые говорят то же самое в другую сторону: «щит
включает зону» — это «зона входит в щит». Такой факт разворачивается при
чтении, иначе половина фактов «часть — целое» не попадала в датасеты.

Пополнять словарь помогает модель: `python georag.py synonyms` находит
похожие имена (по смыслу, через векторы BGE-M3), Qwen3 решает, одно ли это,
и предложения ложатся в `config/synonyms-предложения.yaml`. Нужное оттуда
переносится сюда руками — модель сама словарь не правит.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..text import name_key, normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "synonyms.yaml"
PROPOSALS_PATH = ROOT / "config" / "synonyms-предложения.yaml"

PART_OF = "входит в"            # связь «часть → целое»: по ней собирается датасет


class SynonymsError(ValueError):
    pass


@dataclass
class Synonyms:
    names: dict[str, str] = field(default_factory=dict)       # ключ → каноническое имя
    relations: dict[str, str] = field(default_factory=dict)   # ключ → каноническая связь
    inverse: dict[str, str] = field(default_factory=dict)     # ключ → связь, если развернуть
    groups: dict[str, list[str]] = field(default_factory=dict)  # каноническое → написания
    sha: str = ""
    warnings: list[str] = field(default_factory=list)

    def name(self, raw: str) -> str | None:
        """Каноническое имя из словаря или None, если такого в словаре нет."""
        return self.names.get(name_key(raw))

    def relation(self, raw: str) -> str:
        """Связь из словаря; нет в словаре — как написала модель, строчными."""
        return self.relations.get(name_key(raw)) or clean_relation(raw)

    def relation_dir(self, raw: str) -> tuple[str, bool]:
        """Связь и нужно ли развернуть факт: «включает» → («входит в», True)."""
        key = name_key(raw)
        if key in self.inverse and key not in self.relations:
            return self.inverse[key], True
        return self.relation(raw), False


def clean_relation(raw: str) -> str:
    text = " ".join(str(raw or "").replace("_", " ").split()).strip(" .,;:«»\"'").lower()
    return text


def _pairs(section, where: str):
    if section is None:
        return []
    if not isinstance(section, dict):
        raise SynonymsError(f"{where}: нужен список «имя: [синонимы]»")
    out = []
    for canon, syns in section.items():
        canon = " ".join(str(canon).split())
        if not canon:
            continue
        if syns is None:
            syns = []
        if isinstance(syns, str):
            syns = [syns]
        if not isinstance(syns, list):
            raise SynonymsError(f"{where} → «{canon}»: синонимы — списком в [квадратных скобках]")
        out.append((canon, [" ".join(str(s).split()) for s in syns if str(s).strip()]))
    return out


def parse(data: dict | None, sha: str = "") -> Synonyms:
    data = data or {}
    if not isinstance(data, dict):
        raise SynonymsError("файл должен начинаться с разделов «имена:» и «связи:»")
    unknown = set(data) - {"имена", "связи", "обратные"}
    if unknown:
        raise SynonymsError(f"неизвестный раздел: {', '.join(sorted(unknown))} "
                            "(бывают только «имена», «связи» и «обратные»)")
    syn = Synonyms(sha=sha)
    for section, target, clean in (("имена", syn.names, lambda s: s),
                                   ("связи", syn.relations, clean_relation),
                                   ("обратные", syn.inverse, clean_relation)):
        for canon, variants in _pairs(data.get(section), section):
            canon = clean(canon)
            if section == "имена":
                syn.groups[canon] = variants
            # В «обратных» сама связь не разворачивается — только её обратные написания.
            for variant in (variants if section == "обратные" else [canon, *variants]):
                key = name_key(variant)
                if not key:
                    continue
                if key in target and target[key] != canon:
                    syn.warnings.append(f"«{variant}» стоит и у «{target[key]}», и у «{canon}» — "
                                        f"взято первое")
                    continue
                target[key] = canon
    return syn


def load(path: str | Path | None = None) -> Synonyms:
    import yaml

    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return Synonyms()
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw.decode("utf-8-sig"))
    except yaml.YAMLError as exc:
        raise SynonymsError(f"{path.name}: не читается — {exc}") from exc
    return parse(data, sha=hashlib.sha256(raw).hexdigest()[:16])


_CACHED: dict = {}


def cached(path: str | Path | None = None) -> Synonyms:
    """Словарь, перечитанный, только если файл поменяли: для поиска на каждый
    запрос. Не читается — пустой словарь (поиск без синонимов, а не ошибка)."""
    path = Path(path or DEFAULT_PATH)
    mtime = path.stat().st_mtime if path.exists() else None
    if _CACHED.get("key") != (str(path), mtime):
        try:
            syn = load(path)
        except SynonymsError:
            syn = Synonyms()
        _CACHED.update(key=(str(path), mtime), syn=syn)
    return _CACHED["syn"]


# --------------------------------------------------------------------------- #
#  Синонимы в поиске
# --------------------------------------------------------------------------- #
def _spelling_pattern(spelled: str):
    from ..text import normalize, phrase_pattern

    # Сокращение (SAM, PCA, ДЗЗ) — только целым словом: иначе «SAM» находится в «sample».
    if spelled.isupper() and len(spelled) <= 6:
        return re.compile(r"(?<![а-яa-z0-9])" + re.escape(normalize(spelled)) + r"(?![а-яa-z0-9])")
    return phrase_pattern(spelled)


def synonym_variants(query: str, syn: Synonyms, limit: int = 6) -> list[str]:
    """Тот же запрос с другими написаниями из словаря.

    «золото Донбасса» → «золото donetsk basin», «золото donbass»: полнотекстовый
    поиск ищет слова буквально и англоязычную статью по русскому названию не
    находит. Вектор переводит сам, а точное название — нет."""
    from ..text import normalize

    text = normalize(query)
    out: list[str] = []
    for canon, variants in syn.groups.items():
        spellings = [canon, *variants]
        for spelled in spellings:
            if len(spelled) < 3:
                continue
            try:
                match = _spelling_pattern(spelled).search(text)
            except re.error:
                continue
            if not match:
                continue
            for other in spellings:
                if normalize(other) == normalize(spelled):
                    continue
                alt = f"{text[:match.start()]}{normalize(other)}{text[match.end():]}".strip()
                if alt != text and alt not in out:
                    out.append(alt)
            break
    return out[:limit]


# --------------------------------------------------------------------------- #
#  Предложения: модель находит одинаковое, человек решает
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = ("Ты сводишь в справочник названия из геологических статей. "
                "Отвечай только JSON.")

JUDGE_USER = """Для каждой пары скажи, одно ли это и то же: тот же объект, место, метод,
процесс или понятие, записанное иначе — перевод, сокращение, другое написание.
Часть и целое — НЕ одно и то же («Eastern Donbass» и «Donetsk basin» — разное).
Похожие, но разные вещи — НЕ одно и то же («Leimengou intrusion» и «Leimengou ore
district» — разное; «окварцевание» и «серицитизация» — разное).

{pairs}

Верни JSON: {{"одно": [номера пар, где одно и то же]}}"""

BATCH = 15


def judge_pairs(llm, pairs: list[tuple[str, str]]) -> list[bool]:
    """Qwen3 решает по парам, одно ли это. Не ответила — «нет» (ничего не сводим)."""
    out: list[bool] = []
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        text = "\n".join(f"{i}. «{a}» — «{b}»" for i, (a, b) in enumerate(chunk, start=1))
        try:
            data = llm.chat_json(JUDGE_SYSTEM, JUDGE_USER.format(pairs=text))
            same = {int(m) for m in re.findall(r"\d+", str(data.get("одно") or []))}
        except Exception:  # noqa: BLE001
            same = set()
        out += [i in same for i in range(1, len(chunk) + 1)]
    return out


def _is_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def suggest(names: dict[str, tuple[str, int]], syn: Synonyms, embed, judge,
            threshold: float = 0.8, max_pairs: int = 400) -> list[dict]:
    """Похожие имена → группы «одно и то же».

    names — {ключ: (как пишется чаще всего, сколько раз)}. embed(список строк)
    → векторы (нормированные), judge(пары) → [да/нет]. Уже сведённое словарём
    не предлагается; новое, похожее на имя из словаря, предлагается к нему.
    """
    keys = list(names)
    if len(keys) < 2:
        return []
    shown = [syn.names.get(k) or names[k][0] for k in keys]
    import numpy as np

    vectors = np.asarray(embed(shown), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    scores = vectors @ vectors.T
    pairs = []
    for i, j in zip(*np.nonzero(np.triu(scores, k=1) >= threshold)):
        a, b = shown[i], shown[j]
        if normalize(a) == normalize(b):
            continue                          # уже одно имя (сведены словарём)
        pairs.append((float(scores[i, j]), int(i), int(j)))
    pairs.sort(reverse=True)
    pairs = pairs[:max_pairs]
    verdicts = judge([(shown[i], shown[j]) for _, i, j in pairs]) if pairs else []

    parent = list(range(len(keys)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (_, i, j), same in zip(pairs, verdicts):
        if same:
            parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(keys)):
        groups.setdefault(root(i), []).append(i)
    out = []
    for members in groups.values():
        variants = sorted({shown[i] for i in members})
        if len(variants) < 2:
            continue
        known = [v for v in variants if v in syn.groups]
        if known:
            canon = known[0]
        else:
            count = {shown[i]: names[keys[i]][1] for i in members}
            canon = max(variants, key=lambda v: (_is_cyrillic(v), count.get(v, 0), -len(v)))
        out.append({"canon": canon, "variants": [v for v in variants if v != canon],
                    "count": sum(names[keys[i]][1] for i in members), "known": bool(known)})
    out.sort(key=lambda g: -g["count"])
    return out


def proposals_text(names: list[dict], relations: list[dict]) -> str:
    """Предложения в том же виде, что словарь: нужные строки — копировать в synonyms.yaml."""
    lines = [
        "# ПРЕДЛОЖЕНИЯ В СЛОВАРЬ СИНОНИМОВ — их сделала модель, словарь она не трогает.",
        "#",
        "# Просмотрите и перенесите нужные строки в config/synonyms.yaml, в тот же",
        "# раздел («имена» или «связи»). Первое — каноническое имя (его можно заменить),",
        "# в скобках — как ещё пишут. Ошибочные — просто не переносите.",
        "# «(уже в словаре)» — к имени из словаря нашлись новые написания: допишите их",
        "# в его список.",
        "",
    ]
    for title, groups in (("имена", names), ("связи", relations)):
        lines.append(f"{title}:")
        if not groups:
            lines.append("  # предложений нет")
        for g in groups:
            note = "  # уже в словаре" if g.get("known") else ""
            quoted = ", ".join(_yaml_str(v) for v in g["variants"])
            lines.append(f"  {_yaml_str(g['canon'])}: [{quoted}]{note}")
        lines.append("")
    return "\n".join(lines)


def _yaml_str(text: str) -> str:
    if re.search(r"[:#\[\]{},&*!|>'\"%@`]", text) or text != text.strip():
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text
