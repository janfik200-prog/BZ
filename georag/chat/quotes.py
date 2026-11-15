"""Сверка выписок с фрагментами — проверка ответа у промта «выписки» (messages.QUOTE_SYSTEM).

Модель отвечает не пересказом, а выписками: номер фрагмента, за ним — текст из него. Значит,
проверить ответ может код, без модели: каждое предложение выписки должно стоять в своём
фрагменте. Чего во фрагменте нет — слова модели, а не источника — убирается; выписка без
номера фрагмента — тоже. Нашлось в другом фрагменте — остаётся: это тоже текст базы.

Перевод иноязычного фрагмента дословно не сверить: такие выписки остаются как есть.
Служебные фразы промта («В предоставленных фрагментах есть частные примеры:», «Не раскрыта
часть вопроса о…») — не выписки и не сверяются.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .checks import (
    _BULLET,
    _CITE,
    _HEADING,
    _formula_lines,
    _split_sentences,
    _with_cite,
    is_claim,
    is_foreign,
    rewrite,
)
from .messages import fragment_block

MIN_CHARS = 40  # предложение короче не сверяется: «См. рис. 7.», обрывок перечня
MIN_RUN = 3  # совпадение считается от трёх слов подряд
SUPPORTED = 0.8  # доля слов предложения, стоящих во фрагменте теми же цепочками

# Фраза о самих фрагментах или о вопросе, а не выписка.
_META = re.compile(
    r"фрагмент|предпосылк|не\s+раскрыт|нет\s+(?:ответа|сведений|данных|информации)|"
    r"частные\s+примеры|в\s+базе",
    re.IGNORECASE,
)
_TRANSLATION = re.compile(r"перевод", re.IGNORECASE)
# Раздел своими словами (правило 11 QUOTE_SYSTEM): с него и до конца ответа — не выписки.
_SYNTHESIS = re.compile(r"сопоставлени\w*\s+по\s+выписк", re.IGNORECASE)
_MARK = re.compile(r"^\s*перевод\s+фрагмента\s*[:.—-]?\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+")
STEM = 5  # слова сравниваются по началу: модель правит падежи («природный концентраций» →
# «природных концентраций» в словаре, где сокращения раскрыты без согласования)
# Разметка вокруг выписки: цитата «>», жирный, кавычки-ёлочки по краям.
_MARKUP = re.compile(r"[*_`>#]+")


def _tokens(text: str) -> list[str]:
    plain = _MARKUP.sub(" ", text.lower().replace("ё", "е").replace("­", ""))
    return [w[:STEM] for w in _WORD.findall(plain)]


def _share(words: list[str], hay: list[str]) -> float:
    """Доля слов предложения, которые стоят во фрагменте цепочками от MIN_RUN слов."""
    if not words:
        return 1.0
    need = min(MIN_RUN, len(words))
    blocks = SequenceMatcher(None, words, hay, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= need) / len(words)


def _is_header(line: str, n: int, titles: dict[int, list[str]]) -> bool:
    """Строка — заголовок выписки: номер фрагмента и его название, как в заголовке фрагмента.
    Его не режут на предложения: «… Часть I. Постановка … (2025, с. 12, 13)» распадался на
    обрывки «Часть I.» и «12, 13)»."""
    rest = _tokens(_CITE.sub(" ", line, count=1))
    title = titles.get(n) or []
    return bool(title) and rest[: min(4, len(title))] == title[:4]


def split_synthesis(answer: str) -> tuple[str, str]:
    """Ответ → (выписки, раздел «Сопоставление по выпискам» с заголовком); раздела нет —
    второе пусто."""
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        if _SYNTHESIS.search(line) and len(_CITE.sub("", line).strip(" *#:")) < 60:
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i:]).strip()
    return answer, ""


def check_synthesis(section: str, sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Раздел сопоставления без предложений, у которых нет ссылки на показанный фрагмент:
    (текст, убранное). Смысл сверяет модель (support.check_support) — это делает answer.py."""
    if not section:
        return "", []
    known = {s["n"] for s in sources}
    removed: list[str] = []
    lines = section.split("\n")
    out = [lines[0]]  # заголовок
    for line in lines[1:]:
        if not line.strip():
            out.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        kept = []
        for part in _split_sentences(line[len(prefix) :]):
            bare = _CITE.sub("", part).strip()
            if not bare.strip(" .;,"):
                continue  # строка из одних номеров «[7][8]» — не предложение
            numbers = {
                int(x) for g in _CITE.findall(part) for x in re.split(r"[,;]", g) if x.strip()
            }
            if len(bare) <= MIN_CHARS or (numbers and numbers <= known):
                kept.append(part)
            elif _META.search(bare) and not numbers:
                kept.append(part)  # «для сравнения нет сведений по …»
            else:
                removed.append(part)
        if kept:
            out.append(prefix + " ".join(kept))
    body = [x for x in out[1:] if x.strip()]
    return ("\n".join(out).strip() if body else ""), removed


def check_quotes(
    answer: str, sources: list[dict[str, Any]]
) -> tuple[str, list[str], tuple[int, int]]:
    """Ответ без того, чего нет во фрагментах: (текст, убранное, (подтверждено, сверено))."""
    seen = {s["n"]: fragment_block(s) for s in sources}
    hay = {n: _tokens(block) for n, block in seen.items()}
    # Тело фрагмента — без строки заголовка: выписка из одного заголовка — не ответ.
    body = {n: _tokens(block.split("\n", 1)[-1]) for n, block in seen.items()}
    foreign = {n: is_foreign(block.split("\n", 1)[-1]) for n, block in seen.items()}
    titles = {s["n"]: _tokens(str(s.get("title") or "")) for s in sources}
    removed: list[str] = []
    # Та же выписка второй раз — не нужна. Повторы ищет здесь, а не drop_repeats: тот
    # резал повторённый заголовок статьи («[7] «Сухой Лог. Часть I…»» → «Часть I. 12, 13)»).
    said: set[tuple[str, ...]] = set()
    kept_count = total = 0

    # Блок — строки от номера фрагмента до следующего номера или заголовка раздела.
    blocks: list[dict[str, Any]] = [{"n": None, "lines": [], "content": True}]
    source = answer.split("\n")
    for line, formula in zip(source, _formula_lines(source), strict=True):
        block = blocks[-1]
        if not line.strip():
            block["lines"].append(line)
            continue
        if _HEADING.fullmatch(line) and not _CITE.search(line):
            blocks.append({"n": None, "lines": [line], "content": True, "heading": True})
            blocks.append({"n": None, "lines": [], "content": True})
            continue
        cite = _CITE.search(line)
        if cite:
            first = int(re.split(r"[,;]", cite.group(1))[0])
            block = {
                "n": first if first in seen else -1,
                "lines": [],
                "content": False,
                "translated": bool(_TRANSLATION.search(line)),
            }
            blocks.append(block)
            if _is_header(line, first, titles):
                block["lines"].append(line)
                continue
        n = block["n"]
        if formula:
            block["lines"].append(line)
            block["content"] = block["content"] or n in seen
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        kept: list[str] = []
        for part in _split_sentences(line[len(prefix) :]):
            # «Перевод фрагмента [2]: текст» — пометка не часть выписки.
            bare = _MARK.sub("", _CITE.sub("", part)).strip()
            words = _tokens(bare)
            if len(bare) <= MIN_CHARS or len(words) < MIN_RUN:
                kept.append(part)
                continue
            if n == -1:  # номер фрагмента, которого нет
                removed.append(part)
                continue
            if n is None:
                # До первой выписки — только служебная фраза промта, не рассуждение модели.
                if _META.search(bare) and len(bare) < 200:
                    kept.append(part)
                else:
                    removed.append(part)
                continue
            if foreign[n]:  # перевод сверяется, только если фрагмент правда не по-русски
                kept.append(part)  # перевод дословно не сверить
                block["content"] = True
                continue
            if tuple(words) in said:
                continue
            said.add(tuple(words))
            total += 1
            if _share(words, hay[n]) >= SUPPORTED:
                kept.append(part)
                kept_count += 1
                block["content"] = block["content"] or _share(words, body[n]) >= SUPPORTED
            elif any(_share(words, other) >= SUPPORTED for m, other in body.items() if m != n):
                kept.append(part)  # текст другого фрагмента базы
                kept_count += 1
                block["content"] = True
            elif _META.search(bare) and len(bare) < 200:
                total -= 1
                kept.append(part)
            else:
                removed.append(part)
        rest = " ".join(p for p in kept if p)
        if rest.strip():
            block["lines"].append(prefix + rest)

    lines: list[str] = []
    for block in blocks:
        if not block["content"]:
            continue  # от выписки остался один заголовок
        lines += block["lines"]
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() and _HEADING.fullmatch(line) and not _CITE.search(line):
            after = next((x for x in lines[i + 1 :] if x.strip()), "")
            if not after or (_HEADING.fullmatch(after) and not _CITE.search(after)):
                continue
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip(), removed, (kept_count, total)


COPIED = 0.6  # доля слов фразы без ссылки, стоящих в одном фрагменте цепочками
# Пересказ фрагмента: доля значимых слов (от 4 букв) фразы, которые есть во фрагменте, — от
# RETOLD, слов — от RETOLD_WORDS, и другой фрагмент отстаёт на RETOLD_GAP. На ответах по
# золотой базе (11.11.2026) пересказ словаря давал 0.75–0.91, слова модели — 0.25–0.6.
RETOLD = 0.75
RETOLD_WORDS = 8
RETOLD_GAP = 0.2


def _significant(words: list[str]) -> list[str]:
    return [w for w in words if len(w) >= 4]


def cite_by_words(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Фразе без ссылки, которая почти дословно стоит в одном фрагменте, — ссылка на него.

    Модель часто пишет определение из словаря первой фразой без номера, а номер ставит во
    второй («Главная зона нефтеобразования — зона в разрезе…. По Геологическому словарю
    [10], она приурочена…»). drop_uncited убирал определение, и ответ начинался с «она»
    (oil02, oil05, oil16, прогон 11.11.2026 21:45). Как cite_by_numbers, только по словам:
    фраза от MIN_CHARS знаков, не меньше COPIED её слов — цепочками в одном фрагменте; из
    нескольких таких — тот, где совпало больше. Пересказ фрагмента своими словами — тоже
    (RETOLD): иначе первая фраза раздела уходила, а раздел начинался с «Такие ловушки…».
    Смысл ссылки потом сверяет support.py — как у любой фразы со ссылкой.
    Возвращает (текст, получившие ссылку фразы)."""
    hays = [(s["n"], _tokens(fragment_block(s))) for s in sources]
    bags = [(n, set(hay)) for n, hay in hays]
    cited: list[str] = []

    def change(part: str) -> str:
        if not is_claim(part) or _CITE.search(part) or len(part) < MIN_CHARS:
            return part
        words = _tokens(part)
        best, best_n = 0.0, None
        for n, hay in hays:
            share = _share(words, hay)
            if share > best:
                best, best_n = share, n
        if best_n is None or best < COPIED:
            best_n = _retold(_significant(words))
        if best_n is None:
            return part
        cited.append(part)
        return _with_cite(part, [best_n])

    def _retold(words: list[str]) -> int | None:
        if len(words) < RETOLD_WORDS:
            return None
        scores = sorted(
            ((sum(w in bag for w in words) / len(words), n) for n, bag in bags), reverse=True
        )
        if not scores or scores[0][0] < RETOLD:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < RETOLD_GAP:
            return None
        return int(scores[0][1])

    text = rewrite(answer, change)
    return (text, cited) if cited else (answer, cited)
