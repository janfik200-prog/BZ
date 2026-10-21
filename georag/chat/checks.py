"""Проверка ответа модели: номера фрагментов, отказ, утверждения без ссылки."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

NO_ANSWER = "В базе знаний нет ответа"  # начало фразы-отказа из правила 3

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
CITE_MARK = re.compile(r"[ \t]*\[\d+(?:\s*[,;]\s*\d+)*\]")
# Фразы о самих фрагментах, а не о геологии: «по данным статей базы», «во
# фрагментах нет», «в базе знаний об этом ничего».
_ABOUT_SOURCES = re.compile(
    r"фрагмент|в базе|статей базы|статьях базы|по данным статей|" r"в найденных", re.IGNORECASE
)


def _norm(text: str) -> str:
    return re.sub(r"[«»\"'*_.\s]+", " ", text.lower().replace("ё", "е")).strip()


def is_refusal(start: str) -> bool:
    """Начинается ли ответ с фразы «в базе знаний нет ответа» (правило 3)."""
    return _norm(start).startswith(_norm(NO_ANSWER))


# Отказ другими словами: «В базе знаний нет информации о металлогении Анабарского щита.»
# Отказ — только если ответ на этом и кончается (короче REFUSAL_CHARS): длинный ответ с
# такой первой фразой — частичный ответ (правило 6).
REFUSAL_CHARS = 240


_NOT_IN_BASE = re.compile(r"^в базе знани\w* нет")


def says_not_in_base(text: str) -> bool:
    return bool(_NOT_IN_BASE.match(_norm(text)))


def citations(answer: str, count: int) -> tuple[list[int], list[int]]:
    """Какие номера фрагментов в ответе есть и какие из них выдуманы."""
    found: set[int] = set()
    for group in _CITE.findall(answer):
        for part in re.split(r"[,;]", group):
            if part.strip().isdigit():
                found.add(int(part))
    used = sorted(n for n in found if 1 <= n <= count)
    unknown = sorted(n for n in found if not 1 <= n <= count)
    return used, unknown


# Строка-заголовок: «### Методы», «**1) Признаки рудных узлов**». Считалась
# утверждением без ссылки — у ответов из трёх разделов треть «утверждений» были
# заголовками.
_HEADING = re.compile(r"(?m)^\s*(?:#{1,6}\s+[^\n]*|\*\*[^*\n]+\*\*:?|__[^_\n]+__:?)\s*$")


# Строка — одна формула ($$…$$, можно со знаком препинания после).
_FORMULA_LINE = re.compile(r"\s*\$\$[^\n]*?\$\$\s*[.,;:]?\s*")


def _formula_lines(lines: list[str]) -> list[bool]:
    """Какие строки ответа — формула (правило 8): строка из одной формулы и формула на
    несколько строк — от строки, где $$ открылась, до строки, где закрылась. Пара $$
    считается по всем строкам: открыться формула может и в конце строки текста («по
    формуле (2) [3]: $$»). Раньше такая строка шла как текст — тело формулы считалось
    утверждением без ссылки, а весь ответ после неё — формулой, без проверки."""
    flags: list[bool] = []
    inside = False
    for line in lines:
        odd = line.count("$$") % 2 == 1
        flags.append(inside or odd or bool(_FORMULA_LINE.fullmatch(line)))
        if odd:
            inside = not inside
    return flags


def coverage(answer: str) -> tuple[int, int]:
    """Сколько утверждений ответа подкреплены ссылкой: (со ссылкой, всего).

    Утверждение — предложение или пункт списка длиннее 40 знаков. Не считаются
    фразы о самих фрагментах («По данным статей базы…», «Во фрагментах нет…»):
    это не факт из статьи, ссылка к ним не нужна. Ссылка может стоять и сразу
    после точки — она относится к предыдущему предложению.
    """
    lines = answer.split("\n")
    # Формула — не утверждение (ссылка — во фразе перед ней), заголовок раздела — тоже.
    text = "\n".join("" if f else s for s, f in zip(lines, _formula_lines(lines), strict=True))
    text = _HEADING.sub("", text)
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    text = re.sub(r"([.!?…])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)", r" \2\1", text)
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    claims = [
        p
        for p in parts
        if len(_CITE.sub("", p).strip()) > 40
        and not (_ABOUT_SOURCES.search(p) and not _CITE.search(p))
    ]
    cited = [p for p in claims if _CITE.search(p)]
    return len(cited), len(claims)


# Конец предложения вместе со ссылками после точки: «…разломов. [2][3] Далее…». После
# точки — не строчная буква: «120 млн. лет» — одна фраза, а не две.
_SENTENCE_END = re.compile(r"([.!?…])((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])*)\s+(?![a-zа-яё])")
_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
# Следующая фраза держится за убранную. Связка («Например,», «Также») — снимается:
# фраза верна и без неё. Указание назад («Эти длины волн…», «Таким образом…») — без
# убранной фразы непонятно, о чём речь, поэтому она остаётся. На ответах оценки
# 14.10.2026 каждый третий раздел начинался с «Например,» или «Эти…» — первая,
# обобщающая фраза раздела уходила как фраза без ссылки.
_CONNECTIVE = re.compile(
    r"^(?:(?:Например|В частности|Кроме того|При этом|Так|Помимо этого),\s*|Также\s+)"
)
_BACK_REF = re.compile(
    r"^(?:Эт(?:и|о|а|от|им|ой|их|ого)|Так(?:ие|ой|ая|ое|им\s+образом)|Они|Их|Он|Она|"
    r"Там|Здесь|Данн\w+)\b"
)


def _lead(text: str) -> str:
    """Фраза без связки в начале — с большой буквы."""
    rest = _CONNECTIVE.sub("", text, count=1)
    return rest[:1].upper() + rest[1:] if rest != text else text


_REPEAT_MIN = 30  # строки короче не сравниваются: «- пирит [1]» повторять можно
# Почти тот же абзац: модель пишет раздел ещё раз, выбросив пару слов («глубинных
# корово-мантийных разломов» → «глубинных разломов»; вопрос r02, 14.10.2026). Длинная
# строка без единого нового слова против уже сказанной — повтор. Пункты списка по
# одному шаблону («кварцевые жилы содержат пирит», «… антимонит») — нет: слово новое.
_NEAR_MIN = 80
_WORDS = re.compile(r"\w+")


def _plain(line: str) -> str:
    """Строка для сравнения: без ссылок, разметки, маркера пункта и регистра."""
    text = _BULLET.sub("", CITE_MARK.sub("", line))
    return " ".join(re.sub(r"[*_#«»\"']", "", text).lower().split()).strip(" .;,…")


def drop_repeats(answer: str) -> str:
    """Ответ без строк, дословно повторяющих уже сказанное (ссылки и разметка не в счёт).

    Модель иногда зацикливается и пишет те же разделы ещё раз (вопросы о связях
    14.10.2026: «кварцевые жилы», «золото» — по два одинаковых раздела в конце).
    Заголовок, под которым после этого ничего не осталось, уберёт drop_uncited."""
    seen: set[str] = set()
    long_lines: list[set[str]] = []
    sentences: set[str] = set()
    out = []
    for line in answer.split("\n"):
        plain = _plain(line)
        if len(plain) >= _REPEAT_MIN and not _HEADING.fullmatch(line):
            if plain in seen:
                continue
            words = set(_WORDS.findall(plain))
            if len(plain) >= _NEAR_MIN and any(words <= other for other in long_lines):
                continue
            seen.add(plain)
            if len(plain) >= _NEAR_MIN:
                long_lines.append(words)
            line = _drop_seen_sentences(_drop_list_repeats(line), sentences)
            if not _plain(line):
                continue
        out.append(line)
    return "\n".join(out)


# Пункт перечня внутри строки: «эпидотом [7][9], кальцитом [10], …».
_LIST_ITEM = re.compile(r"[^,;]+(?:[,;]|$)")


def _drop_list_repeats(line: str) -> str:
    """Строка-перечень без повторов пунктов: модель, перечисляя минералы, зацикливалась и
    писала тот же кусок перечня трижды (хлорит, 15.10.2026). Пункт сравнивается без ссылок и
    разметки; первый остаётся, повторы уходят. Короткие строки и строки не-перечни — как были."""
    items = _LIST_ITEM.findall(line)
    if len(items) < 6:
        return line
    seen: set[str] = set()
    kept = []
    for item in items:
        key = _plain(item).strip(" ,;.")
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(item)
    if len(kept) == len(items):
        return line
    text = "".join(kept).rstrip(" ,;")
    return text if text.endswith((".", "!", "?", "…")) else text + "."


def _drop_seen_sentences(line: str, seen: set[str]) -> str:
    """Строка без фраз, уже сказанных в ответе: «Это также отмечено в составе жильных
    минералов [2].» в шести пунктах подряд (вопрос r11, 14.10.2026)."""
    bullet = _BULLET.match(line)
    prefix = bullet.group(1) if bullet else ""
    parts = _split_sentences(line[len(prefix) :])
    kept = []
    for part in parts:
        plain = _plain(part)
        if len(plain) >= _REPEAT_MIN:
            if plain in seen:
                continue
            seen.add(plain)
        kept.append(part)
    return prefix + " ".join(kept) if len(kept) < len(parts) else line


_LOOP_MIN = 60  # фраза короче может повториться и законно


def is_looping(text: str) -> bool:
    """Модель пошла по кругу: последняя законченная фраза уже была в ответе дословно
    (ссылки и разметка не в счёт). Ответ о хлорите — 21 разная фраза, повторенная до
    8 раз: 26 тысяч знаков и 147 с (21.10.2026); drop_repeats убирал повторы, но время
    уже было потрачено."""
    parts = _split_sentences(text)
    if len(parts) < 3:
        return False
    last = _plain(parts[-2])  # parts[-1] — ещё не дописана
    return len(last) >= _LOOP_MIN and any(_plain(p) == last for p in parts[:-2])


def _split_sentences(body: str) -> list[str]:
    return [
        p.strip()
        for p in _SENTENCE_END.sub(lambda m: m.group(1) + m.group(2) + "\0", body).split("\0")
    ]


def drop_uncited(answer: str) -> tuple[str, list[str]]:
    """Ответ без утверждений, у которых нет ссылки на фрагмент.

    Правило 2 просит модель писать только то, что подкреплено фрагментом, но
    ручная проверка 190 утверждений (.local/logs/quality-20261005) нашла 9 % фраз без
    ссылки — общие слова модели, а не статьи. Здесь они убираются: утверждение
    (как в coverage) длиннее 40 знаков, без номера фрагмента и не о самих
    фрагментах. Заголовок, под которым ничего не осталось, уходит тоже. Обрывков не
    остаётся: связка в начале следующей фразы снимается, а фраза, на которую следующая
    ссылается («Эти…»), остаётся (_CONNECTIVE, _BACK_REF).
    Возвращает (новый текст, убранные фразы)."""
    removed: list[str] = []
    lines: list[str] = []
    source = answer.split("\n")
    for line, formula in zip(source, _formula_lines(source), strict=True):
        # Формула ($$…$$, правило 8) — не утверждение: ссылка на фрагмент стоит во фразе
        # перед ней. Формула на несколько строк — тоже (_formula_lines).
        if formula:
            lines.append(line)
            continue
        if not line.strip() or _HEADING.fullmatch(line):
            lines.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(prefix) :]
        parts = _split_sentences(body)
        drop = []
        for part in parts:
            bare = _CITE.sub("", part).strip()
            lead_in = bare.endswith(":")  # «используют следующие методы:» — подводка к списку
            drop.append(
                len(bare) > 40
                and not lead_in
                and not _CITE.search(part)
                and not _ABOUT_SOURCES.search(part)
            )
        kept: list[str] = []
        for i, part in enumerate(parts):
            after = parts[i + 1] if i + 1 < len(parts) and not drop[i + 1] else ""
            if drop[i] and _BACK_REF.match(after):
                drop[i] = False  # без неё следующая фраза непонятна
            if drop[i]:
                removed.append(part)
                if after:
                    parts[i + 1] = _lead(after)
                continue
            bare = _CITE.sub("", part).strip(" .;,")
            # Хвост убранной фразы, разрезанной на сокращении («…120 млн. лет [2].»),
            # и строка из одних ссылок («[3]. [7].») — не фразы.
            tail = i > 0 and drop[i - 1] and bare[:1].islower() and len(bare) < 20
            if not bare or tail:
                continue
            kept.append(part)
        rest = " ".join(p for p in kept if p)
        if rest:
            lines.append(prefix + rest)
    # Заголовок без текста под ним и лишние пустые строки.
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() and _HEADING.fullmatch(line):
            after = next((x for x in lines[i + 1 :] if x.strip()), "")
            if not after or _HEADING.fullmatch(after):
                continue
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip(), removed


def cited_numbers(part: str) -> list[int]:
    """Номера фрагментов, на которые ссылается кусок ответа, по порядку и без повторов."""
    out: list[int] = []
    for group in _CITE.findall(part):
        for n in re.split(r"[,;]", group):
            if n.strip().isdigit() and int(n) not in out:
                out.append(int(n))
    return out


def about_sources(part: str) -> bool:
    """Фраза о самих фрагментах, а не о геологии («во фрагментах нет…»)."""
    return bool(_ABOUT_SOURCES.search(part))


def is_claim(part: str) -> bool:
    """Утверждение — как у drop_uncited: длиннее 40 знаков без ссылок, не подводка к
    списку («используют следующие методы:») и не фраза о самих фрагментах без ссылки."""
    bare = _CITE.sub("", part).strip()
    if len(bare) <= 40 or bare.endswith(":"):
        return False
    return bool(_CITE.search(part)) or not about_sources(part)


def rewrite(answer: str, change: Callable[[str], str | None]) -> str:
    """Ответ, в котором каждое предложение (и пункт списка) прошло через change: строка —
    новый вид, None — убрать. Разбор — тот же, что у drop_uncited: формулы и заголовки не
    трогаются, связка в начале фразы после убранной снимается (_CONNECTIVE), заголовок, под
    которым ничего не осталось, уходит. Так режут ответ cite_by_numbers и сверка ссылок
    (support.py)."""
    lines: list[str] = []
    source = answer.split("\n")
    for line, formula in zip(source, _formula_lines(source), strict=True):
        if formula or not line.strip() or _HEADING.fullmatch(line):
            lines.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        kept: list[str] = []
        dropped = False
        for part in _split_sentences(line[len(prefix) :]):
            new = change(part)
            if new is None:
                dropped = True
                continue
            kept.append(_lead(new) if dropped else new)
            dropped = False
        rest = " ".join(p for p in kept if p)
        if rest:
            lines.append(prefix + rest)
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() and _HEADING.fullmatch(line):
            after = next((x for x in lines[i + 1 :] if x.strip()), "")
            if not after or _HEADING.fullmatch(after):
                continue
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()


# Число в тексте: 1.660, 5343,33 (5343.33), 50–500 (две штуки), 1:50 000 (50000). Значимое —
# из двух цифр и больше или с дробью: одиночные 1–9 есть почти в любом фрагменте.
_GROUP_SPACE = "[   ]"  # пробел, неразрывный и узкий неразрывный между разрядами
_NUMBER = re.compile(rf"\d{{1,3}}(?:{_GROUP_SPACE}\d{{3}})+(?![.,]?\d)|\d+(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    """Значимые числа текста в одном виде: без пробелов в разрядах, с точкой в дроби.
    Номера ссылок [n] — не числа текста."""
    out = set()
    for raw in _NUMBER.findall(_CITE.sub(" ", text)):
        value = re.sub(_GROUP_SPACE, "", raw).replace(",", ".")
        if len(value.replace(".", "")) >= 2:
            out.add(value)
    return out


def cite_by_numbers(
    answer: str, sources: list[dict[str, Any]], seen: Callable[[dict[str, Any]], str] | None = None
) -> tuple[str, list[str]]:
    """Утверждению без ссылки, все значимые числа которого есть в одном фрагменте, —
    ссылка на этот фрагмент, а не вычёркивание: верная цифра («Максимальная концентрация
    радона в точке TTN-07 составила 5343,33 Бк/м³») уходила как фраза без ссылки, а в
    ответе оставалось «конкретные значения не указаны» (AgentKB, 20.10.2026).
    seen(источник) — текст фрагмента, каким его видела модель; по умолчанию — text.
    Возвращает (текст, получившие ссылку фразы)."""
    show = seen or (lambda s: str(s.get("text") or ""))
    known = [(s["n"], numbers(show(s))) for s in sources]
    cited: list[str] = []

    def change(part: str) -> str:
        if not is_claim(part) or _CITE.search(part):
            return part
        need = numbers(part)
        match = next((n for n, have in known if need and need <= have), None)
        if match is None:
            return part
        cited.append(part)
        end = re.search(r"[.!?…]$", part)
        return (part[:-1] + f" [{match}]" + part[-1]) if end else f"{part} [{match}]"

    text = rewrite(answer, change)
    return (text, cited) if cited else (answer, cited)
