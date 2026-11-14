"""Проверка ответа модели: номера фрагментов, отказ, утверждения без ссылки."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

NO_ANSWER = "В базе знаний нет ответа"  # начало фразы-отказа из правила 3
# Та же фраза у промта «выписки» (messages.QUOTE_SYSTEM, правило 8).
NO_ANSWER_QUOTE = "В предоставленных фрагментах нет ответа"

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
CITE_MARK = re.compile(r"[ \t]*\[\d+(?:\s*[,;]\s*\d+)*\]")
# Фразы о самих фрагментах, а не о геологии: «по данным статей базы», «во
# фрагментах нет», «в базе знаний об этом ничего».
_ABOUT_SOURCES = re.compile(
    r"фрагмент|в базе|статей базы|статьях базы|по данным статей|" r"в найденных", re.IGNORECASE
)


def _norm(text: str) -> str:
    return re.sub(r"[«»\"'*_.\s]+", " ", text.lower().replace("ё", "е")).strip()


_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_LETTER = re.compile(r"[^\W\d_]")


def is_foreign(text: str) -> bool:
    """Текст не по-русски: кириллицы меньше трети букв."""
    letters = _LETTER.findall(text)
    return bool(letters) and len(_CYRILLIC.findall(text)) < len(letters) / 3


def is_refusal(start: str) -> bool:
    """Начинается ли ответ с фразы «в базе знаний нет ответа» (правило 3) или с её вида у
    промта «выписки»."""
    return _norm(start).startswith((_norm(NO_ANSWER), _norm(NO_ANSWER_QUOTE)))


# Отказ другими словами: «В базе знаний нет информации о металлогении Анабарского щита.»
# Отказ — только если ответ на этом и кончается (короче REFUSAL_CHARS): длинный ответ с
# такой первой фразой — частичный ответ (правило 6).
REFUSAL_CHARS = 240


_NOT_IN_BASE = re.compile(r"^(?:в базе знани\w* нет|в предоставленных фрагментах нет)")


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
# Ссылка после точки в конце абзаца («…сульфатные породы. [4]») — тоже часть фразы: без
# этого «[4]» отрывалась в отдельный кусок, а фраза оставалась без ссылки и уходила
# (oil04 на золотой базе, 11.11.2026).
_SENTENCE_END = re.compile(
    r"([.!?…])((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])*)(?:\s+(?![a-zа-яё])|\s*$)"
)
_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
# Следующая фраза держится за убранную. Связка («Например,», «Также») — снимается:
# фраза верна и без неё. Указание назад («Эти длины волн…», «Таким образом…») — без
# убранной фразы непонятно, о чём речь, поэтому она остаётся. На ответах оценки
# 14.10.2026 каждый третий раздел начинался с «Например,» или «Эти…» — первая,
# обобщающая фраза раздела уходила как фраза без ссылки.
_CONNECTIVE = re.compile(
    r"^(?:(?:Например|В частности|Кроме того|Кроме этого|При этом|Так|Помимо этого|Однако|"
    r"Наоборот|Тем не менее|Вместе с тем|В то же время|Впрочем),\s*|(?:Также|Однако)\s+)"
)
_BACK_REF = re.compile(
    r"^(?:Эт(?:и|о|а|от|им|ой|их|ого)|Так(?:ие|ой|ая|ое|им\s+образом)|Они|Их|Он|Она|Оно|"
    r"Там|Здесь|Данн\w+)\b"
)
# Указание назад после вводных слов: «По Геологическому словарю [10], она приурочена…» — без
# убранной фразы-определения ответ начинался с «она» (oil02, прогон 11.11.2026 21:45).
_INTRO_REF = re.compile(r"^(?:По|Согласно)\b[^,.]{0,80},\s*(?:она|он|оно|они|её|его|их)\b")


def _lead(text: str) -> str:
    """Фраза без связки в начале — с большой буквы."""
    rest = _CONNECTIVE.sub("", text, count=1)
    return rest[:1].upper() + rest[1:] if rest != text else text


# Указание назад в середине фразы: «Диагностическая полоса поглощения для этих минералов…» —
# без убранной фразы, где минералы названы, пункт списка был ни о чём (k01, 13.11.2026).
_INNER_REF = re.compile(
    r"\b(?:для|у|в|из|от|к|по|при|с|среди|между)\s+(?:эт(?:их|ими|им|ого|ой|ом)|так(?:их|ими|им|ого|ой)|"
    r"данн(?:ых|ыми|ым|ого|ой))\s+[а-яё]{4,}",
    re.IGNORECASE,
)


def _refers_back(text: str) -> bool:
    """Фраза держится за предыдущую: «Эти…», «Наоборот, они…», «По словарю [3], она…»,
    «…для этих минералов…»."""
    return bool(
        _BACK_REF.match(text)
        or _BACK_REF.match(_lead(text))
        or _INTRO_REF.match(text)
        or _INNER_REF.search(text[:160])
    )


_REPEAT_MIN = 30  # строки короче не сравниваются: «- пирит [1]» повторять можно
# Почти тот же абзац: модель пишет раздел ещё раз, выбросив пару слов («глубинных
# корово-мантийных разломов» → «глубинных разломов»; вопрос r02, 14.10.2026). Длинная
# строка без единого нового слова против уже сказанной — повтор. Пункты списка по
# одному шаблону («кварцевые жилы содержат пирит», «… антимонит») — нет: слово новое.
# И только против строки близкой длины: пункт списка, пересказывающий часть длинного
# абзаца, — не повтор. Иначе из перечня трёх типов керогена уходили пункты 1 и 3, слова
# которых все есть во вводном абзаце (m04, 12.11.2026), и оставалось «три типа: 2. …».
_NEAR_MIN = 80
_NEAR_SHARE = 0.7  # повтор — не короче этой доли слов строки, которую он повторяет
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
            if len(plain) >= _NEAR_MIN and any(
                words <= other and len(words) >= _NEAR_SHARE * len(other) for other in long_lines
            ):
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
# Запятая внутри числа — не граница: «от 1,6 до 2,43 мкм» резалось на «…до 2» и «43 мкм»,
# второй кусок совпадал с таким же выше, и в ответе оставалось «от 1,6 до 2.» (f01).
_LIST_ITEM = re.compile(r"(?:[^,;]|(?<=\d),(?=\d))+(?:[,;]|$)")


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


# Точка после инициала или сокращения перед именем — не конец фразы: «по Н.Б. Вассоевичу»
# резалось на «…по Н.Б.» и «Вассоевичу (1967 г.) [1].», «пос. Соболох-Майян» — на обрывок
# «Соболох-Майян, и на юге…» (прогон 11.11.2026 21:45). «г.» сюда не входит: «в 2015 г.
# Затем…» — обычно конец фразы.
_ABBREV = re.compile(
    r"(?:^|[\s(.])(?:[А-ЯЁ]|р|пос|им|скв|оз|пгт|ст|руч|хр|д|п|см|рис|табл|т|св|ок|мас|об|вес|мол|"
    r"ат|отн|ед|усл)$"
)


def _split_sentences(body: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(body):
        if m.group(1) == "." and not m.group(2) and _ABBREV.search(body, 0, m.start()):
            continue
        # Точка внутри названия в кавычках — не конец фразы: «Геологический словарь. Т. 2
        # (К–П). ВСЕГЕИ, 2017» резалось на куски, и в ответе оставался обрывок «Т. 2 (К–П).
        # ВСЕГЕИ, 2017» [3].» (d05, 13.11.2026).
        # «…оксидов железа и др. [1][2].» — точка сокращения перед ссылкой: фраза резалась на
        # «…и др.» без ссылки (её убирали) и «[1][2].» (k03, 13.11.2026).
        if m.group(1) == "." and not m.group(2) and re.match(r"\s*\[\d", body[m.start() + 1 :]):
            continue
        # «3,1–5,9 мас. % NaCl» — после точки знак процента: это не конец фразы (c02).
        if m.group(1) == "." and re.match(r"\s*[%‰]", body[m.start() + 1 :]):
            continue
        inside = body[start : m.start()]
        if inside.count("«") > inside.count("»") and "»" in body[m.end() :]:
            continue
        parts.append(body[start : m.start()] + m.group(1) + m.group(2))
        start = m.end()
    # Пустой хвост — только после пробела в конце, как раньше: is_looping считает
    # последний кусок недописанным.
    if start < len(body) or not parts or body[-1:].isspace():
        parts.append(body[start:])
    return [p.strip() for p in parts]


def _with_cite(part: str, cites: list[int]) -> str:
    """Фраза со ссылкой перед точкой в конце: «… зона.» → «… зона [3].»."""
    mark = "[" + ", ".join(str(n) for n in cites) + "]"
    end = re.search(r"[.!?…]$", part)
    return (part[:-1] + f" {mark}" + part[-1]) if end else f"{part} {mark}"


# Служебный текст модели — не ответ. Вводная о происхождении фрагментов («Все фрагменты
# взяты из одной статьи «<id отчёта>» (…, отчёт <id>)…», «Ниже приведён ответ…») в прогоне
# 11.11.2026 открывала 8 ответов из 93, а в f16 от ответа осталась она одна; строка-заглушка
# «По Геологическому словарю [3][4].» стояла вместо раздела о различиях (oil15).
_SERVICE_INTRO = re.compile(
    r"^(?:Все\s+фрагменты\b|Фрагменты\s+\[[^\]]+\][^.]*\bвзят|Ниже\s+привед[её]н\w*\s+ответ|"
    r"Словарь\s+\[\d+\]\s+используется|Сведения\s+о\s+терминах\b)",
    re.IGNORECASE,
)
_STUB = re.compile(
    r"^(?:По|Согласно)\s+Геологическ\w+\s+словар\w+[\s,]*(?:\[[\d,;\s]+\][\s,]*)*[.:]?$",
    re.IGNORECASE,
)
_RULE_LINE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def drop_service(answer: str) -> str:
    """Ответ без служебного текста: вводной о том, откуда взяты фрагменты (в начале, до
    первой фразы по существу), строк-заглушек из одной ссылки на словарь и линий «---»."""
    out: list[str] = []
    intro = True
    for line in answer.split("\n"):
        if _RULE_LINE.match(line):
            continue
        if not line.strip() or _HEADING.fullmatch(line):
            intro = intro and not line.strip()
            out.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(prefix) :].strip()
        if _STUB.match(body):
            continue
        # «…кулисные складки волочения. По Геологическому словарю [9].» — заглушка в конце
        # абзаца: её ссылка — у фразы перед ней (дальше её по абзацу разносит drop_uncited),
        # а сама заглушка уходит. Иначе определение из словаря оставалось без ссылки и
        # убиралось (s03, 14.11.2026).
        parts = [p for p in _split_sentences(body) if p.strip()]
        if len(parts) > 1 and _STUB.match(parts[-1].strip()):
            cites = cited_numbers(parts[-1])
            head = parts[-2] if _CITE.search(parts[-2]) else _with_cite(parts[-2], cites)
            line = prefix + " ".join([*parts[:-2], head])
            body = line[len(prefix) :].strip()
        if intro:
            parts = [p for p in _split_sentences(body) if p.strip()]
            kept = [p for p in parts if not _SERVICE_INTRO.match(p.strip())]
            if not kept:
                continue
            intro = False
            if len(kept) < len(parts):
                line = prefix + " ".join(kept)
        out.append(line)
    return "\n".join(out).strip()


_BARE_REFUSAL = re.compile(
    r"(?:^|(?<=[.!?…\]])[ \t]+|\n)\**В базе знаний нет ответа на (?:этот|данный) вопрос\.?\**"
    r"(?=\s|$)"
)


def drop_bare_refusal(answer: str) -> str:
    """Ответ по существу без фразы-отказа рядом: модель отвечала по фрагментам со ссылками
    и дописывала «В базе знаний нет ответа на этот вопрос.» (f05, f17, oil59 — 11.11.2026).
    Если кроме отказа ничего со ссылкой нет — текст как был: это и есть отказ."""
    rest = _BARE_REFUSAL.sub("", answer)
    if rest == answer or not _CITE.search(rest):
        return answer
    return re.sub(r"\n{3,}", "\n\n", rest).strip()


# Исправление посылки в начале ответа: «Это не так: линеаменты связаны с разломами.» Ссылку
# модель ставит в следующих фразах, где приводит сказанное во фрагментах, и drop_uncited
# убирал само исправление — оставалось изложение без вывода (g06, 12.11.2026).
_CORRECTION = re.compile(
    r"^(?:Это не так|Это неверно|Вопрос содержит (?:неверн|ошибочн)|Предпосылка вопроса неверн)",
    re.IGNORECASE,
)


def cite_correction(answer: str) -> str:
    """Исправление посылки в первой фразе получает ссылки следующей фразы со ссылкой (не
    дальше трёх фраз): его проверит сверка, как любое утверждение со ссылкой."""
    lines = answer.split("\n")
    first = next((i for i, x in enumerate(lines) if x.strip()), None)
    if first is None:
        return answer
    parts = [p for p in _split_sentences(lines[first].strip()) if p.strip()]
    if not parts or not _CORRECTION.match(parts[0].strip("* ")) or _CITE.search(parts[0]):
        return answer
    following = parts[1:]
    for line in lines[first + 1 :]:
        if len(following) >= 3:
            break
        if line.strip() and not _HEADING.fullmatch(line):
            following += [p for p in _split_sentences(_BULLET.sub("", line)) if p.strip()]
    cites = next((cited_numbers(p) for p in following[:3] if cited_numbers(p)), [])
    if not cites:
        return answer
    lines[first] = " ".join([_with_cite(parts[0], cites), *parts[1:]])
    return "\n".join(lines)


def keep_correction(before: str, after: str, dropped: list[str]) -> tuple[str, list[str]]:
    """Исправление посылки («Это не так: …»), убранное сверкой, возвращается в начало ответа.

    Сверка спрашивает, сказано ли утверждение во фрагменте, а «оруденение не всегда
    приурочено к центрам массивов» — отрицание посылки вопроса: дословно его во фрагменте нет,
    его подтверждает то, что ответ приводит дальше. Без него оставались примеры без вывода
    (f11, 13.11.2026). Возвращается, только если после сверки в ответе остались ссылки."""
    first = next((x for x in before.split("\n") if x.strip()), "")
    parts = [p for p in _split_sentences(first.strip()) if p.strip()]
    if not parts or not _CORRECTION.match(parts[0].strip("* ")) or not _CITE.search(after):
        return after, dropped
    opener = parts[0]
    if _norm(_CITE.sub("", opener)) in _norm(_CITE.sub("", after)):
        return after, dropped
    key = _norm(_CITE.sub("", opener))
    rest = [d for d in dropped if _norm(_CITE.sub("", d)) != key]
    if len(rest) == len(dropped):
        return after, dropped  # убрала не сверка
    return f"{opener} {after.lstrip()}", rest


# «Чего нет» — только о том, что спрошено. Модель дописывала в конец пробелы, о которых
# вопрос не спрашивал: на «Как растительный покров мешает выделению изменённых пород?» —
# «В базе знаний нет ответа на вопрос о том, как… в разных климатических зонах» (h06,
# 13.11.2026; то же в оценке прогона 11.11.2026: «придумывает требования к ответу и сама
# объявляет сведения недостаточными»). Фраза о пробеле остаётся, если хотя бы три четверти её
# значимых слов — из вопроса.
_GAP = re.compile(
    r"^\**(?:В базе знаний|Во? фрагментах|В статьях базы)(?: также)? (?:нет|не (?:указан|упомина|"
    r"описан|содерж|привод|сказано|уточня))",
    re.IGNORECASE,
)
_GAP_WORDS = re.compile(r"[а-яёa-z]{5,}", re.IGNORECASE)
_GAP_STOP = {
    "знани", "фрагм", "стать", "ответ", "вопро", "указа", "упоми", "описа", "содер", "приво",
    "сказа", "уточн", "инфор", "сведе", "данны", "конкр", "именн", "также", "котор", "каким",
    "какие", "какой", "каков", "может", "могут", "этого", "таких", "такие", "прямо", "подро",
}  # fmt: skip


def drop_invented_gaps(answer: str, question: str) -> str:
    """Ответ без заключительной фразы о пробеле, о котором вопрос не спрашивал."""
    lines = answer.rstrip().split("\n")
    last = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
    if last is None:
        return answer
    parts = [p for p in _split_sentences(lines[last].strip()) if p.strip()]
    if not parts or not _GAP.match(parts[-1]) or _CITE.search(parts[-1]):
        return answer
    asked = {w.lower().replace("ё", "е")[:5] for w in _GAP_WORDS.findall(question)}
    words = {w.lower().replace("ё", "е")[:5] for w in _GAP_WORDS.findall(parts[-1])} - _GAP_STOP
    if not words or len(words & asked) * 4 >= len(words) * 3:
        return answer
    rest = "\n".join([*lines[:last], " ".join(parts[:-1])]).rstrip()
    # Кроме этой фразы в ответе ничего со ссылкой — это и есть ответ «в базе нет».
    return rest if _CITE.search(rest) else answer


_BARE_LEAD = re.compile(r"(?:Например|В частности|Так|А именно|Среди них)\s*[:,—-]?", re.IGNORECASE)
LIST_CITES = 3  # ссылок от пунктов списка у фразы-подводки — не больше


def _list_cites(lines: list[str], at: int) -> list[int]:
    """Номера фрагментов из пунктов списка сразу под строкой at (по порядку, без повторов)."""
    out: list[int] = []
    for line in lines[at + 1 :]:
        if not line.strip():
            continue
        if not _BULLET.match(line):
            break
        for n in cited_numbers(line):
            if n not in out:
                out.append(n)
    return out[:LIST_CITES]


_NOT_IN_SOURCES = re.compile(
    r"\bнет\b|\bне\s+(?:указ|упомина|содерж|описа|привод|сказано|говорится|приведен|рассматрива|"
    r"уточня|подтвержда|назван)|отсутству",
    re.IGNORECASE,
)
# Догадка модели в конце фразы: «…, что может быть связано с образованием руд [10]»,
# «…, что может указывать на…». Во фрагменте этого нет — это её вывод; сверка его
# пропускала, если сами факты фразы во фрагменте есть (t03, 14.11.2026). Отрезается,
# факт и ссылка остаются.
_GUESS = re.compile(
    r",?\s+(?:что|это)\s+(?:(?:также|вероятно|возможно)\s+)?(?:может|могут|могло|могли|"
    r"вероятно|возможно)\s+(?:быть|служить|указывать|свидетельствовать|говорить|"
    r"объясняться|определяться|способствовать)\b[^\[]*",
    re.IGNORECASE,
)


def cut_guesses(answer: str) -> str:
    """Ответ без догадок модели в конце фраз («…, что может быть связано с …»)."""

    def change(part: str) -> str:
        m = _GUESS.search(part)
        if not m:
            return part
        marks = "".join(re.findall(r"\[[\d,;\s]+\]", part[m.end() :]))
        head = part[: m.start()].rstrip(" ,;")
        return f"{head} {marks}".rstrip() + "." if head else part

    return rewrite(answer, change)


def drop_empty_leads(answer: str) -> str:
    """Подводка к списку («…известны следующие:»), под которой списка нет, — не ответ."""
    lines = answer.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().endswith(":") and not _BULLET.match(line):
            after = next((x for x in lines[i + 1 :] if x.strip()), "")
            if not _BULLET.match(after):
                continue
        out.append(line)
    return "\n".join(out).strip()


# Пункт перечня, где ярлык и описание называют объект по-разному: «**Пояс Рио-де-ла-Плата**
# — кратон…», «**Пояс Осло** — грабен…». Модель натягивала найденное на слово вопроса
# («Какие пояса описаны?»), сверка ссылок это пропускала: название во фрагменте есть (w01,
# 14.11.2026). Такой пункт — неверное утверждение, он уходит.
_KINDS = (
    "пояс", "кратон", "грабен", "чехол", "провинц", "мегаблок", "бассейн", "щит", "платформ",
    "массив", "свит", "впадин", "прогиб", "блок", "зон", "разлом", "купол", "интрузи",
    "комплекс", "район", "узел", "поле", "месторожден",
)  # fmt: skip
_LABELLED = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)\*\*(?P<label>[^*]+)\*\*\s*[—–-]\s*(?P<what>[а-яё]+)")


def _kind(word: str) -> str | None:
    low = word.lower().replace("ё", "е")
    return next((k for k in _KINDS if low.startswith(k)), None)


def drop_mislabelled(answer: str) -> tuple[str, list[str]]:
    """Ответ без пунктов перечня, где ярлык («Пояс X») и описание («— кратон») — разные объекты."""
    removed: list[str] = []
    out = []
    for line in answer.split("\n"):
        m = _LABELLED.match(line)
        if m:
            label = next((_kind(w) for w in m.group("label").split() if _kind(w)), None)
            what = _kind(m.group("what"))
            if label and what and label != what:
                removed.append(line.strip())
                continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)), removed


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
    # Последняя фраза выше убрана — связка в начале следующей строки («Также…», «Однако…»)
    # снимается: абзац или пункт списка начинался с «Также литологические ловушки…».
    lost = False
    source = answer.split("\n")
    for at, (line, formula) in enumerate(zip(source, _formula_lines(source), strict=True)):
        # Формула ($$…$$, правило 8) — не утверждение: ссылка на фрагмент стоит во фразе
        # перед ней. Формула на несколько строк — тоже (_formula_lines).
        if formula or _HEADING.fullmatch(line):
            lost = False
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
                # Фраза о самих фрагментах без ссылки остаётся, только если говорит, чего в
                # них нет. «Лиственит упоминается во фрагментах статей как одна из формаций…» —
                # утверждение без ссылки: уходит, как любое (d05, 14.11.2026).
                and not (_ABOUT_SOURCES.search(part) and _NOT_IN_SOURCES.search(part))
            )
        # Справа налево: фраза без ссылки остаётся, если на неё ссылается следующая
        # оставшаяся («Орогенные месторождения… Они характеризуются… Такие месторождения… [1]» —
        # без прохода с конца первая уходила, и ответ начинался с «Они», c02, 13.11.2026).
        # В пункте списка первая фраза — его заголовок («**Аргиллизация** — …»): остаётся,
        # если дальше в пункте есть фраза со ссылкой (k01).
        for i in range(len(parts) - 2, -1, -1):
            if not drop[i] or drop[i + 1]:
                continue
            head = i == 0 and bool(prefix) and any(_CITE.search(p) for p in parts[1:])
            # Ссылка в конце абзаца — на весь абзац: модель пишет две-три фразы по одному
            # фрагменту и ставит номер после последней. Фраза прямо перед фразой со ссылкой
            # получает ту же ссылку и идёт на сверку (support.py), а не выбрасывается: без
            # неё оставалось «Веса могут варьироваться от 1 до 0…» — чьи веса, неясно (c03).
            # Связка («Например, …», «Кроме того, …») начинает отдельное утверждение: общая
            # фраза перед ним — слова модели, ссылка примера её не подтверждает.
            paragraph = bool(cited_numbers(parts[i + 1])) and not _CONNECTIVE.match(parts[i + 1])
            if _refers_back(parts[i + 1]) or head or paragraph:
                cites = next((cited_numbers(p) for p in parts[i + 1 :] if cited_numbers(p)), [])
                if cites:
                    drop[i] = False
                    parts[i] = _with_cite(parts[i], cites)
        # Подводка к списку: «Вокруг рудных тел выделяют зоны…, включая метасоматические
        # изменения. Например:» и дальше пункты со ссылками. Фраза перед «Например:» — итог
        # пунктов: остаётся со ссылками первых из них, иначе ответ начинался с «Например:»
        # (w03, 13.11.2026).
        if len(parts) > 1 and not prefix and _CITE.sub("", parts[-1]).strip().endswith(":"):
            below = _list_cites(source, at)
            for i in range(len(parts) - 1):
                if drop[i] and below:
                    drop[i] = False
                    parts[i] = _with_cite(parts[i], below)
        kept: list[str] = []
        if lost and parts and not drop[0]:
            parts[0] = _lead(parts[0])
        for i, part in enumerate(parts):
            after = parts[i + 1] if i + 1 < len(parts) and not drop[i + 1] else ""
            if drop[i] and _refers_back(after):
                # Без неё следующая фраза непонятна. Ссылка — та же, что у следующей: её
                # проверит сверка (support.py), как любую фразу со ссылкой.
                drop[i] = False
                cites = cited_numbers(after)
                if cites:
                    part = _with_cite(part, cites)
            if drop[i]:
                removed.append(part)
                lost = True
                if after:
                    parts[i + 1] = _lead(after)
                continue
            bare = _CITE.sub("", part).strip(" .;,")
            # Хвост убранной фразы, разрезанной на сокращении («…120 млн. лет [2].»),
            # и строка из одних ссылок («[3]. [7].») — не фразы.
            tail = i > 0 and drop[i - 1] and bare[:1].islower() and len(bare) < 20
            if not bare or tail:
                continue
            # Только после оставшейся фразы: пустой хвост строки («…зон. ») сбрасывал
            # отметку, и следующий абзац начинался со связки «В частности, …» (h01).
            lost = False
            kept.append(part)
        rest = " ".join(p for p in kept if p)
        if rest and not (lost and _BARE_LEAD.fullmatch(rest.strip())):
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


_ORPHAN = re.compile(
    r"^(?:(?:Кроме того|Также|При этом),?\s+)?(?:Они|Он|Она|Оно|Их|Его|Е[её]|Там|Здесь|"
    r"Эт(?:и|от|а|о|их|им|ими)\s+[а-яё]{4,}|Так(?:ие|их|им|ими)\s+[а-яё]{4,}|"
    r"В эт(?:ой|ом|их) (?:област|район|зон|регион|част|мест)[а-яё]*)\b",
    re.IGNORECASE,
)


def rewrite(answer: str, change: Callable[[str], str | None]) -> str:
    """Ответ, в котором каждое предложение (и пункт списка) прошло через change: строка —
    новый вид, None — убрать. Разбор — тот же, что у drop_uncited: формулы и заголовки не
    трогаются, связка в начале фразы после убранной снимается (_CONNECTIVE), заголовок, под
    которым ничего не осталось, уходит. Так режут ответ cite_by_numbers и сверка ссылок
    (support.py)."""
    lines: list[str] = []
    dropped = False  # убрана последняя фраза выше, в том числе в конце прошлой строки
    source = answer.split("\n")
    for line, formula in zip(source, _formula_lines(source), strict=True):
        if formula or _HEADING.fullmatch(line):
            dropped = False
        if formula or not line.strip() or _HEADING.fullmatch(line):
            lines.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        kept: list[str] = []
        for part in _split_sentences(line[len(prefix) :]):
            new = change(part)
            if new is None or (dropped and _ORPHAN.match(new)):
                # «В этой области прогнозируются…», «Они характеризуются…» после убранной
                # фразы — ни о чём: уходит вместе с ней (t03, 13.11.2026).
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
_NUMBER = re.compile(
    rf"\d{{1,3}}(?:{_GROUP_SPACE}\d{{3}})+(?![.,]?\d)|[1-9]\d{{0,2}}(?:,\d{{3}}){{2,}}(?![.,]?\d)"
    r"|\d+(?:[.,]\d+)?"
)
_THOUSANDS = re.compile(r"[1-9]\d{0,2}(?:,\d{3})+")  # английская запятая между разрядами


def numbers(text: str) -> set[str]:
    """Значимые числа текста в одном виде: без пробелов в разрядах, с точкой в дроби.
    Номера ссылок [n] — не числа текста."""
    out = set()
    for raw in _NUMBER.findall(_CITE.sub(" ", text)):
        if _THOUSANDS.fullmatch(raw):
            # «1:500,000» английской статьи — пятьсот тысяч, а не дробь: на живой проверке
            # 21.10.2026 «масштаб 1:500 000 [2]» уходил с верного фрагмента. Одна группа
            # («1,660») неоднозначна — оба прочтения; утверждению нужны оба, то есть
            # ссылка по такому числу не ставится, а у фрагмента годится любое.
            out.add(raw.replace(",", ""))
            if raw.count(",") > 1:
                continue
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
        # Ровно один фрагмент: год «2005» стоит в нескольких, и фраза получала ссылку не
        # на тот, где сказано (живая проверка 21.10.2026).
        matches = [n for n, have in known if need and need <= have]
        if len(matches) != 1:
            return part
        match = matches[0]
        cited.append(part)
        end = re.search(r"[.!?…]$", part)
        return (part[:-1] + f" [{match}]" + part[-1]) if end else f"{part} [{match}]"

    text = rewrite(answer, change)
    return (text, cited) if cited else (answer, cited)
