"""Привязка к объекту вопроса: утверждение о названном в вопросе объекте — только по
фрагментам, где этот объект назван.

Сравнение replay 05.11.2026 (93 вопроса, те же фрагменты, температура 0): короткий промт
приписывал Линденской впадине баженовские отложения и свойства из материалов о других
участках (oil59), а Тимано-Печорской провинции — классификацию комплексов из словаря (oil36).
Правило «ссылка подтверждает утверждение для указанного объекта» в промте было — модель его
нарушила. Здесь это проверяет код: объект вопроса — имя собственное («Линденская впадина»,
«Тимано-Печорская провинция»); предложение, которое о нём говорит, остаётся, только если хотя
бы один процитированный в нём фрагмент называет этот объект (в тексте, заголовке или
названии статьи, в любом падеже). Отрицание («во фрагментах не указано, что…», «посылка не
подтверждается») остаётся: это не утверждение об объекте.
"""

from __future__ import annotations

import re
from typing import Any

from .checks import _refers_back, cited_numbers, rewrite

_WORD = re.compile(r"[A-Za-zА-ЯЁа-яё][\w-]*")
# Имя собственное — слово с заглавной не в начале вопроса и не после точки.
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZА-ЯЁ][a-zа-яё][\w-]*(?:\s+[A-ZА-ЯЁ][a-zа-яё][\w-]*)*")
# Обозначение элементов — не имя: «Au-Ag оруденение», «Fe-Mn конкреции», «Cu/Ni» — с
# заглавной, но это химия, а не объект; иначе верное «Au-Ag оруденение известно в Якутии [1]»
# убиралось, если фрагмент пишет «золото-серебряное» (10.11.2026).
_ELEMENTS = re.compile(r"[A-Z][a-z]?(?:[-–/][A-Z][a-z]?)+")
# Отрицание — не утверждение об объекте. По-английски тоже: оба промта велят отвечать
# по-русски, но модель изредка отвечает на языке вопроса.
_NEGATION = re.compile(
    r"не\s+(?:подтвержд|указ|упомина|содерж|описан|сказано|привод|назван|следует|связыва)|"
    r"нет\s+(?:сведений|данных|информации|ответа)|отсутству|неизвестно|не\s+удалось|"
    r"\bnot\s+(?:mention|stat|indicat|specif|confirm|describ|report|support|given|found|known)|"
    r"\bno\s+(?:data|information|evidence|mention|answer)\b|\babsent\b|\bunknown\b",
    re.IGNORECASE,
)


_COORDINATED = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:[ыои]й|ая|ое|[ыи]е|[ои]м|ую|[ое]го|[ыи]х|[ое]му))\s+и\s+"
    r"([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)"
)


def _stem(word: str) -> str:
    """Начало слова: падеж не мешает («Линденской» — «линденск», «Сибири» — «сиби»)."""
    w = word.lower().replace("ё", "е")
    return w if len(w) <= 4 else w[: max(4, len(w) - 3)]


def question_objects(question: str) -> list[list[str]]:
    """Объекты вопроса — имена собственные, каждое списком основ слов."""
    out = []
    # «в Западной и Восточной Сибири» — два объекта, а не «Западной» и «Восточной Сибири».
    question = _COORDINATED.sub(r"\1 \3 и \2 \3", question or "")
    for m in _NAME.finditer(question.strip()):
        words = [w for w in m.group().split() if not _ELEMENTS.fullmatch(w)]
        stems = [_stem(w) for w in _WORD.findall(" ".join(words)) if len(w) >= 3]
        if stems and stems not in out:
            out.append(stems)
    return out


def _words(text: str) -> list[str]:
    """Слова текста строчными, «ё» как «е», составные через дефис — по частям."""
    return [
        part
        for w in _WORD.findall(text or "")
        for part in w.lower().replace("ё", "е").split("-")
        if part
    ]


def _places(stems: list[str], words: list[str]) -> list[int]:
    """Где в словах стоит имя: его основы подряд и по порядку (номера первого слова)."""
    need = [part for s in stems for part in s.split("-") if part]
    if not need:
        return []
    return [
        i
        for i in range(len(words) - len(need) + 1)
        if all(words[i + k].startswith(s) for k, s in enumerate(need))
    ]


def _names(stems: list[str], text: str) -> bool:
    """Назван ли объект в тексте. Имя из нескольких слов — слова подряд и по порядку:
    «Восточная Сибирь» не названа в тексте, где есть «Восточно-Уральская область» и
    «Западно-Сибирская плита» по отдельности (oil40, 11.11.2026: сведения об Урале шли как
    Восточная Сибирь). Составное через дефис — тоже подряд: «Восточно-Сибирская»."""
    return bool(_places(stems, _words(text)))


def _source_text(source: dict[str, Any]) -> str:
    return " ".join(
        [
            str(source.get("title") or ""),
            *map(str, source.get("headings") or []),
            str(source.get("text") or ""),
        ]
    )


# Часть объекта: «на севере Сибирской платформы», «в южной части Урала». Объект во фрагменте
# назван, а часть — другая: на вопрос о севере Сибирской платформы ответ шёл по Сухому Логу
# и рассолам юга платформы (t03, 13.11.2026). Сторона света должна стоять рядом с именем.
_SIDES = {
    "север": ("север",), "юг": ("юг", "южн"), "южн": ("юг", "южн"), "запад": ("запад",),
    "восток": ("восток", "восточ"), "восточ": ("восток", "восточ"),
}  # fmt: skip
# Со строчной буквы: «на севере Сибирской платформы» — часть объекта, а «Восточной Сибири» —
# само имя (с заглавной), стороной света не считается.
_SIDE = re.compile(
    r"\b(север|юг|южн|запад|восток|восточ)[а-яё]*\s+(?:(?:част[ьи]|окраин[а-яё]+|борт[а-яё]*|"
    r"фланг[а-яё]*|склон[а-яё]*|обрамлени[а-яё]+)\s+)?(?=[А-ЯЁ])",
)
SIDE_WINDOW = 8  # слов между стороной света и именем объекта во фрагменте — не больше


def question_side(question: str) -> tuple[str, ...]:
    """Сторона света при объекте вопроса — основами слов («север» или «юг», «южн»); нет — ()."""
    m = _SIDE.search(question or "")
    return _SIDES.get(m.group(1).lower(), ()) if m else ()


def _names_side(stems: list[str], side: tuple[str, ...], text: str) -> bool:
    """Объект назван, и рядом с ним — та же сторона света."""
    words = _words(text)
    span = sum(1 for s in stems for part in s.split("-") if part) + SIDE_WINDOW
    return any(
        any(w.startswith(side) for w in words[max(0, i - SIDE_WINDOW) : i + span])
        for i in _places(stems, words)
    )


# Названный в вопросе метод: «метод весов доказательств» — не «взвешенное наложение слоёв» и
# не «NGETF». Модель описывала под его именем другой метод из найденных фрагментов и
# соглашалась с неверной посылкой вопроса (f04, h02 — 13–14.11.2026). Утверждение о методе
# остаётся, только если хотя бы один процитированный фрагмент называет этот метод — по-русски
# или одним из английских написаний словаря переводов (config/term-translations.yaml).
def question_methods(question: str) -> list[list[list[str]]]:
    """Методы вопроса из словаря переводов: у каждого — написания, каждое списком основ."""
    try:
        from ..index.concepts import translations
        from ..text import name_key

        table = translations()
    except Exception:  # noqa: BLE001 — словаря нет: проверка не делается
        return []
    asked = name_key(question).split()
    out = []
    for key, spellings in table.items():
        words = key.split()
        if not words or words[0] != name_key("метод") or len(words) < 2:
            continue
        if not any(asked[i : i + len(words)] == words for i in range(len(asked))):
            continue
        ru = [w for w in _WORD.findall(question) if name_key(w) in words[1:]]
        names = [[_stem(w) for w in ru]] if ru else []
        names += [[w.lower() for w in _WORD.findall(s)] for s in spellings if s.strip()]
        out.append([n for n in names if n])
    return out


_DEFINED = re.compile(
    r"^\s*(?:что|кто)\s+(?:такое|такой|такая|такие|называют|понимают|значит|означает)\s+"
    r"(?:под\s+)?(?P<term>[^?,.]+)",
    re.IGNORECASE,
)


_RUSSIAN_WORD = re.compile(r"[А-Яа-яЁё]{4,}")  # фрагмент по-русски


def _has_all(stems: list[str], text: str) -> bool:
    """Все основы есть в тексте — в любом порядке (термин вопроса-определения)."""
    words = _words(text)
    return bool(stems) and all(any(w.startswith(s) for w in words) for s in stems)


def defined_term(question: str) -> list[str]:
    """Термин вопроса «Что такое X?» — основами слов; другой вопрос — пусто."""
    m = _DEFINED.match(question or "")
    if not m:
        return []
    # Термин — до союза или второй части вопроса: «Что такое сдвиговые зоны и как с ними
    # связано оруденение?» — термин «сдвиговые зоны», а не «…ними связано» (eval s03, f04).
    head = re.split(
        r"\s+(?:и|или|а|как|где|чем|из|в|на|при|для|у|с|по)\s+", m.group("term"), maxsplit=1
    )[0]
    words = [w for w in _WORD.findall(head) if len(w) >= 4][:2]
    return [_term_stem(w) for w in words]


def _term_stem(word: str) -> str:
    """Основа слова термина — общая для падежей и для прилагательного и существительного:
    «сдвиговые зоны» и «зона сдвига» (s03), «узел» и «узла», «лиственит» и «листвениты»."""
    w = word.lower().replace("ё", "е")
    if len(w) >= 6:
        return w[:5]
    if re.search(r"[бвгджзклмнпрстфхцчшщ][ео][лкцнр]$", w):
        return w[:-2]  # беглая гласная: узел — узла
    return w[:3]


def strict(question: str) -> bool:
    """Вопрос о части объекта, о названном методе или «Что такое X?»: если привязка убрала
    все утверждения, ответа нет (лучше отказ, чем ответ о другом)."""
    return bool(question_side(question) or question_methods(question) or defined_term(question))


def bind_objects(
    question: str, answer: str, sources: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    """Ответ без предложений об объекте вопроса, которые не подтверждены фрагментами,
    называющими этот объект: (текст, убранные)."""
    objects = question_objects(question)
    methods = question_methods(question)
    # Вопрос «Что такое X?»: фраза со ссылкой на статью, где X не назван, — пересказ чужого
    # фрагмента (d05: на «Что такое лиственит?» — фрагменты об альбититах и березитах).
    term = defined_term(question)
    english: list[list[str]] = []
    if term:
        objects = [*objects, term]
        try:
            from .plan import term_hints

            english = [
                [w.lower() for w in _WORD.findall(s)]
                for hint in term_hints(question)
                for s in hint.split(" / ")
            ]
        except Exception:  # noqa: BLE001 — без словаря переводов: язык фрагмента не проверить
            english = []
    if not objects and not methods:
        return answer, []
    by_n = {s["n"]: s for s in sources}
    removed: list[str] = []
    side = question_side(question)
    off_method: list[str] = []

    def change(part: str) -> str | None:
        cited = [by_n[n] for n in cited_numbers(part) if n in by_n]
        if not cited or _NEGATION.search(part):
            return part
        for stems in objects:
            # Вопрос о части объекта («на севере платформы»): проверяется каждая фраза со
            # ссылкой — иначе после убранной оставалось «Эти месторождения расположены в
            # Мамско-Бодайбинской зоне [1]», то есть ответ о юге (t03, 14.11.2026).
            # Термин вопроса «Что такое X?» — тоже: фраза со ссылкой на фрагмент без X —
            # пересказ чужого («В фрагменте [2] отмечается, что в альбититах…», d05).
            every = side or stems is term
            # Термин — все слова где угодно и в любом порядке: «сдвиговые зоны» — «зона сдвига».
            match = _has_all if stems is term else _names
            if not every and not match(stems, part):
                continue
            named = [s for s in cited if match(stems, _source_text(s))]
            # Термин по-русски, фрагмент — на другом языке: «сдвиговые зоны» в статье о shear
            # zones не найти по русским основам (s03). Такой фрагмент проверяется английскими
            # написаниями из словаря переводов, а если их нет — термин в нём не проверить, и
            # фраза уходит на сверку ссылок (support.py), а не выбрасывается здесь.
            if stems is term:
                named += [
                    s
                    for s in cited
                    if s not in named
                    and not _RUSSIAN_WORD.search(str(s.get("text") or ""))
                    and (not english or any(_names(n, _source_text(s)) for n in english))
                ]
            if side and not _NEGATION.search(part):
                named = [s for s in named if _names_side(stems, side, _source_text(s))]
                # Фраза об объекте целиком («Сибирская платформа сложена архейскими
                # комплексами [10]») — не о спрошенной части: на вопрос о севере платформы
                # она не отвечает (t03, 14.11.2026).
                if match(stems, part) and not any(w.startswith(side) for w in _words(part)):
                    named = []
            # Сам термин во фрагменте может стоять мимоходом («формации: альбититы, эйситы,
            # листвениты…»), а фраза ответа пересказывает другое место того же фрагмента —
            # об альбититах (d05). На вопрос «Что такое X?» фраза должна называть X или
            # продолжать фразу о нём («Он…», «Эти породы…»).
            if stems is term and not _NEGATION.search(part):
                if not match(stems, part) and not _refers_back(part.strip()):
                    named = []
            if not named:
                removed.append(part.strip())
                return None
        for names in methods:
            if any(_names(n, part) for n in names) and not any(
                _names(n, _source_text(s)) for n in names for s in cited
            ):
                removed.append(part.strip())
                off_method.append(part)
                return None
        return part

    text = rewrite(answer, change)
    # Фразы о названном методе убраны, и в остатке метод не назван ни разу: остаток — о
    # другом методе («модель обучается на не-золотых минералах [2]», f04, 14.11.2026).
    # В сравнении («чем метод X отличается от нейронных сетей») фраза о X остаётся — и
    # остальное с ней.
    # То же, если фразу с именем метода убрали раньше (она была без ссылки): ответ, где
    # названный в вопросе метод не назван ни разу, — не о нём (h02: «Весовые функции
    # позволяют оценить влияние факторов…» по фрагментам о других методах).
    named_method = any(_names(n, text) for names in methods for n in names)
    if methods and text and not named_method and not _NEGATION.search(text):
        return "", [*removed, *([] if off_method else [text])]
    return text, removed
