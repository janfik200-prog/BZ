"""Чистка текста после разбора: то, из-за чего слово не находит поиск по словам.

Обе беды — от PDF, а не от парсера, и чинятся одинаково для любого парсера:

1. Мягкий перенос (U+00AD) — место, где слово можно разорвать в конце строки. В
   части PDF (CyberLeninka и др.) им размечены все переносы, и при склейке строк
   слово остаётся разорванным: «ра¬ - бот», «рай¬ онов», «недоста¬ то¬ чно». Мягкий
   перенос всегда внутри слова — по нему слово склеивается вместе с пробелами и
   видимым дефисом. Обычный дефис в конце строки так не починить: «госу- дарственных»
   склеивать, а «рудно- магматических» — нет, без словаря не различить.
2. Буквы-двойники: в слове из кириллицы латинская «p» или «x» («веpxняя»), в формуле
   из латиницы кириллическая «О» («TiО»). Глазом не видно, поиск по словам не
   находит. Буквы меньшинства, у которых есть двойник в алфавите большинства,
   заменяются; если хоть одной двойника нет («Pbизотопного», «Feместорождение») —
   это не подмена, а пропущенный пробел, слово не трогается.

3. Буквы со сдвигом: шрифт рисует русские буквы, а коды у них сдвинуты на 0x1D6 —
   «Категория» читается «Ʉɚɬɟɝɨɪɢɹ», пробел — точкой («Зона.насыщения»). Глазом
   не видно (таблица выглядит как обычная), поиск не находит ничего. Сдвиг один на
   все буквы — возвращаются на место, если у документа таких букв много и после
   сдвига доля гласных как у русского текста: одно слово в транскрипции МФА
   («ɪ», «ə») не трогается.

Замер 13.10.2026 по корпусу (104 статьи): разорванных слов — 255 в двух статьях
(мягкие переносы — в 8), слов из смеси алфавитов — 60 в 15 статьях. Буквы со
сдвигом — таблица запасов из закрытой БЗ: 3032 буквы из 5872 знаков (29.10.2026).
"""

from __future__ import annotations

import re

from . import hyphens

# После мягкого переноса — пробелы, перевод строки (у PyMuPDF строки не склеены) и
# видимый дефис; абзац мягким переносом не кончается, поэтому строка — одна.
_SOFT_HYPHEN = re.compile("\u00ad[ \t]*\n?[ \t]*(?:-[ \t]*\n?[ \t]*)?")

_LATIN_TO_CYRILLIC = dict(zip("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху", strict=True))
_CYRILLIC_TO_LATIN = {c: lat for lat, c in _LATIN_TO_CYRILLIC.items()}
_MIXED_WORD = re.compile(
    r"\b(?=[A-Za-zА-Яа-яЁё]*[А-Яа-яЁё])(?=[A-Za-zА-Яа-яЁё]*[A-Za-z])[A-Za-zА-Яа-яЁё]+"
)
_CYRILLIC = set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
# Символ элемента, приклеенный к слову («Moпорфировых» — Mo-порфировых): две
# латинские буквы-двойника, но это не подмена, а пропущенный дефис.
_ELEMENT_PREFIX = re.compile(r"[A-Z][a-z][А-Яа-яЁё]")


# Мягкий перенос между двумя кусками слова — решается по кускам (join_soft_hyphens).
_SOFT_BREAK = re.compile(r"([^\W_]+)" + _SOFT_HYPHEN.pattern + r"([^\W_]+)")


def _soft_break(m: re.Match[str]) -> str:
    """Мягкий перенос между кусками: часть PDF (из LaTeX, из Word с особой разметкой) пишет
    им и настоящий дефис на конце строки — «Северо¬ Западный», «Au¬ содержащие», «КП¬1»
    (тест-план 6.08, 7.02). У букв с цифрой переноса не бывает — дефис; остальное — по
    правилам дефиса на конце строки (hyphens.py), которые не решили — склеить: мягкий
    перенос обычно внутри слова."""
    first, second = m.group(1), m.group(2)
    if first[-1].isdigit() or second[0].isdigit():
        return f"{first}-{second}"  # «КП¬1», «2¬3»
    if second.lower() in hyphens.HANGING:
        return f"{first}- {second}"  # «северо¬ и юго-западный»
    kept = hyphens.line_hyphen(first, second)
    return kept if kept and "-" in kept else first + second


def join_soft_hyphens(text: str) -> str:
    """Слово, разорванное мягким переносом, — целиком: «ра¬ - бот» → «работ»; настоящий
    дефис, записанный мягким переносом, — дефисом: «Северо¬ Западный» → «Северо-Западный»."""
    if "\u00ad" not in text:
        return text
    for _ in range(3):  # «недоста¬ то¬ чно»: кусок после склейки — снова первый
        joined = _SOFT_BREAK.sub(_soft_break, text)
        if joined == text:
            break
        text = joined
    return _SOFT_HYPHEN.sub("", text)


def _one_script(match: re.Match[str]) -> str:
    word = match.group()
    cyrillic = sum(ch in _CYRILLIC for ch in word)
    latin = len(word) - cyrillic
    if cyrillic == latin:
        return word  # не ясно, какой алфавит свой
    to_cyrillic = cyrillic > latin
    if to_cyrillic and _ELEMENT_PREFIX.match(word):
        return word
    swap = _LATIN_TO_CYRILLIC if to_cyrillic else _CYRILLIC_TO_LATIN
    minority = [ch for ch in word if (ch in _CYRILLIC) != to_cyrillic]
    if not all(ch in swap for ch in minority):
        return word
    return "".join(swap[ch] if (ch in _CYRILLIC) != to_cyrillic else ch for ch in word)


def fix_homoglyphs(text: str) -> str:
    """Слово из смеси алфавитов — в одном алфавите, если чужие буквы — двойники."""
    return _MIXED_WORD.sub(_one_script, text)


_SHIFT = 0x1D6  # «а» U+0430 записана как «ɚ» U+025A, «К» U+041A — как «Ʉ» U+0244
_SHIFTED = {chr(code - _SHIFT): chr(code) for code in (*range(0x410, 0x450), 0x401, 0x451)}
_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
SHIFTED_MIN = 20  # букв со сдвигом у документа — меньше этого не исправляется
# Пробел у такого шрифта — точка перед буквой после буквы или знака («Зона.насыщения»,
# «доли,.m») или служебный знак («тыс\x11м»); перенос «насы-.щения» — «насы- щения», его
# решают правила дефисов. Точка перед цифрой («скв.620») не трогается.
_DOT_SPACE = re.compile(r"(?<=[^\W\d_]|[-,;:)])\.(?=[^\W\d_])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")


def shifted_letters(texts: list[str]) -> int:
    """Сколько у документа букв со сдвигом — 0, если сдвиг не подтвердился: после
    сдвига доля гласных не как у русского текста (около 42 %), значит, это не он."""
    letters = [_SHIFTED[ch] for text in texts for ch in text if ch in _SHIFTED]
    if len(letters) < SHIFTED_MIN:
        return 0
    vowels = sum(ch in _VOWELS for ch in letters) / len(letters)
    return len(letters) if 0.30 <= vowels <= 0.55 else 0


def unshift_cyrillic(text: str) -> str:
    """Буквы со сдвигом — на место, если их в тексте не меньше половины всех букв: в
    обычном абзаце с одним словом в транскрипции ничего не меняется. Зовут, когда сдвиг
    у документа подтвердил shifted_letters."""
    shifted = sum(ch in _SHIFTED for ch in text)
    if not shifted or shifted * 2 < sum(ch.isalpha() for ch in text):
        return text
    text = "".join(_SHIFTED.get(ch, ch) for ch in text)
    return _CONTROL.sub(" ", _DOT_SPACE.sub(" ", text))


def tidy_text(text: str) -> str:
    """Всё вместе: мягкие переносы, затем буквы-двойники."""
    return fix_homoglyphs(join_soft_hyphens(text)) if text else text
