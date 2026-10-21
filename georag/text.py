"""Работа с русским текстом — одна на весь проект.

Раньше окончания русских слов снимались в двух местах двумя разными списками
(проверка разбора и словарь понятий), беглая гласная («узел → узла») была только
в одном, а отбор статей брал внутренние функции проверки разбора. Теперь всё
здесь:

* normalize — строка для сравнения: без регистра, «ё» как «е», пробелы схлопнуты;
* ru_stem / stem_variants — основа слова и вариант с выпавшей гласной;
* phrase_pattern — шаблон фразы в любой форме: «Анабарский щит» находит
  «Анабарского щита», «рудный узел» — «рудного узла»;
* keyword_hits — ключевые слова оценки в тексте (evaluation.py; ими же
  генератор вопросов graph/questions.py проверяет свои ключи);
* translit / translit_key — имя другим алфавитом («Kular» → «Кулар») и ключ
  написания, одинаковый у «Koldar», «Колдар» и «Кольдар»: им keyword_hits узнаёт
  имя собственное, написанное по-русски, по латинскому ключу (и наоборот);
* split_sentences — текст на предложения, сокращения и инициалы не рвут.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import product

# Окончания, которые снимаются с русского слова. Длинные первыми: «-иями»
# раньше «-ями», иначе останется лишняя «и».
RU_ENDINGS = tuple(
    sorted(
        {
            "ического",
            "ическому",
            "ическими",
            "ическая",
            "ические",
            "ический",
            "иями",
            "ями",
            "ами",
            "ией",
            "иях",
            "ием",
            "ого",
            "его",
            "ому",
            "ему",
            "ыми",
            "ими",
            "ов",
            "ев",
            "ей",
            "ой",
            "ий",
            "ый",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ых",
            "их",
            "ую",
            "юю",
            "ом",
            "ем",
            "ам",
            "ям",
            "ах",
            "ях",
            "ия",
            "ии",
            "ию",
            "ья",
            "ье",
            "ьи",
            "ью",
            "а",
            "я",
            "о",
            "е",
            "ы",
            "и",
            "у",
            "ю",
            "ь",
            "й",
        },
        key=len,
        reverse=True,
    )
)

_CONS = "бвгджзклмнпрстфхцчшщ"
# Беглая гласная: узел → узла, песок → песка.
_FLEETING = re.compile(rf"^(.*[{_CONS}])[еоё]([{_CONS}])$")


def normalize(text: str) -> str:
    """Строка для сравнения: без регистра, «ё» как «е», пробелы схлопнуты."""
    text = unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def ru_stem(word: str, min_len: int = 3) -> str:
    """Основа русского слова: срезано типовое окончание. Латиница не трогается."""
    if not re.search(r"[а-яё]", word):
        return word
    for ending in RU_ENDINGS:
        if len(word) - len(ending) >= min_len and word.endswith(ending):
            return word[: -len(ending)]
    return word


def stem_variants(word: str, min_len: int = 3) -> list[str]:
    """Основа плюс вариант с выпавшей беглой гласной, длинные первыми."""
    variants = {ru_stem(word, min_len)}
    match = _FLEETING.match(word)
    if match:
        variants.add(match.group(1) + match.group(2))
    return sorted(variants, key=len, reverse=True)


def phrase_pattern(phrase: str, min_len: int | None = None) -> re.Pattern[str]:
    """Фраза в любой форме, по нормализованному тексту (см. normalize).

    Одиночный термин требует основы длиннее: его не подпирают соседние слова,
    и короткий корень начинает совпадать с чем попало. Исключение — короткие
    слова («зона», «руда»): при основе от четырёх букв у них не снимается
    окончание, и «зоны», «руды» не находились. Совпадение — только с начала
    слова: «щит» не находится в «защиты», «узел» — в «узелковые».
    """
    words = [w for w in re.split(r"[\s\-]+", normalize(phrase)) if w]
    single = len(words) == 1
    parts = []
    for w in words:
        n = min_len if min_len is not None else (4 if single and len(w) > 5 else 3)
        parts.append("(?:" + "|".join(re.escape(v) for v in stem_variants(w, n)) + r")[а-яa-z]*")
    return re.compile(r"(?<![а-яa-z0-9])" + r"[\s\-]+".join(parts))


# --------------------------------------------------------------------------- #
#  Ключевые слова оценки
# --------------------------------------------------------------------------- #
_ACRONYM = re.compile(r"[A-ZА-ЯЁ0-9\-]{2,6}")
_SYMBOL = re.compile(r"[A-Z][a-z]?")  # символ элемента: Au, U, Pb
_EN_WORD = re.compile(r"[a-z]{4,}")


def _en_stem(word: str) -> str:
    """Английское слово без множественного числа — общее начало всех форм:
    valleys → valley, anomalies и anomaly → anomal, processes → process.
    На ss, us, is не кончается множественное: gneiss, basis остаются как есть."""
    if word.endswith("ies") and len(word) >= 7:
        return word[:-3]
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("y") and len(word) >= 5 and word[-2] not in "aeiou":
        return word[:-1]  # anomaly → anomal: находит и anomalies
    return word


# Основа на две согласные: в начальной форме между ними бывает беглая гласная.
_CLUSTER = re.compile(rf"[{_CONS}][клнрц]$")


def _fleeting_forms(word: str) -> list[str]:
    """Слово с беглой гласной, вставленной обратно: «пески» → «песок», «сланцы» →
    «сланец». phrase_pattern выпадение гласной знает («песок» находит «песка»), а
    обратное — нет: ключ «пески» не находил «песок»."""
    stem = ru_stem(word)
    if not re.search(r"[а-я]", word) or len(stem) < 3 or not _CLUSTER.search(stem):
        return []
    return [stem[:-1] + vowel + stem[-1] for vowel in "ое"]


def keyword_pattern(word: str) -> re.Pattern[str]:
    """Ключевое слово в любой форме, с начала слова: у русского снимается окончание
    (и беглая гласная — в обе стороны), у английского — множественное число.
    Аббревиатура и символ элемента — целиком: ROC не находится в «rock», Au — в
    «author». Ищется в normalize(текст)."""
    if _ACRONYM.fullmatch(word) or _SYMBOL.fullmatch(word):
        return re.compile(rf"(?<!\w){re.escape(normalize(word))}(?!\w)")
    stems = _EN_WORD.sub(lambda m: _en_stem(m.group()), normalize(word))
    words = [w for w in re.split(r"[\s\-]+", stems) if w]
    forms = product(*([w, *_fleeting_forms(w)] for w in words))
    return re.compile(
        r"(?<!\w)(?:" + "|".join(phrase_pattern(" ".join(f)).pattern for f in forms) + ")"
    )


def keyword_hits(groups: list[list[str]], text: str) -> tuple[int, list[str]]:
    """Сколько ключевых мыслей встретилось в тексте и какие — нет. Мысль — группа
    вариантов («гематит|hematite»): засчитывается любой.

    Имя собственное из одного слова («Kular», «Okhotsk-Koryak») ищется по ключу
    написания (translit_key) — в любом алфавите и транслитерации: «район Кулар»,
    «массив Кольдар», — но не как начало составного имени: «Kular» и «Кулар» не находят
    «Куларо-Нерский». Так же, вдобавок к обычному поиску, — заимствованный термин,
    если в группе он есть в обоих алфавитах с одним ключом («альюнит|alunite» находит
    «алюнит»)."""
    norm = normalize(text)
    found, missing = 0, []
    words: list[tuple[str, bool]] | None = None
    for group in groups:
        loose = _translit_keys(group)
        if any(keyword_pattern(w).search(norm) for w in group if not loose.get(w, ([], False))[1]):
            found += 1
            continue
        if loose:
            words = _text_keys(norm) if words is None else words
            if any(_sounds_in(parts, words) for parts, _ in loose.values()):
                found += 1
                continue
        missing.append(group[0])
    return found, missing


# --------------------------------------------------------------------------- #
#  Транслитерация имён: Kular ↔ Кулар
# --------------------------------------------------------------------------- #
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_RU_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}  # fmt: skip
_TO_LATIN = str.maketrans({**_RU_LAT, **{k.upper(): v.capitalize() for k, v in _RU_LAT.items()}})
_LAT_PAIRS = {
    "shch": "щ", "kh": "х", "zh": "ж", "ch": "ч", "sh": "ш", "ts": "ц", "ya": "я", "yu": "ю",
    "ye": "е", "yo": "йо", "ph": "ф", "th": "т", "ck": "к", "gh": "г", "ou": "у", "qu": "кв",
    "dj": "дж",
}  # fmt: skip
_LAT_ONE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "z": "з",
}  # fmt: skip
_LAT_UNIT = re.compile("|".join(_LAT_PAIRS) + "|.")


def _to_cyrillic(word: str) -> str:
    low, out = word.lower(), []
    for m in _LAT_UNIT.finditer(low):
        unit, prev = m.group(), low[m.start() - 1] if m.start() else ""
        if unit == "ts" and low[m.end() : m.end() + 1] == "k":
            out.append("тс")  # Okhotsk → Охотск, а не «Охоцк»
        elif unit == "yo":
            out.append("ьо" if prev and prev not in "aeiou" else "йо")  # Nyong → Ньонг
        elif unit in _LAT_PAIRS:
            out.append(_LAT_PAIRS[unit])
        elif unit == "y" and prev == "k" and m.end() == len(low):
            out.append("ий")  # Nezhdaninsky → Нежданинский
        elif unit == "y":
            out.append("й" if prev in "aeiou" else "и")  # Aktogay → Актогай
        elif unit == "e" and not prev:
            out.append("э")  # Effon → Эффон
        elif unit == "a" and prev == "i" and m.end() == len(low):
            out.append("я")  # Bahia → Бахия
        else:
            out.append(_LAT_ONE.get(unit, unit))
    text = "".join(out)
    return text[:1].upper() + text[1:] if word[:1].isupper() else text


def translit(text: str) -> str:
    """Имя другим алфавитом, по простым правилам: «Kular» → «Кулар», «Isfahan» →
    «Исфахан», «Кулар» → «Kular». Как пишут по-русски на самом деле, правила не
    знают («Bahia» — «Бахия», а не «Баия»), поэтому сравнивают имена ключом
    translit_key, а не этой строкой."""
    if _CYR.search(text or ""):
        return (text or "").translate(_TO_LATIN)
    return re.sub(r"[A-Za-z]+", lambda m: _to_cyrillic(m.group()), text or "")


# Ключ написания: что по-разному передают при транслитерации, сводится к одному.
_KEY_RU = {**_RU_LAT, "ж": "j", "й": "i", "х": "h", "щ": "sh", "ы": "i", "ю": "u", "я": "a"}
_KEY_FOLDS = (
    (re.compile(r"shch|sch"), "sh"),
    (re.compile(r"kh"), "h"),
    (re.compile(r"dzh|dj|zh"), "j"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"th"), "t"),
    (re.compile(r"gh"), "g"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"qu"), "kv"),
    (re.compile(r"q"), "k"),
    (re.compile(r"w"), "v"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"ou"), "u"),
    (re.compile(r"c(?!h)"), "k"),
    (re.compile(r"y(?=[aeou])"), ""),  # Koryak, Коряк → korak
    (re.compile(r"y"), "i"),
    (re.compile(r"[^a-z]"), ""),
    (re.compile(r"(.)\1+"), r"\1"),  # двойные буквы: Kullar, Куллар → kular
)


def translit_key(word: str) -> str:
    """Ключ написания слова — один у разных транслитераций одного имени: «Kular»,
    «Кулар» → kular; «Koldar», «Колдар», «Кольдар» → koldar; «Isfahan», «Исфахан» →
    isfahan; «alunite», «алюнит», «альюнит» → alunit. Мягкий знак, «й/ы/и», «я/ю»
    против «a/u», «х/kh/h», «дж/j», двойные буквы — не различаются."""
    low = normalize(word)
    if _CYR.search(low):
        low = "".join(_KEY_RU.get(ch, ch) for ch in low)
    for pattern, repl in _KEY_FOLDS:
        low = pattern.sub(repl, low)
    return low


def _key_parts(variant: str) -> list[str]:
    """Ключи написания частей варианта («Okhotsk-Koryak» → ohotsk, korak): у русского
    слова — без окончания, у английского — без множественного числа и немого «e»
    (alunite → alunit: по-русски «алунит»)."""
    parts = []
    for part in re.split(r"[\s\-]+", normalize(variant)):
        if _CYR.search(part):
            part = ru_stem(part)
        elif part:
            part = _en_stem(part)
            if len(part) >= 5 and re.search(r"[^aeiouy]e$", part):
                part = part[:-1]
        parts.append(translit_key(part))
    return parts


def _translit_keys(group: list[str]) -> dict[str, tuple[list[str], bool]]:
    """Варианты группы, которые узнаются по ключу написания, → (ключи частей, имя ли).
    Это одно слово (можно через дефис) с ключом от пяти букв, и оно — имя собственное
    (с заглавной, не сокращение) или термин, который в группе есть в обоих алфавитах
    с одним ключом («альюнит|alunite»). Обычные слова так не сравниваются: «area»
    нашлась бы в «ареале»; короткие — тоже: ключ «Якут» (akut) нашёлся бы в «acute»."""
    single = [v for v in group if v.strip() and not re.search(r"\s", v.strip())]
    keyed = {v: _key_parts(v) for v in single}
    out: dict[str, tuple[list[str], bool]] = {}
    for v, parts in keyed.items():
        if sum(map(len, parts)) < 5 or min(map(len, parts)) < 3:
            continue
        proper = v[:1].isupper() and not v.isupper()
        twin = any(
            o != v and keyed[o] == parts and bool(_CYR.search(o)) != bool(_CYR.search(v))
            for o in keyed
        )
        if proper or twin:
            out[v] = (parts, proper)
    return out


_TEXT_WORD = re.compile(r"[a-zа-я0-9]+(?:['’][a-zа-я]+)*")


def _text_keys(norm: str) -> list[tuple[str, bool]]:
    """Слова текста — ключами написания; и идёт ли за словом через дефис другое слово."""
    return [
        (translit_key(m.group()), bool(re.match(r"-[a-zа-я]", norm[m.end() : m.end() + 2])))
        for m in _TEXT_WORD.finditer(norm)
    ]


def _sounds_in(parts: list[str], words: list[tuple[str, bool]]) -> bool:
    """Есть ли в тексте подряд слова, которые начинаются с этих ключей. Последнее
    слово — не первая часть составного имени: «Kular» не находит «Куларо-Нерский»."""
    n = len(parts)
    for i in range(len(words) - n + 1):
        if not all(words[i + j][0].startswith(parts[j]) for j in range(n)):
            continue
        last, joined = words[i + n - 1]
        if joined and last[len(parts[-1]) :] in ("", "o", "e"):
            continue
        return True
    return False


# --------------------------------------------------------------------------- #
#  Предложения
# --------------------------------------------------------------------------- #
# Сокращения, после которых точка не кончает предложение.
_ABBREV = re.compile(
    r"(?:\b(?:т\.\s?е|т\.\s?к|т\.\s?н|и\s?др|и\s?т\.\s?д|и\s?т\.\s?п|см|рис|табл|гг?|вв?|"
    r"с|стр|им|ок|млн|млрд|тыс|кв|e\.g|i\.e|et\s?al|fig|figs|tab|vs|approx)"
    r"|\b[А-ЯЁA-Z])\.$",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЁA-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Текст → предложения. Сокращения и инициалы предложение не рвут."""
    text = " ".join((text or "").split())
    if not text:
        return []
    sentences: list[str] = []
    for piece in _SPLIT.split(text):
        if sentences and _ABBREV.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


# --------------------------------------------------------------------------- #
#  Цитаты и имена — для фактов графа
# --------------------------------------------------------------------------- #
_QUOTES = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "ё": "е",
        "Ё": "е",
    }
)


def _canon_with_map(text: str) -> tuple[str, list[int]]:
    """Текст для сравнения и карта: символ сравниваемого текста → место в исходном.

    Сравнение не различает регистр, «ё/е», виды кавычек и тире и пробелы —
    модель переписывает их как хочет. Карта нужна, чтобы вернуть цитату
    в том виде, в каком она стоит в статье, а не в пересказе модели.
    """
    out: list[str] = []
    where: list[int] = []
    last_space = True
    for i, ch in enumerate(text or ""):
        if ch == "\xad":  # мягкий перенос
            continue
        if ch.isspace():
            if not last_space:
                out.append(" ")
                where.append(i)
            last_space = True
            continue
        out.append(ch.translate(_QUOTES).lower())
        where.append(i)
        last_space = False
    return "".join(out), where


def locate_quote(quote: str, text: str) -> str | None:
    """Найти цитату во фрагменте; вернуть её в написании статьи или None.

    Модель иногда сокращает цитату многоточием — тогда каждая часть должна
    найтись в тексте по порядку, и возвращается кусок от первой до последней.
    """
    parts = [p.strip(" ,;:") for p in re.split(r"…|\.\.\.", quote or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 20:
        return None
    canon_text, where = _canon_with_map(text)
    start_at, first, last = 0, None, None
    for part in parts:
        canon_part = _canon_with_map(part)[0].strip()
        pos = canon_text.find(canon_part, start_at)
        if pos < 0:
            return None
        if first is None:
            first = pos
        last = pos + len(canon_part)
        start_at = last
    assert first is not None and last is not None  # цикл выше прошёл хотя бы раз
    begin = where[first]
    end = where[last - 1] + 1
    return " ".join(text[begin:end].split())


_WORDS = re.compile(r"[a-zа-я0-9]+(?:['’][a-z]+)?")
# Слова ключа имени — и с греческими буквами: «β-кварц» и «α-кварц» — разные
# минералы, «δ18O» и «18O», «εNd» и «Nd» — разные величины. Без них буква
# терялась, и «β-кварц», «α-кварц», «кварц» получали один ключ «кварц». mentions
# слова текста статьи берёт по-прежнему через _WORDS: одна буква («β») там всё
# равно не значимое слово (короче трёх), а греческие буквы в тексте из PDF часто
# потеряны или искажены.
_KEY_WORDS = re.compile(r"[a-zа-яα-ω0-9]+(?:['’][a-z]+)?")
# Беглая гласная для ключа — только там, где она и бывает (узел, угол, песок,
# конец, ветер) и у основы от четырёх букв. Шире нельзя: «зона» давала «зн»,
# «метод» — «метд», а «соль» и «сель» сходились в одно «сл».
_KEY_FLEETING = re.compile(rf"^(?=.{{4}})(.+[{_CONS}])[еоё]([лкцнр])$")
# Существительное на -ом, -ов, -ем, -ев: у «разлом» это не окончание, но срезается
# как окончание, а у «разлома» — нет. Поэтому хвост срезается у любой основы:
# «разлом», «разлома», «разломов» — одно «разл»; «покров», «покрова» — «покр».
_KEY_TAIL = re.compile(r"(?<=...)(?:ом|ов|ем|ев)$")


def _key_word(word: str) -> str:
    if re.search(r"[а-я]", word):
        stem = _KEY_TAIL.sub("", ru_stem(word))
        match = _KEY_FLEETING.match(stem)
        return match.group(1) + match.group(2) if match else stem
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        if word.endswith(("sses", "xes", "ches", "shes")):
            return word[:-2]  # processes → process
        return word[:-1]  # basins → basin
    return word


def name_key(name: str) -> str:
    """Ключ имени: одно и то же в любом падеже и числе даёт один ключ.

    «Донецкий бассейн», «Донецкого бассейна», «донецкий  бассейн» → один
    ключ; «Donetsk basins» и «Donetsk basin» — тоже. Разные языки ключ не
    сводит — это делает словарь синонимов. Греческая буква — часть имени:
    «β-кварца» и «β-кварц» — один ключ, «β-кварц» и «α-кварц» — разные.
    """
    return " ".join(_key_word(w) for w in _KEY_WORDS.findall(normalize(name).replace("-", " ")))


def mentions(name: str, text: str) -> bool:
    """Упомянуто ли имя в тексте: целиком в любой форме — или почти все его
    значимые слова (модель пишет «Leimengou porphyry system», а в тексте
    «Leimengou porphyry Mo system»)."""
    name = (name or "").strip()
    if len(name) < 2:
        return False
    norm = normalize(text)
    # Целиком — но как отдельные слова: «магнетит» не упомянут в «титаномагнетит».
    if re.search(r"(?<![а-яa-z0-9])" + re.escape(normalize(name)) + r"(?![а-яa-z0-9])", norm):
        return True
    try:
        if phrase_pattern(name).search(norm):
            return True
    except re.error:
        pass
    original = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", name)
    words = [(w, normalize(w)) for w in original if len(w) >= 3]
    if len(words) < 3:
        return False  # короткое имя — только целиком
    have = {_key_word(w) for w in _WORDS.findall(norm.replace("-", " "))}
    missing = [w for w, low in words if not any(h.startswith(_key_word(low)) for h in have)]
    # Можно пропустить одно общее слово, но не имя собственное и не число:
    # «Онежский рудный район» по тексту «рудного района» — не упоминание.
    return len(missing) <= 1 and not any(w[0].isupper() or w[0].isdigit() for w in missing)
