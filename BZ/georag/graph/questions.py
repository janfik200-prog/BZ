"""Вопросы для оценки — из фактов графа, с заранее известным ответом.

Оценка (evaluation.py) честна настолько, насколько хороши вопросы. Писать их
руками долго, а у каждого факта графа уже есть всё нужное: цитата (ответ),
фрагмент и статья, где он сказан. Поэтому вопрос строится из факта:

    факт:     «Малеевское месторождение» — входит в — «Зыряновский рудный район»
    вопрос:   «В какой рудный район входит Малеевское месторождение?»  (пишет модель)
    ключевые: «Зыряновский рудный район»    (ответ — второй конец факта)
    статьи:   doc_id статьи с цитатой       (что поиск обязан найти)

Первый набор (6.10; разбор — .local/logs/research-20261015) наполовину проверял
сам себя: ключ — не тот конец факта или повтор вопроса, «в данном регионе»,
«согласно источнику [13]», английский ключ при русских ответах. Поэтому теперь:

* Ответ — второй конец факта («к» в направлении связи графа), и ключ — он же в
  нескольких написаниях: по-русски и как в статье («гематит|hematite»; русское
  даёт модель). Ключ сверяется тем же сопоставлением, что и оценка
  (text.keyword_hits): он должен найтись в цитате — иначе оценка не узнала бы
  верный фрагмент — и не должен стоять в вопросе ни в каком написании.
* Перевод модели — не единственное написание: верные ответы не засчитывались из-за
  него («провинция Исфаган» при ответе «провинции Исфахан», «альюнит» — «алюнитом»).
  Поэтому в ключ идут и написания той же сущности из словаря синонимов и графа
  (spellings), а у места — его имя без вида места и оно же другим алфавитом
  (name_forms: «Isfahan Province» → «Isfahan», «Исфахан»; «Kular area» → «Kular»,
  «Кулар»). Готовому набору ключи пересобираются так же, без модели (rekey).
* Модель отвечает на готовый вопрос по цитате, и её ответ должен попасть в ключ.
  Так отсеиваются вопросы «наоборот» (спросили, что образуется, а ключ — из
  чего) и вопросы, на которые цитата не отвечает.
* Вопрос понятен без статьи: в нём есть имя собственное — объект, район,
  месторождение из цитаты или названия статьи; нет «данного региона», «этой
  площади», ссылок на источник, номеров ссылок и рисунков, чисел.
* Факты — только с типом связи (kinds.py) и без «предположительно»; ответ —
  конкретного вида (ANSWER_KINDS), до трёх слов, не перечень и не общее слово
  («породы», «deposits»). Сначала факты, где спрашиваемое — имя собственное и
  проверок больше; по очереди из разных статей — чтобы проверялась вся база.

Не прошёл проверку — берётся следующий факт: лучше меньше вопросов, чем вопрос,
который проверяет сам себя. Вопрос об одном факте помечен «кратко: да»: оценка
не требует от ответа развёрнутости (evaluation.py).

    python georag.py questions              → .local/eval-graph.yaml
    python georag.py eval --questions .local/eval-graph.yaml

Нет модели — вопрос по шаблону, без этих проверок (только для пробы).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..paths import LOCAL
from ..text import keyword_hits, keyword_pattern, name_key, normalize, translit
from .kinds import HEDGE

DEFAULT_OUT = LOCAL / "eval-graph.yaml"
LIMIT = 40  # вопросов в наборе
WORKERS = 4  # запросов к модели одновременно: модель общая
PER_DOC = 3  # фактов-кандидатов из одной статьи
QUOTE_CHARS = 600

SYSTEM = "Ты составляешь вопросы для проверки поиска по геологическим статьям. Отвечай только JSON."
USER = """Статья: «{title}»
Цитата из статьи: «{quote}»
Факт из цитаты: «{src}» — {relation} — «{dst}».

Составь вопрос геолога, ответ на который — «{dst}» (а не «{src}»).
1. Вопрос понятен без статьи: назови в нём «{src}» и конкретный объект по имени —
месторождение, рудное поле, район, пояс, массив, свиту — из цитаты или названия статьи.
2. Имена собственные пиши точно как в статье: латинские — латиницей, не переводи и не
транслитерируй («Maru Schist Belt», а не «Мару шистовый пояс»).
3. Не называй «{dst}», его перевод и синонимы — это ответ.
4. Нельзя: «данный», «этот», «изучаемый» регион или район, «в статье», «согласно
источнику», номера ссылок и рисунков, числа.
5. По-русски, 8–25 слов, без кавычек.
Если ни в цитате, ни в названии статьи нет имени объекта, о котором факт, — верни
пустой вопрос.

Верни JSON: {{"вопрос": "..."}}"""

# Ответ по-русски — отдельным запросом: попутно с вопросом модель переводила наугад
# («siltstones» — «песчаники», «carbonatite intrusion» — «углеродистый массив»).
TERM_SYSTEM = "Ты переводчик геологических терминов. Отвечай только JSON."
TERM = """Термин из геологической статьи: «{term}»
Контекст: «{quote}»

Напиши этот термин так, как его пишут в русских геологических статьях, и по-английски:
точный термин, а не пересказ; 1–3 слова, именительный падеж, без пояснений. Примеры:
hematite — гематит; pillow basalts — подушечные базальты; shear zone — зона сдвига;
porphyry copper deposit — медно-порфировое месторождение; березиты — beresites.
Имя собственное (месторождение, район, свита, комплекс) оставь как в статье, латиницей.

Верни JSON: {{"ru": "...", "en": "..."}}"""
# Перевод проверяется обратным: «глинистые сланцы» обратно — «clay shales», а не
# «siltstones», и такое написание в ключ не идёт.
BACK = """Переведи геологический термин {direction}: «{term}». Точный термин, 1–3 слова,
без пояснений.

Верни JSON: {{"term": "..."}}"""

CHECK_SYSTEM = "Ты отвечаешь на вопрос по цитате из геологической статьи. Отвечай только JSON."
CHECK = """Цитата из статьи: «{quote}»

Вопрос: {question}

Ответь на вопрос только по цитате: сам ответ, 1–3 слова, как в цитате, без пояснений.
Если цитата на вопрос не отвечает — пустая строка.

Верни JSON: {{"ответ": "..."}}"""

# Ответ — конкретное: что найти в статье и назвать одним-тремя словами.
ANSWER_KINDS = ("минерал", "порода", "объект", "оруденение", "метод")
MAX_ANSWER_WORDS = 3
MAX_NAME_WORDS = 6
# Общее слово — не ответ: «Какие породы слагают…? — Породы».
GENERIC = frozenset(
    {
        "rock", "rocks", "host rock", "host rocks", "mineral", "minerals", "mineralization",
        "mineralisation", "deposit", "deposits", "ore", "ores", "intrusion", "intrusions", "unit",
        "units", "formation", "formations", "sediments", "basement", "vein", "veins", "zone",
        "zones", "area", "region", "study area", "succession", "successions", "rock succession",
        "rock successions", "sequence", "sequences", "complex", "порода", "породы",
        "вмещающие породы", "минерал", "минералы", "минерализация", "оруденение",
        "месторождение", "месторождения", "руда", "руды", "интрузия", "интрузии", "жила", "жилы",
        "толща", "отложения", "зона", "зоны", "район", "регион", "площадь", "участок", "комплекс",
    }
)  # fmt: skip
_GENERIC_KEYS = frozenset(name_key(x) for x in GENERIC)  # в любом падеже и числе
_GENERIC_HEAD = frozenset({"rock", "rocks", "порода", "породы", "пород"})  # «породы комплекса»
# Одно прилагательное — не ответ: «dolomitic», «доломитовая» не находят «доломиты».
_ADJECTIVE = re.compile(r"(?:ic|ical|ous|ive|ый|ий|ой|ая|яя|ое|ее|ые|ие|ых|их)$")
_NOUN_IE = re.compile(r"(?:ни|ти)е$")  # «окварцевание», «развитие» — существительные
# Имя — только буквы, пробел, дефис, апостроф, точка: без чисел (количества, возраст,
# номера), скобок и «=» из строк таблиц, подчёркиваний и мягких переносов разметки.
_NAME = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\s\-'’.]*")
_GLUED = re.compile(r"[a-z]{2,}[A-Z]")  # слова, склеенные разметкой: «AcidIntermediate»
_LIST = re.compile(r",|/|;|\b(?:and|or|и|или)\b", re.IGNORECASE)
# Вопрос, которому нужна статья перед глазами: номер ссылки или рисунка, число,
# «данный район», «согласно статье», «в контексте», «по данным Иванова».
_VAGUE = re.compile(
    r"\d|\[|рис\.|рисун|\bfig|табл"
    r"|(?<![а-яё])(?:данн|эт|указанн|изучаем|исследуем|рассматриваем)\w*\s+"
    r"(?:регион|район|участ|площад|территор|месторожд|объект|зон|бассейн|пояс|стать|работ)"
    r"|согласно|стать|текст|контекст|упомян|результат\w* исследован|в источник|в работе"
    r"|автор|по данным",
    re.IGNORECASE,
)
_MIXED = re.compile(r"[A-Za-z][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]")  # «Пенinsula», «Зharma»
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_WORD = re.compile(r"[^\W\d_]+")


def _usable(name: str, max_words: int = MAX_NAME_WORDS) -> bool:
    """Годится ли имя для вопроса: коротко, только слова — без чисел и разметки."""
    name = " ".join((name or "").split())
    return (
        bool(_NAME.fullmatch(name)) and not _GLUED.search(name) and len(name.split()) <= max_words
    )


def _generic(name: str) -> bool:
    """Общее слово, а не ответ: «породы», «deposits», «породы комплекса», одно
    прилагательное («dolomitic»)."""
    words = normalize(name).split()
    if not words or name_key(name) in _GENERIC_KEYS:
        return True
    if len(words) > 1:
        return words[0] in _GENERIC_HEAD
    return bool(_ADJECTIVE.search(words[0])) and not _NOUN_IE.search(words[0])


def _answer_ok(name: str) -> bool:
    """Годится ли конец факта в ответ: до трёх слов, не перечень и не общее слово."""
    return _usable(name, MAX_ANSWER_WORDS) and not _LIST.search(name) and not _generic(name)


def _ends(fact: dict[str, Any]) -> tuple[str, str, str, str]:
    """Концы факта в направлении связи графа, как написано в статье: (от, к, вид от,
    вид к). Тип связи бывает развёрнут (rel_flip): «щит включает зону» — «зона входит
    в щит», и тогда «к» — это src факта."""
    src, dst = fact["src"], fact["dst"]
    src_kind, dst_kind = fact.get("src_kind") or "", fact.get("dst_kind") or ""
    if fact.get("rel_flip"):
        return dst, src, dst_kind, src_kind
    return src, dst, src_kind, dst_kind


def _hits(variants: list[str], text: str) -> bool:
    """Нашёлся ли ключ в тексте — тем же сопоставлением, что у оценки."""
    return keyword_hits([variants], text)[0] > 0


def candidates(g: Any, per_doc: int = PER_DOC) -> list[tuple[Any, dict[str, Any]]]:
    """Факты для вопросов: (связь, факт). По очереди из разных статей, в статье —
    сначала спрашиваемое с именем собственным, потом подтверждённое большим числом
    проверок и с ответом покороче. От связи — один факт, самый подходящий."""
    by_doc: dict[str, list[tuple[tuple[int, int, int, int], Any, dict[str, Any]]]] = {}
    for link in g.links.values():
        if link.relation.startswith(HEDGE):  # предположение статьи — не факт для проверки
            continue
        best: tuple[tuple[int, int, int, int], Any, dict[str, Any]] | None = None
        for fact in link.facts:
            src, dst, _, dst_kind = _ends(fact)
            if not fact.get("rel_type") or dst_kind not in ANSWER_KINDS:
                continue
            if not (_usable(src) and _answer_ok(dst) and _hits([dst], fact["quote"])):
                continue
            rank = (
                0 if g.proper(link.src) else 1,
                -int(fact.get("votes") or 0),
                len(dst.split()),
                int(fact.get("id") or 0),
            )
            if best is None or rank < best[0]:
                best = (rank, link, fact)
        if best is not None:
            by_doc.setdefault(best[2]["doc_id"], []).append(best)
    queues = [sorted(found, key=lambda x: x[0])[:per_doc] for found in by_doc.values()]
    return [
        (queue[i][1], queue[i][2]) for i in range(per_doc) for queue in queues if i < len(queue)
    ]


def template(link: Any) -> str:
    """Вопрос без модели: имя и связь, ответ — пропуск."""
    return f"«{link.src}» — {link.relation} — что именно, по статьям базы?"


def _variants(*names: Any) -> list[str]:
    """Написания ответа для ключа: без повторов, короткие, не общие слова; русские —
    первыми (их и покажет отчёт, если ответ не назван)."""
    out: list[str] = []
    for raw in names:
        name = " ".join(str(raw or "").split()).strip(" «»\"'.")
        if (
            _usable(name, MAX_ANSWER_WORDS + 1)
            and not _generic(name)
            and normalize(name) not in {normalize(v) for v in out}
        ):
            out.append(name)
    return sorted(out, key=lambda v: not _CYRILLIC.search(v))


def _more(have: list[str], *names: str) -> list[str]:
    """Написания have и ещё names — те, которых в have нет ни в каком падеже и числе."""
    keys = {name_key(v) for v in have}
    extra = []
    for name in names:
        key = name_key(name)
        if key and key not in keys:
            keys.add(key)
            extra.append(name)
    return _variants(*have, *extra)


def spellings(g: Any, *names: str) -> list[str]:
    """Другие написания той же сущности графа: из словаря синонимов (каноническое имя
    и его список) и как её писали в фактах — в том числе сведённые автосинонимами
    (таблица entity_names, autosyn.py): граф их уже свёл в один узел."""
    out: list[str] = []
    for name in names:
        canon = g.syn.name(name) or name
        out += [canon, *g.syn.groups.get(canon, [])]
        entity = g.entities.get(name) or g.entities.get(canon) or {}
        out += sorted(entity.get("spellings") or ())
    return out


# Вид места в его имени: «Isfahan Province», «Bahia State», «Zgounder Mine», «провинция
# Исфаган». Слова со строчной — тоже вид места («Kular area», «Koldar intrusive massif»).
_PLACE_WORDS = frozenset(
    name_key(w)
    for w in (
        "province", "state", "area", "district", "region", "county", "prefecture",
        "department", "territory", "zone", "belt", "field", "massif", "intrusion", "intrusive",
        "pluton", "complex", "formation", "group", "supergroup", "subgroup", "series", "system",
        "suite", "sequence", "basin", "craton", "shield", "block", "terrane", "uplift",
        "deposit", "mine", "prospect", "occurrence", "cluster", "провинция", "штат", "район",
        "область", "округ", "регион", "зона", "пояс", "поле", "массив", "интрузия", "интрузив",
        "плутон", "комплекс", "свита", "серия", "система", "бассейн", "кратон", "щит", "блок",
        "террейн", "поднятие", "месторождение", "рудник", "рудопроявление", "узел",
    )
)  # fmt: skip
# Слова с заглавной, которые не имя: стороны света и «верхний — нижний».
_NOT_NAMES = frozenset(
    name_key(w)
    for w in (
        "north", "south", "east", "west", "central", "upper", "lower", "middle", "inner",
        "outer", "greater", "lesser", "great", "little", "new", "old", "northeast", "northwest",
        "southeast", "southwest",
    )
)  # fmt: skip
# Прилагательное — не имя: «Nigerian», «Dalradian», «Chinese», «Eastern», «Кольдарский».
_NAME_ADJECTIVE = re.compile(r"(?:ic|ian|ean|ese|ish|ern|ous|ive|ий|ый|ой|ая|яя|ое|ее|ые|ие)$")
_NAME_PART = re.compile(r"[A-ZА-ЯЁ][a-zа-яё'’]+")


def _name_word(word: str) -> bool:
    """Имя собственное одним словом (можно через дефис: «Okhotsk-Koryak»): части с
    заглавной, дальше строчные; от четырёх букв; не прилагательное, не сторона света."""
    return (
        all(_NAME_PART.fullmatch(part) for part in word.split("-"))
        and len(re.sub(r"[^A-Za-zА-Яа-яЁё]", "", word)) >= 4
        and not _NAME_ADJECTIVE.search(normalize(word))
        and name_key(word) not in _NOT_NAMES
    )


def _proper_word(name: str) -> str:
    """Отличительное слово имени места: «Koldar» в «Koldar intrusive massif», «Isfahan»
    в «Isfahan Province», «Исфаган» в «провинция Исфаган». Только когда такое слово с
    заглавной одно, кроме вида места, и оно — имя (_name_word). В «East Greenland», «Red
    Sea Hills», «Chinese Altai» его нет: «Greenland» или «Altai» — уже другой ответ."""
    words = name.split()
    if len(words) < 2:
        return ""
    capital = [w for w in words if w[:1].isupper() and name_key(w) not in _PLACE_WORDS]
    return capital[0] if len(capital) == 1 and _name_word(capital[0]) else ""


def name_forms(variants: list[str], article: list[str], kind: str, question: str) -> list[str]:
    """Ещё написания ответа-места (вид «объект»): имя без вида места (_proper_word) и
    имя из статьи другим алфавитом (text.translit: «Kular» → «Кулар», «Koldar» →
    «Колдар»). Переводу модели транслит не нужен — он и так по-русски. Что стоит в
    вопросе — не ответ: не берётся. У минерала, породы, метода форм нет: «Kujura» из
    «Kujura felsites» нашлось бы и в «Kujura syncline»."""
    if kind != "объект":
        return []
    forms: list[str] = []
    article = [" ".join(a.split()) for a in article]
    for name in dict.fromkeys(" ".join(n.split()) for n in [*variants, *article]):
        word = _proper_word(name) or (name if " " not in name and _name_word(name) else "")
        if not word:
            continue
        forms.append(word)
        if name in article:
            forms.append(translit(word))
    return [f for f in dict.fromkeys(forms) if not _hits([f], question)]


def augment(
    g: Any, link: Any, fact: dict[str, Any], variants: list[str], question: str
) -> list[str]:
    """Ключ без модели: к написаниям ответа — написания той же сущности из словаря и
    графа (spellings; что стоит в вопросе — нет) и формы имени места (name_forms)."""
    _, dst, _, kind = _ends(fact)
    out = _more(variants, *(s for s in spellings(g, dst, link.dst) if not _hits([s], question)))
    return _more(out, *name_forms(out, [dst, link.dst], kind, question))


def rekey(items: list[dict[str, Any]], g: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Ключи готового набора — заново, без модели (augment): прежние написания
    остаются, к ним добавляются новые. Факт вопроса ищется в графе по фрагменту и
    строке «факт»; не нашёлся — вопрос остаётся как был. → (вопросы, id без факта)."""
    by_chunk = g.by_chunk()
    out, lost = [], []
    for item in items:
        pair = next(
            (
                (link, fact)
                for link, fact in by_chunk.get(int(item.get("фрагмент") or 0), [])
                if f"{link.src} — {link.relation} — {link.dst}" == item.get("факт")
            ),
            None,
        )
        keys = [str(k) for k in item.get("ключевые") or []]
        if pair is None or not keys:
            lost.append(str(item.get("id")))
            out.append(item)
            continue
        given = [w.strip() for w in keys[0].split("|") if w.strip()]
        variants = augment(g, pair[0], pair[1], given, str(item.get("вопрос") or ""))
        out.append(
            {**item, "ключевые": ["|".join(v.replace("|", "/") for v in variants), *keys[1:]]}
        )
    return out, lost


def _named(
    question: str, proper: Callable[[str], bool], latin: bool = False, context: str = ""
) -> bool:
    """Есть ли в вопросе имя собственное. Первое слово не в счёт — оно с заглавной
    всегда; символ элемента (Au) — не имя. Русское слово с заглавной посреди вопроса —
    имя («Сухой Лог»), латинское — если в графе оно не пишется и со строчной («Pyrite
    grain» из подписи — не имя). latin — статья не русская: имя должно быть латиницей,
    как в ней («Ламбли Крик» поиск по английской статье не найдёт). context — цитата и
    название статьи: имя должно найтись в них, а не быть придумано моделью."""
    where = normalize(context)
    for w in _WORD.findall(question)[1:]:
        if len(w) < 3 or not w[:1].isupper() or w.isupper():
            continue
        if not ((not latin) if _CYRILLIC.match(w) else proper(w)):
            continue
        if not context or keyword_pattern(w).search(where):
            return True
    return False


def problem(
    question: str,
    variants: list[str],
    proper: Callable[[str], bool],
    latin: bool = False,
    context: str = "",
) -> str:
    """Чем плох вопрос (пусто — ничем): без статьи не понять, о чём, или в нём ответ.
    latin и context — как у _named."""
    if not 15 <= len(question) <= 250:
        return "вопрос слишком короткий или длинный"
    if _VAGUE.search(question):
        return "в вопросе номер, число, «данный район» или «в статье»"
    if _MIXED.search(question):
        return "в вопросе слово из двух алфавитов"
    if not _named(question, proper, latin, context):
        return "в вопросе не назван объект из статьи" + (" латиницей" if latin else "")
    if _hits(variants, question):
        return "ответ — в самом вопросе"
    return ""


def phrase(llm: Any, g: Any, link: Any, fact: dict[str, Any]) -> tuple[str, list[str], str]:
    """Вопрос по факту и его ключ: (вопрос, написания ответа, почему отброшен).

    Модель пишет вопрос; код проверяет его (problem); модель пишет ответ на другом
    языке (TERM), и перевод проверяется обратным (BACK); модель отвечает на вопрос
    по цитате — ответ должен попасть в ключ. Отброшен — вопрос пустой, причина —
    для сводки."""
    src, dst, _, dst_kind = _ends(fact)
    quote = " ".join(str(fact["quote"]).split())[:QUOTE_CHARS]
    english = not _CYRILLIC.search(dst)  # ответ — как в нерусской статье
    try:
        data = llm.chat_json(
            SYSTEM,
            USER.format(
                title=fact.get("title") or "",
                quote=quote,
                src=src,
                relation=link.relation,
                dst=dst,
            ),
            temperature=0,  # набор пересоздаётся тем же: те же факты — те же вопросы
            max_tokens=300,
        )
        question = " ".join(str(data.get("вопрос") or "").split()).strip(' «»"')
        if not question:
            return "", [], "у факта нет названного объекта"
        question = question if question.endswith("?") else question + "?"
        why = problem(
            question,
            _variants(dst, link.dst),
            g.proper,
            latin=not _CYRILLIC.search(quote),
            context=f"{quote} {fact.get('title') or ''}",
        )
        if why:
            return "", [], why
        term = llm.chat_json(
            TERM_SYSTEM, TERM.format(term=dst, quote=quote), temperature=0, max_tokens=100
        )
        other, same = (
            (term.get("ru"), term.get("en")) if english else (term.get("en"), term.get("ru"))
        )
        translated = _variants(other)
        if translated:
            back = llm.chat_json(
                TERM_SYSTEM,
                BACK.format(
                    direction="на английский" if english else "на русский", term=translated[0]
                ),
                temperature=0,
                max_tokens=60,
            )
            if not _hits(_variants(dst, same, link.dst), str(back.get("term") or "")):
                translated = []  # обратно не сходится — перевод неточный, в ключ не идёт
        # Написания той же сущности из словаря и графа — как перевод: стоят в вопросе —
        # ответ в вопросе.
        variants = _more(_variants(*translated, dst, same, link.dst), *spellings(g, dst, link.dst))
        if not any(_CYRILLIC.search(v) for v in variants) and not g.proper(dst):
            return "", [], "нет точного русского написания ответа"
        if _hits(variants, question):
            return "", [], "ответ — в самом вопросе"
        variants = _more(variants, *name_forms(variants, [dst, link.dst], dst_kind, question))
        said = llm.chat_json(
            CHECK_SYSTEM,
            CHECK.format(quote=quote, question=question),
            temperature=0,
            max_tokens=100,
        )
    except Exception:  # noqa: BLE001 — модель не ответила: берётся следующий факт
        return "", [], "модель не ответила"
    if not _hits(variants, str(said.get("ответ") or "")):
        return "", [], "по цитате ответ не тот конец факта"
    return question, variants, ""


def _item(
    n: int, link: Any, fact: dict[str, Any], question: str, variants: list[str]
) -> dict[str, Any]:
    return {
        "id": f"g{n:02d}",
        "вопрос": question,
        "ожидание": "база",
        "кратко": "да",  # вопрос об одном факте: короткий ответ — не отписка
        "понятие": link.src,  # о чём вопрос: по нему отчёт ищет, где мало статей
        "ключевые": ["|".join(v.replace("|", "/") for v in variants)],
        # Вопрос назван по объекту из этой статьи — найти поиск обязан её.
        "статьи": [fact["doc_id"]],
        "фрагмент": fact["chunk_id"],
        "факт": f"{link.src} — {link.relation} — {link.dst}",
        "цитата": " ".join(str(fact["quote"]).split())[:300],
    }


def build(
    g: Any,
    llm: Any = None,
    limit: int = LIMIT,
    log: Any = print,
    workers: int = WORKERS,
) -> list[dict[str, Any]]:
    """Набор вопросов: факты по очереди (candidates), пока не наберётся limit.
    Модель спрашивают по workers фактов разом; один и тот же ответ — один раз."""
    pool = candidates(g)
    out: list[dict[str, Any]] = []
    answers: set[str] = set()
    why: Counter[str] = Counter()
    workers = max(1, workers)

    def make(pair: tuple[Any, dict[str, Any]]) -> tuple[str, list[str], str]:
        link, fact = pair
        if llm is None:
            question = template(link)
            return question, augment(g, link, fact, [_ends(fact)[1], link.dst], question), ""
        return phrase(llm, g, link, fact)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(pool), workers):
            if len(out) >= limit:
                break
            batch = [p for p in pool[start : start + workers] if _answer_key(p) not in answers]
            made: Iterable[tuple[str, list[str], str]] = (
                ex.map(make, batch) if llm is not None else map(make, batch)
            )
            for pair, (question, variants, reason) in zip(batch, made, strict=True):
                key = _answer_key(pair)
                if reason or key in answers:
                    why[reason or "ответ уже есть в наборе"] += 1
                    continue
                if len(out) >= limit:
                    break
                answers.add(key)
                out.append(_item(len(out) + 1, pair[0], pair[1], question, variants))
                if log and len(out) % 10 == 0:
                    log(f"  вопросов: {len(out)}")
    if log and why:
        log("Отброшено: " + "; ".join(f"{k} — {v}" for k, v in why.most_common()))
    return out


def _answer_key(pair: tuple[Any, dict[str, Any]]) -> str:
    return name_key(_ends(pair[1])[1])


def to_yaml(items: list[dict[str, Any]], note: str = "") -> str:
    """Набор вопросов файлом. note — строка-пояснение в начало (откуда набор)."""
    import yaml

    head = (
        [f"# {line}" for line in note.splitlines()]
        + ["#"] * bool(note)
        + [
            "# ВОПРОСЫ ОЦЕНКИ ИЗ ГРАФА — сделаны командой python georag.py questions,",
            "# пересоздаются ею же (правки руками затрутся — переносите удачные вопросы",
            "# в config/eval-questions.yaml).",
            "#",
            "# У каждого вопроса известен ответ: «факт» из графа с «цитатой»; «ключевые» —",
            "# второй конец факта, по-русски и как в статье (любое написание засчитывается);",
            "# «статьи» и «фрагмент» — где он сказан. Поиск должен найти статью, ответ —",
            "# назвать ключевое. «кратко: да» — вопрос об одном факте: короткий ответ",
            "# не считается отпиской (graph/questions.py, evaluation.py).",
            "#",
            "# Запуск: python georag.py eval --questions <этот файл>",
            "",
        ]
    )
    body = yaml.safe_dump({"вопросы": items}, allow_unicode=True, sort_keys=False, width=100)
    return "\n".join(head) + body
