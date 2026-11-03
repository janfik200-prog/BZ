"""Ключевые понятия вопроса: найденные фрагменты, где они есть, — выше.

Поиск по смыслу и по словам находит и фрагменты «рядом с темой»: тот же термин в другом
значении, общие слова. Модель выделяет из вопроса одно–три обязательных понятия (метод,
показатель, минерал или процесс, тип оруденения, полезное ископаемое, место) с русским и
английским написанием; фрагменты упорядочиваются по доле найденных в них понятий, а при
равной доле — как нашёл поиск. Ничего не выбрасывается: фрагмент без понятий просто ниже.

Проверка 30.10.2026 (research/v15, 25 вопросов, разметка «полезен ли фрагмент»): доля
полезных среди пяти лучших 0,50 → 0,72. Включается GEORAG_CONCEPTS=1.

Модель переводит термины сама и ошибается; где понятие есть в словаре
config/term-translations.yaml, английские написания берутся оттуда.

Место вопроса (район, месторождение, рудное поле) модель называет отдельно. Его написание
сверяется с названиями, которые есть в базе (resolve: ключ транслитерации и близость
векторов названий), и поиск идёт в два слоя: сначала статьи, где это место упомянуто
(place_docs), потом ответ — только в них. Статей о месте нет — поиск по всей базе.
Проверка на 20 новых вопросах (research/v15/search_eval2.py) — до этих правок: 0,37 → 0,49.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..paths import ROOT
from ..text import name_key, translit_key

TRANSLATIONS = ROOT / "config" / "term-translations.yaml"
MAX_CONCEPTS = 3
MIN_SPELLING = 3  # «L», «F», «Au» — совпали бы почти с любым текстом

SYSTEM = (
    "Ты помогаешь искать сведения в геологических статьях на русском и английском языках. "
    "Выдели от одного до трёх обязательных понятий: без каждого из них фрагмент не может отвечать на "
    "вопрос. Это метод, показатель, минерал или процесс, тип оруденения, полезное ископаемое, название "
    "места. Общие слова («признак», «участок», «месторождение», «связь», «формирование», "
    "«картирование», «метод», «как выделяют») обязательными не бывают. Для каждого понятия дай "
    "написания: русское, английское, общепринятые сокращения. Не придумывай сокращений, которых нет в "
    "литературе. Не дроби понятие на части и не заменяй его частными примерами. Написания — "
    "короткие, в одно–три слова, как их пишут в статьях.\n"
    "Название места (район, месторождение, рудное поле, щит, массив) — отдельно, в поле place, "
    "в must его не повторяй; места в вопросе нет — place: null.\n"
    "Примеры:\n"
    "«Отношение калия к торию как признак изменений» → "
    '{"place": null, "must": [["отношение калия к торию", "K/Th", "potassium-thorium ratio"]]}\n'
    "«Минералы-индикаторы золото-кварцевого оруденения» → "
    '{"place": null, "must": [["минералы-индикаторы", "indicator minerals"], '
    '["золото-кварцевое оруденение", "gold-quartz"]]}\n'
    "«Колчеданные месторождения Алтая» → "
    '{"place": ["Алтай", "Altai"], "must": [["колчеданные месторождения", "VMS", '
    '"massive sulfide"]]}\n'
    'Ответ — только JSON: {"place": ["написание", ...] или null, '
    '"must": [["написание", "написание", ...], ...]}'
)

# Сверка названия места с базой (resolve) — по отличительной части названия, без «щит»,
# «массив», «Shield»: иначе «Anabar Shield» сходился с «Guiana Shield» (вектор 0,8+).
# Проверено 30.10.2026: «Колдар» → Koldar (ключ 1,0), «Camacau» → Camaquã (ключ 0,69,
# вектор 0,68), а «Ludzyn» с «Lead-Zinc» — ключ 0,71 при далёком векторе: нужны оба.
PLACE_KEY = 0.85  # одного ключа транслитерации довольно
PLACE_BOTH = 0.6  # или ключ и вектор, оба не ниже этого
PLACE_NAMES = 4
# Фонетические векторы Symphonym v8 (CC-BY-4.0, .local/external/symphonym/v8; нет папки —
# без них). Проверка 30.10.2026 (research/v15/place_names_eval.py): по русскому написанию
# нашли латинское в 19 случаях из 20 («Лудзин» → Lujing, 0,96), но и выдуманным сходствам
# дают 0,85–0,95 («Кольский полуостров» → False Color Composite) — поэтому ещё и ключ.
SYMPHONYM = ROOT / ".local" / "external" / "symphonym" / "v8"
SYM_MIN = 0.9
SYM_KEY = 0.62  # «Лудзин» — Lujing: 0,67; «Тыва» — Tarkwa: около 0,6
PLACE_WORDS = {
    "щит", "shield", "массив", "massif", "кряж", "ridge", "поле", "field", "рудное", "ore",
    "район", "district", "region", "area", "область", "province", "провинция", "пояс", "belt",
    "горы", "mountains", "mountain", "хребет", "range", "бассейн", "basin", "зона", "zone",
    "месторождение", "deposit", "рудник", "mine", "участок", "site", "узел", "cluster",
    "республика", "republic", "автономная", "autonomous", "северный", "northern", "южный",
    "southern", "восточный", "eastern", "западный", "western", "центральный", "central",
    "средний", "middle", "полуостров", "peninsula", "остров", "island", "the", "of", "do", "de",
}  # fmt: skip
CACHE = ROOT / ".local" / "cache"
_PROPER = re.compile(r"\b[A-ZА-ЯЁ][\w’'-]+(?:[ -][A-ZА-ЯЁ][\w’'-]+)*")

# Общие слова: понятие из одних таких слов отбрасывается (основы, как в _stem).
GENERIC = {
    "признак", "участ", "месторожд", "deposit", "связ", "relat", "формиров", "format",
    "картир", "mapp", "метод", "method", "выделен", "контрол", "contr", "зона", "zone",
    "район", "area", "данн", "data", "изучен", "study", "анализ", "analys", "оценк",
    "оруденен", "mineraliz", "mineralis", "порода", "rock",
}  # fmt: skip

_WORD = re.compile(r"[\w/]+", re.U)


def _stem(word: str) -> str:
    """Начало слова для сравнения: падеж и число не мешают («линеаментов» — «линеамент»)."""
    w = word.lower().replace("ё", "е")
    return w if len(w) <= 4 or "/" in w else w[: max(4, len(w) - 3)]


def _words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text or "")]


def _generic(spelling: str) -> bool:
    """Написание из одних общих слов («месторождения», «mapping»); короткие слова не в счёт."""
    words = [_stem(w) for w in _WORD.findall(spelling) if len(w) > 2]
    return bool(words) and all(
        any(w.startswith(g) or (len(w) >= 4 and g.startswith(w)) for g in GENERIC) for w in words
    )


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[str, list[str]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text("utf-8")) or {}
    return {name_key(str(k)): [str(x) for x in (v or [])] for k, v in data.items()}


def translations(path: Path = TRANSLATIONS) -> dict[str, list[str]]:
    """Словарь переводов (ключ русского понятия → английские написания); нет файла — пусто."""
    if not path.exists():
        return {}
    return _load(str(path), path.stat().st_mtime)


def clean(raw: Any, table: dict[str, list[str]] | None = None) -> list[list[str]]:
    """Ответ модели → понятия: без коротких и общих написаний, со словарными переводами."""
    table = translations() if table is None else table
    out: list[list[str]] = []
    for concept in raw if isinstance(raw, list) else []:
        spellings = [str(s).strip() for s in (concept if isinstance(concept, list) else [concept])]
        spellings = [s for s in spellings if len(s) >= MIN_SPELLING and not _generic(s)]
        if not spellings:
            continue
        known = next((table[name_key(s)] for s in spellings if name_key(s) in table), None)
        if known is not None:
            # Перевод — из словаря: кириллические написания модели остаются, латинские — нет.
            spellings = [s for s in spellings if re.search("[а-яё]", s, re.I)] + known
        out.append(list(dict.fromkeys(spellings)))
    return out[:MAX_CONCEPTS]


@dataclass
class Parsed:
    """Разбор вопроса: понятия (с местом — для порядка кандидатов) и само место."""

    must: list[list[str]] = field(default_factory=list)
    place: list[str] = field(default_factory=list)

    @property
    def concepts(self) -> list[list[str]]:
        return ([self.place] if self.place else []) + self.must


def parse(llm: Any, question: str) -> Parsed:
    """Понятия и место вопроса; модель молчит или ответ не разобрать — пусто."""
    try:
        data = llm.chat_json(SYSTEM, question, temperature=0.0, max_tokens=400)
    except Exception:  # noqa: BLE001 — без понятий поиск работает как прежде
        return Parsed()
    if not isinstance(data, dict):
        return Parsed()
    place = clean([data.get("place")]) if data.get("place") else []
    return Parsed(clean(data.get("must"))[: MAX_CONCEPTS - bool(place)], place[0] if place else [])


def extract(llm: Any, question: str) -> list[list[str]]:
    """Понятия вопроса вместе с местом (для порядка кандидатов)."""
    return parse(llm, question).concepts


def _main(spelling: str) -> list[str]:
    """Основы главных слов написания: без общих («deposits») и служебных («of»)."""
    parts = [_stem(w) for w in _WORD.findall(spelling) if len(w) > 2 or "/" in w]
    main = [p for p in parts if not _generic(p)]
    return main or parts


def has(spelling: str, words: list[str]) -> bool:
    """Написание есть в тексте (по началу слова). В длинном (три главных слова и больше)
    одно слово может не совпасть: «large-scale gold ore deposits» пишут и без «scale»."""
    parts = _main(spelling)
    if not parts:
        return False
    found = sum(any(t.startswith(p) for t in words) for p in parts)
    return found >= (len(parts) - 1 if len(parts) >= 3 else len(parts))


def share(concepts: list[list[str]], text: str) -> float:
    """Доля понятий, хотя бы одно написание которых есть в тексте."""
    if not concepts:
        return 0.0
    words = _words(text)
    return sum(any(has(s, words) for s in c) for c in concepts) / len(concepts)


def rerank(hits: list[Any], concepts: list[list[str]]) -> list[Any]:
    """Сначала фрагменты с большей долей понятий; при равной — прежний порядок."""
    if not concepts:
        return hits
    scored = [(share(concepts, h.text), i, h) for i, h in enumerate(hits)]
    return [h for _, _, h in sorted(scored, key=lambda x: (-x[0], x[1]))]


# --- место вопроса: сверка с базой и статьи о нём ---------------------------------------


def _distinct(name: str) -> str:
    """Отличительная часть названия: «Anabar Shield» → «Anabar», «массив Колдар» → «Колдар».
    Осталось короче пяти букв — название целиком: от «Mountain Pass» одно «Pass» нашлось бы
    в 33 статьях из 114."""
    kept = [
        w
        for w in re.split(r"[\s\-]+", name)
        if w and _stem(w) not in {_stem(x) for x in PLACE_WORDS}
    ]
    short = sum(len(w) for w in kept) < 5
    return name if short or not kept else " ".join(kept)


@dataclass
class _Base:
    """Что о местах знает база: названия и слова каждой статьи; векторы названий —
    по первой нужде (vectors), с записью на диск: считаются около минуты."""

    stamp: tuple[Any, ...]
    names: list[str]
    keys: list[str]
    docs: dict[str, set[str]]  # статья → слова названия, заголовков и текста
    _vectors: Any = None
    _phonetic: Any = None

    def phonetic(self, model: Any) -> Any:
        """Фонетические векторы названий (Symphonym), по первой нужде."""
        if self._phonetic is None:
            items = [
                (_distinct(n), "ru" if re.search("[а-яё]", n, re.I) else "en") for n in self.names
            ]
            self._phonetic = model.batch_embed(items)
        return self._phonetic

    def vectors(self, embedder: Any) -> Any:
        import numpy as np

        if self._vectors is not None:
            return self._vectors
        path = CACHE / f"place-names-{self.stamp[0]}-{self.stamp[1]}.npy"
        if path.exists():
            self._vectors = np.load(path)
        else:
            v = np.array(embedder.encode([_distinct(n) for n in self.names]))
            self._vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
            try:
                CACHE.mkdir(parents=True, exist_ok=True)
                np.save(path, self._vectors)
            except OSError:
                pass  # без записи — посчитается снова в следующий раз
        return self._vectors


_BASE: dict[tuple[Any, ...], _Base] = {}


def _base(conn: Any, embedder: Any) -> _Base:
    """Названия из заголовков статей и разделов и из графа; пересчёт — если база выросла."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), coalesce(max(id), 0) FROM chunks")
        stamp = tuple(cur.fetchone())
        if stamp in _BASE:
            return _BASE[stamp]
        cur.execute(
            "SELECT d.doc_id, d.title, c.headings, c.text FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id"
        )
        rows = cur.fetchall()
        cur.execute("SELECT to_regclass('entity_names') IS NOT NULL")
        graph: list[str] = []
        if cur.fetchone()[0]:
            cur.execute("SELECT name FROM entity_names")
            graph = [n for (n,) in cur.fetchall()]
    docs: dict[str, set[str]] = {}
    found: set[str] = set()
    for doc_id, title, headings, text in rows:
        heads = " ".join(headings or [])
        words = docs.setdefault(doc_id, set(_words(title)))
        words.update(_words(heads), _words(text))
        for source in (title or "", heads):
            found.update(m.group() for m in _PROPER.finditer(source))
    for name in graph:
        found.update(m.group() for m in _PROPER.finditer(name or ""))
    names = sorted(found)
    _BASE.clear()
    _BASE[stamp] = _Base(stamp, names, [translit_key(_distinct(n)) for n in names], docs)
    return _BASE[stamp]


def _in(spelling: str, words: set[str]) -> bool:
    """has() для множества слов статьи: по началу слова."""
    parts = _main(spelling)
    if not parts:
        return False
    found = sum(any(w.startswith(p) for w in words) for p in parts)
    return found >= (len(parts) - 1 if len(parts) >= 3 else len(parts))


@lru_cache(maxsize=1)
def _symphonym(path: str) -> Any:
    """Модель Symphonym из скачанной папки; нет папки или torch — None."""
    import importlib.util
    from functools import partial

    try:
        spec = importlib.util.spec_from_file_location(
            "symphonym_inference", Path(path) / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Словари модели — в UTF-8, а её код открывает их в кодировке системы (в Windows
        # это cp1251): open внутри модуля — с явной кодировкой.
        module.open = partial(open, encoding="utf-8")
        return module.SymphonymModel(model_dir=path)
    except Exception:  # noqa: BLE001 — без фонетики сверка идёт как раньше
        return None


def _cyr(text: str) -> bool:
    return bool(re.search("[а-яё]", text, re.I))


def _close_enough(spelling: str, name: str, key: float, strict: bool = False) -> bool:
    """Название того же алфавита — только почти то же написание: «Кольский» и «Колымский»,
    «Хакасия» и «Якутия» — разные места; другой алфавит — по общим правилам.
    strict — место уже нашлось в статьях своим написанием: тогда и другой алфавит —
    только почти то же («Koldar» → «Кульдур» не годится, «Букантау» → Bukantau — да)."""
    return key >= PLACE_KEY or (not strict and _cyr(spelling) != _cyr(name))


def _phonetic_matches(base: _Base, place: list[str], strict: bool = False) -> dict[str, float]:
    """Названия базы, близкие к месту по звучанию (Symphonym) и по ключу написания."""
    model = _symphonym(str(SYMPHONYM)) if SYMPHONYM.exists() else None
    if model is None or not base.names:
        return {}
    vectors = base.phonetic(model)
    out: dict[str, float] = {}
    for spelling in place:
        lang = "ru" if re.search("[а-яё]", spelling, re.I) else "en"
        sims = vectors @ model.embed(_distinct(spelling), lang)
        key = translit_key(_distinct(spelling))
        for i in sims.argsort()[::-1][:5]:
            f = SequenceMatcher(None, key, base.keys[i]).ratio()
            ok = _close_enough(spelling, base.names[i], f, strict)
            if sims[i] >= SYM_MIN and f >= SYM_KEY and ok:
                out[base.names[i]] = max(out.get(base.names[i], 0.0), float(sims[i]) + f)
    return out


def resolve(conn: Any, embedder: Any, place: list[str]) -> list[str]:
    """Написания места, которые есть в базе: свои, если встречаются в статьях, и
    ближайшие названия базы — по ключу транслитерации и векторам BGE-M3, по звучанию
    (Symphonym). «Лудзин» → Lujing, «Букантау» → и Букантау, и Bukantau: английские статьи
    о месте находятся и тогда, когда русское написание уже есть в русских.
    Ничего похожего — пусто: о месте в базе нет статей."""
    import numpy as np

    base = _base(conn, embedder)
    present = [p for p in place if any(_in(p, w) for w in base.docs.values())]
    if not base.names:
        return present
    strict = bool(present)
    best: dict[str, float] = dict(_phonetic_matches(base, place, strict))
    vectors = base.vectors(embedder)
    for spelling in place:
        key = translit_key(_distinct(spelling))
        v = np.array(embedder.encode_one(_distinct(spelling)))
        sims = vectors @ (v / np.linalg.norm(v))
        for i, name in enumerate(base.names):
            f = SequenceMatcher(None, key, base.keys[i]).ratio()
            vs = float(sims[i])
            if (f >= PLACE_KEY or (f >= PLACE_BOTH and vs >= PLACE_BOTH)) and _close_enough(
                spelling, name, f, strict
            ):
                best[name] = max(best.get(name, 0.0), f + vs)
    found = sorted(best, key=lambda n: -best[n])[:PLACE_NAMES]
    return list(dict.fromkeys(present + found))


def _named(name: str, words: set[str]) -> bool:
    """Место упомянуто: все слова отличительной части названия есть в статье. Без поблажки
    has(): у «Seligdar Ore Field» совпали бы одни «ore» и «field»."""
    parts = [_stem(w) for w in _WORD.findall(_distinct(name)) if len(w) > 2]
    return bool(parts) and all(any(w.startswith(p) for w in words) for p in parts)


def place_docs(conn: Any, embedder: Any, names: list[str]) -> list[str]:
    """Статьи, где место упомянуто — в названии, заголовке раздела или тексте."""
    if not names:
        return []
    base = _base(conn, embedder)
    return sorted(d for d, words in base.docs.items() if any(_named(n, words) for n in names))
