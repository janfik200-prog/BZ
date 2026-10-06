"""Факты из статей: Qwen3 выписывает, код проверяет каждый.

Факт — «от — связь — к» с дословной цитатой из фрагмента. Видов сущностей
нет: «от» и «к» — как названо в тексте (место, разлом, метод, процесс,
признак — всё равно), связь — короткая фраза по-русски. Разные написания
одного и того же сводит словарь синонимов (synonyms.py), а не модель.

Как идёт, по фрагментам статей:

1. Модели уходят правила (config/graph-rules.txt — правится Блокнотом) и
   фрагмент. Модель возвращает JSON: {"факты": [{от, связь, к, цитата}]}.
2. **Код проверяет каждый факт**, не прошедшее в граф не попадает:
   * цитата есть во фрагменте дословно (кавычки, тире, регистр и пробелы не
     важны; сокращение многоточием — части по порядку);
   * в цитате видны оба конца — в любом падеже; из длинного имени можно
     пропустить одно общее слово, но не имя собственное («Онежский рудный
     район» по цитате про «рудный район» не проходит);
   * концы — не пустые, не слишком длинные, не одно и то же, не год и не число,
     не количество («около 80 % сульфидов», «385 ± 2 млн лет»), не ссылка на
     автора («Binotto (2015)»), не положение («южнее разлома»); предлог в начале
     («в аномалиях Bi») снимается;
   * связь — от 1 до 6 слов.
   Что отброшено и почему — сохраняется: `graph --new` показывает, где модель
   ошибалась чаще всего.
3. Принятое пишется в базу по фрагменту. Проход возобновляемый: прервали —
   продолжится с того же фрагмента; `--redo` — всё заново (после правки правил).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..text import locate_quote, mentions, name_key
from . import store
from .synonyms import clean_relation

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "graph-rules.txt"
FRAGMENT_CHARS = 4000
MAX_NAME = 100
MAX_WORDS = 8
MAX_FACTS = 20
_NUMBER = re.compile(r"^[\d\s.,–—\-±~<>=%]+(гг?|г\.|лет|млн|ма|ga|ma)?\.?$", re.IGNORECASE)


def load_rules(path: Path | str | None = None) -> str:
    """Правила из файла, без строк-комментариев. Нет файла — ошибка: без правил не строим."""
    path = Path(path or RULES_PATH)
    text = path.read_text(encoding="utf-8-sig")
    rules = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).strip()
    if len(rules) < 100:
        raise ValueError(f"{path.name}: правил почти нет — граф без правил не строится")
    return rules


def messages(rules: str, text: str) -> tuple[str, str]:
    system = f"{rules}\n\nОтвечай только JSON, без пояснений."
    user = (
        f'Фрагмент статьи:\n"""{(text or "")[:FRAGMENT_CHARS]}"""\n\n'
        'Верни JSON: {"факты": [{"от": "...", "связь": "...", "к": "...", '
        '"цитата": "..."}]}'
    )
    return system, user


@dataclass(frozen=True)
class Fact:
    src: str
    relation: str
    dst: str
    quote: str

    @property
    def src_key(self) -> str:
        return name_key(self.src)

    @property
    def dst_key(self) -> str:
        return name_key(self.dst)


@dataclass
class Checked:
    facts: list[Fact] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


# Количество, а не предмет: «около 80 % сульфидных минералов», «385 ± 2 млн лет»,
# «2 км», «порядка 17,6 млрд т запасов».
_QUANTITY = re.compile(
    r"^(?:около|порядка|примерно|более|менее|свыше|до|от|не\s+(?:более|менее)|"
    r"~|≈|>|<)?\s*[±]?\s*\d[\d.,]*(?:\s*[-–—±]\s*\d[\d.,]*)*(?=\s|%|°|$)",
    re.IGNORECASE,
)  # «3D-модель» — не количество
# Ссылка на работу, а не предмет: «Binotto (2015)», «Иванов и др., 2019», «Smith et al.».
_CITATION = re.compile(
    r"\(\s*(?:1[89]|20)\d{2}[a-zа-я]?\s*\)|\bet\s+al\b|" r"\bи\s+др\.?,?\s*(?:1[89]|20)\d{2}",
    re.IGNORECASE,
)
# Положение, а не предмет: «южнее Персияновского разлома в западной части района».
_POSITION = re.compile(
    r"^(?:южнее|севернее|западнее|восточнее|вдоль|вблизи|около|рядом\s+с|"
    r"между|выше|ниже|внутри|вне)\s",
    re.IGNORECASE,
)
# Ручная проверка 100 фактов (logs/quality-20261005): каждый восьмой «предмет» —
# не предмет. Вторая проверка моделью их не ловила (5 из 24 ошибок при 10 верных
# фактах, отброшенных зря), правила кода — ловят.
# Местоимение вместо названия: «this event», «these structures», «это время».
_PRONOUN = re.compile(
    r"^(?:this|these|those|that|it|they|such|its|their|этот|эта|это|эти|данн\w+|"
    r"такие|такой|он|она|они)\b",
    re.IGNORECASE,
)
# Человек: «В. Н. Бобров», «Smith J.».
_PERSON = re.compile(
    r"^(?:[A-ZА-ЯЁ]\.\s?){1,2}\s?[A-ZА-ЯЁ][a-zа-яё]+$|^[A-ZА-ЯЁ][a-zа-яё]+,?\s(?:[A-ZА-ЯЁ]\.\s?){1,2}$"
)
# Год или десятилетие: «1950s», «1990-е годы».
_YEAR = re.compile(
    r"^(?:early\s+|late\s+|mid-?\s*)?\d{3,4}(?:-?[хxе]|'?s)?"
    r"(?:\s*(?:гг?\.?|годы|годах|year|years))?$",
    re.IGNORECASE,
)
# Счёт без названия: «three areas», «несколько участков».
_COUNTED = re.compile(
    r"^(?:one|two|three|four|five|six|several|many|some|various|other|другие|один|одна|"
    r"два|две|три|четыре|пять|несколько|многие|некоторые|различные)\s",
    re.IGNORECASE,
)
# Общее слово без названия: «месторождение» — какое?
_GENERIC = {
    "месторождение", "месторождения", "структура", "структуры", "район", "участок",
    "участки", "территория", "область", "зона", "зоны", "порода", "породы", "объект",
    "объекты", "изучаемая территория", "исследуемая территория", "area", "areas",
    "region", "deposit", "deposits", "structure", "structures", "zone", "zones", "rocks",
    "study area",
}  # fmt: skip
# Оценка из таблицы вместо предмета: «Barite = High», «Gold = Very high».
_DEGREE = {
    "high", "low", "very", "moderate", "medium", "strong", "weak", "major", "minor",
    "высокий", "высокая", "высокое", "низкий", "низкая", "низкое", "средний", "средняя",
    "среднее", "очень", "умеренный", "умеренная",
}  # fmt: skip
# Предположение в цитате, которого нет в связи: «The ores probably formed…» →
# «ores — образовано при — …». Связь получает пометку «предположительно».
_HEDGE_QUOTE = re.compile(
    r"вероятн|по-видимому|предположительн|возможн|не\s+исключен|считает|считают|"
    r"по\s+мнению|косвенн\w*\s+признак|probabl|likely|possibl|presumabl|"
    r"\b(?:is|are|was|were)\s+(?:thought|believed|inferred|presumed)|suggest|"
    r"may\s+have|might",
    re.IGNORECASE,
)
_HEDGE_RELATION = re.compile(
    r"вероятн|возможн|предполож|может|могут|можно|интерпрет|считает|may|might|likely|"
    r"possib|probab|interpret|suggest|infer",
    re.IGNORECASE,
)
HEDGE = "предположительно"
# Предлог в начале имени модель тащит из текста («в аномалиях Bi») — он снимается.
_PREPOSITION = re.compile(
    r"^(?:в|во|на|к|ко|по|с|со|у|о|об|из|при|для|in|at|on|of)\s+(?=\w)", re.IGNORECASE
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip(" .,;:«»\"'")


def _clean_name(value: Any) -> str:
    return _PREPOSITION.sub("", _clean(value)).strip()


def _bad_name(name: str) -> str | None:
    if not name:
        return "пустое имя"
    if len(name) > MAX_NAME or len(name.split()) > MAX_WORDS:
        return "слишком длинное имя"
    if _NUMBER.match(name) or not name_key(name):
        return "число или год, а не предмет"
    if _QUANTITY.match(name):
        return "количество, а не предмет"
    if _CITATION.search(name):
        return "ссылка на автора, а не предмет"
    if _POSITION.match(name):
        return "положение, а не предмет"
    if _PRONOUN.match(name):
        return "местоимение, а не предмет"
    if _PERSON.match(name):
        return "человек, а не предмет"
    if _YEAR.match(name):
        return "число или год, а не предмет"
    if _COUNTED.match(name):
        return "счёт без названия, а не предмет"
    if name.lower() in _GENERIC:
        return "общее слово без названия"
    if set(name.lower().replace("-", " ").split()) <= _DEGREE:
        return "оценка из таблицы, а не предмет"
    return None


def hedged(relation: str, quote: str) -> str:
    """Связь с пометкой «предположительно», если в цитате это предположение."""
    if _HEDGE_QUOTE.search(quote) and not _HEDGE_RELATION.search(relation):
        return f"{HEDGE} {relation}"
    return relation


def check(answer: Any, text: str) -> Checked:
    """Ответ модели → факты, прошедшие проверку, и причины отказа по остальным."""
    out = Checked()
    if not isinstance(answer, dict):
        out.rejected.append("ответ: не объект JSON")
        return out
    items = answer.get("факты") or answer.get("facts") or []
    if not isinstance(items, list):
        out.rejected.append("ответ: «факты» — не список")
        return out
    seen = set()
    for item in items[: MAX_FACTS * 2]:
        if not isinstance(item, dict):
            out.rejected.append(f"{str(item)[:60]!r}: факт — не объект")
            continue
        src, dst = _clean_name(item.get("от")), _clean_name(item.get("к"))
        relation = clean_relation(item.get("связь"))
        label = f"«{src or '?'}» — {relation or '?'} — «{dst or '?'}»"
        reason = _bad_name(src) or _bad_name(dst)
        if reason:
            out.rejected.append(f"{label}: {reason}")
            continue
        if not relation or len(relation.split()) > 6:
            out.rejected.append(f"{label}: связь пустая или длиннее 6 слов")
            continue
        if name_key(src) == name_key(dst):
            out.rejected.append(f"{label}: связь с самим собой")
            continue
        quote = locate_quote(_clean(item.get("цитата")), text)
        if quote is None:
            out.rejected.append(f"{label}: цитаты нет во фрагменте")
            continue
        if not mentions(src, quote) or not mentions(dst, quote):
            missing = src if not mentions(src, quote) else dst
            out.rejected.append(f"{label}: в цитате нет «{missing}»")
            continue
        fact = Fact(src, hedged(relation, quote), dst, quote)
        key = (fact.src_key, relation, fact.dst_key)
        if key in seen:
            continue
        seen.add(key)
        out.facts.append(fact)
        if len(out.facts) >= MAX_FACTS:
            break
    return out


def run(
    conn: Any,
    llm: Any,
    limit: int | None = None,
    rules_path: Any = None,
    log: Any = print,
    stop_after_errors: int = 3,
) -> dict[str, Any]:
    """Модель проходит фрагменты, которых ещё не видела. Возобновляемо."""
    rules = load_rules(rules_path)
    rows = store.todo(conn, limit=limit)
    model = getattr(llm, "model", "модель")
    stats = {"todo": len(rows), "done": 0, "facts": 0, "rejected": 0, "errors": 0, "stopped": False}
    errors_in_row = 0
    started = time.monotonic()
    for i, (chunk_id, doc_id, text) in enumerate(rows, start=1):
        system, user = messages(rules, text or "")
        try:
            answer = llm.chat_json(system, user)
            errors_in_row = 0
        except Exception as exc:  # noqa: BLE001 — модель молчит: этот фрагмент потом
            stats["errors"] += 1
            errors_in_row += 1
            log(f"  [{i}/{len(rows)}] модель не ответила: {type(exc).__name__}: {exc}")
            if errors_in_row >= stop_after_errors:
                stats["stopped"] = True
                log(
                    "  Модель молчит несколько раз подряд — останавливаюсь. Проверьте Ollama "
                    "и запустите снова: продолжится с этого места."
                )
                break
            continue
        checked = check(answer, text or "")
        store.save(conn, chunk_id, doc_id, checked.facts, checked.rejected, model)
        conn.commit()
        stats["done"] += 1
        stats["facts"] += len(checked.facts)
        stats["rejected"] += len(checked.rejected)
        if i % 10 == 0 or i == len(rows):
            pace = (time.monotonic() - started) / i
            log(
                f"  [{i}/{len(rows)}] фактов {stats['facts']}, отброшено проверкой "
                f"{stats['rejected']} · {pace:.1f} с на фрагмент"
            )
    return stats


def recheck(conn: Any, log: Any = print) -> dict[str, int]:
    """Перепроверить готовый граф правилами кода — без модели.

    Правила проверки меняются (новые виды «не предметов», пометка предположений),
    а прогонять модель заново по всем фрагментам — часы. Здесь каждый факт
    проверяется тем же кодом, что и новый: не прошёл — убирается с причиной,
    предположение — получает пометку в связи."""
    stats = {"facts": 0, "dropped": 0, "hedged": 0}
    for fact_id, chunk_id, src, relation, dst, quote in store.stored_facts(conn):
        stats["facts"] += 1
        reason = _bad_name(src) or _bad_name(dst)
        if reason:
            store.drop_fact(conn, fact_id, chunk_id, f"«{src}» — {relation} — «{dst}»: {reason}")
            stats["dropped"] += 1
            continue
        marked = hedged(relation, quote)
        if marked != relation and store.set_relation(conn, fact_id, marked):
            stats["hedged"] += 1
    conn.commit()
    if log:
        log(
            f"Фактов проверено {stats['facts']}: убрано {stats['dropped']}, "
            f"помечено предположением {stats['hedged']}"
        )
    return stats
