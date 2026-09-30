"""Вопросы для оценки — из фактов графа, с заранее известным ответом.

Оценка (evaluation.py) честна настолько, насколько хороши вопросы. Писать их
руками долго, а у каждого факта графа уже есть всё нужное: цитата (ответ),
фрагмент и статья, где он сказан. Поэтому вопрос строится из факта:

    факт:    «оруденение» — приурочено к — «зоны дробления»
    вопрос:  «К чему приурочено оруденение?»          (модель переформулирует)
    ключевые: «зоны дробления»                         (ответ — второй конец факта)
    статьи:  doc_id статьи с цитатой                  (что поиск обязан найти)

В вопросе нет ответа («к»): иначе поиск находил бы его просто по словам
вопроса, и оценка была бы завышена. Факты берутся из разных статей по очереди —
чтобы проверялась вся база, а не одна подробная статья.

    python georag.py questions              → config/eval-graph.yaml
    python georag.py eval --questions config/eval-graph.yaml

Модель (Qwen3) только переформулирует факт в вопрос геолога; нет модели —
вопрос по шаблону. Проверяет ответы, как и прежде, код.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..text import mentions, normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "config" / "eval-graph.yaml"

SYSTEM = "Ты составляешь вопросы для проверки поиска по геологическим статьям. Отвечай только JSON."
USER = """Факт из статьи: «{src}» — {relation} — «{dst}».
Цитата: «{quote}»

Составь вопрос геолога, на который отвечает эта цитата. В вопросе назови «{src}»,
но НЕ называй «{dst}» и его синонимов — это ответ. По-русски, 6–15 слов, без кавычек.

Верни JSON: {{"вопрос": "..."}}"""

MAX_NAME_WORDS = 6
_DIGITS = re.compile(r"\d")


def _usable(name: str) -> bool:
    """Годится ли имя для вопроса: коротко, без чисел (количества — не предмет)."""
    return bool(name) and len(name.split()) <= MAX_NAME_WORDS and not _DIGITS.search(name)


def pick(g, limit: int = 30) -> list:
    """Факты для вопросов: по очереди из разных статей, сначала подтверждённые
    несколькими статьями и с именами покороче."""
    by_doc: dict[str, list] = {}
    for link in g.links.values():
        if not (_usable(link.src) and _usable(link.dst)):
            continue
        doc = link.facts[0]["doc_id"]
        by_doc.setdefault(doc, []).append(link)
    for links in by_doc.values():
        links.sort(key=lambda lk: (-lk.documents, len(lk.src) + len(lk.dst)))
    out, round_ = [], 0
    while len(out) < limit:
        taken = False
        for links in by_doc.values():
            if round_ < len(links):
                out.append(links[round_])
                taken = True
                if len(out) >= limit:
                    break
        if not taken:
            break
        round_ += 1
    return out


def template(link) -> str:
    """Вопрос без модели: имя и связь, ответ — пропуск."""
    return f"«{link.src}» — {link.relation} — что именно, по статьям базы?"


def phrase(llm, link) -> str:
    """Модель переформулирует факт в вопрос. Не ответила или выдала ответ в вопросе — шаблон."""
    quote = link.facts[0]["quote"]
    try:
        data = llm.chat_json(SYSTEM, USER.format(src=link.src, relation=link.relation,
                                                 dst=link.dst, quote=quote[:600]))
        text = " ".join(str(data.get("вопрос") or "").split()).strip(" «»\"")
    except Exception:  # noqa: BLE001
        return template(link)
    # Ответ в вопросе — в любом падеже («зонам дробления» для «зоны дробления»).
    if not 10 <= len(text) <= 200 or mentions(link.dst, text):
        return template(link)
    return text if text.endswith("?") else text + "?"


def keywords(link) -> list[str]:
    """Ключевое для ответа — второй конец факта; через | — его самое длинное слово,
    чтобы сверка по словам не требовала всего имени дословно."""
    words = sorted(re.findall(r"[A-Za-zА-Яа-яЁё]{5,}", link.dst), key=len, reverse=True)
    group = [link.dst] + ([words[0]] if words and normalize(words[0]) != normalize(link.dst) else [])
    return ["|".join(w.replace("|", "/") for w in group)]


def build(g, llm=None, limit: int = 30, log=print) -> list[dict]:
    out = []
    for i, link in enumerate(pick(g, limit), start=1):
        question = phrase(llm, link) if llm is not None else template(link)
        docs = sorted({f["doc_id"] for f in link.facts})
        out.append({"id": f"g{i:02d}", "вопрос": question, "ожидание": "база",
                    "понятие": link.relation, "ключевые": keywords(link), "статьи": docs,
                    "факт": f"{link.src} — {link.relation} — {link.dst}"})
        if log and i % 10 == 0:
            log(f"  вопросов: {i}")
    return out


def to_yaml(items: list[dict]) -> str:
    import yaml

    head = [
        "# ВОПРОСЫ ОЦЕНКИ ИЗ ГРАФА — сделаны командой python georag.py questions,",
        "# пересоздаются ею же (правки руками затрутся — переносите удачные вопросы",
        "# в config/eval-questions.yaml).",
        "#",
        "# У каждого вопроса известен ответ: «факт» из графа, «ключевые» — его второй",
        "# конец, «статьи» — где он сказан. Поиск должен найти эти статьи, ответ —",
        "# назвать ключевое.",
        "#",
        "# Запуск: python georag.py eval --questions config/eval-graph.yaml",
        "",
    ]
    body = yaml.safe_dump({"вопросы": items}, allow_unicode=True, sort_keys=False, width=100)
    return "\n".join(head) + body
