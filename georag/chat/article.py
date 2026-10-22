"""Вопрос о названной статье: «<название статьи> Используется ли в этой статье X?».

Разбор вопроса делил такой вопрос на части, и название шло отдельной частью: по нему
находилась сама статья, а вторая часть — «используется ли в этой статье X» — искалась по
всей базе, без статьи, и бот отвечал «в базе нет» (живая проверка 22.10.2026, статья о
байесовских моделях: «Метод Гаусса и Крамера используются ли в этой статье?»). Здесь
название статьи из базы узнаётся в вопросе кодом: из вопроса оно убирается, разбирается
остальное, а ищется только во фрагментах этой статьи (Plan.article).

Название засчитывается целиком, без учёта регистра и знаков препинания, и не короче
MIN_WORDS слов: короткое название («Gold deposits») встречается в обычных вопросах.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..index import db
from .plan import Plan, previous_question

MIN_WORDS = 4  # слов в названии, чтобы узнавать его в вопросе
ABOUT = "О чём эта статья?"  # в вопросе одно название — вопрос о статье в целом

_WORD = re.compile(r"[^\W_]+")
# Ссылка на статью из прошлого вопроса: «а в этой статье…», «в ней».
_THIS = re.compile(r"\b(?:эт\w*\s+(?:стать|работ)|в\s+ней\b|this\s+(?:paper|article))", re.I)
_EDGES = " \t\n«»\"“”'.,:;—–-?!()[]"


@dataclass
class NamedArticle:
    """Статья, названная в вопросе, и вопрос без её названия."""

    doc_id: str
    title: str
    rest: str


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def find_title(question: str, titles: list[tuple[str, str]]) -> NamedArticle | None:
    """Самое длинное название из titles (doc_id, название), которое есть в вопросе."""
    asked = f" {' '.join(_words(question))} "
    best: tuple[str, str, list[str]] | None = None
    for doc_id, title in titles:
        words = _words(title or "")
        if len(words) < MIN_WORDS or f" {' '.join(words)} " not in asked:
            continue
        if best is None or len(words) > len(best[2]):
            best = (doc_id, title, words)
    if best is None:
        return None
    doc_id, title, words = best
    # Название в самом вопросе — с теми же словами, а между ними что угодно, кроме букв;
    # кавычки вокруг него уходят вместе с ним.
    body = r"[\W_]+".join(map(re.escape, words))
    pattern = r"[«\"“]?\s*(?<![^\W_])" + body + r"(?![^\W_])[.\s]*[»\"”]?"
    rest = re.sub(pattern, " ", question, count=1, flags=re.IGNORECASE)
    rest = " ".join(rest.split()).strip(_EDGES)
    if not re.search(r"[^\W\d_]{3}", rest):
        rest = ABOUT
    elif not rest.endswith(("?", ".", "!", "…")):
        rest += "?"
    return NamedArticle(doc_id, " ".join(title.split()), rest[:1].upper() + rest[1:])


def named_article(
    conn: Any, question: str, history: list[dict[str, Any]] | None = None
) -> NamedArticle | None:
    """Статья базы, названная в вопросе, а для «в этой статье…» без названия — в прошлом
    вопросе разговора. Заглушка базы в проверках — нет."""
    if conn is None or not db.is_real(conn):
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, title FROM documents WHERE title <> ''")
        titles = [(row[0], row[1]) for row in cur.fetchall()]
    found = find_title(question, titles)
    if found is None and history and _THIS.search(question):
        before = find_title(previous_question(history), titles)
        if before is not None:
            found = NamedArticle(before.doc_id, before.title, question.strip())
    return found


def in_article(plan: Plan, named: NamedArticle) -> Plan:
    """Разбор вопроса → поиск только в названной статье.

    «Собрать всё», граф о предмете и каталог ищут по всей базе — для вопроса о
    статье они не нужны. Статья есть в базе — значит, вопрос по теме базы."""
    plan.article = named.doc_id
    plan.catalog = None
    plan.collect = False
    plan.subject = None
    plan.on_topic = True
    return plan
