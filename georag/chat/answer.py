"""Чат-бот: вопрос → поиск по базе вместе с моделью → ответ Qwen3 по найденному.

Обычный поиск (векторы + слова) на сложном вопросе ошибается в обе стороны:
не находит нужное, если оно сказано другими словами, и приносит похожее по
словам, но не о том. Поэтому поиск ведёт модель:

1. **Разбор.** Qwen3 делит составной вопрос на части (до трёх), на каждую
   составляет запросы — по-русски и по-английски, терминами предметной
   области; отмечает, о чём конкретно вопрос и просят ли «собрать всё».
2. **Широкий набор.** По запросам ищется с запасом — до десяти кандидатов на
   часть, с мягким порогом: лучше показать модели лишнее, чем не показать нужное.
3. **Модель отбирает.** Qwen3 читает кандидатов и называет те, где правда есть
   ответ (похожее по словам, но о другом — отбрасывает), говорит, чего не
   хватает, и составляет запросы, чтобы это найти.
4. **Ищет ещё.** По этим запросам — новый набор, модель отбирает снова. Всего
   до трёх кругов; останавливается, когда всего хватает или новое не находится.
5. **«Собрать всё»** («что известно об Анабарском щите») — из графа: все факты
   о предмете вопроса и о том, что в него входит, выписанные моделью из всех
   статей, с цитатами; поиск только добавляет.
6. **Ответ.** Модель отвечает ТОЛЬКО по отобранным фрагментам, после каждого
   утверждения — номер фрагмента [1]. Длина — по просьбе в самом вопросе:
   «подробно», «кратко», «перечисли» — так и отвечает; настроек нет.
7. **Не нашлось** — статьи не показываются: бот говорит, что в базе такого нет,
   и отвечает из общих знаний модели, с пометкой.

Модель не ответила при отборе (сервер модели выключен или сбой) — отбор по порогу
близости, как у обычного поиска: бот работает, только хуже на сложных вопросах.

Здесь — всё вместе; части — в своих модулях: settings (настройки), plan (разбор
вопроса), search (поиск по порогу), retrieval (поиск с моделью), dataset (граф:
«собрать всё» и вопросы о связях), messages (задание модели), conflicts
(расхождения между статьями), checks (проверка ответа: ссылки, отказ).

Ответ идёт потоком — событиями:

    {"type": "status", "text": "…"}                         что сейчас делается
    {"type": "sources", "queries": […], "sources": […],     на что опирается ответ
     "parts": […], "extra": […], "missing": […], "dataset": "…"|None,
     "collected": {"articles": N, "read": N, …}|None,     (сбор по статьям, collect.py)
     "rounds": 1–3, "checked": N, "judged": true|false}
    {"type": "general", "queries": […], "text": "…", …}     в базе нет — ответ без неё
                                                            (off_topic: true — вопрос не о
                                                            геологии: только отказ, без ответа)
    {"type": "token", "text": "…"}                          кусок ответа, много раз
    {"type": "revised", "text": "…", "removed": […]}       ответ целиком без фраз без
                                                            ссылки на фрагмент (drop_uncited)
    {"type": "done", "mode": "база"|"без базы", "used": [1, 3], "unknown": [],
     "coverage": [5, 6], "removed": 2, "rounds": …, …}
    {"type": "error", "error": "…"}
"""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, replace
from typing import Any

from ..index import db
from ..llm import LLM, LLM_HOST, LLMError
from .checks import (
    NO_ANSWER,
    REFUSAL_CHARS,
    citations,
    coverage,
    drop_repeats,
    drop_uncited,
    is_looping,
    is_refusal,
    says_not_in_base,
)
from .collect import collect, rest_of_summary
from .conflicts import find_conflicts
from .dataset import collect_dataset, dataset_sources, graph_candidates, question_subject
from .messages import build_messages, fit_budget, general_messages
from .plan import Plan, is_follow_up, make_plan, plan_question
from .retrieval import MAX_ROUNDS, Retrieval, judge_fragments, retrieve
from .settings import MODEL, Settings

NOT_IN_BASE = (
    "По вашему запросу в базе знаний такой информации нет. Ниже — ответ модели "
    "из её общих знаний: он не подтверждён статьями базы, проверяйте его."
)

# Вопрос не о геологии и в базе ничего нет: бот не отвечает из общих знаний
# (раньше выдавал, например, рецепт хлеба). Вернуть старое — Settings.general_off_topic.
OFF_TOPIC = (
    "Вопрос не относится к геологии, а бот отвечает только по рудной геологии и "
    "смежным наукам о Земле. В базе знаний статей об этом нет."
)


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
def llm_status(host: str = LLM_HOST, model: str = MODEL, timeout: int = 5) -> dict[str, Any]:
    """Жив ли сервер модели и та ли на нём модель — для подсказки на странице."""
    return LLM(model=model, host=host).status(timeout=timeout)


def stream_llm(messages: list[dict[str, Any]], settings: Settings) -> Iterator[dict[str, Any]]:
    """Ответ модели кусками: {"text": …} и в конце {"done": True, "tokens": …}."""
    return settings.client().stream(messages)


# --------------------------------------------------------------------------- #
#  Всё вместе
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Всё, что нужно для ответа: разбор вопроса, датасет, найденное."""

    plan: Plan
    sources: list[dict[str, Any]]
    retrieval: Retrieval
    dataset: dict[str, Any] | None = None
    graph: int = 0  # сколько кандидатов дал граф (вопрос о названном предмете)

    def info(self) -> dict[str, Any]:
        plan = self.plan
        return {
            "queries": plan.queries + plan.extra,
            "standalone": plan.standalone,
            "graph": self.graph,
            "parts": plan.describe() if len(plan.parts) > 1 else [],
            "extra": plan.extra,
            "missing": plan.missing,
            "dataset": self.dataset["name"] if self.dataset else None,
            "collected": (self.dataset or {}).get("collected"),
            **self.retrieval.info(),
        }


def prepare(
    conn: Any,
    embedder: Any,
    question: str,
    history: list[dict[str, Any]],
    settings: Settings,
    planner: Any = plan_question,
    judge: Any = judge_fragments,
    datasets: Any = collect_dataset,
    graph_hits: Any = graph_candidates,
) -> Generator[Any, Any, Any]:
    """Разбор вопроса и поиск с моделью. Генератор событий status; в конце — Prepared."""
    yield {"type": "status", "text": "разбираю вопрос…"}
    plan = make_plan(question, history, settings, planner)

    # «Собрать всё» о предмете вопроса: сначала — по всем статьям, где он упомянут
    # (collect.py), нет его в статьях — из датасета графа; поиск только добавляет.
    data = None
    base: list[dict[str, Any]] = []
    if plan.collect and settings.collect and db.is_real(conn):
        subject = question_subject(conn, plan.subject, question)
        if subject:
            gathered = yield from collect(
                conn, embedder, settings.client(), subject, question=question
            )
            if gathered is not None:
                base, data = gathered.sources, gathered.info()
    if not base and plan.collect and datasets is not None:
        yield {"type": "status", "text": "собираю факты из графа…"}
        data = datasets(conn, plan.subject, question)
        base = dataset_sources(data) if data else []

    # Вопрос о названном предмете — фрагменты о нём из графа в кандидаты.
    extra: list[Any] = []
    if not data and plan.subject and graph_hits is not None and judge is not None:
        extra = graph_hits(conn, plan.subject, question) or []
        if extra:
            yield {"type": "status", "text": f"беру из графа фрагменты о «{plan.subject}»…"}

    yield {
        "type": "status",
        "text": (
            f"ищу в базе по {len(plan.parts)} частям вопроса…"
            if len(plan.parts) > 1
            else "ищу в базе…"
        ),
    }
    limit = max(settings.top_k - (len(base) - 1), 3) if base else None
    # Вопрос не о геологии — один круг: проверить, что в базе правда пусто, и всё.
    rounds = MAX_ROUNDS if plan.on_topic else 1
    # После сбора по статьям поиск добавляет несколько фрагментов целиком, без заметок:
    # выписки по статьям уже есть, окно модели не резиновое.
    searching = replace(settings, notes=False) if data and "collected" in data else settings
    found = yield from retrieve(
        conn, embedder, question, plan, searching, judge, base, limit, rounds, extra
    )
    return Prepared(plan, base + found.sources, found, data, len(extra))


def answer(
    conn: Any,
    embedder: Any,
    question: str,
    history: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    llm: Any = stream_llm,
    planner: Any = plan_question,
    judge: Any = judge_fragments,
    datasets: Any = collect_dataset,
    prepared: Prepared | None = None,
    conflicts: Any = find_conflicts,
) -> Generator[dict[str, Any], None, None]:
    """Ответ на вопрос событиями (см. начало файла).

    llm, planner, judge и datasets подменяются в проверках: planner может
    вернуть и просто список запросов, judge — None (отбор по порогу), datasets
    (conn, предмет, вопрос) — датасет или None. prepared — уже найденное
    (оценка ищет один раз и для проверки поиска, и для ответа). conflicts
    (вопрос, фрагменты, настройки) → расхождения между статьями (find_conflicts);
    с подменённой моделью ответа настоящая сверка не идёт — она спросила бы модель."""
    settings = settings or Settings()
    history = history or []
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "error": "пустой вопрос"}
        return

    started = time.monotonic()
    if prepared is None:
        prepared = yield from prepare(
            conn, embedder, question, history, settings, planner, judge, datasets
        )
    sources, plan, info = prepared.sources, prepared.plan, prepared.info()

    # Разговор модели нужен только для вопроса вдогонку. В самостоятельном
    # вопросе прошлый ответ — помеха: на живом прогоне в ответ о Персияновском
    # разломе попала фраза из прошлого ответа об Ишимбинском, со ссылкой [1].
    if not is_follow_up(question, plan, history):
        history = []

    text, tokens = "", None
    if sources:
        clashes: list[dict[str, Any]] = []
        real = conflicts is not find_conflicts or llm is stream_llm
        if conflicts is not None and settings.check_conflicts and real:
            yield {"type": "status", "text": "сверяю, не расходятся ли статьи…"}
            clashes = conflicts(question, sources, settings) or []
        info["conflicts"] = clashes
        sources, turns = fit_budget(question, sources, history, settings, plan, clashes)
        yield {"type": "status", "text": "пишу ответ по найденному…"}
        # Отказ при фрагментах, которые модель-отборщик признала подходящими, — второй
        # заход с напоминанием правила 6 (insist). На s01 модель с одним и тем же набором
        # из 6 фрагментов отказывалась 3 раза из 3, хотя ответ в них есть (14.10.2026).
        retries = (False, True) if prepared.retrieval.judged else (False,)
        for insist in retries:
            if insist:
                yield {"type": "status", "text": "перечитываю найденное…"}
            done = yield from _answer_from(
                llm, question, sources, turns, plan, clashes, settings, info, started, insist
            )
            if done:
                return

    if not plan.on_topic and not settings.general_off_topic:
        # Не о геологии и в базе пусто — короткий отказ, без ответа из общих знаний.
        yield {"type": "general", "text": OFF_TOPIC, "off_topic": True, **info}
        yield {
            "type": "done",
            "mode": "без базы",
            "used": [],
            "unknown": [],
            "coverage": [0, 0],
            "model": settings.model,
            "seconds": round(time.monotonic() - started, 1),
            "tokens": None,
            "off_topic": True,
            **info,
        }
        return

    # В базе нет: ни статей, ни номеров — ответ модели с пометкой.
    yield {"type": "general", "text": NOT_IN_BASE, **info}
    yield {"type": "status", "text": "отвечаю без базы…"}
    try:
        for part in _clean(llm(general_messages(question, history, settings.model), settings)):
            if "done" in part:
                tokens = part["done"]
                break
            text += part["text"]
            yield {"type": "token", "text": part["text"]}
    except LLMError as exc:
        yield {"type": "error", "error": str(exc)}
        return
    yield {
        "type": "done",
        "mode": "без базы",
        "used": [],
        "unknown": [],
        "coverage": [0, 0],
        "model": settings.model,
        "seconds": round(time.monotonic() - started, 1),
        "tokens": tokens,
        **info,
    }


def _answer_from(
    llm: Any,
    question: str,
    sources: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    plan: Plan,
    clashes: list[dict[str, Any]],
    settings: Settings,
    info: dict[str, Any],
    started: float,
    insist: bool = False,
) -> Generator[dict[str, Any], None, bool]:
    """Ответ по фрагментам событиями. True — ответ дан (или ошибка сервера модели),
    False — модель отказалась: «в базе знаний нет ответа»."""
    messages = build_messages(question, sources, turns, settings.model, plan, clashes, insist)
    stream = _clean(llm(messages, settings))
    tokens = None
    try:
        # Первые слова держатся, пока не ясно, не отказ ли это: если модель
        # прочла фрагменты и ответа в них нет, статьи не показываются вовсе.
        # «В базе знаний нет информации о …» — отказ, если ответ на этом и кончается.
        head, finished = "", False
        for part in stream:
            if "done" in part:
                tokens, finished = part["done"], True
                break
            head += part["text"]
            short = says_not_in_base(head) and len(head) < REFUSAL_CHARS
            if len(head.strip()) >= len(NO_ANSWER) + 10 and not short:
                break
        if not head.strip() or is_refusal(head) or (finished and says_not_in_base(head)):
            return False
        yield {"type": "sources", "sources": sources, **info}
        text = head
        yield {"type": "token", "text": head}
        if not finished:
            for part in stream:
                if "done" in part:
                    tokens = part["done"]
                    break
                text += part["text"]
                yield {"type": "token", "text": part["text"]}
                if any(mark in part["text"] for mark in ".!?…") and is_looping(text):
                    break  # дальше — те же фразы по кругу; поток закроет finally
        # Фразы без ссылки на фрагмент — не из статей, повторы — зацикливание:
        # ответ заменяется очищенным, убранное показывается отдельно.
        unique = drop_repeats(text)
        cleaned, removed = drop_uncited(unique)
        rest = rest_of_summary(cleaned, sources)
        if rest:
            cleaned = f"{cleaned.rstrip()}\n\n{rest}".strip()
        used, unknown = citations(cleaned, len(sources))
        if not used:
            # Уже показанный поток нужно стереть перед повтором или ответом без базы.
            yield {"type": "revised", "text": "", "removed": removed}
            return False
        if cleaned != text:
            text = cleaned
            yield {"type": "revised", "text": text, "removed": removed}
        used, unknown = citations(text, len(sources))
        cited, claims = coverage(text)
        yield {
            "type": "done",
            "mode": "база",
            "used": used,
            "unknown": unknown,
            "coverage": [cited, claims],
            "removed": len(removed),
            "model": settings.model,
            "seconds": round(time.monotonic() - started, 1),
            "tokens": tokens,
            **({"insisted": True} if insist else {}),
            **info,
        }
        return True
    except LLMError as exc:
        yield {"type": "error", "error": str(exc)}
        return True
    finally:
        stream.close()


def _clean(parts: Any) -> Generator[dict[str, Any], None, None]:
    """Поток модели без блока размышлений: {"text": …} и в конце {"done": токены}."""
    think = ThinkFilter()
    try:
        for part in parts:
            if part.get("done"):
                tail = think.flush()
                if tail:
                    yield {"text": tail}
                yield {"done": part.get("tokens")}
                return
            piece = think.feed(part.get("text", ""))
            if piece:
                yield {"text": piece}
        tail = think.flush()
        if tail:
            yield {"text": tail}
        yield {"done": None}
    finally:
        close = getattr(parts, "close", None)
        if close:
            close()


class ThinkFilter:
    """Срезает блок <think>…</think>, если модель его всё-таки прислала.

    Размышления выключены (llm.NO_THINK), но сервер, запущенный без этой
    поддержки, отдаёт их прямо в тексте. Приходит это кусками, поэтому фильтр
    держит состояние между кусками.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.state = "start"  # start → (think →) text

    def feed(self, chunk: str) -> str:
        if self.state == "text":
            return chunk
        self.buffer += chunk
        if self.state == "start":
            head = self.buffer.lstrip()
            if not head:
                return ""
            if "<think>".startswith(head[:7]) and len(head) < 7:
                return ""  # ещё не ясно, начало ли это блока
            if head.startswith("<think>"):
                self.state = "think"
            else:
                self.state = "text"
                out, self.buffer = self.buffer.lstrip(), ""
                return out
        if self.state == "think" and "</think>" in self.buffer:
            rest = self.buffer.split("</think>", 1)[1].lstrip()
            self.state, self.buffer = "text", ""
            return rest
        return ""

    def flush(self) -> str:
        out = self.buffer if self.state != "think" else ""
        self.buffer = ""
        return out
