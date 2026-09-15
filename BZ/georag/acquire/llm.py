"""Два места, где в добыче участвует LLM: формулировка запросов и фильтрация аннотаций.

Всё остальное делает Python. Модель не решает, что скачивать и куда сохранять —
она только предлагает запросы и отвечает «по теме / не по теме».

Особенности Qwen3, из-за которых наивный код ломается:

* это думающая модель, и по умолчанию она выдаёт блок <think>…</think> перед ответом.
  С `format: json` Ollama ждёт чистый JSON, поэтому размышления либо выключаются
  (`think: false`, плюс подсказка /no_think в промпте), либо срезаются из ответа;
* даже с JSON-режимом ответ иногда приходит с обёрткой — поэтому JSON вытаскивается
  по границам скобок, а не через прямой json.loads всего текста.

Если модель не ответила или ответила мусором, решение по статье принимает
эвристика из heuristic.py — прогон не встаёт. Без неё (флаг --no-fallback)
статья получает статус «нужно решение» и попадает в отчёт человеку.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import Candidate

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

QUERY_SYSTEM = (
    "Ты помогаешь искать научные статьи в библиографической базе. "
    "Отвечай только JSON, без пояснений."
)

QUERY_USER = """Тема поиска: «{topic}».

Тема может быть сформулирована свободно — одной фразой или целым описанием
исследовательского интереса. Разбери её на разные смысловые аспекты.

Составь {n} поисковых запросов для научной базы. Требования:
- часть запросов по-русски, часть по-английски: база двуязычная, и англоязычные
  формулировки находят другие работы, чем русские;
- 2-6 слов, без кавычек, без логических операторов;
- запросы должны отличаться по смыслу, а не быть перестановкой одних слов;
- используй принятую терминологию предметной области.

Верни JSON: {{"queries": ["...", "..."]}}"""

FILTER_SYSTEM = (
    "Ты отбираешь научные статьи по теме для базы знаний. "
    "Строго следуй теме: работы из смежных областей, не отвечающие теме, отклоняй. "
    "Отвечай только JSON, без пояснений."
)

FILTER_USER = """Тема: «{topic}».

Ниже статьи. Для каждой реши, относится ли она к теме.

{items}

Верни JSON: {{"decisions": [{{"i": 1, "relevant": true, "reason": "краткая причина"}}]}}
Причина — не больше 15 слов. Реши по каждому номеру из списка."""


def _extract_json(text: str) -> dict:
    cleaned = _THINK_RE.sub("", text or "")
    if "</think>" in cleaned:  # незакрытый блок размышлений
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"в ответе нет JSON: {cleaned[:200]!r}")
    return json.loads(cleaned[start : end + 1])


@dataclass
class OllamaLLM:
    model: str = "qwen3:14b"
    host: str = "http://localhost:11434"
    timeout: int = 180
    temperature: float = 0.2
    num_ctx: int = 8192

    def chat_json(self, system: str, user: str) -> dict:
        import requests

        if self.model.lower().startswith("qwen3"):
            user = f"{user}\n\n/no_think"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        content = (response.json().get("message") or {}).get("content", "")
        return _extract_json(content)

    # -- шаг 2: формулировка запросов -------------------------------------- #
    def queries(self, topic: str, n: int = 5) -> list[str]:
        data = self.chat_json(QUERY_SYSTEM, QUERY_USER.format(topic=topic, n=n))
        queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
        if not queries:
            raise ValueError("модель не вернула ни одного запроса")
        return queries[:n]

    # -- шаг 4: фильтрация аннотаций --------------------------------------- #
    def filter_batch(self, topic: str, batch: list[Candidate]) -> None:
        items = "\n\n".join(f"{i}. {c.brief()}" for i, c in enumerate(batch, start=1))
        data = self.chat_json(FILTER_SYSTEM, FILTER_USER.format(topic=topic, items=items))

        decided: dict[int, dict] = {}
        for item in data.get("decisions") or []:
            try:
                decided[int(item.get("i"))] = item
            except (TypeError, ValueError):
                continue

        for i, cand in enumerate(batch, start=1):
            decision = decided.get(i)
            if decision is None:
                cand.relevant = None
                cand.reason = "модель не дала решения по этому номеру"
                continue
            cand.relevant = bool(decision.get("relevant"))
            cand.reason = str(decision.get("reason") or "").strip()[:200]


def filter_candidates(
    llm,
    topic: str,
    candidates: list[Candidate],
    batch_size: int = 8,
    fallback=None,
) -> list[str]:
    """Фильтрация партиями. Возвращает ошибки по партиям, не падая целиком.

    Если модель не ответила и задан fallback, решение принимает он — иначе прогон
    без человека встанет: все статьи уйдут в статус «нужно решение».
    """
    errors: list[str] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        try:
            llm.filter_batch(topic, batch)
            for cand in batch:
                cand.filtered_by = getattr(llm, "model", "llm")
        except Exception as exc:  # noqa: BLE001 — одна плохая партия не рушит прогон
            errors.append(f"{type(exc).__name__}: {exc}")
            if fallback is not None:
                fallback.filter_batch(topic, batch)
                for cand in batch:
                    cand.filtered_by = getattr(fallback, "model", "эвристика")
            else:
                for cand in batch:
                    cand.relevant = None
                    cand.reason = f"ошибка фильтрации: {type(exc).__name__}"
                    cand.filtered_by = "нет решения"

    # Модель может вернуть ответ без решения по отдельному номеру — добираем их.
    undecided = [c for c in candidates if c.relevant is None]
    if undecided and fallback is not None:
        fallback.filter_batch(topic, undecided)
        for cand in undecided:
            cand.filtered_by = getattr(fallback, "model", "эвристика")

    return errors
