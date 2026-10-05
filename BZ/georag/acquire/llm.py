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

from dataclasses import dataclass
from typing import Any

from ..llm import Ollama, extract_json
from .models import Candidate

# О чём база. Без этого модель понимает тему буквально: вопрос про «признаки
# на основе гравиметрии в моделях перспективности» давал запросы вроде
# «gravimetry features prospectivity models», а они находили медицину и геодезию.
DOMAIN = (
    "База знаний — для прогноза рудных месторождений: прогнозирование оруденения, "
    "карты и модели перспективности (mineral prospectivity mapping), поисковые "
    "признаки и критерии, рудная геология, геофизика и дистанционное зондирование "
    "в поисках полезных ископаемых."
)

QUERY_SYSTEM = (
    "Ты помогаешь искать научные статьи в библиографической базе. "
    + DOMAIN
    + " Отвечай только JSON, без пояснений."
)

QUERY_USER = """Тема поиска: «{topic}».

Тема может быть сформулирована свободно — фразой, вопросом или описанием
исследовательского интереса. Разбери её на смысловые аспекты.

Составь {n} поисковых запросов для научной базы. Требования:
- часть запросов по-русски, часть по-английски: база двуязычная, и англоязычные
  формулировки находят другие работы, чем русские;
- 2-6 слов, без кавычек, без логических операторов, без вопросительных слов;
- в каждом запросе — привязка к рудной тематике (mineral prospectivity, ore
  deposits, mineral exploration, прогноз оруденения, рудные месторождения):
  без неё слово «gravimetry» или «features» находит медицину и геодезию;
- термины — как пишут в статьях: gravity data, gravity anomalies, Bouguer anomaly,
  magnetic data, remote sensing, evidential layers, predictor maps, а не
  дословный перевод вопроса;
- запросы должны отличаться по смыслу, а не быть перестановкой одних слов.

Пример. Тема «Какие признаки на основе гравиметрии использовались в моделях
перспективности?» → gravity data mineral prospectivity mapping; gravity anomalies
evidential layer prospectivity; гравиметрические данные прогноз оруденения;
potential field data mineral exploration targeting; геофизические признаки
прогнозные модели месторождений.

Верни JSON: {{"queries": ["...", "..."]}}"""

FILTER_SYSTEM = (
    "Ты отбираешь научные статьи для базы знаний. "
    + DOMAIN
    + " Отвечай только JSON, без пояснений."
)

FILTER_USER = """Тема: «{topic}».

Ниже статьи. Для каждой реши, брать ли её в базу по этой теме.

Правила:
1. Берём, только если статья про рудную геологию, поиски или прогноз полезных
   ископаемых И отвечает теме хотя бы по одному главному аспекту. Не нужно,
   чтобы совпали все слова темы.
   Пример: тема про гравиметрические признаки в моделях перспективности — статья
   про карту перспективности, где среди слоёв геофизические или гравиметрические
   данные, подходит; статья про модели перспективности вообще тоже подходит, если
   в ней разбираются поисковые признаки.
2. Слово из темы в другом смысле — мимо: гравиметрия в геодезии, гидрологии (GRACE),
   приборах и спутниках; gravimetric analysis в химии; prospective study в медицине;
   модели в экономике. Нефть и газ, если тема не о них, — тоже мимо.
3. Сомневаешься, а статья про прогноз или поиски руд, — бери.

{items}

Верни JSON: {{"decisions": [{{"i": 1, "relevant": true, "reason": "краткая причина"}}]}}
Причина — не больше 15 слов. Реши по каждому номеру из списка."""


# Разбор JSON и сам клиент — общие для всего проекта, в georag/llm.py.
_extract_json = extract_json


@dataclass
class OllamaLLM(Ollama):
    """Клиент Ollama с двумя вопросами добычи: запросы и отбор аннотаций."""

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

        decided: dict[int, dict[str, Any]] = {}
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
    llm: Any,
    topic: str,
    candidates: list[Candidate],
    batch_size: int = 8,
    fallback: Any = None,
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
