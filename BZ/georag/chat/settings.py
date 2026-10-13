"""Настройки чат-бота: модель и её сервер, сколько фрагментов, запросов и частей вопроса."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import DEFAULT_MODEL, LLM, LLM_HOST, NUM_CTX

MODEL = DEFAULT_MODEL

# Сколько фрагментов давать модели для ответа: восемь, из одной статьи до
# трёх. Вместе с соседними кусками (см. search.NEIGHBOR_CHARS) это около 7 тысяч
# токенов — окно модели 12 тысяч, с запасом на ответ и историю разговора.
TOP_K = 8
PER_DOC = 3
QUERIES = 3  # сколько поисковых запросов на часть вопроса
PARTS = 3  # на сколько частей самое большее делится вопрос


@dataclass
class Settings:
    model: str = MODEL
    host: str = LLM_HOST
    top_k: int = TOP_K
    per_doc: int = PER_DOC
    queries: int = QUERIES
    temperature: float = 0.1
    num_ctx: int = NUM_CTX
    timeout: int = 300
    parts: int = PARTS  # на сколько частей самое большее делить вопрос
    # Вопрос не по теме и в базе пусто: False — короткий отказ, True — ответ из общих знаний.
    general_off_topic: bool = False
    # Сверять, не расходятся ли статьи (find_conflicts): около 4 с на ответ.
    check_conflicts: bool = True

    def client(self) -> LLM:
        """Клиент сервера модели с этими настройками."""
        return LLM(
            model=self.model,
            host=self.host,
            timeout=self.timeout,
            temperature=self.temperature,
        )
