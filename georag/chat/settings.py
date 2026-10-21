"""Настройки чат-бота: модель и её сервер, сколько фрагментов, запросов и частей вопроса."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..llm import DEFAULT_MODEL, LLM, LLM_HOST, NUM_CTX

MODEL = DEFAULT_MODEL

# Сколько фрагментов давать модели для ответа: восемь, из одной статьи до
# трёх. Вместе с соседними кусками (см. search.NEIGHBOR_CHARS) это около 7 тысяч
# токенов — окно модели 12 тысяч, с запасом на ответ и историю разговора.
TOP_K = 8
PER_DOC = 3
QUERIES = 3  # сколько поисковых запросов на часть вопроса
PARTS = 3  # на сколько частей самое большее делится вопрос


def env_flag(name: str) -> bool:
    """Переключатель из окружения: 1, true, yes, да — включено; пусто и прочее — выключено."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "да")


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
    # Вопрос по теме, а в базе пусто: False — отказ «в базе нет», True — ответ из общих
    # знаний модели с пометкой (GEORAG_GENERAL=1). Выключено: в AgentKB на «сколько статей
    # про Марокко» такой ответ сочинил геологию Марокко — с пометкой, но вредно.
    general_knowledge: bool = field(default_factory=lambda: env_flag("GEORAG_GENERAL"))
    # Сверять, не расходятся ли статьи (find_conflicts): около 4 с на ответ.
    check_conflicts: bool = True
    # Искать и по векторам фактов графа: фрагменты, которые привели близкие факты,
    # получают до двух мест среди кандидатов (index.search.FACT_SLOTS). Выключено:
    # поиск это чуть улучшает, ответы — нет (сравнение 14.10.2026 — в УСТРОЙСТВО.md,
    # «Поиск по связям»). Включить и сравнить — eval --facts.
    fact_search: bool = False
    # Заметки: модель, отбирая фрагменты, пишет по каждому подходящему, что в нём сказано
    # по вопросу; ответ видит TOP_K фрагментов целиком и заметки ещё по NOTES (retrieval).
    # Статей в ответе — вдвое больше, но полнота по эталону не выросла надёжно (0,30 → 0,36
    # и 0,35 → 0,33 на 8 широких вопросах, 15.10.2026), а ответ дольше на 25 с — выключено.
    # wide — сколько кандидатов читает модель с заметками и на узкий вопрос (plan.narrow).
    notes: bool = False
    wide: int = 32
    # Сбор по предмету (collect.py): на «что известно о X» и «с чем связан X» модель
    # читает все статьи, где X упомянут, и отвечает по сводке и выпискам из них.
    collect: bool = True

    def client(self) -> LLM:
        """Клиент сервера модели с этими настройками."""
        return LLM(
            model=self.model,
            host=self.host,
            timeout=self.timeout,
            temperature=self.temperature,
        )
