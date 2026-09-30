"""Языковая модель в Ollama — один клиент на весь проект.

Раньше к Ollama ходили четыре места, каждое по-своему: отбор статей, чат-бот
(запросы к базе и ответ потоком), проверка разметки. Адрес, срезание блока
размышлений Qwen3, разбор JSON из ответа и тексты ошибок повторялись. Теперь
всё это здесь, а остальные модули только задают вопросы.

Адрес по умолчанию — localhost:11434; другой задаётся переменной окружения
GEORAG_OLLAMA (например, если Ollama стоит на соседней машине).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterator

DEFAULT_MODEL = "qwen3:14b"
OLLAMA_HOST = os.environ.get("GEORAG_OLLAMA", "").strip() or "http://localhost:11434"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
INSTALL_HINT = "Запустите её или переустановите с ollama.com/download."


class LLMError(RuntimeError):
    """Модель не ответила или ответила ошибкой. Текст — для человека."""


def strip_think(text: str) -> str:
    """Убрать блок размышлений Qwen3, в том числе незакрытый."""
    cleaned = THINK_RE.sub("", text or "")
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned


def extract_json(text: str) -> dict:
    """Первый объект JSON из ответа модели: вокруг него бывает лишний текст."""
    cleaned = strip_think(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"в ответе нет JSON: {cleaned[:200]!r}")
    return json.loads(cleaned[start: end + 1])


def no_think(model: str, text: str) -> str:
    """Qwen3 без размышлений вслух: ответ в разы быстрее, а рассуждать тут не о чем."""
    return f"{text}\n\n/no_think" if model.lower().startswith("qwen3") else text


@dataclass
class Ollama:
    model: str = DEFAULT_MODEL
    host: str = OLLAMA_HOST
    timeout: int = 180
    temperature: float = 0.2
    num_ctx: int = 8192

    def _url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}{path}"

    def chat_json(self, system: str, user: str, *, temperature: float | None = None,
                  num_ctx: int | None = None, timeout=None) -> dict:
        """Один вопрос — один объект JSON в ответ."""
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": no_think(self.model, user)}],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_ctx": num_ctx or self.num_ctx},
        }
        response = requests.post(self._url("/api/chat"), json=payload,
                                 timeout=timeout or self.timeout)
        response.raise_for_status()
        return extract_json((response.json().get("message") or {}).get("content", ""))

    def stream(self, messages: list[dict], *, temperature: float | None = None,
               num_ctx: int | None = None) -> Iterator[dict]:
        """Ответ кусками: {"text": …}, в конце {"done": True, "tokens": …}."""
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_ctx": num_ctx or self.num_ctx},
        }
        try:
            response = requests.post(self._url("/api/chat"), json=payload, stream=True,
                                     timeout=(10, self.timeout))
        except requests.RequestException as exc:
            raise LLMError(f"Ollama не отвечает на {self.host}: {type(exc).__name__}. "
                           f"{INSTALL_HINT}") from None
        with response:
            if response.status_code == 404:
                raise LLMError(f"Модель {self.model} не скачана: ollama pull {self.model}")
            if response.status_code >= 400:
                raise LLMError(f"Ollama ответила ошибкой {response.status_code}: "
                               f"{response.text[:300]}")
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise LLMError(f"Ollama: {data['error']}")
                text = (data.get("message") or {}).get("content") or ""
                if text:
                    yield {"text": text}
                if data.get("done"):
                    yield {"done": True, "tokens": data.get("eval_count")}
                    return

    def status(self, timeout: int = 5) -> dict:
        """Жива ли Ollama и скачана ли модель — для подсказки человеку."""
        import requests

        try:
            response = requests.get(self._url("/api/tags"), timeout=timeout)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "model": self.model,
                    "error": f"Ollama не отвечает ({type(exc).__name__}). {INSTALL_HINT}"}
        base = self.model.split(":")[0]
        present = self.model in names or (
            ":" not in self.model and any(n.split(":")[0] == base for n in names))
        if not present:
            return {"ok": False, "model": self.model, "models": names,
                    "error": f"Модель {self.model} не скачана: ollama pull {self.model}"}
        return {"ok": True, "model": self.model}
