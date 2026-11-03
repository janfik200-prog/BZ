"""WP03. Адаптер модели для экспериментов: тот же протокол OpenAI, что georag/llm.py, но
с журналом каждого вызова — модель (как её назвал сервер), хэш задания, токены задания и
ответа (usage сервера), время, ошибка. «Qwen3-8B использована» пишется только по
такому журналу, а не по настройке.

Невалидный JSON — ok=False и текст ошибки в журнале; пустой «правильный» ответ вместо
ошибки не подставляется.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm import NO_THINK, extract_json
from .contracts import sha256_text


@dataclass
class Result:
    ok: bool
    data: Any
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    model: str
    prompt_sha256: str


class ModelAdapter:
    def __init__(
        self, host: str, model: str, log_path: Path | str | None = None, timeout: int = 300
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()

    def server_models(self) -> list[dict[str, Any]]:
        import requests

        r = requests.get(f"{self.host}/v1/models", timeout=10)
        r.raise_for_status()
        return list(r.json().get("data", []))

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 3000,
        temperature: float = 0.0,
        tag: str = "",
    ) -> Result:
        import requests

        prompt_sha = sha256_text(system + "\n" + user)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            **NO_THINK,
        }
        started = time.monotonic()
        usage: dict[str, Any] = {}
        error, data, served = None, None, self.model
        try:
            r = requests.post(
                f"{self.host}/v1/chat/completions", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage") or {}
            served = body.get("model") or self.model
            text = body["choices"][0]["message"].get("content") or ""
            try:
                data = extract_json(text, repair=True)
            except ValueError as exc:
                error = f"invalid JSON: {exc}"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        result = Result(
            ok=error is None,
            data=data,
            error=error,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=round(time.monotonic() - started, 3),
            model=served,
            prompt_sha256=prompt_sha,
        )
        if self.log_path:
            row = {
                "tag": tag,
                "model": served,
                "prompt_sha256": prompt_sha,
                "ok": result.ok,
                "error": error,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": result.latency_s,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with self._lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result
