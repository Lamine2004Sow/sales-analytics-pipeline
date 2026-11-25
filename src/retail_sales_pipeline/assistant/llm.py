"""Minimal chat client for DeepSeek (OpenAI-compatible /chat/completions), JSON output mode."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from retail_sales_pipeline.config import ENV_FILE
from retail_sales_pipeline.load import _read_env_file

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
TIMEOUT_S = 60


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> Completion: ...


def _setting(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or _read_env_file(ENV_FILE).get(key) or default


class DeepSeekChat:
    """Deterministic settings (temperature 0) so that eval runs are comparable."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        self.name = model or _setting("DEEPSEEK_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        key = api_key or _setting("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set: export it or add it to .env")
        self._base_url = (base_url or _setting("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._http = requests.Session()
        self._http.headers["Authorization"] = f"Bearer {key}"

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        start = time.perf_counter()
        response = self._http.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self.name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
                "response_format": {"type": "json_object"},
            },
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return Completion(
            content=body["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - start,
        )
