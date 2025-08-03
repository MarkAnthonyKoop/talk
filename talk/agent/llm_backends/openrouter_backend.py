from __future__ import annotations

import importlib
import os
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError


class OpenRouterBackend(LLMBackend):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMBackendError("OPENROUTER_API_KEY not set")

        try:
            self.requests = importlib.import_module("requests")
        except ImportError as exc:
            raise LLMBackendError("pip install requests") from exc

        self.model_name = cfg.get("model_name", "openrouter/openai/gpt-3.5-turbo")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    def api_key_header(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "backend_smoke_test",
        }

    def complete(self, messages: List[Message]) -> Message:
        headers = self.api_key_header()
        headers["Content-Type"] = "application/json"

        payload = {
            "model": self.model_name,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            r = self.requests.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            return Message.from_provider(data["choices"][0]["message"])
        except Exception as exc:
            raise LLMBackendError(exc) from exc

