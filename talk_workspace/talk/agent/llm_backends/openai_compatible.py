# ~/sandbox/agent/llm_backends/openai_compatible.py
"""
Mixin/helper for providers that mimic the OpenAI Chat API.

Concrete subclasses must define:

    BASE_URL: str
    api_key_header(): -> dict[str, str]  (adds auth header)
"""

from __future__ import annotations

import importlib
import json
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError


class OpenAICompatibleBackend(LLMBackend):
    BASE_URL: str = ""
    TIMEOUT = 30

    def __init__(self, cfg: Dict, api_key: str):
        super().__init__(cfg)
        self.api_key = api_key

        try:
            self.requests = importlib.import_module("requests")
        except ImportError as exc:
            raise LLMBackendError("pip install requests") from exc

        self.model_name = cfg.get("model_name", "gpt-3.5-turbo")

        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    # ----- helpers ---------------------------------------------------- #
    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", **self.api_key_header()}

    def api_key_header(self) -> Dict[str, str]:  # override in subclass
        return {}

    # ----- main call --------------------------------------------------- #
    def complete(self, messages: List[Message]) -> Message:
        payload = {
            "model": self.model_name,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            r = self.requests.post(
                f"{self.BASE_URL}/chat/completions",
                data=json.dumps(payload),
                headers=self._headers(),
                timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            return Message.from_provider(data["choices"][0]["message"])
        except Exception as exc:
            raise LLMBackendError(exc) from exc

