from __future__ import annotations

import importlib
import os
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError


class PerplexityBackend(LLMBackend):
    BASE_URL = "https://api.perplexity.ai"   # ← corrected

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise LLMBackendError("PERPLEXITY_API_KEY not set")

        try:
            self.requests = importlib.import_module("requests")
        except ImportError as exc:
            raise LLMBackendError("pip install requests") from exc

        self.model_name = cfg.get("model_name", "pplx-7b-online")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    def complete(self, messages: List[Message]) -> Message:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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

