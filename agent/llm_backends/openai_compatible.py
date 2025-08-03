from __future__ import annotations

import importlib
import os
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError

class OpenAICompatibleBackend(LLMBackend):
    BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "gpt-3.5-turbo"

    def __init__(self, cfg: Dict, api_key: str | None = None):
        super().__init__(cfg)
        self.api_key = api_key or os.getenv(cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not self.api_key:
            raise LLMBackendError("Missing required API key")

        try:
            openai_mod = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMBackendError("pip install openai") from exc

        self.client = openai_mod.OpenAI(
            api_key=self.api_key,
            base_url=cfg.get("base_url", self.BASE_URL),
        )

        self.model_name = cfg.get("model_name", self.DEFAULT_MODEL)
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

        # Sub-classes may set this
        self._extra_headers: Dict[str, str] | None = None

    def complete(self, messages: List[Message]) -> Message:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[m.to_dict() for m in messages],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **({"extra_headers": self._extra_headers} if self._extra_headers else {}),
            )

            return Message.from_provider(resp.choices[0].message.to_dict())
        except Exception as exc:
            raise LLMBackendError(exc) from exc

