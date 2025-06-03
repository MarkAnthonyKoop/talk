from __future__ import annotations

import importlib
import logging
import os
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError

log = logging.getLogger(__name__)

class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key_var = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_var) or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError(f"environment variable '{api_key_var}' not set")

        try:
            openai_mod = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMBackendError("pip install openai") from exc

        self.client = openai_mod.OpenAI(api_key=api_key)
        self.model_name = cfg.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    def complete(self, messages: List[Message]) -> Message:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[m.to_dict() for m in messages],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            msg_raw = resp.choices[0].message.to_dict()
            return Message.from_provider(msg_raw)
        except Exception as exc:
            raise LLMBackendError(exc) from exc
