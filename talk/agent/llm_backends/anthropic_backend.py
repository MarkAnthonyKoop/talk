"""
Anthropic Claude backend (v1 `messages` endpoint).
"""

from __future__ import annotations

import importlib
from typing import Dict, List

from agent.messages import Message
from agent.settings import settings
from . import LLMBackend, LLMBackendError


class AnthropicBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key = settings.get_config_value("CLAUDE_API_TOKEN")  # type: ignore[attr-defined]
        if not api_key:
            raise LLMBackendError("CLAUDE_API_TOKEN not set")

        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError as exc:
            raise LLMBackendError("pip install anthropic") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = cfg.get("model_name", "claude-3-haiku-20240307")

    def complete(self, messages: List[Message]) -> Message:
        conv = [
            {"role": m.role.value, "content": m.content or ""}
            for m in messages
            if m.content
        ]
        try:
            resp = self.client.messages.create(
                model=self.model_name,
                max_tokens=cfg.get("max_tokens", 1500),
                temperature=cfg.get("temperature", 0.7),
                messages=conv,
            )
            text = resp.content[0].text.strip()
            return Message(role="assistant", content=text)  # type: ignore[arg-type]
        except Exception as exc:
            raise LLMBackendError(exc) from exc

