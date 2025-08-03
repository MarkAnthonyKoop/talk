from __future__ import annotations

import importlib
from typing import Dict, List

from agent.messages import Message
from agent.settings import settings
from . import LLMBackend, LLMBackendError


class GeminiBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key = settings.get_config_value("GEMINI_API_KEY")
        if not api_key:
            raise LLMBackendError("GEMINI_API_KEY not set")

        try:
            genai = importlib.import_module("google.generativeai")
        except ImportError:
            raise LLMBackendError("pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(cfg.get("model_name", "gemini-pro"))
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    def complete(self, messages: List[Message]) -> Message:
        prompt = "\n".join(
            f"{m.role.value}: {m.content or ''}" for m in messages if m.content
        )
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={'temperature': self.temperature, 'max_output_tokens': self.max_tokens},
            )
            text = getattr(resp, "text", "").strip()
            return Message(role="assistant", content=text)  # type: ignore[arg-type]
        except Exception as exc:
            raise LLMBackendError(exc) from exc

