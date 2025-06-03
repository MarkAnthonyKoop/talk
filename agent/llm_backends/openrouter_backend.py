from __future__ import annotations

import os
from typing import Dict

from . import LLMBackendError
from .openai_compatible import OpenAICompatibleBackend

class OpenRouterBackend(OpenAICompatibleBackend):
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "gpt-4o-mini"  # updated default

    def __init__(self, cfg: Dict):
        api_key = os.getenv("OPENROUTER_API_KEY") or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError("OPENROUTER_API_KEY not set")
        super().__init__(cfg, api_key)
        referer = (
            os.getenv("OPENROUTER_REFERER")
            or cfg.get("referer")
            or "https://github.com/your/project"
        )
        self._extra_headers = {"HTTP-Referer": referer}
