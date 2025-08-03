# ~/sandbox/agent/llm_backends/fireworks_backend.py
from __future__ import annotations

import os
from typing import Dict

from agent.settings import settings
from . import LLMBackendError
from .openai_compatible import OpenAICompatibleBackend


class FireworksBackend(OpenAICompatibleBackend):
    BASE_URL = "https://api.fireworks.ai/inference/v1"
    DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v3"
    _GENERIC_MODELS = {"gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-3.5"}  # tweak if needed

    def __init__(self, cfg: Dict):
        api_key = (
            os.getenv("FIREWORKS_API_KEY")
            or settings.get_config_value("FIREWORKS_API_KEY")  # type: ignore[attr-defined]
        )
        if not api_key:
            raise LLMBackendError("FIREWORKS_API_KEY not set")

        super().__init__(cfg, api_key)

        # Override if caller left generic placeholder
        if cfg.get("model_name") in self._GENERIC_MODELS:
            self.model_name = self.DEFAULT_MODEL

    def api_key_header(self):
        return {"Authorization": f"Bearer {self.api_key}"}

