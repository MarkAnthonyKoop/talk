from __future__ import annotations

import os
from typing import Dict

from . import LLMBackendError
from .openai_compatible import OpenAICompatibleBackend

class FireworksBackend(OpenAICompatibleBackend):
    BASE_URL = "https://api.fireworks.ai/inference/v1"
    DEFAULT_MODEL = "accounts/fireworks/models/deepseek-r1"
    
    def __init__(self, cfg: Dict):
        api_key = os.getenv("FIREWORKS_API_KEY") or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError("FIREWORKS_API_KEY not set")
        super().__init__(cfg, api_key)
        # Fireworks uses standard Bearer auth; no extra headers needed.
