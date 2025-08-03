from __future__ import annotations

import os
from typing import Dict

from . import LLMBackendError
from .openai_compatible import OpenAICompatibleBackend

class PerplexityBackend(OpenAICompatibleBackend):
    BASE_URL = "https://api.perplexity.ai"
    DEFAULT_MODEL = "pplx-7b-online"

    def __init__(self, cfg: Dict):
        api_key = os.getenv("PPLX_API_KEY") or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError("PPLX_API_KEY not set")
        super().__init__(cfg, api_key)

