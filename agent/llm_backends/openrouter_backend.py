from __future__ import annotations

import os
from typing import Dict

from agent.llm_backends.openai_compatible import OpenAICompatibleBackend
from . import LLMBackendError

class OpenrouterBackend(OpenAICompatibleBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key_var = cfg.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(api_key_var) or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError(f"environment variable '{api_key_var}' not set")
        
        # Set base URL and headers for OpenRouter
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers.update({
            "HTTP-Referer": "https://talk.agent.framework",
            "X-Title": "Talk Agent Framework",
        })

