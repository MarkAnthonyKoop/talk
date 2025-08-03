# ~/sandbox/agent/llm_backends/openai_backend.py
"""
Thin wrapper around `openai` Chat Completions v1.

Requirements
------------
    pip install openai>=1.2.0

Environment
-----------
    export OPENAI_API_KEY=sk-...

Notes
-----
* Supports `temperature` + `max_tokens` from settings.
* Converts internal `Message` objects to the provider’s format.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List

from agent.messages import Message
from agent.settings import settings
from . import LLMBackend, LLMBackendError  # relative import from pkg __init__

log = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        api_key_var = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = settings.get_config_value(api_key_var)  # type: ignore[attr-defined]
        if not api_key:
            raise LLMBackendError(
                f"environment variable '{api_key_var}' not set – cannot use OpenAIBackend"
            )

        try:
            openai_mod = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMBackendError(
                "package 'openai' not installed; run `pip install openai`"
            ) from exc

        self.client = openai_mod.OpenAI(api_key=api_key)

        # Pull model+params from cfg
        self.model_name = cfg.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)

    # ------------------------------------------------------------------ #
    # Mandatory override
    # ------------------------------------------------------------------ #
    def complete(self, messages: List[Message]) -> Message:
        """Convert messages → provider format → response."""
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

