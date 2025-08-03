# pyright: reportUnusedImport=false
"""
LLM backend package – factory & base classes.

Add a new provider by:
1. Writing `<name>_backend.py` with a class `<Name>Backend`.
2. Importing the module lazily in `get_backend()`.

Recognised `type` strings (case‑insensitive):

    openai, shell, gemini, anthropic, claude, perplexity, openrouter, stub
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Dict, List

from agent.messages import Message
from agent.settings import settings

log = logging.getLogger(__name__)


class LLMBackendError(RuntimeError):
    pass


class LLMBackend:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.model_name: str = self.cfg.get("model_name", "unknown")

    def complete(self, messages: List[Message]) -> Message:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Stub (always available)
# --------------------------------------------------------------------------- #

class StubBackend(LLMBackend):
    def complete(self, messages: List[Message]) -> Message:
        last = next((m for m in reversed(messages) if m.role == "user"), None)
        echo = last.content[:100] if last else ""
        return Message(role="assistant", content=f"[stub] {echo}")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Internal helper
# --------------------------------------------------------------------------- #

_BACKEND_MAP = {
    "openai": "openai_backend.OpenAIBackend",
    "shell": "shell_backend.ShellBackend",
    "gemini": "gemini_backend.GeminiBackend",
    "anthropic": "anthropic_backend.AnthropicBackend",
    "claude": "anthropic_backend.AnthropicBackend",
    "perplexity": "perplexity_backend.PerplexityBackend",
    "openrouter": "openrouter_backend.OpenRouterBackend",
    "fireworks": "fireworks_backend.FireworksBackend",     #  ← new
    "stub": "stub",
}



def _load(symbol_path: str):
    module_name, class_name = symbol_path.split(".")
    mod = import_module(f"agent.llm_backends.{module_name}")
    return getattr(mod, class_name)


def get_backend(provider_cfg: Dict[str, Any] | None = None) -> LLMBackend:
    cfg = provider_cfg or settings.provider.model_dump()
    key = cfg.get("type", "stub").lower()

    if key == "stub":
        return StubBackend(cfg)

    target = _BACKEND_MAP.get(key)
    if target is None:
        log.warning(f"[llm] unknown backend '{key}', falling back to stub.")
        return StubBackend(cfg)

    try:
        cls = _load(target)
        return cls(cfg)
    except (ImportError, AttributeError) as exc:
        log.error(f"[llm] failed to load backend '{key}': {exc}")
        return StubBackend(cfg)

