# ~/sandbox/agent/agent.py
"""
High‑level Agent class.

Changes 2025‑04‑17
------------------
• robustly handles cfg_overrides that turn `self.cfg.provider`
  from a Pydantic model into a plain dict (avoids AttributeError).
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.messages import Message, Role
from agent.settings import Settings, settings

# need BaseSettings only for isinstance check
try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:  # pydantic‑v1 fallback
    from pydantic import BaseSettings  # type: ignore

from agent.storage import ConversationLog
from agent.llm_backends import LLMBackend, LLMBackendError, get_backend

log = logging.getLogger(__name__)


class Agent:
    """
    Parameters
    ----------
    cfg_overrides : dict | None
        Extra settings merged into the global singleton.
    conversation_path : str | Path | None
        Where to store the JSONL log; if None auto‑generated.
    backend : LLMBackend | None
        Inject a pre‑built backend (mostly for unit tests).
    system_prompt : str | None
        Override the default system prompt.
    """

    def __init__(
        self,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        conversation_path: Optional[str | Path] = None,
        backend: Optional[LLMBackend] = None,
        system_prompt: Optional[str] = None,
    ):
        # ------------------------------------------------------------------ #
        # Local, immutable settings instance
        # ------------------------------------------------------------------ #
        self.cfg: Settings = settings.resolve(cfg_overrides)

        # ------------------------------------------------------------------ #
        # Conversation log
        # ------------------------------------------------------------------ #
        if conversation_path is None:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_path = (
                self.cfg.paths.conversations_dir / f"conv_{ts}.jsonl"
            )
        self.log = ConversationLog(conversation_path)

        # ------------------------------------------------------------------ #
        # Backend – tolerate dict or BaseSettings
        # ------------------------------------------------------------------ #
        if backend:
            self.backend = backend
        else:
            provider_raw = self.cfg.provider
            if isinstance(provider_raw, BaseSettings):
                provider_cfg = provider_raw.model_dump()
            else:  # already a dict due to overrides
                provider_cfg = dict(provider_raw)
            self.backend = get_backend(provider_cfg)

        # ------------------------------------------------------------------ #
        # System prompt
        # ------------------------------------------------------------------ #
        prompt = system_prompt or self.cfg.default_system_prompt
        if prompt:
            self._ensure_system_prompt(prompt)

        log.info(
            "[agent] ready – backend=%s  log=%s",
            type(self.backend).__name__,
            self.log._messages,
        )

    # ================================================================== #
    # Public interface
    # ================================================================== #
    def run(self, prompt: str) -> str:
        if not prompt.strip():
            return "[error: empty prompt]"

        self.log.append(Message.user(prompt))

        try:
            assistant_msg = self.backend.complete(list(self.log))
        except LLMBackendError as exc:
            err = f"[agent backend‑error] {exc}"
            log.error(err)
            return err

        self.log.append(assistant_msg)
        return assistant_msg.content or ""

    def get_history(self) -> List[Message]:
        return list(self.log)

    def last_assistant(self) -> Optional[Message]:
        return self.log.last_assistant()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_system_prompt(self, text: str):
        if len(self.log) == 0 or self.log._messages[0].role is not Role.system:
            self.log.insert(0, Message.system(text))
        else:
            self.log._messages[0].content = text
            self.log._save_full_log()


__all__ = ["Agent"]

