# agent/agent.py

"""
Unified Agent – no dict history, only Message objects.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.messages import Message, Role
from agent.settings import Settings
from agent.storage import ConversationLog
from agent.llm_backends import LLMBackend, LLMBackendError, get_backend

log = logging.getLogger(__name__)

class Agent:
    """
    Minimal chat-agent façade:
        • owns its own Settings copy
        • writes history via ConversationLog
        • delegates completions to pluggable LLM back-ends
    """

    def __init__(
        self, 
        *, 
        roles: List[str] | None = None, 
        overrides: Dict[str, Any] | None = None,
        name: Optional[str] = None,
        id: Optional[str] = None
    ):
        # —— Config ————————————————————————————————————————————
        self.cfg: Settings = Settings.resolve(overrides)
        
        # —— Identity ————————————————————————————————————————————
        self.name = name or self.__class__.__name__
        self.id = id or f"{self.name}-{uuid.uuid4().hex[:8]}"

        # —— Conversation log ————————————————————————————————————
        conv_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation = ConversationLog(
            run_name=conv_name,
            system_roles=roles or [],
            root_dir=self.cfg.paths.logs_dir,
            flush_every=self.cfg.logging.flush_every,
        )

        # —— Backend ————————————————————————————————————————————
        self._setup_backend()

    # ======================================================================
    # Public API
    # ======================================================================

    def run(self, user_text: str) -> str:
        self._append("user", user_text)
        reply = self.call_ai()
        self._append("assistant", reply)
        return reply

    def call_ai(self) -> str:
        if self.cfg.debug.mock_mode:
            return "[mock] This would be the assistant reply."

        try:
            assistant_msg = self.backend.complete(self._pydantic_msgs())
            return assistant_msg.content or ""
        except LLMBackendError as exc:
            log.error("Backend error: %s", exc)
            return f"[backend error] {exc}"

    def switch_provider(self, **provider_kw: Any) -> bool:
        """
        Dynamically switch to a different LLM provider / model.

        Keyword Args:
            type / provider : new provider name (e.g. ``openai``, ``google`` …)
            model           : model identifier for the new backend
            Any additional backend-specific keyword is passed through and, if
            an attribute with that name exists on either ``self.cfg.llm`` or
            the legacy ``self.cfg.provider`` namespace, it is updated.

        Returns
        -------
        bool
            ``True`` when the backend could be re-initialised, otherwise
            ``False`` (the previous backend will remain active).
        """
        try:
            # Normalise common aliases -------------------------------------
            if "type" in provider_kw and "provider" not in provider_kw:
                provider_kw["provider"] = provider_kw["type"]

            # Mutate the active Settings instance --------------------------
            for key, value in provider_kw.items():
                # Primary, preferred location (new Settings schema)
                if hasattr(self.cfg.llm, key):
                    setattr(self.cfg.llm, key, value)
                    continue

                # Fallback to legacy place – kept for backward compatibility
                if hasattr(self.cfg.provider, key):
                    setattr(self.cfg.provider, key, value)

            # Re-establish backend with new configuration ------------------
            self._setup_backend()
            return True

        except Exception as exc:  # pylint: disable=broad-except
            log.exception("Could not switch provider: %s", exc)
            return False

    # ======================================================================
    # Internals
    # ======================================================================

    def _setup_backend(self):
        try:
            # Handle both Pydantic v1 and v2 model serialization
            try:
                provider_dict = self.cfg.provider.model_dump(mode="python")
            except AttributeError:
                # Fallback for Pydantic v1
                provider_dict = self.cfg.provider.dict()
            
            self.backend: LLMBackend = get_backend(provider_dict)
            
            if self.cfg.debug.verbose:
                log.info("[agent] Using backend=%s model=%s",
                         self.backend.__class__.__name__,
                         getattr(self.backend, "model_name", "n/a"))
        except ImportError as exc:
            log.exception("Backend module not found: %s. Falling back to stub backend.", exc)
            self.cfg.debug.mock_mode = True
            self.backend = get_backend({"type": "stub"})
        except LLMBackendError as exc:
            log.exception("Backend initialization error: %s. Falling back to stub backend.", exc)
            self.cfg.debug.mock_mode = True
            self.backend = get_backend({"type": "stub"})
        except Exception as exc:  # pylint: disable=broad-except
            log.exception("Unexpected error setting up backend: %s. Falling back to stub backend.", exc)
            self.cfg.debug.mock_mode = True
            self.backend = get_backend({"type": "stub"})

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #

    def _append(self, role: str, content: str):
        """Append a Message to ConversationLog."""
        self.conversation.append(Message(role=Role(role), content=content))

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    def _pydantic_msgs(self) -> List[Message]:
        """Return history as Message objects (ConversationLog already stores them)."""
        return list(self.conversation.messages)

    # Convenience accessor
    def get_config(self) -> Settings:
        return self.cfg

__all__ = ["Agent"]

