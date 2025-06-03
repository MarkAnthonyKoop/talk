# ~/sandbox/agent/messages.py

"""
Typed message primitives shared by the whole framework.

This replaces the loose `dict` structures used in Implementations A & B with
Pydantic models that guarantee each message is valid *before* it reaches a
backend or the log.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    Json,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Enum for allowed roles
# --------------------------------------------------------------------------- #

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"

# --------------------------------------------------------------------------- #
# Core message model
# --------------------------------------------------------------------------- #

class Message(BaseModel):
    """
    A single chat–turn.

    Attributes
    ----------
    role : Role
        Who said it.
    content : str | None
        Natural–language text; can be None when the assistant only emitted a
        tool call (OpenAI spec).
    name : str | None
        Optional speaker name (if you need more granularity than `role`).
    tool_calls : Json | None
        Raw JSON from the provider when the assistant requests functions/tools.
    tool_call_id : str | None
        Correlates a returned tool result to the originating call.
    """

    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[Json] = Field(default=None, alias="tool_calls")
    tool_call_id: Optional[str] = Field(default=None, alias="tool_call_id")

    # ------------------------------------------------------------- #
    # Validators
    # ------------------------------------------------------------- #

    @field_validator("content", mode="after")
    def _strip_content(cls, v):
        # Normalise empty strings to None
        if v is not None and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def _assistant_content_or_tool_calls(self):
        """
        The OpenAI spec allows assistant messages with *either*
        textual content OR tool_calls but not neither.
        """
        if self.role is Role.assistant:
            if self.content is None and self.tool_calls is None:
                raise ValueError(
                    "assistant message must have content OR tool_calls"
                )
        return self

    # ------------------------------------------------------------- #
    # Convenience helpers
    # ------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Return OpenAI–style dict (drop None values)."""
        result = {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
        }
        # Remove keys whose values are None; OpenAI rejects them.
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_provider(cls, data: Dict[str, Any]) -> "Message":
        """
        Factory to build a Message from a provider–supplied dict
        (e.g. `openai.types.chat.chat_completion_message.ChatCompletionMessage`).
        """
        return cls.parse_obj(data)  # Pydantic handles aliases

    # Allow `Message(...)` shortcuts like `Message.user("hi")`
    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.system, content=text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.user, content=text)

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role=Role.assistant, content=text)

# --------------------------------------------------------------------------- #
# MessageList – preserves order & adds helpers
# --------------------------------------------------------------------------- #

class MessageList(MutableSequence):
    """
    A simple thin wrapper around `list[Message]` that offers:
      * type safety,
      * convenience filters,
      * direct `.to_dict_list()` for backend calls.
    """

    def __init__(self, init: Optional[List[Message | Dict[str, Any]]] = None):
        self._messages: List[Message] = []
        if init:
            for m in init:
                self.append(m)

    # ------------------------------------------------------------------ #
    # MutableSequence protocol implementation
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx):
        return self._messages[idx]

    def __setitem__(self, idx, value):
        self._messages[idx] = self._coerce(value)

    def __delitem__(self, idx):
        del self._messages[idx]

    def __len__(self):
        return len(self._messages)

    def insert(self, idx: int, value):
        self._messages.insert(idx, self._coerce(value))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def append(self, value):
        self._messages.append(self._coerce(value))

    def _coerce(self, value: Message | Dict[str, Any]) -> Message:
        if isinstance(value, Message):
            return value
        if isinstance(value, dict):
            return Message.from_provider(value)
        raise TypeError("MessageList only accepts Message objects or dicts.")

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return list ready for provider API calls."""
        return [m.to_dict() for m in self._messages]

    # Quick filters

    def last_user(self) -> Optional[Message]:
        for msg in reversed(self._messages):
            if msg.role is Role.user:
                return msg
        return None

    def last_assistant(self) -> Optional[Message]:
        for msg in reversed(self._messages):
            if msg.role is Role.assistant:
                return msg
        return None

    # Representation for debugging

    def __repr__(self) -> str:
        preview = ", ".join(f"{m.role.value}:{(m.content or '')[:20]!r}" for m in self._messages[-3:])
        return f"<MessageList len={len(self)} [{preview}]>"
