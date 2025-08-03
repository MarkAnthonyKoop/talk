# ~/sandbox/agent/storage.py
"""
Conversation storage layer.

Provides `ConversationLog`, a thin wrapper around `MessageList` that can persist
each turn to a JSONL file.  Keeps disk I/O isolated from agent logic.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from agent.messages import Message, MessageList
from agent.settings import settings

log = logging.getLogger(__name__)


class ConversationLog:
    """
    Parameters
    ----------
    path : str | Path | None
        File destination for JSONL persistence.  If *None*, operates purely
        in‑memory.
    auto_save : bool
        If *True* (default) every `append()` immediately writes that message
        to disk.  Set to *False* when you want manual control (e.g. unit tests).
    """

    def __init__(self, path: Optional[str | Path] = None, *, auto_save: bool = True):
        self.path: Optional[Path] = Path(path).expanduser() if path else None
        self.auto_save = auto_save
        self._messages = MessageList()

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------ #
    # Public API mirroring MutableSequence where useful
    # ------------------------------------------------------------------ #

    def append(self, msg: Message):
        self._messages.append(msg)
        if self.auto_save:
            self._append_to_disk(msg)

    def insert(self, idx: int, msg: Message):
        self._messages.insert(idx, msg)
        self._save_full_log()  # safest way

    def extend(self, msgs: Iterable[Message]):
        for m in msgs:
            self.append(m)

    def clear(self):
        self._messages = MessageList()
        if self.path and self.path.exists():
            self.path.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def last_user(self) -> Optional[Message]:
        return self._messages.last_user()

    def last_assistant(self) -> Optional[Message]:
        return self._messages.last_assistant()

    # ------------------------------------------------------------------ #
    # Disk I/O helpers
    # ------------------------------------------------------------------ #

    def _load_from_disk(self):
        if not self.path or not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self._messages.append(Message.from_provider(json.loads(line)))
            log.debug(f"[storage] loaded {len(self)} rows from {self.path}")
        except Exception as exc:
            log.warning(f"[storage] failed to load log {self.path}: {exc}")
            # fall back to empty

    def _append_to_disk(self, msg: Message):
        if not self.path:
            return
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error(f"[storage] could not append message to {self.path}: {exc}")

    def _save_full_log(self):
        if not self.path:
            return
        try:
            with self.path.open("w", encoding="utf-8") as fh:
                for m in self._messages:
                    fh.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error(f"[storage] could not save full log to {self.path}: {exc}")

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    @classmethod
    def temp(cls) -> "ConversationLog":
        """
        Create an in‑memory log that still writes to a timestamped file inside
        `settings.paths.logs_dir` – useful for debugging transient chats.
        """
        p = settings.paths.logs_dir / f"conv_{os.getpid()}_{id(cls)}.jsonl"
        return cls(p)

    def to_provider_dicts(self) -> List[dict]:
        """Return history as list[dict] suitable for backend call."""
        return self._messages.to_dict_list()

    # ------------------------------------------------------------------ #
    # Introspection API
    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        """
        Returns a summary string about the current database/log:
        - number of messages
        - last user message
        - last assistant message
        - location of storage file (if any)
        """
        info = [
            f"ConversationLog: {len(self)} message(s)",
            f"Path: {self.path}" if self.path else "[in-memory only]",
        ]
        last_user = self.last_user()
        if last_user:
            info.append(f"Last user: {repr(str(last_user.content)[:80])}")
        else:
            info.append("Last user: [none]")
        last_assistant = self.last_assistant()
        if last_assistant:
            info.append(f"Last assistant: {repr(str(last_assistant.content)[:80])}")
        else:
            info.append("Last assistant: [none]")
        return "\n".join(info)

