"""
Conversation storage layer.

`ConversationLog` wraps an in-memory MessageList and persists each turn to
JSONL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from agent.messages import Message, MessageList

log = logging.getLogger(__name__)

class ConversationLog:
    def __init__(
        self,
        *,
        run_name: str,
        system_roles: List[str],
        root_dir: Path | str,
        flush_every: int = 1,
    ):
        self.run_name = run_name
        self.system_roles = system_roles
        self.flush_every = max(int(flush_every), 1)
        root = Path(root_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{run_name}.jsonl"
        self._messages = MessageList()
        self._since_flush = 0
        if self.path.exists():
            self._load_from_disk()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    @property
    def messages(self) -> MessageList:
        return self._messages

    def append(self, msg: Message):
        self._messages.append(msg)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._append_to_disk(msg)
            self._since_flush = 0

    def extend(self, msgs: Iterable[Message]):
        for m in msgs:
            self.append(m)

    def insert(self, idx: int, msg: Message):
        self._messages.insert(idx, msg)
        self._save_full_log()

    def clear(self):
        self._messages = MessageList()
        if self.path.exists():
            self.path.unlink(missing_ok=True)

    def last_user(self):
        return self._messages.last_user()

    def last_assistant(self):
        return self._messages.last_assistant()

    def _load_from_disk(self):
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        data = json.loads(line)
                        self._messages.append(Message.from_provider(data))
            log.debug("[storage] loaded %d rows from %s", len(self), self.path)
        except Exception as exc:
            log.warning("[storage] failed to load %s: %s", self.path, exc)

    def _append_to_disk(self, msg: Message):
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error("[storage] could not append message to %s: %s", self.path, exc)

    def _save_full_log(self):
        try:
            with self.path.open("w", encoding="utf-8") as fh:
                for m in self._messages:
                    fh.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error("[storage] rewrite failed %s: %s", self.path, exc)
