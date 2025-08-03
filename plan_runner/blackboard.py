# plan_runner/blackboard.py

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

def _uuid() -> str:  # helper
    return uuid.uuid4().hex

@dataclass(slots=True)
class BlackboardEntry:
    """
    A single record on the blackboard.

    section : logical namespace (e.g. "tasks", "artifacts")
    role    : usually "user", "assistant", or domain-specific tag
    author  : agent id that wrote the entry (provenance)
    label   : plan-step label that produced the entry
    meta    : arbitrary small dict for extra flags / scores
    """

    id: str = field(default_factory=_uuid)
    section: str = "default"
    role: str = "assistant"
    author: str = "system"
    label: str = "unlabelled"
    content: Any = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

class Blackboard:
    """
    Async-safe, keyed blackboard with O(1) lookup.

    *No event log yet*  — we can layer that on later.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, BlackboardEntry] = {}
        self._lock = asyncio.Lock()

    async def add(self, entry: BlackboardEntry) -> None:
        async with self._lock:
            self._entries[entry.id] = entry

    async def update(self, entry_id: str, **patch) -> None:
        async with self._lock:
            ent = self._entries[entry_id]
            for k, v in patch.items():
                setattr(ent, k, v)

    async def get(self, entry_id: str) -> Optional[BlackboardEntry]:
        return self._entries.get(entry_id)

    async def query(
        self,
        *,
        section: str | None = None,
        role: str | None = None,
        author: str | None = None,
        label: str | None = None,
    ) -> List[BlackboardEntry]:
        return [
            e
            for e in self._entries.values()
            if (section is None or e.section == section)
            and (role is None or e.role == role)
            and (author is None or e.author == author)
            and (label is None or e.label == label)
        ]

    # Synchronous convenience methods for non-async contexts
    def add_sync(self, label: str, content: Any, **kwargs) -> None:
        """Synchronous convenience method to add an entry."""
        entry = BlackboardEntry(
            label=label,
            content=content,
            **kwargs
        )
        self._entries[entry.id] = entry
    
    def add(self, label: str, content: Any, **kwargs) -> None:
        """Alias for add_sync for backward compatibility."""
        self.add_sync(label, content, **kwargs)
    
    def entries(self) -> List[BlackboardEntry]:
        """Return all entries synchronously."""
        return list(self._entries.values())
    
    def query_sync(
        self,
        *,
        section: str | None = None,
        role: str | None = None,
        author: str | None = None,
        label: str | None = None,
    ) -> List[BlackboardEntry]:
        """Synchronous version of query."""
        return [
            e
            for e in self._entries.values()
            if (section is None or e.section == section)
            and (role is None or e.role == role)
            and (author is None or e.author == author)
            and (label is None or e.label == label)
        ]

    # simple helper for debugging
    def __repr__(self) -> str:  # noqa: D401
        return f"<Blackboard {len(self._entries)} entries>"
