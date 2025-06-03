# runtime/step.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(slots=True)
class Step:
    label: str
    agent_key: str
    message: str | None = None
    on_success: Optional[str] = None
    on_fail: Optional[str] = None
    steps: Optional[List["Step"]] = field(default_factory=list)  # nested
