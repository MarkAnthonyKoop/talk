# runtime/context.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from runtime.step import Step
from runtime.blackboard import Blackboard

@dataclass(slots=True)
class ExecContext:
    """
    Metadata PlanRunner can hand to an agent.

    step        : the mutable Step currently executing
    blackboard  : shared Blackboard instance
    plan        : the full ordered list of Step objects
    runner_data : scratch-pad dict PlanRunner (or agents) may use
    """

    step: Step
    blackboard: Blackboard
    plan: Sequence[Step]
    runner_data: Dict[str, Any] = field(default_factory=dict)

