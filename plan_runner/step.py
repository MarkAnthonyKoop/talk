# plan_runner/step.py
#
# Changes from Original & Architectural Theory:
# The original `Step` class was a simple Python class. This has been converted
# to a `dataclass` for convenience and clarity.
#
# The architectural changes are significant:
# 1. Addition of `steps: List["Step"]`: This field allows a step to contain a
#    list of its own sub-steps that should be executed serially. This enables
#    arbitrarily deep, recursive plan execution, all handled by the PlanRunner's
#    unified execution logic.
# 2. Addition of `parallel_steps: List["Step"]`: This field allows a step to
#    contain a list of sub-steps that should be executed concurrently. This is
#    the hook the `PlanRunner` uses to invoke its thread-based parallelism.
# 3. Auto-Labeling: The `__post_init__` method now automatically assigns a
#    unique label (e.g., `_step0`) if one isn't provided. This ensures that
#    every step can be referenced, which is vital for the `on_success` logic.
# 4. The previous validation error for having both `steps` and `parallel_steps`
#    was removed, as per the final requirement to process both if present.
#

"""
Defines the Step class, the fundamental building block of an execution plan.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional

# A global counter to generate unique labels for anonymous steps.
_anon = count()

@dataclass(slots=True)
class Step:
    """
    Represents a single node in an execution plan.

    A Step can define an action (via `agent_key`), a list of serial sub-steps
    (`steps`), and a list of parallel sub-steps (`parallel_steps`). The PlanRunner
    will execute these in a defined order: agent -> parallel -> serial.
    """
    # A human-readable label for this step, used for logging and `on_success` jumps.
    label: Optional[str] = None

    # The key for the agent that should execute this step's primary action.
    agent_key: str = ""
    
    # If set, the plan will jump to the step with this label upon successful completion.
    on_success: Optional[str] = None
    
    # If set, the plan will jump to the step with this label when an exception is raised.
    on_fail: Optional[str] = None
    
    # A list of sub-steps to be executed serially after this step's action.
    steps: List["Step"] = field(default_factory=list)
    
    # A list of sub-steps to be executed in parallel after this step's action.
    parallel_steps: List["Step"] = field(default_factory=list)

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        # Auto-generate a label if one wasn't provided, ensuring all steps are referenceable.
        if not self.label:
            # Use module-level counter – single underscore avoids name-mangling issues.
            self.label = f"_step{next(_anon)}"
