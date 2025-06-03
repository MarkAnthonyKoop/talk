# special_agents/branching_agent.py

"""
BranchingAgent
==============

Tiny control-flow helper.  It owns a **reference** to its Step object, so it
can rewrite `step.on_success` in place and make PlanRunner loop or stop
without PlanRunner knowing any new tricks.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from agent.agent import Agent as CoreAgent
from runtime.step import Step

log = logging.getLogger(__name__)

class BranchingAgent(CoreAgent):
    """
    Parameters
    ----------
    step : Step
        *Live* step object this agent is responsible for.
    loop_target : str
        Label to jump back to while under `max_iters`.
    max_iters : int
        After this many loops the agent sets `step.on_success = None` to stop.
    """

    def __init__(
        self,
        *,
        step: Step,
        loop_target: str,
        max_iters: int = 5,
        overrides: Optional[Dict] = None,
    ):
        super().__init__(overrides=overrides)
        self.step = step
        self.loop_target = loop_target
        self.max_iters = max_iters
        self.count = 0

    # ------------------------------------------------------------------ #

    def run(self, prompt: str) -> str:  # noqa: D401
        self.count += 1
        if self.count < self.max_iters:
            self.step.on_success = self.loop_target   # keep looping
            msg = f"[loop {self.count}/{self.max_iters-1}]"
        else:
            self.step.on_success = None               # fall through / end
            msg = "[exit loop]"
        log.info(msg)
        return msg

__all__ = ["BranchingAgent"]
