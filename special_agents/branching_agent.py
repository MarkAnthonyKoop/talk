# special_agents/branching_agent.py

"""
BranchingAgent
==============

Tiny control-flow helper. It owns a **reference** to its Step object, so it
can rewrite `step.on_success` in place and make PlanRunner loop or stop
without PlanRunner knowing any new tricks.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional

from agent.agent import Agent  # ✅ Use golden Agent, not aliased CoreAgent
from runtime.step import Step

log = logging.getLogger(__name__)

class BranchingAgent(Agent):  # ✅ Inherit from golden Agent
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

    # ✅ Golden interface: run(user_text: str) -> str
    def run(self, user_text: str) -> str:
        """
        Control flow logic for looping or termination.
        
        Parameters
        ----------
        user_text : str
            Input prompt (typically test results from previous step)
            
        Returns
        -------
        str
            Status message indicating loop state
        """
        self.count += 1
        
        # Check if tests passed (simple heuristic) OR max iterations reached
        tests_passed = "EXIT 0" in user_text
        max_reached = self.count >= self.max_iters
        
        if tests_passed or max_reached:
            self.step.on_success = None  # stop looping
            msg = f"[exit loop: tests_passed={tests_passed}, count={self.count}]"
        else:
            self.step.on_success = self.loop_target  # keep looping
            msg = f"[loop {self.count}/{self.max_iters}]"
            
        log.info(msg)
        return msg

__all__ = ["BranchingAgent"]
