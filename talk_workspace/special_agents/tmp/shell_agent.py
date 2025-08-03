# special_agents/shell_agent.py

"""
ShellAgent
==========

High-level orchestrator that runs an iterative
    code → apply → test
cycle until the tests pass or `max_cycles` is hit.

It wires its sub-agents with explicit constructor injections:
    * each sub-agent that needs a `Step` gets the live object
    * no base-class or PlanRunner changes required
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from agent.agent import Agent as CoreAgent
from runtime.blackboard import Blackboard
from runtime.plan_runner import PlanRunner
from runtime.step import Step

from .code_agent import CodeAgent
from .file_agent import FileAgent
from .test_agent import TestAgent
from .branching_agent import BranchingAgent

log = logging.getLogger(__name__)

class ShellAgent(CoreAgent):
    """
    Orchestrates iterative code modifications until the test suite passes or
    `max_cycles` is reached.
    """

    def __init__(
        self,
        *,
        max_cycles: int = 5,
        test_cmd: str = "pytest -q",
        overrides: Optional[Dict] = None,
    ):
        super().__init__(overrides=overrides)
        self.max_cycles = max_cycles
        self.test_cmd = test_cmd

        # -------------------------------------------------- create plan steps
        self.steps: List[Step] = [
            Step(label="code",  agent_key="code"),
            Step(label="apply", agent_key="apply"),
            Step(label="test",  agent_key="test"),
            Step(label="loop",  agent_key="loop"),
        ]

        # manual wiring of success chain
        self.steps[0].on_success = "apply"
        self.steps[1].on_success = "test"
        self.steps[2].on_success = "loop"

        # self.steps[3] (loop) gets rewired by BranchingAgent at runtime
        # ---------------------------------------------- build sub-agents map
        self.agents: Dict[str, CoreAgent] = {
            "code":  CodeAgent(),                            # needs nothing extra
            "apply": FileAgent(),                            # ·
            "test":  TestAgent(cmd=test_cmd),                # ·
            "loop":  BranchingAgent(
                step=self.steps[3],
                loop_target="code",
                max_iters=max_cycles,
            ),
        }

    # ------------------------------------------------------------------ #

    def run(self, prompt: str) -> str:  # noqa: D401 — public API
        """
        Runs the full plan once.  BranchingAgent internally loops the plan
        until either the tests pass (exit code 0) or `max_cycles` is hit.
        """
        bb = Blackboard()
        runner = PlanRunner(self.steps, self.agents, bb)
        asyncio.run(runner.run(prompt))

        # return the latest test output (if any) -------------------------
        test_logs = [
            e.content for e in bb._entries.values() if e.label == "test"   # pylint: disable=protected-access
        ]
        return test_logs[-1] if test_logs else "<no test run>"

__all__ = ["ShellAgent"]

