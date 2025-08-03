# special_agents/shell_agent.py

from __future__ import annotations
from typing import Dict, List, Optional

from agent.agent import Agent  # ✅ Use golden Agent, not CoreAgent
from runtime.blackboard import Blackboard
from runtime.parallel import run_sync_in_thread
from runtime.plan_runner import PlanRunner
from runtime.step import Step
from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent

class ShellAgent(Agent):  # ✅ Inherit from golden Agent
    """
    High-level orchestrator that repeatedly:
    1. asks LLM for a patch
    2. applies it
    3. runs tests
    until tests pass or `max_cycles` is reached.
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

        # Plan definition
        self.steps: List[Step] = [
            Step("code", agent_key="code", on_success="apply"),
            Step("apply", agent_key="apply", on_success="test"),
            Step("test", agent_key="test", on_success="loop"),
            Step("loop", agent_key="loop"),  # BranchingAgent will mutate on_success
        ]

        self.agents: Dict[str, Agent] = {  # ✅ Use golden Agent type
            "code": CodeAgent(overrides=overrides),
            "apply": FileAgent(overrides=overrides),
            "test": TestAgent(cmd=test_cmd, overrides=overrides),
            "loop": BranchingAgent(
                step=self.steps[3],
                loop_target="code",
                max_iters=max_cycles,
                overrides=overrides,
            ),
        }

    # ✅ Golden interface: run(user_text: str) -> str
    def run(self, user_text: str) -> str:
        """
        Execute the code-apply-test plan and return the last test report.
        Maintains conversation with the outside world through this interface.
        """
        bb = Blackboard()
        runner = PlanRunner(self.steps, self.agents, bb)

        run_sync_in_thread(lambda: runner.run(user_text))

        # Last "test" entry is authoritative
        tests = [e.content for e in bb.entries() if e.label == "test"]
        return tests[-1] if tests else "<no test run>"
