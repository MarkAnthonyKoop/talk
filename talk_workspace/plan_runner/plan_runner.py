# plan_runner/plan_runner.py
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .blackboard import Blackboard
from .step import Step

log = logging.getLogger(__name__)


class PlanRunner:
    """
    Purely synchronous execution engine with optional fork/join when a
    wrapper step defines `parallel_steps`.
    """

    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        steps: List[Step],
        agents: Dict[str, "BaseAgent"],
        blackboard: Blackboard,
    ):
        self.order = steps
        self.index = {s.label: s for s in steps}  # labels are now guaranteed
        self.agents = agents
        self.bb = blackboard

    # ──────────────────────────────────────────────────────────────
    def run(self, user_prompt: str) -> str:
        current = self.order[0]
        prev_out = user_prompt

        while current:
            if current.parallel_steps:
                prev_out = self._run_parallel(current, prev_out)
            else:
                prev_out = self._run_single(current, prev_out)

            # execute serial children, if any
            for child in current.steps:
                prev_out = self._run_single(child, prev_out)

            current = self._next_step(current)

        return prev_out

    # -------- helpers --------------------------------------------------
    def _run_single(self, step: Step, prompt: str) -> str:
        log.debug("→ %s", step.label)
        agent = self.agents[step.agent_key]
        result = agent.run(prompt)
        self.bb.add(step.label, result)
        return result

    # ------------------------------------------------------------------
    def _run_parallel(self, wrapper: Step, prompt: str) -> str:
        log.debug("⇉ fork %s", wrapper.label)

        def _worker(st: Step) -> str:
            return self._run_single(st, prompt)

        last_out: Optional[str] = None
        with ThreadPoolExecutor(max_workers=len(wrapper.parallel_steps)) as pool:
            fut_map = {
                pool.submit(_worker, st): st.label for st in wrapper.parallel_steps
            }
            for fut in as_completed(fut_map):
                last_out = fut.result()

        # record wrapper result for downstream logic
        self.bb.add(wrapper.label, last_out)
        log.debug("⇇ join %s", wrapper.label)
        return last_out or ""

    # ------------------------------------------------------------------
    def _next_step(self, step: Step) -> Optional[Step]:
        if step.on_success:
            return self.index.get(step.on_success)

        # fall back to linear ordering
        idx = self.order.index(step)
        return self.order[idx + 1] if idx + 1 < len(self.order) else None
