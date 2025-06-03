# runtime/plan_runner.py

from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence

from runtime.blackboard import Blackboard, BlackboardEntry
from runtime.step import Step

class PlanRunner:
    """
    Executes a list-of-Step workflow against a dict of agents.

    Agents must expose:  `async def run(prompt:str) -> str`
                         and have an `.id` attribute for provenance.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        agents: Dict[str, "Agent"],
        blackboard: Blackboard,
    ) -> None:
        self.index = {s.label: s for s in steps}
        self.order = list(self.index)
        self.agents = agents
        self.bb = blackboard

    # ------------------------------------------------------------------ #

    async def run(self, user_input: str) -> str:
        current = self.order[0]
        prev_output = user_input
        while current:
            step = self.index[current]
            agent = self.agents[step.agent_key]
            prompt = step.message or prev_output

            # run agent (await even if underlying impl is sync by design)
            if asyncio.iscoroutinefunction(agent.run):
                result = await agent.run(prompt)
            else:
                result = agent.run(prompt)  # type: ignore[arg-type]

            # write to blackboard
            await self.bb.add(
                BlackboardEntry(
                    section="conversation",
                    role="assistant",
                    author=agent.id,
                    label=step.label,
                    content=result,
                )
            )

            prev_output = result

            # pick next label
            current = step.on_success or self._next_in_sequence(step.label)

        return prev_output

    # ------------------------------------------------------------------ #

    def _next_in_sequence(self, lbl: str) -> str | None:
        i = self.order.index(lbl)
        return self.order[i + 1] if i + 1 < len(self.order) else None
