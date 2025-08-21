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
        
        # Enhance prompt with blackboard context for certain agents
        enhanced_prompt = self._enhance_prompt_with_context(step, prompt)
        
        result = agent.run(enhanced_prompt)
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
    
    # ------------------------------------------------------------------
    def _enhance_prompt_with_context(self, step: Step, prompt: str) -> str:
        """
        Enhance the prompt with blackboard context for agents that need it.
        
        This ensures agents like CodeAgent get the full task context,
        not just "generate_code" from the previous agent.
        """
        # List of agents that need full context
        context_needing_labels = ["generate_code", "apply_files", "run_tests"]
        
        if step.label not in context_needing_labels:
            return prompt
        
        # Build context from blackboard
        import json
        context = {
            "immediate_instruction": prompt,
            "original_task": None,
            "planning_context": None,
            "recent_actions": []
        }
        
        # Get original task
        task_entries = self.bb.query_sync(label="task_description")
        if task_entries:
            context["original_task"] = task_entries[0].content
        
        # Get latest planning recommendation
        planning_entries = self.bb.query_sync(label="plan_next")
        if planning_entries:
            # Get the most recent planning entry
            latest_plan = planning_entries[-1]
            context["planning_context"] = latest_plan.content
        
        # Get recent actions for context
        all_entries = self.bb.entries()
        for entry in all_entries[-5:]:  # Last 5 entries
            if entry.label != "task_description":
                context["recent_actions"].append({
                    "step": entry.label,
                    "summary": entry.content[:200] if len(entry.content) > 200 else entry.content
                })
        
        # For CodeAgent specifically, provide rich context
        if step.label == "generate_code":
            enhanced = f"""CONTEXT:
Original Task: {context['original_task'] or 'Not specified'}

Planning Recommendation: {context['planning_context'] or 'Not available'}

Recent Actions: {json.dumps(context['recent_actions'], indent=2) if context['recent_actions'] else 'None'}

Instruction: {prompt}

IMPORTANT: Generate code that directly addresses the original task above, not generic templates."""
            return enhanced
        
        # For other agents, provide simpler context
        return json.dumps(context, indent=2)
