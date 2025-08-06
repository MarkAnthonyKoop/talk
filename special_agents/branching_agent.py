#!/usr/bin/env python3
"""
BranchingAgent - Intelligent label selector for workflow branching.

This agent uses the LLM to interpret PlanningAgent recommendations and select
the appropriate Step label from the available steps in the plan. It modifies
its Step's on_success field to control workflow execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from agent.agent import Agent
from plan_runner.step import Step

log = logging.getLogger(__name__)


class BranchingAgent(Agent):
    """
    Intelligent branching agent that uses LLM to select next steps.
    
    This agent:
    1. Receives the PlanningAgent's recommendations
    2. Uses the LLM to interpret and select the best matching Step
    3. Modifies its own Step's on_success field to that label
    4. Returns the LLM's reasoning and selection
    """
    
    def __init__(self, 
                 step: Step,
                 plan: List[Step],
                 **kwargs):
        """
        Initialize the BranchingAgent.
        
        Args:
            step: Reference to this agent's own Step object (can modify on_success)
            plan: The complete List[Step] plan to choose from
            **kwargs: Additional arguments for the base Agent
        """
        # Define roles for the LLM
        roles = [
            "You are a workflow branching controller in a multi-agent orchestration system.",
            "Your job is to select the next Step label based on strategic recommendations from the PlanningAgent.",
            "You have access to the complete execution plan and must choose the most appropriate next step.",
            "",
            "ARCHITECTURAL CONTEXT:",
            "- The system executes Steps sequentially based on labels",
            "- Each Step has a label and an associated agent",
            "- You control flow by selecting which Step executes next",
            "- Setting next step to 'complete' or None ends the workflow",
            "",
            "IMPORTANT RULES:",
            "1. You must select exactly ONE label from the available steps",
            "2. If the planning recommendation includes a 'next_action' field, prefer that",
            "3. If the task is complete, select 'complete' to end the workflow",
            "4. Always explain your reasoning for the selection",
            "5. Your output must include a clear 'SELECTED:' line with the chosen label"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        self.step = step  # Reference to our own Step (can modify on_success)
        self.plan = plan  # Complete plan with all available Steps
        self.available_labels = self._extract_labels(plan)
        self.scratch_dir = Path.cwd() / ".talk_scratch"
    
    def _extract_labels(self, plan: List[Step]) -> List[str]:
        """Extract all available Step labels from the plan."""
        labels = []
        for step in plan:
            if step.label and step.label != self.step.label:
                labels.append(step.label)
        return labels
    
    def run(self, prompt: str) -> str:
        """
        Select the next Step label based on planning recommendation using the LLM.
        
        Args:
            prompt: Contains PlanningAgent's recommendation
            
        Returns:
            LLM completion with selection and reasoning
        """
        try:
            # Try to load planning data from scratch if available
            planning_data = self._load_planning_data(prompt)
            
            # Build the selection prompt
            selection_prompt = self._build_selection_prompt(planning_data, prompt)
            
            # Get LLM to make the selection
            self._append("user", selection_prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Extract the selected label from the completion
            selected_label = self._extract_selected_label(completion)
            
            # Apply the selection to our Step
            if selected_label:
                if selected_label.lower() == "complete" or selected_label.lower() == "none":
                    self.step.on_success = None
                    log.info("Workflow marked as complete")
                elif selected_label in self.available_labels:
                    self.step.on_success = selected_label
                    log.info(f"Selected next step: {selected_label}")
                else:
                    # Try to find closest match
                    closest = self._find_closest_match(selected_label)
                    if closest:
                        self.step.on_success = closest
                        log.warning(f"Using closest match '{closest}' for '{selected_label}'")
                    else:
                        # Fallback
                        self.step.on_success = "generate_code" if "generate_code" in self.available_labels else self.available_labels[0]
                        log.warning(f"No match found, using fallback: {self.step.on_success}")
            else:
                # Could not extract selection, use fallback
                self.step.on_success = "generate_code" if "generate_code" in self.available_labels else self.available_labels[0]
                log.warning(f"Could not extract selection, using fallback: {self.step.on_success}")
            
            return completion
            
        except Exception as e:
            log.error(f"BranchingAgent error: {e}")
            # Return error message and set safe fallback
            self.step.on_success = "error_recovery" if "error_recovery" in self.available_labels else None
            return f"Error in branch selection: {e}\nFalling back to: {self.step.on_success}"
    
    def _load_planning_data(self, prompt: str) -> Optional[Dict]:
        """Try to load planning data from scratch or parse from prompt."""
        # First try to parse prompt as JSON
        try:
            return json.loads(prompt)
        except json.JSONDecodeError:
            pass
        
        # Try to load from scratch file
        try:
            planning_file = self.scratch_dir / "latest_planning.json"
            if planning_file.exists():
                with open(planning_file) as f:
                    return json.load(f)
        except Exception as e:
            log.debug(f"Could not load planning from scratch: {e}")
        
        return None
    
    def _build_selection_prompt(self, planning_data: Optional[Dict], raw_prompt: str) -> str:
        """Build the prompt for label selection."""
        # Extract key information from planning data
        if planning_data:
            recommendation = planning_data.get("recommendation", "")
            next_action = planning_data.get("next_action", "")
            analysis = planning_data.get("analysis", {})
            todo_hierarchy = planning_data.get("todo_hierarchy", "")
        else:
            recommendation = raw_prompt
            next_action = ""
            analysis = {}
            todo_hierarchy = ""
        
        # Build list of available steps with descriptions
        step_descriptions = []
        for step in self.plan:
            if step.label and step.label != self.step.label:
                agent = step.agent_key or "terminal"
                if step.label == "plan_next":
                    desc = "Return to planning for next strategic decision"
                elif step.label == "generate_code":
                    desc = "Generate code implementation"
                elif step.label == "apply_files":
                    desc = "Apply code changes to files"
                elif step.label == "run_tests":
                    desc = "Execute tests to validate"
                elif step.label == "research":
                    desc = "Search for information"
                elif step.label == "error_recovery":
                    desc = "Handle errors and retry"
                elif step.label == "complete":
                    desc = "Mark workflow as complete"
                else:
                    desc = f"Execute {step.label}"
                step_descriptions.append(f"- {step.label}: {desc} (agent: {agent})")
        
        prompt_parts = ["PLANNING RECOMMENDATION:"]
        
        if todo_hierarchy:
            prompt_parts.append(f"\nTodo Hierarchy:\n{todo_hierarchy}")
        
        if analysis:
            prompt_parts.append(f"\nAnalysis: {json.dumps(analysis, indent=2)}")
        
        if next_action:
            prompt_parts.append(f"\nRecommended Action: {next_action}")
        
        if recommendation and recommendation != raw_prompt:
            prompt_parts.append(f"\nReasoning: {recommendation}")
        
        prompt_parts.append(f"\n\nAVAILABLE NEXT STEPS:\n{chr(10).join(step_descriptions)}")
        
        prompt_parts.append("""\n\nBased on the planning recommendation above, select the most appropriate next step.

Consider:
1. What action does the PlanningAgent recommend?
2. Which available step best matches that recommendation?
3. Is the task complete or should work continue?

Respond with your selection and reasoning. Your response MUST include a line starting with "SELECTED:" followed by the exact label.

Example format:
The PlanningAgent recommends [action] because [reasoning].
Looking at the available steps, [label] best matches this recommendation.
This will [what it will accomplish].

SELECTED: [exact_label]

Now make your selection:""")
        
        return "\n".join(prompt_parts)
    
    def _extract_selected_label(self, completion: str) -> Optional[str]:
        """Extract the selected label from LLM completion."""
        lines = completion.strip().split('\n')
        
        # Look for SELECTED: line
        for line in lines:
            if line.strip().upper().startswith("SELECTED:"):
                label = line.split(":", 1)[1].strip()
                # Remove quotes and extra whitespace
                label = label.strip('"').strip("'").strip()
                return label
        
        # Fallback: try to find a label mentioned in context
        completion_lower = completion.lower()
        
        # Check for "complete" or "done" signals
        if "task is complete" in completion_lower or "workflow complete" in completion_lower:
            return "complete"
        
        # Look for exact label mentions
        for step in self.plan:
            if step.label and step.label in completion:
                return step.label
        
        # Look for action keywords
        if "generate" in completion_lower and "code" in completion_lower:
            return "generate_code"
        elif "apply" in completion_lower and "file" in completion_lower:
            return "apply_files"
        elif "test" in completion_lower or "run" in completion_lower:
            return "run_tests"
        elif "research" in completion_lower or "search" in completion_lower:
            return "research"
        elif "error" in completion_lower or "recover" in completion_lower:
            return "error_recovery"
        
        return None
    
    def _find_closest_match(self, target: str) -> Optional[str]:
        """Find the closest matching label."""
        target_lower = target.lower()
        
        # Exact match (case insensitive)
        for label in self.available_labels:
            if label.lower() == target_lower:
                return label
        
        # Partial match
        for label in self.available_labels:
            if target_lower in label.lower() or label.lower() in target_lower:
                return label
        
        # Keyword matching
        if "code" in target_lower:
            for label in self.available_labels:
                if "code" in label.lower():
                    return label
        
        if "file" in target_lower:
            for label in self.available_labels:
                if "file" in label.lower():
                    return label
                    
        if "test" in target_lower:
            for label in self.available_labels:
                if "test" in label.lower():
                    return label
        
        return None