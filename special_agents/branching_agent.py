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
    1. Receives the PlanningAgent's plain English recommendations
    2. Uses the LLM to interpret and select the best matching Step
    3. Modifies its own Step's on_success field to that label
    4. Returns the selected action
    """
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Interpret recommendations and select next workflow step"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest branching is needed."""
        return ["select", "choose", "branch", "decide", "next", "workflow"]
    
    def __init__(self, 
                 step: Step,
                 plan: List[Step],
                 agents: Optional[Dict[str, Agent]] = None,
                 **kwargs):
        """
        Initialize the BranchingAgent.
        
        Args:
            step: Reference to this agent's own Step object (can modify on_success)
            plan: The complete List[Step] plan to choose from
            agents: Optional dict of agent instances for getting descriptions
            **kwargs: Additional arguments for the base Agent
        """
        self.step = step  # Reference to our own Step (can modify on_success)
        self.plan = plan  # Complete plan with all available Steps
        self.agents = agents or {}  # Agent instances for descriptions
        self.available_labels = self._extract_labels(plan)
        self.loop_count = 0  # Track loops to prevent infinite recursion
        self.recent_selections = []  # Track recent selections
        
        # Build available steps descriptions from actual agents
        step_descriptions = self._build_step_descriptions(plan, agents)
        
        # Define roles for the LLM
        roles = [
            "You are the workflow controller. You decide which step to execute next.",
            "Your job: Interpret recommendations and select next workflow step",
            "",
            "AVAILABLE STEPS:",
            step_descriptions,
            "",
            "RULES:",
            "1. Select the step that best matches the planner's recommendation",
            "2. If recommendation says 'complete' or 'done', select 'complete'",
            "3. If stuck in a loop (same step 3+ times), try something different",
            "4. When in doubt, prefer moving forward over planning again",
            "",
            "You receive plain English recommendations like:",
            "- 'We need to generate code for the hello function'  → select: generate_code",
            "- 'The code is ready, let's save it to a file'  → select: apply_files",
            "- 'Testing shows an error, we should fix it'  → select: error_recovery or generate_code",
            "- 'Everything is complete'  → select: complete",
            "",
            "Respond with just the step label to execute next."
        ]
        
        super().__init__(roles=roles, **kwargs)
        self.scratch_dir = Path.cwd() / ".talk_scratch"
    
    def _extract_labels(self, plan: List[Step]) -> List[str]:
        """Extract all available Step labels from the plan."""
        labels = []
        for step in plan:
            if step.label and step.label != self.step.label:
                labels.append(step.label)
        return labels
    
    def _build_step_descriptions(self, plan: List[Step], agents: Optional[Dict[str, Agent]]) -> str:
        """Build descriptions from actual agents in the plan."""
        descriptions = []
        
        for step in plan:
            if step.label and step.label != self.step.label:
                # Get description from agent if available
                if agents and step.agent_key and step.agent_key in agents:
                    agent = agents[step.agent_key]
                    if hasattr(agent, 'brief_description'):
                        desc = agent.brief_description
                    else:
                        desc = f"Execute {step.label}"
                # Use predefined descriptions for common steps
                elif step.label == "plan_next":
                    desc = "Return to planning for next strategic decision"
                elif step.label == "generate_code":
                    desc = "Generate code implementation"
                elif step.label == "apply_files":
                    desc = "Save code to files"
                elif step.label == "run_tests":
                    desc = "Execute tests to validate"
                elif step.label == "research":
                    desc = "Search for information"
                elif step.label == "error_recovery":
                    desc = "Handle errors and retry"
                elif step.label == "complete":
                    desc = "End workflow when task is complete"
                else:
                    desc = f"Execute {step.label}"
                
                descriptions.append(f"- {step.label}: {desc}")
        
        return "\n".join(descriptions)
    
    def run(self, prompt: str) -> str:
        """
        Select the next Step label based on plain English planning recommendation.
        
        Args:
            prompt: Plain English recommendation from PlanningAgent
            
        Returns:
            Selected step label
        """
        try:
            self.loop_count += 1
            
            # Build the selection prompt
            selection_prompt = self._build_selection_prompt(prompt)
            
            # Get LLM to make the selection
            self._append("user", selection_prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Extract the selected label (should be just the label)
            selected_label = completion.strip().lower()
            
            # Track recent selections
            self.recent_selections.append(selected_label)
            if len(self.recent_selections) > 5:
                self.recent_selections.pop(0)
            
            # Check for loops
            if self.recent_selections.count(selected_label) >= 3:
                log.warning(f"Detected loop with '{selected_label}', breaking pattern")
                # Try to break the loop
                if selected_label == "plan_next":
                    selected_label = "generate_code" if "generate_code" in self.available_labels else "complete"
                elif selected_label == "generate_code":
                    selected_label = "apply_files" if "apply_files" in self.available_labels else "complete"
            
            # Apply the selection to our Step
            if selected_label == "complete" or selected_label == "none":
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
                    log.info(f"Using closest match '{closest}' for '{selected_label}'")
                else:
                    # Smart fallback based on context
                    self.step.on_success = self._smart_fallback(prompt)
                    log.warning(f"No match found, using smart fallback: {self.step.on_success}")
            
            return f"Selected: {self.step.on_success}"
            
        except Exception as e:
            log.error(f"BranchingAgent error: {e}")
            # Return error message and set safe fallback
            self.step.on_success = "error_recovery" if "error_recovery" in self.available_labels else None
            return f"Error in branch selection: {e}. Falling back to: {self.step.on_success}"
    
    def _smart_fallback(self, prompt: str) -> str:
        """Smart fallback based on recommendation content."""
        prompt_lower = prompt.lower()
        
        # Check for keywords in the recommendation
        if "code" in prompt_lower or "generate" in prompt_lower or "implement" in prompt_lower:
            return "generate_code" if "generate_code" in self.available_labels else self.available_labels[0]
        elif "save" in prompt_lower or "file" in prompt_lower or "write" in prompt_lower:
            return "apply_files" if "apply_files" in self.available_labels else self.available_labels[0]
        elif "test" in prompt_lower or "run" in prompt_lower or "validate" in prompt_lower:
            return "run_tests" if "run_tests" in self.available_labels else self.available_labels[0]
        elif "error" in prompt_lower or "fix" in prompt_lower or "debug" in prompt_lower:
            return "error_recovery" if "error_recovery" in self.available_labels else "generate_code"
        elif "complete" in prompt_lower or "done" in prompt_lower or "finish" in prompt_lower:
            return None  # Complete
        else:
            # Default to generate_code as most common next step
            return "generate_code" if "generate_code" in self.available_labels else self.available_labels[0]
    
    def _build_selection_prompt(self, recommendation: str) -> str:
        """Build the prompt for label selection."""
        # Add metrics to help with decision
        metrics = f"""
METRICS:
- Loop count: {self.loop_count}
- Recent selections: {', '.join(self.recent_selections[-3:]) if self.recent_selections else 'none'}
"""
        
        prompt = f"""
PLANNING RECOMMENDATION:
{recommendation}

{metrics}

Based on the recommendation above, which step should execute next?

Respond with ONLY the step label (no explanation needed).

Example responses:
- generate_code
- apply_files
- run_tests
- complete

Your selection:"""
        
        return prompt
    
    # Removed - no longer needed with plain output
    
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