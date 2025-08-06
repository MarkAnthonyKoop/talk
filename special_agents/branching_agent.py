#!/usr/bin/env python3
"""
BranchingAgent - Intelligent label selector for workflow branching.

This agent interprets PlanningAgent recommendations and selects the appropriate
Step label from the available steps in the plan. It modifies its Step's 
on_success field to control workflow execution.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from agent.agent import Agent
from plan_runner.step import Step

log = logging.getLogger(__name__)


class BranchingAgent(Agent):
    """
    Simplified branching agent that directly uses PlanningAgent's next_action.
    
    This agent:
    1. Receives the PlanningAgent's output with next_action field
    2. Validates the action against available Steps
    3. Modifies its own Step's on_success field to that label
    4. Returns confirmation of the selection
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
        super().__init__(**kwargs)
        
        self.step = step  # Reference to our own Step (can modify on_success)
        self.plan = plan  # Complete plan with all available Steps
        self.available_labels = self._extract_labels(plan)
    
    def _extract_labels(self, plan: List[Step]) -> List[str]:
        """Extract all available Step labels from the plan."""
        labels = []
        for step in plan:
            if step.label and step.label != self.step.label:
                labels.append(step.label)
        return labels
    
    def run(self, prompt: str) -> str:
        """
        Select the next Step label based on planning recommendation.
        
        Args:
            prompt: Contains PlanningAgent's recommendation with next_action field
            
        Returns:
            Confirmation of the selected label
        """
        try:
            # Parse the planning recommendation
            planning_data = self._parse_planning_input(prompt)
            
            # Get the next action directly
            selected_label = self._get_next_action(planning_data)
            
            # Validate the selection
            if selected_label == "complete" or selected_label == "COMPLETE":
                self.step.on_success = None
                log.info("Workflow complete - setting on_success to None")
                return "Workflow marked as complete"
            elif selected_label in self.available_labels:
                self.step.on_success = selected_label
                log.info(f"Selected next step: {selected_label}")
                return f"Next step: {selected_label}"
            else:
                # Try to find a close match
                fallback = self._find_best_match(selected_label)
                if fallback:
                    self.step.on_success = fallback
                    log.warning(f"Using closest match for '{selected_label}': {fallback}")
                    return f"Next step: {fallback}"
                else:
                    # Last resort fallback
                    fallback = "generate_code" if "generate_code" in self.available_labels else self.available_labels[0]
                    self.step.on_success = fallback
                    log.warning(f"No match for '{selected_label}', using fallback: {fallback}")
                    return f"Next step: {fallback}"
                
        except Exception as e:
            log.error(f"BranchingAgent error: {e}")
            # On error, try to generate code
            if "generate_code" in self.available_labels:
                self.step.on_success = "generate_code"
                return "Error occurred, falling back to generate_code"
            else:
                self.step.on_success = None
                return f"Error in branching: {e}"
    
    def _parse_planning_input(self, prompt: str) -> Dict:
        """Parse the input from PlanningAgent."""
        try:
            # Try to parse as JSON
            if prompt.strip().startswith('{'):
                return json.loads(prompt)
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        return {
            "recommendation": prompt,
            "context": prompt
        }
    
    def _get_next_action(self, planning_data: Dict) -> str:
        """Extract the next action from planning data."""
        # Check for direct next_action field
        if "next_action" in planning_data:
            return planning_data["next_action"]
        
        # Check in analysis section
        if "analysis" in planning_data and isinstance(planning_data["analysis"], dict):
            if "action_needed" in planning_data["analysis"]:
                return planning_data["analysis"]["action_needed"]
        
        # Check recommendation field
        if "recommendation" in planning_data:
            rec = planning_data["recommendation"]
            if isinstance(rec, str):
                # Parse action from string
                return self._parse_action_from_string(rec)
        
        # Default fallback
        return "generate_code"
    
    def _parse_action_from_string(self, text: str) -> str:
        """Parse an action from text recommendation."""
        text_lower = text.lower()
        
        # Direct matches
        if "generate_code" in text_lower or "generate code" in text_lower:
            return "generate_code"
        elif "apply_files" in text_lower or "apply files" in text_lower:
            return "apply_files"
        elif "run_tests" in text_lower or "run tests" in text_lower:
            return "run_tests"
        elif "complete" in text_lower or "finish" in text_lower:
            return "complete"
        elif "research" in text_lower:
            return "research"
        elif "error" in text_lower:
            return "error_recovery"
        
        # Default
        return "generate_code"
    
    def _find_best_match(self, target: str) -> Optional[str]:
        """Find the best matching label for a target string."""
        target_lower = target.lower()
        
        # Exact match (case insensitive)
        for label in self.available_labels:
            if label.lower() == target_lower:
                return label
        
        # Partial match
        for label in self.available_labels:
            if target_lower in label.lower() or label.lower() in target_lower:
                return label
        
        # Keyword match
        if "code" in target_lower or "generate" in target_lower:
            for label in self.available_labels:
                if "code" in label.lower() or "generate" in label.lower():
                    return label
        
        if "file" in target_lower or "apply" in target_lower:
            for label in self.available_labels:
                if "file" in label.lower() or "apply" in label.lower():
                    return label
        
        if "test" in target_lower or "run" in target_lower:
            for label in self.available_labels:
                if "test" in label.lower() or "run" in label.lower():
                    return label
        
        return None