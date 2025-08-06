#!/usr/bin/env python3
"""
BranchingAgent - Makes control flow decisions in the orchestration workflow.

This agent analyzes the current state and results to determine the next
step in the workflow, enabling conditional branching and intelligent
flow control.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Any
from enum import Enum

from agent.agent import Agent

log = logging.getLogger(__name__)


class BranchDecision(Enum):
    """Possible branching decisions."""
    CONTINUE = "continue"              # Continue to next step
    LOOP_REFINEMENT = "loop_refinement"  # Loop back to refinement
    COMPLETE = "complete"              # Task is complete
    ESCALATE = "escalate"              # Needs human intervention
    RESTART = "restart"                # Start over from planning


class BranchingAgent(Agent):
    """
    Agent that makes control flow decisions based on current state.
    
    This agent analyzes results from previous steps and determines
    the optimal next action in the workflow.
    """
    
    def __init__(self, **kwargs):
        """Initialize the BranchingAgent."""
        roles = [
            "You are an expert workflow control agent.",
            "You analyze results and make intelligent branching decisions.",
            "You determine when tasks are complete or need more work.",
            "You optimize workflow efficiency by choosing the right path."
        ]
        super().__init__(roles=roles, **kwargs)
        
    def run(self, input_text: str) -> str:
        """
        Analyze input and make a branching decision.
        
        Args:
            input_text: Current state/results to analyze
            
        Returns:
            JSON string with branching decision and target
        """
        decision = self.make_decision(input_text)
        
        # Return decision as JSON
        return json.dumps({
            "decision": decision["decision"].value,
            "target": decision.get("target"),
            "reason": decision.get("reason", ""),
            "confidence": decision.get("confidence", 0.8)
        }, indent=2)
    
    def make_decision(self, state: str) -> Dict[str, Any]:
        """
        Make a branching decision based on current state.
        
        Args:
            state: Current workflow state/results
            
        Returns:
            Dictionary with decision details
        """
        # Parse refinement results if available
        refinement_data = self._parse_refinement_results(state)
        
        if refinement_data:
            return self._decide_from_refinement(refinement_data)
        
        # Otherwise use LLM to analyze state
        return self._decide_from_analysis(state)
    
    def _parse_refinement_results(self, state: str) -> Optional[Dict[str, Any]]:
        """Try to parse refinement agent results from state."""
        try:
            # Look for JSON in the state
            if "{" in state and "}" in state:
                start = state.index("{")
                end = state.rindex("}") + 1
                json_str = state[start:end]
                return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass
        return None
    
    def _decide_from_refinement(self, refinement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on refinement results."""
        status = refinement_data.get("status", "")
        iterations = refinement_data.get("iterations", 0)
        
        if status == "success":
            return {
                "decision": BranchDecision.COMPLETE,
                "target": None,
                "reason": "All tests passed, task completed successfully",
                "confidence": 0.95
            }
        elif status == "max_iterations":
            return {
                "decision": BranchDecision.ESCALATE,
                "target": "human_review",
                "reason": f"Reached max iterations ({iterations}) without success",
                "confidence": 0.9
            }
        elif status == "needs_improvement":
            if iterations < 3:
                return {
                    "decision": BranchDecision.LOOP_REFINEMENT,
                    "target": "development_cycle",
                    "reason": "Tests failing but still have iterations left",
                    "confidence": 0.85
                }
            else:
                return {
                    "decision": BranchDecision.ESCALATE,
                    "target": "expert_review",
                    "reason": "Multiple refinement attempts failed",
                    "confidence": 0.8
                }
        else:  # failed
            return {
                "decision": BranchDecision.RESTART,
                "target": "planning",
                "reason": "Critical failure, needs replanning",
                "confidence": 0.7
            }
    
    def _decide_from_analysis(self, state: str) -> Dict[str, Any]:
        """Use LLM to analyze state and make decision."""
        analysis_prompt = f"""
Analyze this workflow state and determine the next action:

CURRENT STATE:
{state}

Consider:
1. Are all requirements met?
2. Are there any failures or errors?
3. Is the quality acceptable?
4. Should we continue, loop back, or complete?

Provide your decision in this format:
{{
    "decision": "continue|loop_refinement|complete|escalate|restart",
    "target": "step_label or null",
    "reason": "explanation",
    "confidence": 0.0-1.0,
    "analysis": {{
        "requirements_met": true/false,
        "has_errors": true/false,
        "quality_acceptable": true/false,
        "iterations_used": number
    }}
}}
"""
        
        response = self.reply(analysis_prompt)
        
        try:
            result = json.loads(response)
            # Convert string decision to enum
            decision_str = result.get("decision", "continue")
            result["decision"] = BranchDecision(decision_str)
            return result
        except (json.JSONDecodeError, ValueError):
            # Default conservative decision
            return {
                "decision": BranchDecision.CONTINUE,
                "target": None,
                "reason": "Unable to parse state, continuing with default flow",
                "confidence": 0.5
            }