#!/usr/bin/env python3
"""
PlanningAgent - Strategic decision maker with hierarchical todo tracking.

This agent uses the LLM to analyze the current state of execution and provide
strategic recommendations for what to do next. It maintains context through
the conversation history and outputs structured recommendations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from agent.agent import Agent

log = logging.getLogger(__name__)


class PlanningAgent(Agent):
    """
    Strategic planning agent that uses LLM to analyze state and recommend actions.
    
    This agent:
    1. Analyzes the current blackboard state and task progress
    2. Uses the LLM to reason about what should happen next
    3. Returns structured recommendations that include specific action labels
    4. Maintains conversation history for context
    """
    
    def __init__(self, **kwargs):
        """Initialize the planning agent with strategic roles."""
        roles = [
            "You are a strategic planning agent for a multi-agent orchestration system.",
            "You analyze the current state of task execution and recommend the next action.",
            "You maintain a mental model of the task progress and what remains to be done.",
            "",
            "IMPORTANT: You must recommend specific action labels that map to workflow steps:",
            "- generate_code: Generate code implementation",
            "- apply_files: Apply code changes to files", 
            "- run_tests: Execute tests to validate changes",
            "- research: Search for information (if available)",
            "- complete: Mark the task as complete",
            "- error_recovery: Handle errors and retry",
            "",
            "Your output should be structured JSON that includes:",
            "1. A hierarchical todo list showing task breakdown",
            "2. Analysis of the current situation",
            "3. A specific next_action recommendation (must be one of the labels above)",
            "4. Reasoning for your recommendation"
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Track state across calls
        self.task_description = None
        self.actions_taken = []
        self.scratch_dir = None
    
    def run(self, input_text: str) -> str:
        """
        Analyze state and provide planning recommendations using the LLM.
        
        Args:
            input_text: Current blackboard state and context (usually JSON)
            
        Returns:
            LLM completion with structured planning output
        """
        try:
            # Parse input if it's JSON
            blackboard_state = self._parse_input(input_text)
            
            # Initialize or update task description
            if "task_description" in blackboard_state:
                self.task_description = blackboard_state["task_description"]
            
            # Build the prompt for the LLM
            prompt = self._build_planning_prompt(blackboard_state)
            
            # Get LLM's strategic analysis
            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Track this action
            if "last_action" in blackboard_state and blackboard_state["last_action"]:
                self.actions_taken.append(blackboard_state["last_action"])
            
            # Optionally save to scratch for other agents
            self._save_to_scratch(completion)
            
            return completion
            
        except Exception as e:
            log.error(f"Planning error: {e}")
            # Return a valid planning response even on error
            error_response = f"""{{
    "error": "{str(e)}",
    "todo_hierarchy": "[ ] Recover from error\\n    [ ] Analyze error\\n    [ ] Retry task",
    "analysis": {{
        "situation": "An error occurred during planning",
        "action_needed": "generate_code",
        "confidence": "low"
    }},
    "next_action": "generate_code",
    "recommendation": "Starting with code generation due to planning error"
}}"""
            return error_response
    
    def _parse_input(self, input_text: str) -> Dict[str, Any]:
        """Parse the input to extract blackboard state."""
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            # If not JSON, treat as plain text task description
            return {
                "task_description": input_text,
                "blackboard_state": {},
                "last_action": "",
                "last_result": ""
            }
    
    def _build_planning_prompt(self, blackboard_state: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for strategic planning."""
        task = blackboard_state.get("task_description", "No task specified")
        last_action = blackboard_state.get("last_action", "")
        last_result = blackboard_state.get("last_result", "")
        
        # Build context about what's happened so far
        context_parts = [f"TASK: {task}"]
        
        if self.actions_taken:
            context_parts.append(f"\nACTIONS TAKEN SO FAR: {', '.join(self.actions_taken)}")
        
        if last_action:
            context_parts.append(f"\nLAST ACTION: {last_action}")
            
        if last_result:
            # Truncate very long results
            if len(last_result) > 500:
                last_result = last_result[:500] + "... (truncated)"
            context_parts.append(f"\nLAST RESULT: {last_result}")
        
        # Add any additional blackboard state
        if "blackboard_state" in blackboard_state:
            state_info = blackboard_state["blackboard_state"]
            if state_info:
                context_parts.append(f"\nADDITIONAL STATE: {json.dumps(state_info, indent=2)}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""{context}

Based on the above context, provide strategic planning for the next step.

Create a hierarchical todo list showing the overall task breakdown and current progress.
Analyze what has been accomplished and what remains to be done.
Recommend the specific next action to take.

Your response must be valid JSON in this format:
{{
    "todo_hierarchy": "[ ] Main task\\n    [ ] Subtask 1\\n    [✓] Subtask 2 (completed)\\n    [ ] Subtask 3",
    "current_path": ["Main task", "Current subtask"],
    "stats": {{
        "total": 5,
        "completed": 2,
        "pending": 3
    }},
    "analysis": {{
        "situation": "Current state assessment",
        "next_focus": "What to focus on next",
        "action_needed": "Specific action recommendation",
        "potential_problems": "Any issues to watch for",
        "confidence": "high/medium/low"
    }},
    "next_action": "generate_code|apply_files|run_tests|research|complete|error_recovery",
    "recommendation": "Explanation of why this action is recommended"
}}

IMPORTANT: The "next_action" field must be exactly one of: generate_code, apply_files, run_tests, research, complete, error_recovery

Think step by step:
1. What is the task asking for?
2. What progress has been made?
3. What's the logical next step?
4. Which specific action label best matches that step?"""
        
        return prompt
    
    def _save_to_scratch(self, completion: str):
        """Save planning output to scratch directory for other agents."""
        try:
            # Create scratch directory if needed
            if not self.scratch_dir:
                base_dir = Path.cwd()
                scratch_dir = base_dir / ".talk_scratch"
                scratch_dir.mkdir(exist_ok=True)
                self.scratch_dir = scratch_dir
            
            # Save the latest planning output
            planning_file = self.scratch_dir / "latest_planning.json"
            
            # Try to parse and save as formatted JSON
            try:
                planning_data = json.loads(completion)
                with open(planning_file, "w") as f:
                    json.dump(planning_data, f, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, save as text
                with open(planning_file, "w") as f:
                    f.write(completion)
                    
        except Exception as e:
            log.debug(f"Could not save to scratch: {e}")
            # Not critical, so we just log and continue