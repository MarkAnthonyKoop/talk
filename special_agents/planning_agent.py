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
import time
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
    3. Returns plain English recommendations for next actions
    4. Maintains TALK.md with todos and progress
    5. Maintains conversation history for context
    """
    
    @property
    def brief_description(self) -> str:
        """Brief description for BranchingAgent."""
        return "Analyze progress, maintain todos, recommend next actions"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest planning is needed."""
        return ["plan", "strategy", "analyze", "todo", "progress", "next"]
    
    def __init__(self, **kwargs):
        """Initialize the planning agent with strategic roles."""
        # Extract base_dir before passing to parent
        self.base_dir = Path(kwargs.pop('base_dir', Path.cwd()))
        
        roles = [
            "You are a strategic planning agent.",
            "Your job: Analyze progress, maintain todos, recommend next actions",
            "",
            "You maintain TALK.md with:",
            "- Task description and goals",
            "- Todo list with [x] completed and [ ] pending items",
            "- Progress notes and observations",
            "- Scratchpad for ideas",
            "",
            "MEMORY-AWARE CAPABILITIES:",
            "- You have access to memories from similar past tasks",
            "- Learn from previous successful approaches and avoid past mistakes",
            "- Consider patterns and best practices from historical experiences",
            "",
            "OUTPUT PLAIN ENGLISH recommendations like:",
            "- 'We need to generate code for the hello function'",
            "- 'The code is ready, let's save it to a file'",
            "- 'Testing shows an error in line 15, we should fix it'",
            "- 'Everything works correctly, the task is complete'",
            "",
            "DO NOT mention workflow steps or agent names.",
            "Just describe what needs to happen next in natural language.",
            "",
            "Also maintain a simple todo list format:",
            "[ ] Todo item (pending)",
            "[x] Completed item",
            "[~] In progress item"
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Track state across calls
        self.task_description = None
        self.actions_taken = []
        self.scratch_dir = None
        self.memory_context = None  # Store memory context from ReminiscingAgent
        self.base_dir = kwargs.get('base_dir', Path.cwd())  # For TALK.md
    
    def run(self, input_text: str) -> str:
        """
        Analyze state and provide planning recommendations using the LLM.
        
        Args:
            input_text: Current blackboard state and context
            
        Returns:
            Plain English planning recommendation
        """
        try:
            # Parse input if it's JSON
            blackboard_state = self._parse_input(input_text)
            
            # Initialize or update task description
            if "task_description" in blackboard_state:
                self.task_description = blackboard_state["task_description"]
            
            # Load current TALK.md if it exists
            talk_md_content = self._load_talk_md()
            
            # Build the prompt for the LLM
            prompt = self._build_planning_prompt(blackboard_state, talk_md_content)
            
            # Get LLM's strategic analysis
            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Track this action
            if "last_action" in blackboard_state and blackboard_state["last_action"]:
                self.actions_taken.append(blackboard_state["last_action"])
            
            # Update TALK.md with the new todos and recommendation
            self._update_talk_md(completion, blackboard_state)
            
            # Extract just the recommendation for BranchingAgent
            recommendation = self._extract_recommendation(completion)
            
            return recommendation
            
        except Exception as e:
            log.error(f"Planning error: {e}")
            # Return plain English error response
            return f"An error occurred during planning: {e}. Let's try generating code to move forward."
    
    def _parse_input(self, input_text: str) -> Dict[str, Any]:
        """Parse the input to extract blackboard state."""
        try:
            parsed = json.loads(input_text)
            # Extract and store memory context if present
            if "memory_context" in parsed:
                self.memory_context = parsed["memory_context"]
            return parsed
        except json.JSONDecodeError:
            # If not JSON, treat as plain text task description
            return {
                "task_description": input_text,
                "blackboard_state": {},
                "last_action": "",
                "last_result": "",
                "memory_context": None
            }
    
    def _build_planning_prompt(self, blackboard_state: Dict[str, Any], talk_md: str) -> str:
        """Build a comprehensive prompt for strategic planning."""
        task = blackboard_state.get("task_description", "No task specified")
        last_action = blackboard_state.get("last_action", "")
        last_result = blackboard_state.get("last_result", "")
        
        # Build context about what's happened so far
        context_parts = [f"TASK: {task}"]
        
        # Add current TALK.md content if exists
        if talk_md:
            context_parts.append(f"\nCURRENT TALK.md CONTENT:\n{talk_md}")
        
        # Add memory context if available (from ReminiscingAgent)
        memory_context = blackboard_state.get("memory_context") or self.memory_context
        if memory_context:
            context_parts.append(f"\nRELEVANT MEMORIES FROM SIMILAR TASKS:")
            context_parts.append(f"{memory_context[:1500]}")  # Truncate if too long
            context_parts.append("\nConsider these past experiences when planning the approach.")
        
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
                context_parts.append(f"\nADDITIONAL STATE: {state_info}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""{context}

Based on the above context, provide strategic planning for the next step.

Provide:
1. An updated todo list showing what's done and what remains
2. Your analysis of the current situation
3. A clear recommendation for what to do next (in plain English)

Format your response like this:

## Todo List
[ ] Main task
  [x] Completed subtask
  [~] In-progress subtask
  [ ] Pending subtask

## Analysis
Describe the current situation and what has been accomplished.

## Recommendation
Describe what should happen next in plain English. For example:
- "We need to generate code for the hello function"
- "The code is ready, let's save it to a file"
- "There's an error in the test, we should fix it"
- "Everything is complete"

Do NOT mention agent names or workflow labels. Just describe the action needed."""
        
        return prompt
    
    def _load_talk_md(self) -> str:
        """Load current TALK.md content if it exists."""
        talk_md_path = self.base_dir / "TALK.md"
        if talk_md_path.exists():
            try:
                with open(talk_md_path, 'r') as f:
                    return f.read()
            except Exception as e:
                log.debug(f"Could not load TALK.md: {e}")
        return ""
    
    def _update_talk_md(self, completion: str, blackboard_state: Dict[str, Any]):
        """Update TALK.md with current todos and progress."""
        from datetime import datetime
        
        talk_md_path = self.base_dir / "TALK.md"
        try:
            # Extract todo list from completion
            todos = self._extract_todos(completion)
            
            content = f"""# TALK.md - Project State

## Task
{blackboard_state.get('task_description', 'No task specified')}

## Todo List
{todos}

## Progress Notes
- Last updated: {datetime.now().isoformat()}
- Actions taken: {len(self.actions_taken)}
- Last action: {blackboard_state.get('last_action', 'None')}

## Latest Recommendation
{self._extract_recommendation(completion)}

## Scratchpad
{completion}
"""
            
            with open(talk_md_path, 'w') as f:
                f.write(content)
                
        except Exception as e:
            log.debug(f"Could not update TALK.md: {e}")
    
    def _extract_todos(self, completion: str) -> str:
        """Extract todo list from completion."""
        # Look for todo list section
        lines = completion.split('\n')
        in_todos = False
        todos = []
        
        for line in lines:
            if '## Todo' in line or '## TODO' in line:
                in_todos = True
                continue
            elif in_todos and line.startswith('##'):
                break
            elif in_todos and (line.strip().startswith('[') or line.strip().startswith('-')):
                todos.append(line)
        
        return '\n'.join(todos) if todos else "[ ] No todos extracted"
    
    def _extract_recommendation(self, completion: str) -> str:
        """Extract the recommendation from completion."""
        # Look for recommendation section
        lines = completion.split('\n')
        in_rec = False
        rec_lines = []
        
        for line in lines:
            if '## Recommendation' in line or '## Next' in line:
                in_rec = True
                continue
            elif in_rec and line.startswith('##'):
                break
            elif in_rec and line.strip():
                rec_lines.append(line.strip())
        
        if rec_lines:
            return ' '.join(rec_lines)
        
        # Fallback: look for sentences about what to do
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['need to', 'should', 'let\'s', 'we must', 'next']):
                return line.strip()
        
        return "Continue with the task"
    
    def _save_to_scratch(self, completion: str):
        """Save planning output to scratch directory for other agents."""
        try:
            # Create scratch directory if needed
            if not self.scratch_dir:
                base_dir = Path.cwd()
                scratch_dir = base_dir / ".talk_scratch"
                scratch_dir.mkdir(exist_ok=True)
                self.scratch_dir = scratch_dir
            
            # Save as plain text
            planning_file = self.scratch_dir / "latest_planning.txt"
            with open(planning_file, "w") as f:
                f.write(completion)
            
            # Also save structured context for CodeAgent
            context_file = self.scratch_dir / "planning_context.json"
            context_data = {
                "task_description": self.task_description,
                "latest_recommendation": completion,
                "actions_taken": self.actions_taken,
                "timestamp": time.time()
            }
            with open(context_file, "w") as f:
                json.dump(context_data, f, indent=2)
                    
        except Exception as e:
            log.debug(f"Could not save to scratch: {e}")
            # Not critical, so we just log and continue