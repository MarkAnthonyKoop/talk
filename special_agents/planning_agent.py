#!/usr/bin/env python3
"""
PlanningAgent - Strategic decision maker with hierarchical todo tracking.

This agent analyzes the current state of execution, maintains a hierarchical
todo list, and provides strategic recommendations for what to do next.
It provides SPECIFIC action recommendations that map to Step labels.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agent.agent import Agent

log = logging.getLogger(__name__)


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoNode:
    """A node in the hierarchical todo tree."""
    description: str
    status: TodoStatus = TodoStatus.PENDING
    children: List[TodoNode] = field(default_factory=list)
    
    def to_string(self, indent: int = 0) -> str:
        """Convert to indented string representation."""
        status_map = {
            TodoStatus.PENDING: "[ ]",
            TodoStatus.IN_PROGRESS: "[→]",
            TodoStatus.COMPLETED: "[✓]",
            TodoStatus.FAILED: "[✗]",
            TodoStatus.SKIPPED: "[~]"
        }
        
        result = "    " * indent + f"{status_map[self.status]} {self.description}"
        for child in self.children:
            result += "\n" + child.to_string(indent + 1)
        return result
    
    def count_status(self) -> Dict[str, int]:
        """Count todos by status."""
        counts = {s.value: 0 for s in TodoStatus}
        counts[self.status.value] += 1
        
        for child in self.children:
            child_counts = child.count_status()
            for status, count in child_counts.items():
                counts[status] += count
        
        return counts


class TodoTree:
    """Simple hierarchical todo manager."""
    
    def __init__(self):
        self.root: Optional[TodoNode] = None
        self.current_path: List[TodoNode] = []
    
    def initialize_simple(self, task: str):
        """Initialize with a simple todo structure for any task."""
        self.root = TodoNode(f"Complete: {task}", TodoStatus.IN_PROGRESS)
        
        # Add standard sub-tasks
        self.root.children = [
            TodoNode("Generate code implementation", TodoStatus.PENDING),
            TodoNode("Apply code to files", TodoStatus.PENDING),
            TodoNode("Validate implementation", TodoStatus.PENDING)
        ]
        
        self.current_path = [self.root]
    
    def get_next_todo(self) -> Optional[TodoNode]:
        """Get the next pending todo."""
        def find_pending(node: TodoNode) -> Optional[TodoNode]:
            if node.status == TodoStatus.PENDING:
                return node
            for child in node.children:
                result = find_pending(child)
                if result:
                    return result
            return None
        
        if self.root:
            return find_pending(self.root)
        return None
    
    def mark_todo_complete(self, description: str):
        """Mark a todo as complete by description match."""
        def mark_in_tree(node: TodoNode) -> bool:
            if description.lower() in node.description.lower():
                node.status = TodoStatus.COMPLETED
                return True
            for child in node.children:
                if mark_in_tree(child):
                    return True
            return False
        
        if self.root:
            mark_in_tree(self.root)
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall status."""
        if not self.root:
            return {"total": 0}
        
        counts = self.root.count_status()
        total = sum(counts.values())
        
        return {
            "total": total,
            **counts,
            "completion_rate": counts.get("completed", 0) / total if total > 0 else 0
        }


class PlanningAgent(Agent):
    """
    Simplified planning agent that provides clear action recommendations.
    """
    
    def __init__(self, **kwargs):
        """Initialize the planning agent."""
        super().__init__(**kwargs)
        self.todo_tree = TodoTree()
        self.initialized = False
        self.last_action = None
        self.action_count = 0
    
    def run(self, input_text: str) -> str:
        """
        Analyze state and provide planning recommendations.
        
        Returns:
            JSON string with clear action recommendation
        """
        try:
            # Parse the input
            blackboard_state = self._parse_input(input_text)
            task = blackboard_state.get("task_description", "")
            
            # Initialize on first run
            if not self.initialized:
                self.todo_tree.initialize_simple(task)
                self.initialized = True
            
            # Update based on last action
            last_action = blackboard_state.get("last_action", "")
            last_result = blackboard_state.get("last_result", "")
            
            # Determine next action based on simple state machine
            next_action = self._determine_next_action(last_action, last_result, task)
            
            # Update todo status
            if "generate_code" in last_action and last_result:
                self.todo_tree.mark_todo_complete("Generate code")
            elif "apply_files" in last_action and last_result:
                self.todo_tree.mark_todo_complete("Apply code")
            elif "run_tests" in last_action:
                self.todo_tree.mark_todo_complete("Validate")
            
            # Build output
            todo_status = self.todo_tree.get_status()
            
            output = {
                "todo_hierarchy": self.todo_tree.root.to_string() if self.todo_tree.root else f"[ ] {task}",
                "current_path": [],
                "stats": todo_status,
                "analysis": {
                    "situation": self._get_situation(last_action, last_result),
                    "next_focus": f"Action: {next_action}",
                    "action_needed": next_action,
                    "potential_problems": "",
                    "fallback_plan": "generate_code" if next_action != "generate_code" else "research",
                    "confidence": "high" if self.action_count < 10 else "medium"
                },
                "recommendation": next_action,
                "next_action": next_action,
                "next_todo": self.todo_tree.get_next_todo().description if self.todo_tree.get_next_todo() else "Complete"
            }
            
            self.last_action = next_action
            self.action_count += 1
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            log.error(f"Planning error: {e}")
            # Return a valid action even on error
            return json.dumps({
                "error": str(e),
                "recommendation": "generate_code",
                "next_action": "generate_code",
                "analysis": {
                    "situation": "Error occurred",
                    "action_needed": "generate_code"
                }
            })
    
    def _parse_input(self, input_text: str) -> Dict[str, Any]:
        """Parse the input to extract blackboard state."""
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            # Fall back to simple parsing
            return {
                "task_description": input_text,
                "last_action": "",
                "last_result": ""
            }
    
    def _determine_next_action(self, last_action: str, last_result: str, task: str) -> str:
        """
        Determine the next action based on simple state machine.
        
        This returns ACTUAL Step labels that exist in the workflow.
        """
        # If this is the first run or we have no last action
        if not last_action or last_action == "plan_next":
            # Check if we have completed some work
            if self.action_count == 0:
                return "generate_code"  # Start with code generation
            elif self.action_count > 10:
                return "complete"  # Prevent infinite loops
        
        # State machine for action flow
        if "generate_code" in last_action:
            if "code" in last_result.lower() or "function" in last_result.lower() or "def" in last_result.lower():
                return "apply_files"  # Code was generated, apply it
            else:
                return "generate_code"  # Try again
        
        elif "apply_files" in last_action:
            if "applied" in last_result.lower() or "created" in last_result.lower() or "modified" in last_result.lower():
                return "run_tests"  # Files applied, test them
            else:
                return "run_tests"  # Try testing anyway
        
        elif "run_tests" in last_action:
            # Tests were run, we're done
            return "complete"
        
        elif "research" in last_action:
            # Research done, generate code
            return "generate_code"
        
        elif "error" in last_action.lower():
            # Error occurred, try to generate code
            return "generate_code"
        
        elif "complete" in last_action:
            # Already complete
            return "complete"
        
        # Default: generate code
        return "generate_code"
    
    def _get_situation(self, last_action: str, last_result: str) -> str:
        """Get a description of the current situation."""
        if not last_action:
            return "Starting new task"
        elif "generate_code" in last_action:
            return "Code generation attempted"
        elif "apply_files" in last_action:
            return "Files have been applied"
        elif "run_tests" in last_action:
            return "Tests have been executed"
        elif "complete" in last_action:
            return "Task completed"
        else:
            return "Processing task"