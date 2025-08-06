#!/usr/bin/env python3
"""
ExecutionPlannerAgent - Dynamic execution plan generation for Talk orchestration.

This agent creates concrete execution plans as lists of Step class instances
that can be directly used by TalkOrchestrator. It analyzes tasks and generates
optimal Step sequences with proper agent assignments, success/failure transitions,
and conditional logic.

The agent supports:
- Dynamic Step class instance generation
- Agent assignment optimization
- Success/failure transition planning
- Conditional workflow logic
- Integration with Talk framework orchestration
- Adaptive plan modification based on task requirements
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from agent.agent import Agent
from plan_runner.step import Step

log = logging.getLogger(__name__)

class ExecutionPlanningAgent(Agent):
    """
    Specialized agent for generating executable Step-based plans.
    
    This agent creates actual Step class instances that can be directly
    used by TalkOrchestrator's PlanRunner for task execution.
    """
    
    def __init__(self, **kwargs):
        """Initialize with execution planning capabilities."""
        super().__init__(roles=[
            "You are an expert execution planner for multi-agent orchestration systems.",
            "You create concrete execution plans as Step objects with proper agent assignments.",
            "You understand the Talk framework's agent ecosystem and optimal workflow patterns.", 
            "You design robust execution sequences with error handling and fallback logic."
        ], **kwargs)
        
        # Define available agent types for orchestration
        self.available_agents = {
            "coder": "Generates code files and implements features",
            "file": "Applies code changes to the filesystem", 
            "tester": "Runs tests and validates implementations",
            "researcher": "Performs web research and gathers information",
            "reminiscing": "Retrieves relevant memories and contextual information"
        }
        
        # Define workflow patterns based on task characteristics
        self.workflow_patterns = {
            "simple_implementation": {
                "description": "Basic implementation without research",
                "pattern": ["reminiscing", "coder", "file", "tester", "coder"]
            },
            "research_based": {
                "description": "Implementation requiring external research",
                "pattern": ["researcher", "reminiscing", "coder", "file", "tester", "coder"]
            },
            "complex_system": {
                "description": "Multi-component system requiring iterative development",
                "pattern": ["researcher", "reminiscing", "coder", "file", "tester", "coder", "file", "tester", "coder"]
            },
            "bug_fix": {
                "description": "Debugging and fixing existing code",
                "pattern": ["reminiscing", "coder", "file", "tester", "coder"]
            },
            "feature_addition": {
                "description": "Adding features to existing codebase",
                "pattern": ["reminiscing", "coder", "file", "tester", "coder"]
            }
        }
    
    def run(self, input_text: str) -> List[Step]:
        """
        Generate a concrete execution plan as Step objects.
        
        Args:
            input_text: Task description to create execution plan for
            
        Returns:
            List of Step objects for the execution plan
        """
        try:
            # Analyze the task to determine optimal workflow
            task_analysis = self._analyze_task_requirements(input_text)
            
            # Select appropriate workflow pattern
            workflow_pattern = self._select_workflow_pattern(task_analysis)
            
            # Generate Step objects for the selected pattern
            steps = self._create_step_instances(workflow_pattern, task_analysis)
            
            # Optimize the plan with error handling and transitions
            optimized_steps = self._optimize_step_transitions(steps, task_analysis)
            
            log.info(f"Generated execution plan with {len(optimized_steps)} steps")
            
            return optimized_steps
            
        except Exception as e:
            log.error(f"Error generating execution plan: {e}")
            return []
    
    def generate_steps(self, input_text: str) -> List[Step]:
        """
        Generate Step objects for orchestration (separate from run method).
        
        Args:
            input_text: Task description to create execution plan for
            
        Returns:
            List of Step objects ready for orchestration
        """
        try:
            # Analyze the task to determine optimal workflow
            task_analysis = self._analyze_task_requirements(input_text)
            
            # Select appropriate workflow pattern
            workflow_pattern = self._select_workflow_pattern(task_analysis)
            
            # Generate Step objects for the selected pattern
            steps = self._create_step_instances(workflow_pattern, task_analysis)
            
            # Optimize the plan with error handling and transitions
            optimized_steps = self._optimize_step_transitions(steps, task_analysis)
            
            log.info(f"Generated execution plan with {len(optimized_steps)} steps")
            return optimized_steps
            
        except Exception as e:
            log.error(f"Error generating execution plan: {e}")
            return self._create_fallback_steps()
    
    def _analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """Analyze task to determine execution requirements."""
        analysis_prompt = f"""
Analyze this task for execution planning:

TASK: {task}

Determine:
1. RESEARCH_NEEDED: Does this require web research? [yes/no]
2. COMPLEXITY: How complex is this task? [simple/medium/complex] 
3. EXISTING_CODE: Does this modify existing code? [yes/no]
4. TESTING_SCOPE: What testing is needed? [basic/standard/comprehensive]
5. ITERATIONS: How many code-test cycles likely needed? [1-5]
6. RISK_LEVEL: How risky is this implementation? [low/medium/high]

Respond in exactly this format:
RESEARCH_NEEDED: [yes/no]
COMPLEXITY: [level]
EXISTING_CODE: [yes/no] 
TESTING_SCOPE: [scope]
ITERATIONS: [number]
RISK_LEVEL: [level]
"""
        
        self._append("user", analysis_prompt)
        llm_response = self.call_ai()
        self._append("assistant", llm_response)
        
        # Parse response
        analysis = self._parse_analysis_response(llm_response)
        analysis["original_task"] = task
        analysis["timestamp"] = datetime.now().isoformat()
        
        return analysis
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis into structured data."""
        analysis = {}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()
                
                if key == "research_needed":
                    analysis["research_needed"] = value == "yes"
                elif key == "complexity":
                    analysis["complexity"] = value
                elif key == "existing_code":
                    analysis["existing_code"] = value == "yes"
                elif key == "testing_scope":
                    analysis["testing_scope"] = value
                elif key == "iterations":
                    try:
                        analysis["iterations"] = int(value)
                    except ValueError:
                        analysis["iterations"] = 2
                elif key == "risk_level":
                    analysis["risk_level"] = value
        
        # Set defaults for missing values
        defaults = {
            "research_needed": False,
            "complexity": "medium",
            "existing_code": False,
            "testing_scope": "standard",
            "iterations": 2,
            "risk_level": "medium"
        }
        
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
        
        return analysis
    
    def _select_workflow_pattern(self, analysis: Dict[str, Any]) -> str:
        """Select the most appropriate workflow pattern."""
        if analysis.get("research_needed", False):
            if analysis.get("complexity") == "complex":
                return "complex_system"
            else:
                return "research_based"
        elif analysis.get("existing_code", False):
            if "bug" in analysis.get("original_task", "").lower() or "fix" in analysis.get("original_task", "").lower():
                return "bug_fix"
            else:
                return "feature_addition"
        elif analysis.get("complexity") == "complex":
            return "complex_system"
        else:
            return "simple_implementation"
    
    def _create_step_instances(self, pattern_name: str, analysis: Dict[str, Any]) -> List[Step]:
        """Create actual Step instances based on the selected pattern."""
        pattern = self.workflow_patterns.get(pattern_name, self.workflow_patterns["simple_implementation"])
        agent_sequence = pattern["pattern"]
        
        steps = []
        
        # Create step labels based on agent sequence
        step_labels = self._generate_step_labels(agent_sequence)
        
        # Create Step instances
        for i, (agent_key, label) in enumerate(zip(agent_sequence, step_labels)):
            # Determine success transition
            on_success = step_labels[i + 1] if i + 1 < len(step_labels) else None
            
            # Create the step
            step = Step(
                label=label,
                agent_key=agent_key,
                on_success=on_success,
                on_fail=None  # Will be set in optimization phase
            )
            steps.append(step)
        
        return steps
    
    def _generate_step_labels(self, agent_sequence: List[str]) -> List[str]:
        """Generate meaningful step labels from agent sequence."""
        labels = []
        agent_counts = {}
        
        for agent in agent_sequence:
            # Track how many times we've used each agent
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            # Generate appropriate label
            if agent == "researcher":
                labels.append("research_requirements")
            elif agent == "reminiscing":
                labels.append("recall_memories")
            elif agent == "coder":
                if agent_counts[agent] == 1:
                    labels.append("generate_code")
                elif agent_counts[agent] == 2:
                    labels.append("refine_code")
                else:
                    labels.append(f"iterate_code_{agent_counts[agent]}")
            elif agent == "file":
                if agent_counts[agent] == 1:
                    labels.append("apply_changes")
                else:
                    labels.append(f"apply_changes_{agent_counts[agent]}")
            elif agent == "tester":
                if agent_counts[agent] == 1:
                    labels.append("run_tests")
                else:
                    labels.append(f"run_tests_{agent_counts[agent]}")
        
        # Ensure the final step is always check_results
        if labels and not labels[-1].startswith("check_"):
            if labels[-1].startswith("generate_code") or labels[-1].startswith("refine_code") or labels[-1].startswith("iterate_code"):
                labels[-1] = "check_results"
        
        return labels
    
    def _optimize_step_transitions(self, steps: List[Step], analysis: Dict[str, Any]) -> List[Step]:
        """Add error handling and optimize step transitions."""
        if not steps:
            return steps
        
        # Add failure transitions
        for i, step in enumerate(steps):
            if step.agent_key == "coder":
                # If code generation fails, try research (if not already done) or go to previous coder step
                if not any(s.agent_key == "researcher" for s in steps):
                    # No research step exists, could add one
                    step.on_fail = "research_requirements"
                else:
                    # Research already done, try iterating
                    step.on_fail = "generate_code"
            elif step.agent_key == "tester":
                # If tests fail, go back to code generation
                step.on_fail = "generate_code"
            elif step.agent_key == "file":
                # If file application fails, go back to code generation
                step.on_fail = "generate_code"
        
        # Handle high-risk tasks with additional safety measures
        if analysis.get("risk_level") == "high":
            # Add additional validation steps if needed
            pass
        
        # Ensure proper termination
        if steps:
            steps[-1].on_success = None  # Last step should end the workflow
        
        return steps
    
    def _create_fallback_steps(self) -> List[Step]:
        """Create a simple fallback execution plan."""
        log.warning("Creating fallback execution plan")
        
        return [
            Step(
                label="recall_memories",
                agent_key="reminiscing",
                on_success="generate_code"
            ),
            Step(
                label="generate_code",
                agent_key="coder",
                on_success="apply_changes",
                on_fail="recall_memories"
            ),
            Step(
                label="apply_changes",
                agent_key="file",
                on_success="run_tests",
                on_fail="generate_code"
            ),
            Step(
                label="run_tests",
                agent_key="tester",
                on_success="check_results",
                on_fail="generate_code"
            ),
            Step(
                label="check_results",
                agent_key="coder",
                on_success=None
            )
        ]
    
    def get_plan_summary(self, steps: List[Step]) -> str:
        """Get a human-readable summary of the execution plan."""
        if not steps:
            return "No execution plan generated"
        
        summary = f"Execution Plan ({len(steps)} steps):\n"
        for i, step in enumerate(steps, 1):
            summary += f"{i}. {step.label} [{step.agent_key}]"
            if step.on_success:
                summary += f" -> {step.on_success}"
            if step.on_fail:
                summary += f" (on fail: {step.on_fail})"
            summary += "\n"
        
        return summary
