#!/usr/bin/env python3
"""
RefinementAgent - Handles iterative code development cycles.

This agent encapsulates the code->test->evaluate->refine loop,
managing the iterative improvement of code until it meets quality standards.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from agent.agent import Agent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent

log = logging.getLogger(__name__)


class RefinementStatus(Enum):
    """Status of the refinement cycle."""
    SUCCESS = "success"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class RefinementResult:
    """Result of a refinement cycle."""
    status: RefinementStatus
    iterations: int
    final_output: str
    test_results: Optional[str] = None
    improvements_made: List[str] = None
    
    def __post_init__(self):
        if self.improvements_made is None:
            self.improvements_made = []


class RefinementAgent(Agent):
    """
    Agent that manages iterative code refinement cycles.
    
    This agent orchestrates the process of generating code, applying it,
    testing it, and refining based on test results until the code meets
    quality standards or reaches maximum iterations.
    """
    
    def __init__(self, 
                 base_dir: str,
                 max_iterations: int = 5,
                 **kwargs):
        """
        Initialize the RefinementAgent.
        
        Args:
            base_dir: Base directory for file operations
            max_iterations: Maximum refinement iterations before giving up
            **kwargs: Additional arguments for the base Agent
        """
        roles = [
            "You are an expert code refinement orchestrator.",
            "You manage iterative development cycles to produce high-quality code.",
            "You analyze test results and determine when code meets quality standards.",
            "You coordinate between code generation, file operations, and testing."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.base_dir = base_dir
        self.max_iterations = max_iterations
        
        # Initialize sub-agents
        self.code_agent = CodeAgent(name="RefinementCodeAgent")
        self.file_agent = FileAgent(base_dir=base_dir, name="RefinementFileAgent")
        self.test_agent = TestAgent(base_dir=base_dir, name="RefinementTestAgent")
        
    def run(self, input_text: str) -> str:
        """
        Run the refinement cycle.
        
        Args:
            input_text: Task description or previous cycle output
            
        Returns:
            JSON string containing refinement results
        """
        result = self.refine_code(input_text)
        
        # Convert result to JSON for serialization
        return json.dumps({
            "status": result.status.value,
            "iterations": result.iterations,
            "final_output": result.final_output,
            "test_results": result.test_results,
            "improvements_made": result.improvements_made
        }, indent=2)
    
    def refine_code(self, task: str) -> RefinementResult:
        """
        Execute the refinement cycle until code meets standards.
        
        Args:
            task: Task description or improvement request
            
        Returns:
            RefinementResult with status and details
        """
        iterations = 0
        improvements = []
        current_task = task
        last_test_results = None
        
        while iterations < self.max_iterations:
            iterations += 1
            log.info(f"Starting refinement iteration {iterations}")
            
            try:
                # Step 1: Generate or improve code
                code_output = self.code_agent.run(current_task)
                
                # Step 2: Apply code changes
                file_output = self.file_agent.run(code_output)
                
                # Step 3: Run tests
                test_output = self.test_agent.run("Run all tests")
                last_test_results = test_output
                
                # Step 4: Evaluate results
                evaluation = self._evaluate_test_results(test_output, task)
                
                if evaluation["success"]:
                    log.info(f"Refinement successful after {iterations} iterations")
                    return RefinementResult(
                        status=RefinementStatus.SUCCESS,
                        iterations=iterations,
                        final_output=code_output,
                        test_results=test_output,
                        improvements_made=improvements
                    )
                
                # Prepare for next iteration
                improvements.append(evaluation["improvement_summary"])
                current_task = self._prepare_improvement_prompt(
                    original_task=task,
                    test_results=test_output,
                    evaluation=evaluation,
                    previous_attempts=iterations
                )
                
            except Exception as e:
                log.error(f"Error in refinement iteration {iterations}: {e}")
                return RefinementResult(
                    status=RefinementStatus.FAILED,
                    iterations=iterations,
                    final_output=f"Failed with error: {str(e)}",
                    test_results=last_test_results,
                    improvements_made=improvements
                )
        
        # Reached max iterations
        log.warning(f"Reached maximum iterations ({self.max_iterations})")
        return RefinementResult(
            status=RefinementStatus.MAX_ITERATIONS,
            iterations=iterations,
            final_output="Maximum iterations reached without achieving success",
            test_results=last_test_results,
            improvements_made=improvements
        )
    
    def _evaluate_test_results(self, test_output: str, original_task: str) -> Dict[str, Any]:
        """
        Evaluate test results to determine if refinement is needed.
        
        Args:
            test_output: Output from test execution
            original_task: Original task description
            
        Returns:
            Dictionary with evaluation results
        """
        # Use LLM to evaluate test results
        evaluation_prompt = f"""
Evaluate these test results for the following task:

TASK: {original_task}

TEST OUTPUT:
{test_output}

Analyze and determine:
1. SUCCESS: Did all tests pass? [true/false]
2. CRITICAL_ERRORS: Are there any critical failures? [list]
3. IMPROVEMENTS_NEEDED: What specific improvements are needed? [list]
4. IMPROVEMENT_PRIORITY: What should be fixed first? [string]

Respond in JSON format:
{{
    "success": true/false,
    "critical_errors": ["error1", "error2"],
    "improvements_needed": ["improvement1", "improvement2"],
    "improvement_priority": "Most important fix",
    "improvement_summary": "Brief summary of what needs fixing"
}}
"""
        
        response = super().run(evaluation_prompt)
        
        try:
            # Parse JSON response
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: Simple heuristic
            success = "PASSED" in test_output and "FAILED" not in test_output
            return {
                "success": success,
                "critical_errors": ["Failed to parse test results"],
                "improvements_needed": ["Fix test failures"],
                "improvement_priority": "Make tests pass",
                "improvement_summary": "Tests failed - needs debugging"
            }
    
    def _prepare_improvement_prompt(self, 
                                   original_task: str,
                                   test_results: str,
                                   evaluation: Dict[str, Any],
                                   previous_attempts: int) -> str:
        """
        Prepare prompt for the next refinement iteration.
        
        Args:
            original_task: Original task description
            test_results: Latest test results
            evaluation: Test evaluation results
            previous_attempts: Number of previous attempts
            
        Returns:
            Prompt for next iteration
        """
        prompt = f"""
Iteration {previous_attempts + 1} of code refinement.

ORIGINAL TASK: {original_task}

LATEST TEST RESULTS:
{test_results}

EVALUATION:
- Critical Errors: {', '.join(evaluation.get('critical_errors', []))}
- Priority Fix: {evaluation.get('improvement_priority', 'Unknown')}
- Improvements Needed: {', '.join(evaluation.get('improvements_needed', []))}

Please fix the issues identified above. Focus on:
1. {evaluation.get('improvement_priority', 'Making tests pass')}
2. Ensuring all tests pass
3. Maintaining code quality and style

Generate the necessary code changes to address these issues.
"""
        return prompt