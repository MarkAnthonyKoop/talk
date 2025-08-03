#!/usr/bin/env python3
"""
TaskAnalysisAgent - Intelligent task analysis and verbalization.

This agent analyzes incoming tasks and provides detailed analysis of complexity,
requirements, and optimal approaches. It verbalizes execution strategies and
breaks down complex requirements into understandable components for better
project planning and decision making.

The agent supports:
- Task complexity analysis and decomposition
- Requirement verbalization and explanation
- Risk assessment and mitigation strategies
- Technology stack recommendations
- Development approach suggestions
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from agent.agent import Agent
from agent.messages import Message, Role
from plan_runner.step import Step

log = logging.getLogger(__name__)

class TaskAnalysisAgent(Agent):
    """
    Specialized agent for analyzing and verbalizing development tasks.
    
    This agent provides detailed analysis of task requirements, complexity
    assessment, and strategic recommendations for development approaches.
    """
    
    def __init__(self, **kwargs):
        """Initialize with planning-specific capabilities."""
        super().__init__(roles=[
            "You are an expert project planning specialist for software development.",
            "You analyze tasks and create detailed execution plans with optimal step sequences.",
            "You break down complex requirements into manageable, actionable steps.",
            "You understand software development workflows and can design appropriate execution strategies."
        ], **kwargs)
        
        # Define available agent types for plan generation
        self.available_agents = {
            "coder": "Generates code files and implements features",
            "file": "Applies code changes to the filesystem", 
            "tester": "Runs tests and validates implementations",
            "researcher": "Performs web research and gathers information",
            "reminiscing": "Retrieves relevant memories and contextual information"
        }
        
        # Define plan templates for common scenarios
        self.plan_templates = {
            "simple_api": {
                "description": "Basic API or web service implementation",
                "steps": ["generate_code", "apply_changes", "run_tests", "check_results"]
            },
            "complex_system": {
                "description": "Multi-component system with database, API, and frontend",
                "steps": ["research_requirements", "design_architecture", "generate_database", 
                         "generate_api", "generate_frontend", "integrate_components", "run_tests", "check_results"]
            },
            "bug_fix": {
                "description": "Debugging and fixing existing code",
                "steps": ["analyze_issue", "research_solution", "apply_fix", "run_tests", "verify_fix"]
            },
            "feature_addition": {
                "description": "Adding new features to existing codebase",
                "steps": ["understand_codebase", "design_feature", "implement_feature", "integrate_feature", "run_tests", "check_results"]
            },
            "research_project": {
                "description": "Research-heavy project requiring information gathering",
                "steps": ["research_requirements", "gather_resources", "analyze_information", "generate_code", "apply_changes", "run_tests"]
            }
        }
        
        # Complexity indicators for task analysis
        self.complexity_indicators = {
            "high": [
                "microservice", "distributed", "scalable", "enterprise", "production-ready",
                "authentication", "authorization", "database", "multiple", "complex",
                "real-time", "notification", "integration", "api gateway", "deployment"
            ],
            "medium": [
                "api", "backend", "frontend", "crud", "validation", "testing",
                "models", "endpoints", "forms", "dashboard", "admin"
            ],
            "low": [
                "hello world", "simple", "basic", "example", "demo", "prototype",
                "single", "minimal", "quick", "small"
            ]
        }
    
    def run(self, input_text: str) -> str:
        """
        Analyze the task and generate a custom execution plan.
        
        Args:
            input_text: Task description to analyze and plan for
            
        Returns:
            JSON-formatted execution plan with steps and metadata
        """
        try:
            # Parse input to extract task details
            task_analysis = self._analyze_task(input_text)
            
            # Generate appropriate execution plan
            execution_plan = self._generate_plan(task_analysis)
            
            # Validate and optimize the plan
            optimized_plan = self._optimize_plan(execution_plan, task_analysis)
            
            # Format the response
            return self._format_plan_response(optimized_plan, task_analysis)
            
        except Exception as e:
            log.error(f"Error generating execution plan: {e}")
            return self._create_fallback_plan(input_text)
    
    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze the task to determine complexity and requirements."""
        # Use LLM to analyze the task
        analysis_prompt = f"""
Analyze this software development task and provide a structured assessment:

TASK: {task}

Please analyze and categorize this task along the following dimensions:

1. COMPLEXITY: [low/medium/high] - Based on technical requirements and scope
2. TYPE: [api, frontend, backend, fullstack, database, testing, research, bugfix, feature] - Primary development type
3. COMPONENTS: List the main technical components needed (e.g., database, authentication, API endpoints, frontend, tests)
4. TECHNOLOGIES: Identify specific technologies mentioned (e.g., FastAPI, React, PostgreSQL, Docker)
5. ESTIMATED_STEPS: How many major development phases would this require? [1-10]
6. RESEARCH_NEEDED: [true/false] - Does this require external research or documentation lookup?
7. TESTING_COMPLEXITY: [simple/medium/complex] - How extensive should testing be?
8. DEPENDENCIES: List any external dependencies or prerequisites
9. RISK_FACTORS: Identify potential challenges or complex aspects

Respond in exactly this format:
COMPLEXITY: [level]
TYPE: [type]  
COMPONENTS: [component1, component2, ...]
TECHNOLOGIES: [tech1, tech2, ...]
ESTIMATED_STEPS: [number]
RESEARCH_NEEDED: [true/false]
TESTING_COMPLEXITY: [level]
DEPENDENCIES: [dep1, dep2, ...]
RISK_FACTORS: [risk1, risk2, ...]
"""
        
        self._append("user", analysis_prompt)
        llm_response = self.call_ai()
        self._append("assistant", llm_response)
        
        # Parse LLM response into structured data
        analysis = self._parse_task_analysis(llm_response)
        
        # Add original task for reference
        analysis["original_task"] = task
        analysis["timestamp"] = datetime.now().isoformat()
        
        # Supplement with pattern-based analysis
        analysis.update(self._pattern_based_analysis(task))
        
        return analysis
    
    def _parse_task_analysis(self, response: str) -> Dict[str, Any]:
        """Parse the LLM analysis response into structured data."""
        analysis = {}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "complexity":
                    analysis["complexity"] = value.lower()
                elif key == "type":
                    analysis["type"] = value.lower()
                elif key == "components":
                    analysis["components"] = [c.strip() for c in value.split(',') if c.strip()]
                elif key == "technologies":
                    analysis["technologies"] = [t.strip() for t in value.split(',') if t.strip()]
                elif key == "estimated_steps":
                    try:
                        analysis["estimated_steps"] = int(value)
                    except ValueError:
                        analysis["estimated_steps"] = 5  # Default
                elif key == "research_needed":
                    analysis["research_needed"] = value.lower() == "true"
                elif key == "testing_complexity":
                    analysis["testing_complexity"] = value.lower()
                elif key == "dependencies":
                    analysis["dependencies"] = [d.strip() for d in value.split(',') if d.strip()]
                elif key == "risk_factors":
                    analysis["risk_factors"] = [r.strip() for r in value.split(',') if r.strip()]
        
        # Set defaults for missing values
        defaults = {
            "complexity": "medium",
            "type": "api",
            "components": [],
            "technologies": [],
            "estimated_steps": 4,
            "research_needed": False,
            "testing_complexity": "medium",
            "dependencies": [],
            "risk_factors": []
        }
        
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
        
        return analysis
    
    def _pattern_based_analysis(self, task: str) -> Dict[str, Any]:
        """Supplement LLM analysis with pattern-based detection."""
        task_lower = task.lower()
        analysis = {}
        
        # Detect complexity based on keywords
        complexity_score = 0
        for level, keywords in self.complexity_indicators.items():
            for keyword in keywords:
                if keyword in task_lower:
                    if level == "high":
                        complexity_score += 3
                    elif level == "medium":
                        complexity_score += 2
                    else:
                        complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 8:
            analysis["pattern_complexity"] = "high"
        elif complexity_score >= 4:
            analysis["pattern_complexity"] = "medium"
        else:
            analysis["pattern_complexity"] = "low"
        
        # Detect if testing is mentioned
        testing_keywords = ["test", "testing", "unit test", "integration test", "pytest"]
        analysis["explicit_testing"] = any(keyword in task_lower for keyword in testing_keywords)
        
        # Detect if research might be needed
        research_keywords = ["best practice", "how to", "research", "learn", "documentation", "examples"]
        analysis["might_need_research"] = any(keyword in task_lower for keyword in research_keywords)
        
        return analysis
    
    def _generate_plan(self, analysis: Dict[str, Any]) -> List[Step]:
        """Generate execution plan based on task analysis."""
        complexity = analysis.get("complexity", "medium")
        task_type = analysis.get("type", "api")
        research_needed = analysis.get("research_needed", False)
        testing_complexity = analysis.get("testing_complexity", "medium")
        
        # Select base template
        template_key = self._select_template(complexity, task_type, analysis)
        template = self.plan_templates.get(template_key, self.plan_templates["simple_api"])
        
        log.info(f"Selected template: {template_key} for task type: {task_type}, complexity: {complexity}")
        
        # Generate steps based on template and analysis
        steps = []
        
        # Add research step if needed
        if research_needed or analysis.get("might_need_research", False):
            research_step = Step(
                label="research_requirements",
                agent_key="researcher",
                on_success="recall_memories"
            )
            steps.append(research_step)
            
            # Add memory retrieval step
            memory_step = Step(
                label="recall_memories", 
                agent_key="reminiscing",
                on_success="generate_code"
            )
            steps.append(memory_step)
        else:
            # Add memory retrieval without research
            memory_step = Step(
                label="recall_memories",
                agent_key="reminiscing", 
                on_success="generate_code"
            )
            steps.append(memory_step)
        
        # Add core development steps
        if complexity == "high" or len(analysis.get("components", [])) > 3:
            # Break down into multiple generation phases
            components = analysis.get("components", [])
            
            if "database" in components or "models" in components:
                db_step = Step(
                    label="generate_database",
                    agent_key="coder",
                    on_success="generate_api"
                )
                steps.append(db_step)
            
            if "api" in components or "backend" in components:
                api_step = Step(
                    label="generate_api",
                    agent_key="coder",
                    on_success="apply_changes"
                )
                steps.append(api_step)
            else:
                # General code generation
                code_step = Step(
                    label="generate_code",
                    agent_key="coder",
                    on_success="apply_changes"
                )
                steps.append(code_step)
        else:
            # Simple single-phase generation
            code_step = Step(
                label="generate_code",
                agent_key="coder",
                on_success="apply_changes"
            )
            steps.append(code_step)
        
        # Apply changes step
        apply_step = Step(
            label="apply_changes",
            agent_key="file",
            on_success="run_tests"
        )
        steps.append(apply_step)
        
        # Testing steps based on complexity
        if testing_complexity == "complex" or analysis.get("explicit_testing", False):
            # Multiple test phases
            unit_test_step = Step(
                label="run_unit_tests",
                agent_key="tester",
                on_success="run_integration_tests"
            )
            steps.append(unit_test_step)
            
            integration_test_step = Step(
                label="run_integration_tests", 
                agent_key="tester",
                on_success="check_results"
            )
            steps.append(integration_test_step)
        else:
            # Standard testing
            test_step = Step(
                label="run_tests",
                agent_key="tester",
                on_success="check_results"
            )
            steps.append(test_step)
        
        # Final validation step
        check_step = Step(
            label="check_results",
            agent_key="coder",
            on_success=None  # End of workflow
        )
        steps.append(check_step)
        
        return steps
    
    def _select_template(self, complexity: str, task_type: str, analysis: Dict[str, Any]) -> str:
        """Select the most appropriate plan template."""
        components = analysis.get("components", [])
        
        # Decision logic for template selection
        if "bug" in task_type or "fix" in task_type:
            return "bug_fix"
        elif analysis.get("research_needed", False) or analysis.get("might_need_research", False):
            return "research_project"
        elif complexity == "high" or len(components) > 3:
            return "complex_system"
        elif "feature" in task_type:
            return "feature_addition"
        else:
            return "simple_api"
    
    def _optimize_plan(self, steps: List[Step], analysis: Dict[str, Any]) -> List[Step]:
        """Optimize the execution plan based on analysis."""
        # Add error handling transitions
        for i, step in enumerate(steps):
            if step.agent_key == "coder" and i < len(steps) - 1:
                # If code generation fails, try research if not already done
                if not any(s.label == "research_requirements" for s in steps):
                    step.on_fail = "research_requirements"
            elif step.agent_key == "tester":
                # If tests fail, go back to code generation
                step.on_fail = "generate_code"
        
        # Add parallel execution opportunities for complex tasks
        if analysis.get("complexity") == "high":
            # Look for steps that could run in parallel
            # For now, keep it simple - future enhancement
            pass
        
        return steps
    
    def _format_plan_response(self, steps: List[Step], analysis: Dict[str, Any]) -> str:
        """Format the execution plan as a structured response."""
        # Convert steps to serializable format
        steps_data = []
        for step in steps:
            step_data = {
                "label": step.label,
                "agent_key": step.agent_key,
                "on_success": step.on_success,
                "on_fail": step.on_fail,
                "steps": [asdict(s) for s in step.steps] if step.steps else [],
                "parallel_steps": [asdict(s) for s in step.parallel_steps] if step.parallel_steps else []
            }
            steps_data.append(step_data)
        
        response = {
            "plan_type": "custom_generated",
            "analysis": analysis,
            "execution_steps": steps_data,
            "total_steps": len(steps),
            "estimated_complexity": analysis.get("complexity", "medium"),
            "research_required": analysis.get("research_needed", False),
            "testing_strategy": analysis.get("testing_complexity", "medium"),
            "success_criteria": self._define_success_criteria(analysis),
            "risk_mitigation": self._define_risk_mitigation(analysis),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent_version": "1.0.0",
                "template_used": self._select_template(
                    analysis.get("complexity", "medium"),
                    analysis.get("type", "api"), 
                    analysis
                )
            }
        }
        
        return json.dumps(response, indent=2)
    
    def _define_success_criteria(self, analysis: Dict[str, Any]) -> List[str]:
        """Define success criteria based on task analysis."""
        criteria = [
            "All code files are generated successfully",
            "Code applies to filesystem without errors", 
            "All tests pass successfully"
        ]
        
        if analysis.get("research_needed", False):
            criteria.insert(0, "Relevant research information is gathered")
        
        if analysis.get("testing_complexity") == "complex":
            criteria.append("Both unit and integration tests pass")
        
        if "api" in analysis.get("components", []):
            criteria.append("API endpoints are functional and accessible")
        
        if "database" in analysis.get("components", []):
            criteria.append("Database models and operations work correctly")
        
        return criteria
    
    def _define_risk_mitigation(self, analysis: Dict[str, Any]) -> List[str]:
        """Define risk mitigation strategies."""
        mitigations = []
        
        risk_factors = analysis.get("risk_factors", [])
        
        for risk in risk_factors:
            if "complex" in risk.lower():
                mitigations.append("Break down complex tasks into smaller, manageable steps")
            elif "integration" in risk.lower():
                mitigations.append("Test individual components before integration")
            elif "dependency" in risk.lower():
                mitigations.append("Verify all dependencies are available and compatible")
        
        # Default mitigations
        if not mitigations:
            mitigations = [
                "Monitor each step for errors and provide fallback options",
                "Use incremental development approach",
                "Validate intermediate results before proceeding"
            ]
        
        return mitigations
    
    def _create_fallback_plan(self, task: str) -> str:
        """Create a simple fallback plan when analysis fails."""
        log.warning("Creating fallback plan due to analysis error")
        
        fallback_steps = [
            {
                "label": "recall_memories",
                "agent_key": "reminiscing",
                "on_success": "generate_code",
                "on_fail": None,
                "steps": [],
                "parallel_steps": []
            },
            {
                "label": "generate_code", 
                "agent_key": "coder",
                "on_success": "apply_changes",
                "on_fail": None,
                "steps": [],
                "parallel_steps": []
            },
            {
                "label": "apply_changes",
                "agent_key": "file", 
                "on_success": "run_tests",
                "on_fail": None,
                "steps": [],
                "parallel_steps": []
            },
            {
                "label": "run_tests",
                "agent_key": "tester",
                "on_success": "check_results",
                "on_fail": "generate_code",
                "steps": [],
                "parallel_steps": []
            },
            {
                "label": "check_results",
                "agent_key": "coder",
                "on_success": None,
                "on_fail": None,
                "steps": [],
                "parallel_steps": []
            }
        ]
        
        response = {
            "plan_type": "fallback",
            "analysis": {
                "original_task": task,
                "complexity": "unknown",
                "error": "Failed to analyze task, using fallback plan"
            },
            "execution_steps": fallback_steps,
            "total_steps": len(fallback_steps),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent_version": "1.0.0",
                "template_used": "fallback"
            }
        }
        
        return json.dumps(response, indent=2)
    
    def create_steps_from_plan(self, plan_json: str) -> List[Step]:
        """Convert JSON plan back to Step objects for execution."""
        try:
            plan_data = json.loads(plan_json)
            steps = []
            
            for step_data in plan_data.get("execution_steps", []):
                step = Step(
                    label=step_data.get("label"),
                    agent_key=step_data.get("agent_key", ""),
                    on_success=step_data.get("on_success"),
                    on_fail=step_data.get("on_fail")
                )
                steps.append(step)
            
            return steps
        except Exception as e:
            log.error(f"Error converting plan to steps: {e}")
            return []