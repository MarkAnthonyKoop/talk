#!/usr/bin/env python3
"""
AssessorAgent - Analyzes task complexity and determines optimal workflow.

This agent is the first line of defense in Talk's dynamic orchestration.
It analyzes incoming tasks and determines:
- Task complexity (simple, moderate, complex, epic)
- Required agents and their order
- Estimated time and resources
- Success criteria
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

from agent.agent import Agent

log = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels that determine workflow selection."""
    SIMPLE = "simple"        # Single command, direct execution
    MODERATE = "moderate"    # Standard workflow, 2-5 steps
    COMPLEX = "complex"      # Multi-phase workflow, 6-20 steps
    EPIC = "epic"           # Major system build, 20+ steps


class TaskDomain(Enum):
    """Task domains that influence agent selection."""
    FILESYSTEM = "filesystem"
    CODE_GENERATION = "code_generation"
    SYSTEM_DESIGN = "system_design"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"


class AssessorAgent(Agent):
    """
    Analyzes tasks and determines optimal execution strategy.
    
    This agent uses both pattern matching and LLM intelligence to:
    1. Classify task complexity
    2. Identify required domains
    3. Recommend agent composition
    4. Estimate resource requirements
    """
    
    def __init__(self, **kwargs):
        """Initialize the AssessorAgent."""
        roles = [
            "You are an expert task assessor for an advanced orchestration system.",
            "Your job is to analyze tasks and determine the optimal execution strategy.",
            "You must classify tasks by complexity and identify required agents.",
            "Always err on the side of using more sophisticated workflows for ambiguous tasks."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Pattern mappings for quick classification
        self.simple_patterns = [
            r"^(ls|list|show)\s+(files?|dir|directory)",
            r"^(cat|read|show)\s+[\w./]+",
            r"^(mkdir|create\s+directory)",
            r"^(rm|delete|remove)\s+[\w./]+",
            r"^run\s+[\w\s]+",
            r"^execute\s+[\w\s]+",
        ]
        
        self.complex_patterns = [
            r"build\s+(a|an)?\s*(system|framework|application|platform)",
            r"create\s+(a|an)?\s*(complete|full|entire)",
            r"implement\s+(a|an)?\s*(comprehensive|full-featured)",
            r"design\s+and\s+implement",
            r"architect\s+(a|an)?\s*",
        ]
        
        self.epic_patterns = [
            r"production[- ]ready",
            r"enterprise[- ]grade",
            r"scalable\s+.*(system|platform)",
            r"microservice",
            r"kubernetes|k8s",
            r"full[- ]stack",
        ]
    
    def assess_task(self, task: str) -> Dict[str, any]:
        """
        Assess a task and return execution strategy.
        
        Args:
            task: The task description
            
        Returns:
            Dictionary containing:
            - complexity: TaskComplexity enum
            - domains: List of TaskDomain enums
            - recommended_agents: List of agent names in order
            - estimated_steps: Number of execution steps
            - requires_research: Boolean
            - requires_planning: Boolean
            - success_criteria: List of success conditions
        """
        # Quick pattern-based assessment
        complexity = self._pattern_assess(task)
        
        # LLM-based deep analysis
        analysis = self._llm_assess(task, complexity)
        
        # Combine results
        return self._combine_assessments(complexity, analysis)
    
    def _pattern_assess(self, task: str) -> TaskComplexity:
        """Quick pattern-based complexity assessment."""
        task_lower = task.lower()
        
        # Check for epic patterns first
        for pattern in self.epic_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.EPIC
        
        # Check for complex patterns
        for pattern in self.complex_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.COMPLEX
        
        # Check for simple patterns
        for pattern in self.simple_patterns:
            if re.match(pattern, task_lower):
                return TaskComplexity.SIMPLE
        
        # Default to moderate
        return TaskComplexity.MODERATE
    
    def _llm_assess(self, task: str, initial_complexity: TaskComplexity) -> Dict:
        """Use LLM for deep task analysis."""
        prompt = f"""Analyze this task and provide a detailed assessment:

Task: {task}
Initial Complexity Assessment: {initial_complexity.value}

Provide your analysis in the following format:

COMPLEXITY: [simple|moderate|complex|epic]
DOMAINS: [comma-separated list from: filesystem, code_generation, system_design, data_analysis, research, testing, deployment, documentation]
REQUIRES_RESEARCH: [yes|no]
REQUIRES_PLANNING: [yes|no]
ESTIMATED_STEPS: [number]
KEY_CHALLENGES: [bullet points]
SUCCESS_CRITERIA: [bullet points]
RECOMMENDED_AGENTS: [ordered list of agents needed]

Consider:
- Does this require multiple iterations?
- Does it need architecture/design phase?
- Does it require external research?
- What quality standards should apply?
- What agents would produce the best result?
"""
        
        response = self.run(prompt)
        return self._parse_llm_response(response)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse structured response from LLM."""
        result = {
            "domains": [],
            "requires_research": False,
            "requires_planning": False,
            "estimated_steps": 5,
            "key_challenges": [],
            "success_criteria": [],
            "recommended_agents": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("COMPLEXITY:"):
                complexity_str = line.split(":", 1)[1].strip().lower()
                # Parsed separately
            elif line.startswith("DOMAINS:"):
                domains_str = line.split(":", 1)[1].strip()
                result["domains"] = [d.strip() for d in domains_str.split(",")]
            elif line.startswith("REQUIRES_RESEARCH:"):
                result["requires_research"] = "yes" in line.lower()
            elif line.startswith("REQUIRES_PLANNING:"):
                result["requires_planning"] = "yes" in line.lower()
            elif line.startswith("ESTIMATED_STEPS:"):
                try:
                    result["estimated_steps"] = int(line.split(":", 1)[1].strip())
                except:
                    result["estimated_steps"] = 10
            elif line.startswith("KEY_CHALLENGES:"):
                current_section = "challenges"
            elif line.startswith("SUCCESS_CRITERIA:"):
                current_section = "criteria"
            elif line.startswith("RECOMMENDED_AGENTS:"):
                current_section = "agents"
            elif line.startswith("-") or line.startswith("•"):
                item = line.lstrip("-•").strip()
                if current_section == "challenges":
                    result["key_challenges"].append(item)
                elif current_section == "criteria":
                    result["success_criteria"].append(item)
                elif current_section == "agents":
                    result["recommended_agents"].append(item)
        
        return result
    
    def _combine_assessments(self, pattern_complexity: TaskComplexity, 
                            llm_analysis: Dict) -> Dict:
        """Combine pattern and LLM assessments into final strategy."""
        # Convert domains to enums
        domains = []
        for domain_str in llm_analysis.get("domains", []):
            try:
                domains.append(TaskDomain(domain_str.lower()))
            except ValueError:
                log.warning(f"Unknown domain: {domain_str}")
        
        # Build final assessment
        assessment = {
            "complexity": pattern_complexity,
            "domains": domains,
            "recommended_agents": llm_analysis.get("recommended_agents", []),
            "estimated_steps": llm_analysis.get("estimated_steps", 5),
            "requires_research": llm_analysis.get("requires_research", False),
            "requires_planning": llm_analysis.get("requires_planning", False),
            "success_criteria": llm_analysis.get("success_criteria", []),
            "key_challenges": llm_analysis.get("key_challenges", [])
        }
        
        # Adjust based on complexity
        if pattern_complexity == TaskComplexity.EPIC:
            assessment["estimated_steps"] = max(assessment["estimated_steps"], 20)
            assessment["requires_planning"] = True
            assessment["requires_research"] = True
        elif pattern_complexity == TaskComplexity.COMPLEX:
            assessment["estimated_steps"] = max(assessment["estimated_steps"], 10)
            assessment["requires_planning"] = True
        
        return assessment
    
    def recommend_workflow(self, assessment: Dict) -> List[str]:
        """
        Recommend a workflow based on assessment.
        
        Returns:
            List of agent names in execution order
        """
        workflow = []
        complexity = assessment["complexity"]
        
        # Always start with assessment (this agent)
        workflow.append("assessor")
        
        # Add research phase if needed
        if assessment["requires_research"]:
            workflow.append("web_search")
            workflow.append("research_synthesizer")
        
        # Add planning phase if needed
        if assessment["requires_planning"]:
            workflow.append("planner")
            workflow.append("architect")
        
        # Core execution based on complexity
        if complexity == TaskComplexity.SIMPLE:
            # Direct execution
            if TaskDomain.FILESYSTEM in assessment["domains"]:
                workflow.append("shell")
            else:
                workflow.append("code")
        else:
            # Multi-phase execution
            workflow.extend(["code", "file", "test"])
            
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.EPIC]:
                # Add quality loops
                workflow.extend(["critic", "refine", "test"])
                
                if complexity == TaskComplexity.EPIC:
                    # Add advanced phases
                    workflow.extend([
                        "benchmark",
                        "optimize", 
                        "security_audit",
                        "documentation",
                        "deployment"
                    ])
        
        # Always end with verification
        workflow.append("verifier")
        
        return workflow