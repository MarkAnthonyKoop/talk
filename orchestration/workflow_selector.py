#!/usr/bin/env python3
"""
WorkflowSelector - Dynamic workflow generation based on task assessment.

This component takes assessments from AssessorAgent and generates
optimal execution workflows using Talk's available agents.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from plan_runner.step import Step
from special_agents.assessor_agent import TaskComplexity, TaskDomain

log = logging.getLogger(__name__)


@dataclass
class WorkflowTemplate:
    """Template for common workflow patterns."""
    name: str
    description: str
    steps: List[Step]
    min_complexity: TaskComplexity
    required_domains: List[TaskDomain]


class WorkflowSelector:
    """
    Dynamically generates execution workflows based on task assessment.
    
    This selector understands Talk's agent capabilities and creates
    optimal execution plans that leverage agent specializations.
    """
    
    def __init__(self):
        """Initialize the WorkflowSelector with templates."""
        self.templates = self._initialize_templates()
        self.agent_capabilities = self._map_agent_capabilities()
    
    def select_workflow(self, assessment: Dict) -> List[Step]:
        """
        Generate workflow based on task assessment.
        
        Args:
            assessment: Task assessment from AssessorAgent
            
        Returns:
            List of Step objects for execution
        """
        complexity = assessment["complexity"]
        domains = assessment["domains"]
        
        # Route based on complexity
        if complexity == TaskComplexity.SIMPLE:
            return self._simple_workflow(assessment)
        elif complexity == TaskComplexity.MODERATE:
            return self._moderate_workflow(assessment)
        elif complexity == TaskComplexity.COMPLEX:
            return self._complex_workflow(assessment)
        else:  # EPIC
            return self._epic_workflow(assessment)
    
    def _simple_workflow(self, assessment: Dict) -> List[Step]:
        """Generate workflow for simple tasks."""
        steps = []
        
        # Determine primary agent
        if TaskDomain.FILESYSTEM in assessment["domains"]:
            steps.append(Step(
                label="execute_command",
                agent_key="shell",
                on_success="verify_result"
            ))
        elif TaskDomain.CODE_GENERATION in assessment["domains"]:
            steps.append(Step(
                label="generate_code",
                agent_key="coder",
                on_success="apply_changes"
            ))
            steps.append(Step(
                label="apply_changes",
                agent_key="file",
                on_success="verify_result"
            ))
        else:
            # Default to shell for simple tasks
            steps.append(Step(
                label="execute_task",
                agent_key="shell",
                on_success="verify_result"
            ))
        
        # Always verify
        steps.append(Step(
            label="verify_result",
            agent_key="verifier"
        ))
        
        return steps
    
    def _moderate_workflow(self, assessment: Dict) -> List[Step]:
        """Generate workflow for moderate complexity tasks."""
        steps = []
        
        # Research phase if needed
        if assessment["requires_research"]:
            steps.append(Step(
                label="research_task",
                agent_key="researcher",
                on_success="analyze_research"
            ))
            steps.append(Step(
                label="analyze_research", 
                agent_key="analyzer",
                on_success="generate_code"
            ))
            
        # Standard code generation workflow
        steps.extend([
            Step(
                label="generate_code",
                agent_key="coder",
                on_success="apply_changes"
            ),
            Step(
                label="apply_changes",
                agent_key="file",
                on_success="run_tests"
            ),
            Step(
                label="run_tests",
                agent_key="tester",
                on_success="check_results"
            ),
            Step(
                label="check_results",
                agent_key="checker"
            )
        ])
        
        return steps
    
    def _complex_workflow(self, assessment: Dict) -> List[Step]:
        """Generate workflow for complex tasks."""
        steps = []
        
        # Planning phase
        steps.extend([
            Step(
                label="create_plan",
                agent_key="planner",
                on_success="design_architecture"
            ),
            Step(
                label="design_architecture",
                agent_key="architect",
                on_success="review_design"
            ),
            Step(
                label="review_design",
                agent_key="critic",
                on_success="research_components"
            )
        ])
        
        # Research phase
        if assessment["requires_research"]:
            steps.append(Step(
                label="research_components",
                agent_key="researcher",
                on_success="generate_components"
            ))
        
        # Implementation phase with quality loops
        steps.extend([
            Step(
                label="generate_components",
                agent_key="coder",
                on_success="review_code"
            ),
            Step(
                label="review_code",
                agent_key="critic",
                on_success="refine_code"
            ),
            Step(
                label="refine_code",
                agent_key="refiner",
                on_success="apply_changes"
            ),
            Step(
                label="apply_changes",
                agent_key="file",
                on_success="run_tests"
            ),
            Step(
                label="run_tests",
                agent_key="tester",
                on_success="analyze_results"
            ),
            Step(
                label="analyze_results",
                agent_key="analyzer",
                on_success="generate_documentation"
            ),
            Step(
                label="generate_documentation",
                agent_key="documenter",
                on_success="final_review"
            ),
            Step(
                label="final_review",
                agent_key="critic"
            )
        ])
        
        return steps
    
    def _epic_workflow(self, assessment: Dict) -> List[Step]:
        """Generate workflow for epic-scale tasks."""
        steps = []
        
        # Phase 1: Research and Planning
        steps.extend([
            Step(
                label="deep_research",
                agent_key="researcher",
                parallel_steps=[
                    Step(label="research_patterns", agent_key="pattern_researcher"),
                    Step(label="research_technologies", agent_key="tech_researcher"),
                    Step(label="research_competitors", agent_key="competitor_analyzer")
                ],
                on_success="synthesize_research"
            ),
            Step(
                label="synthesize_research",
                agent_key="synthesizer",
                on_success="create_masterplan"
            ),
            Step(
                label="create_masterplan",
                agent_key="masterplanner",
                on_success="design_system"
            )
        ])
        
        # Phase 2: Architecture and Design
        steps.extend([
            Step(
                label="design_system",
                agent_key="architect",
                parallel_steps=[
                    Step(label="design_components", agent_key="component_designer"),
                    Step(label="design_interfaces", agent_key="api_designer"),
                    Step(label="design_data_model", agent_key="data_architect")
                ],
                on_success="review_architecture"
            ),
            Step(
                label="review_architecture",
                agent_key="senior_architect",
                on_success="create_specifications"
            ),
            Step(
                label="create_specifications",
                agent_key="spec_writer",
                on_success="implement_core"
            )
        ])
        
        # Phase 3: Implementation
        steps.extend([
            Step(
                label="implement_core",
                agent_key="senior_coder",
                parallel_steps=[
                    Step(label="implement_backend", agent_key="backend_coder"),
                    Step(label="implement_frontend", agent_key="frontend_coder"),
                    Step(label="implement_infrastructure", agent_key="infra_coder")
                ],
                on_success="integrate_components"
            ),
            Step(
                label="integrate_components",
                agent_key="integration_specialist",
                on_success="test_system"
            )
        ])
        
        # Phase 4: Quality Assurance
        steps.extend([
            Step(
                label="test_system",
                agent_key="qa_lead",
                parallel_steps=[
                    Step(label="unit_tests", agent_key="unit_tester"),
                    Step(label="integration_tests", agent_key="integration_tester"),
                    Step(label="performance_tests", agent_key="performance_tester"),
                    Step(label="security_tests", agent_key="security_tester")
                ],
                on_success="analyze_quality"
            ),
            Step(
                label="analyze_quality",
                agent_key="quality_analyst",
                on_success="optimize_system"
            )
        ])
        
        # Phase 5: Optimization and Polish
        steps.extend([
            Step(
                label="optimize_system",
                agent_key="optimizer",
                parallel_steps=[
                    Step(label="optimize_performance", agent_key="perf_optimizer"),
                    Step(label="optimize_security", agent_key="security_hardener"),
                    Step(label="optimize_ux", agent_key="ux_optimizer")
                ],
                on_success="document_system"
            )
        ])
        
        # Phase 6: Documentation and Deployment
        steps.extend([
            Step(
                label="document_system",
                agent_key="tech_writer",
                parallel_steps=[
                    Step(label="api_docs", agent_key="api_documenter"),
                    Step(label="user_docs", agent_key="user_doc_writer"),
                    Step(label="ops_docs", agent_key="ops_doc_writer")
                ],
                on_success="prepare_deployment"
            ),
            Step(
                label="prepare_deployment",
                agent_key="devops_engineer",
                on_success="final_validation"
            ),
            Step(
                label="final_validation",
                agent_key="senior_validator",
                on_success="deploy_system"
            ),
            Step(
                label="deploy_system",
                agent_key="deployment_specialist"
            )
        ])
        
        return steps
    
    def _initialize_templates(self) -> List[WorkflowTemplate]:
        """Initialize reusable workflow templates."""
        return [
            WorkflowTemplate(
                name="simple_command",
                description="Direct command execution",
                steps=[
                    Step(label="execute", agent_key="shell"),
                    Step(label="verify", agent_key="verifier")
                ],
                min_complexity=TaskComplexity.SIMPLE,
                required_domains=[TaskDomain.FILESYSTEM]
            ),
            WorkflowTemplate(
                name="code_review_loop",
                description="Code generation with review cycle",
                steps=[
                    Step(label="generate", agent_key="coder"),
                    Step(label="review", agent_key="critic"),
                    Step(label="refine", agent_key="refiner"),
                    Step(label="test", agent_key="tester")
                ],
                min_complexity=TaskComplexity.MODERATE,
                required_domains=[TaskDomain.CODE_GENERATION]
            ),
            WorkflowTemplate(
                name="research_synthesis",
                description="Research and synthesis workflow",
                steps=[
                    Step(label="research", agent_key="researcher"),
                    Step(label="analyze", agent_key="analyzer"),
                    Step(label="synthesize", agent_key="synthesizer")
                ],
                min_complexity=TaskComplexity.MODERATE,
                required_domains=[TaskDomain.RESEARCH]
            )
        ]
    
    def _map_agent_capabilities(self) -> Dict[str, List[str]]:
        """Map agents to their capabilities."""
        return {
            "shell": ["execute_command", "file_operations", "system_info"],
            "coder": ["generate_code", "refactor_code", "fix_bugs"],
            "file": ["write_files", "read_files", "modify_files"],
            "tester": ["run_tests", "validate_output", "check_errors"],
            "researcher": ["web_search", "analyze_docs", "find_patterns"],
            "critic": ["review_code", "assess_quality", "find_issues"],
            "architect": ["design_systems", "create_diagrams", "plan_structure"],
            "planner": ["create_plans", "estimate_effort", "identify_risks"],
            "documenter": ["write_docs", "create_examples", "explain_code"],
            "optimizer": ["improve_performance", "reduce_complexity", "enhance_quality"]
        }