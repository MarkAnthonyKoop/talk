#!/usr/bin/env python3
"""
Enhanced WorkflowSelector with iterative quality loops and completion verification.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from plan_runner.step import Step
from special_agents.assessor_agent import TaskComplexity, TaskDomain

log = logging.getLogger(__name__)


class EnhancedWorkflowSelector:
    """
    Advanced workflow selector with quality loops and completion verification.
    """
    
    def __init__(self):
        """Initialize with enhanced templates and agent mappings."""
        self.agent_registry = self._build_agent_registry()
    
    def select_workflow(self, assessment: Dict) -> List[Step]:
        """Generate enhanced workflow with quality loops."""
        complexity = assessment["complexity"]
        
        if complexity == TaskComplexity.SIMPLE:
            return self._simple_workflow(assessment)
        elif complexity == TaskComplexity.MODERATE:
            return self._iterative_workflow(assessment)
        elif complexity == TaskComplexity.COMPLEX:
            return self._multi_phase_workflow(assessment)
        else:  # EPIC
            return self._epic_orchestration_workflow(assessment)
    
    def _simple_workflow(self, assessment: Dict) -> List[Step]:
        """Simple workflow with basic verification."""
        return [
            Step(label="execute_task", agent_key="shell", on_success="verify_completion"),
            Step(label="verify_completion", agent_key="completion_verifier")
        ]
    
    def _iterative_workflow(self, assessment: Dict) -> List[Step]:
        """Moderate workflow with quality loops."""
        steps = []
        
        # Phase 1: Analysis and Planning
        if assessment.get("requires_research"):
            steps.extend([
                Step(label="research_domain", agent_key="researcher", on_success="analyze_requirements"),
                Step(label="analyze_requirements", agent_key="task_analyzer", on_success="plan_execution")
            ])
        
        steps.append(Step(
            label="plan_execution", 
            agent_key="execution_planner", 
            on_success="implement_solution"
        ))
        
        # Phase 2: Implementation with Quality Loop
        steps.extend([
            Step(label="implement_solution", agent_key="coder", on_success="apply_changes"),
            Step(label="apply_changes", agent_key="file", on_success="test_solution"),
            Step(label="test_solution", agent_key="tester", on_success="verify_iteration"),
            
            # Quality Gate
            Step(label="verify_iteration", agent_key="completion_verifier", on_success="finalize"),
            
            # Refinement Loop (if needed)
            Step(label="critique_solution", agent_key="critic", on_success="refine_solution"),
            Step(label="refine_solution", agent_key="coder", on_success="apply_changes"),
            
            # Final Steps
            Step(label="generate_docs", agent_key="documenter", on_success="final_verification"),
            Step(label="final_verification", agent_key="completion_verifier")
        ])
        
        return steps
    
    def _multi_phase_workflow(self, assessment: Dict) -> List[Step]:
        """Complex workflow with multiple quality gates."""
        steps = []
        
        # Phase 1: Deep Analysis
        steps.extend([
            Step(
                label="comprehensive_analysis",
                agent_key="task_analyzer",
                parallel_steps=[
                    Step(label="research_patterns", agent_key="researcher"),
                    Step(label="analyze_complexity", agent_key="assessor"),
                    Step(label="identify_requirements", agent_key="task_analyzer")
                ],
                on_success="synthesize_analysis"
            ),
            Step(label="synthesize_analysis", agent_key="execution_planner", on_success="design_architecture")
        ])
        
        # Phase 2: Architecture and Design
        steps.extend([
            Step(
                label="design_architecture",
                agent_key="architect",
                parallel_steps=[
                    Step(label="design_components", agent_key="architect"),
                    Step(label="design_interfaces", agent_key="architect"),
                    Step(label="plan_testing", agent_key="tester")
                ],
                on_success="review_design"
            ),
            Step(label="review_design", agent_key="critic", on_success="implement_core")
        ])
        
        # Phase 3: Iterative Implementation
        for iteration in range(3):  # Up to 3 improvement cycles
            iteration_suffix = f"_iter{iteration}"
            
            steps.extend([
                Step(
                    label=f"implement_features{iteration_suffix}",
                    agent_key="coder",
                    on_success=f"apply_implementation{iteration_suffix}"
                ),
                Step(
                    label=f"apply_implementation{iteration_suffix}",
                    agent_key="file",
                    on_success=f"test_implementation{iteration_suffix}"
                ),
                Step(
                    label=f"test_implementation{iteration_suffix}",
                    agent_key="tester",
                    on_success=f"analyze_quality{iteration_suffix}"
                ),
                Step(
                    label=f"analyze_quality{iteration_suffix}",
                    agent_key="metrics",
                    on_success=f"verify_iteration{iteration_suffix}"
                ),
                Step(
                    label=f"verify_iteration{iteration_suffix}",
                    agent_key="completion_verifier",
                    on_success=f"critique_iteration{iteration_suffix}"
                ),
                Step(
                    label=f"critique_iteration{iteration_suffix}",
                    agent_key="critic",
                    on_success=f"refine_implementation{iteration_suffix}" if iteration < 2 else "finalize_implementation"
                )
            ])
            
            if iteration < 2:  # Add refinement step for non-final iterations
                steps.append(Step(
                    label=f"refine_implementation{iteration_suffix}",
                    agent_key="coder",
                    on_success=f"implement_features_iter{iteration+1}"
                ))
        
        # Phase 4: Finalization
        steps.extend([
            Step(label="finalize_implementation", agent_key="coder", on_success="generate_documentation"),
            Step(label="generate_documentation", agent_key="documenter", on_success="create_examples"),
            Step(label="create_examples", agent_key="coder", on_success="final_verification"),
            Step(label="final_verification", agent_key="completion_verifier")
        ])
        
        return steps
    
    def _epic_orchestration_workflow(self, assessment: Dict) -> List[Step]:
        """Epic workflow with massive parallel execution."""
        steps = []
        
        # Phase 1: Strategic Intelligence Gathering (Parallel)
        steps.append(Step(
            label="strategic_intelligence",
            agent_key="task_analyzer",
            parallel_steps=[
                Step(label="domain_research", agent_key="researcher"),
                Step(label="technology_research", agent_key="researcher"),
                Step(label="competition_analysis", agent_key="researcher"),
                Step(label="best_practices_research", agent_key="researcher"),
                Step(label="security_research", agent_key="researcher")
            ],
            on_success="synthesize_intelligence"
        ))
        
        steps.append(Step(
            label="synthesize_intelligence",
            agent_key="execution_planner",
            on_success="master_architecture"
        ))
        
        # Phase 2: Master Architecture Design (Parallel)
        steps.append(Step(
            label="master_architecture",
            agent_key="architect",
            parallel_steps=[
                Step(label="system_architecture", agent_key="architect"),
                Step(label="component_design", agent_key="architect"),
                Step(label="api_design", agent_key="architect"),
                Step(label="data_architecture", agent_key="architect"),
                Step(label="security_architecture", agent_key="architect"),
                Step(label="deployment_architecture", agent_key="architect")
            ],
            on_success="review_master_plan"
        ))
        
        steps.append(Step(
            label="review_master_plan",
            agent_key="critic",
            on_success="parallel_implementation"
        ))
        
        # Phase 3: Massive Parallel Implementation
        steps.append(Step(
            label="parallel_implementation",
            agent_key="coder",
            parallel_steps=[
                # Backend Team
                Step(label="core_backend", agent_key="coder"),
                Step(label="api_backend", agent_key="coder"),
                Step(label="data_backend", agent_key="coder"),
                
                # Frontend Team  
                Step(label="ui_frontend", agent_key="coder"),
                Step(label="dashboard_frontend", agent_key="coder"),
                
                # Infrastructure Team
                Step(label="deployment_scripts", agent_key="coder"),
                Step(label="monitoring_setup", agent_key="coder"),
                Step(label="security_implementation", agent_key="coder"),
                
                # Configuration Team
                Step(label="config_management", agent_key="coder"),
                Step(label="environment_setup", agent_key="coder")
            ],
            on_success="integration_phase"
        ))
        
        # Phase 4: Integration and Quality Assurance
        steps.extend([
            Step(label="integration_phase", agent_key="coder", on_success="comprehensive_testing"),
            
            Step(
                label="comprehensive_testing",
                agent_key="tester",
                parallel_steps=[
                    Step(label="unit_testing", agent_key="tester"),
                    Step(label="integration_testing", agent_key="tester"),
                    Step(label="performance_testing", agent_key="tester"),
                    Step(label="security_testing", agent_key="tester"),
                    Step(label="load_testing", agent_key="tester")
                ],
                on_success="quality_analysis"
            ),
            
            Step(
                label="quality_analysis",
                agent_key="metrics",
                parallel_steps=[
                    Step(label="code_quality_metrics", agent_key="metrics"),
                    Step(label="performance_metrics", agent_key="metrics"),
                    Step(label="security_metrics", agent_key="metrics")
                ],
                on_success="quality_review"
            ),
            
            Step(label="quality_review", agent_key="critic", on_success="optimization_phase")
        ])
        
        # Phase 5: Optimization and Polish
        steps.append(Step(
            label="optimization_phase",
            agent_key="coder",
            parallel_steps=[
                Step(label="performance_optimization", agent_key="coder"),
                Step(label="security_hardening", agent_key="coder"),
                Step(label="ui_polish", agent_key="coder"),
                Step(label="error_handling", agent_key="coder")
            ],
            on_success="documentation_phase"
        ))
        
        # Phase 6: Comprehensive Documentation
        steps.append(Step(
            label="documentation_phase",
            agent_key="documenter",
            parallel_steps=[
                Step(label="api_documentation", agent_key="documenter"),
                Step(label="user_documentation", agent_key="documenter"),
                Step(label="developer_documentation", agent_key="documenter"),
                Step(label="deployment_documentation", agent_key="documenter"),
                Step(label="examples_and_tutorials", agent_key="documenter")
            ],
            on_success="deployment_preparation"
        ))
        
        # Phase 7: Deployment and Validation
        steps.extend([
            Step(label="deployment_preparation", agent_key="coder", on_success="final_testing"),
            Step(label="final_testing", agent_key="tester", on_success="epic_verification"),
            Step(label="epic_verification", agent_key="completion_verifier")
        ])
        
        return steps
    
    def _build_agent_registry(self) -> Dict[str, Dict]:
        """Build registry of available agents and their capabilities."""
        return {
            "assessor": {
                "class": "AssessorAgent",
                "capabilities": ["complexity_analysis", "resource_estimation"],
                "specialties": ["task_analysis", "planning"]
            },
            "task_analyzer": {
                "class": "TaskAnalysisAgent", 
                "capabilities": ["requirement_analysis", "risk_assessment"],
                "specialties": ["planning", "decomposition"]
            },
            "execution_planner": {
                "class": "ExecutionPlannerAgent",
                "capabilities": ["workflow_generation", "step_optimization"],
                "specialties": ["orchestration", "planning"]
            },
            "completion_verifier": {
                "class": "CompletionVerifierAgent",
                "capabilities": ["completion_verification", "quality_assessment"],
                "specialties": ["quality_control", "validation"]
            },
            "coder": {
                "class": "CodeAgent",
                "capabilities": ["code_generation", "implementation"],
                "specialties": ["development", "programming"]
            },
            "file": {
                "class": "FileAgent", 
                "capabilities": ["file_operations", "content_management"],
                "specialties": ["file_system", "content"]
            },
            "tester": {
                "class": "TestAgent",
                "capabilities": ["testing", "validation"],
                "specialties": ["quality_assurance", "verification"]
            },
            "metrics": {
                "class": "MetricsAgent",
                "capabilities": ["code_analysis", "metrics_collection"],
                "specialties": ["analysis", "measurement"]
            },
            "researcher": {
                "class": "WebSearchAgent",
                "capabilities": ["research", "information_gathering"],
                "specialties": ["intelligence", "analysis"]
            },
            "architect": {
                "class": "ArchitectAgent",
                "capabilities": ["system_design", "architecture"],
                "specialties": ["design", "planning"]
            },
            "critic": {
                "class": "CriticAgent",
                "capabilities": ["code_review", "quality_assessment"],
                "specialties": ["review", "quality"]
            },
            "documenter": {
                "class": "DocumenterAgent",
                "capabilities": ["documentation", "examples"],
                "specialties": ["writing", "communication"]
            }
        }