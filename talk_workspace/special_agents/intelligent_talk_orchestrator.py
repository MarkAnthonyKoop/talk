#!/usr/bin/env python3
"""
IntelligentTalkOrchestrator - Enhanced orchestrator with dynamic planning capabilities.

This enhanced version of the Talk orchestrator integrates the PlanningAgent
to create custom execution plans based on task analysis, rather than using
a fixed workflow. It provides more intelligent and adaptive behavior for
complex software development tasks.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import base Talk components
from talk.talk import TalkOrchestrator
from plan_runner.step import Step

# Import specialized agents
from special_agents.planning_agent import PlanningAgent
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

log = logging.getLogger("intelligent_talk")

class IntelligentTalkOrchestrator(TalkOrchestrator):
    """
    Enhanced Talk orchestrator with intelligent planning capabilities.
    
    This orchestrator uses the PlanningAgent to analyze tasks and generate
    custom execution plans rather than using a fixed workflow. It integrates
    memory capabilities through the ReminiscingAgent for contextual awareness.
    """
    
    def __init__(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout_minutes: int = 30,
        interactive: bool = False,
        resume_session: Optional[str] = None,
        enable_web_search: bool = True,
        enable_planning: bool = True,
        enable_memory: bool = True
    ):
        """
        Initialize the intelligent orchestrator.
        
        Args:
            task: The code generation task description
            working_dir: Directory where code changes will be applied
            model: LLM model to use for agents
            timeout_minutes: Maximum runtime in minutes
            interactive: Whether to run in interactive mode
            resume_session: Path to previous session directory to resume
            enable_web_search: Whether to enable web search for research
            enable_planning: Whether to use intelligent planning (vs fixed workflow)
            enable_memory: Whether to enable memory/reminiscing capabilities
        """
        # Set attributes before calling super().__init__
        self.enable_planning = enable_planning
        self.enable_memory = enable_memory
        self.generated_plan = None
        self.plan_analysis = None
        
        # Initialize base TalkOrchestrator
        super().__init__(
            task=task,
            working_dir=working_dir,
            model=model,
            timeout_minutes=timeout_minutes,
            interactive=interactive,
            resume_session=resume_session,
            enable_web_search=enable_web_search
        )
        
        # Initialize intelligent agents
        if self.enable_planning:
            log.info("Initializing PlanningAgent...")
            self.planning_agent = PlanningAgent()
        
        if self.enable_memory:
            log.info("Initializing ReminiscingAgent...")
            self.reminiscing_agent = ReminiscingAgent()
            # Add to agent pool
            self.agents["reminiscing"] = self.reminiscing_agent
    
    def _create_plan(self) -> List[Step]:
        """
        Create execution plan using PlanningAgent if enabled, otherwise use base logic.
        
        Returns:
            List of Step objects defining the workflow
        """
        if not self.enable_planning:
            log.info("Using standard fixed workflow")
            return super()._create_plan()
        
        log.info("Generating intelligent execution plan...")
        
        try:
            # Ensure PlanningAgent is initialized
            if not hasattr(self, 'planning_agent'):
                log.info("Initializing PlanningAgent...")
                self.planning_agent = PlanningAgent()
            
            # Use PlanningAgent to analyze task and generate plan
            plan_response = self.planning_agent.run(self.task)
            self.generated_plan = json.loads(plan_response)
            self.plan_analysis = self.generated_plan.get("analysis", {})
            
            # Log plan details
            log.info(f"Generated plan type: {self.generated_plan.get('plan_type', 'unknown')}")
            log.info(f"Estimated complexity: {self.generated_plan.get('estimated_complexity', 'unknown')}")
            log.info(f"Total steps: {self.generated_plan.get('total_steps', 0)}")
            
            # Store plan in blackboard for reference
            self.blackboard.add_sync(
                label="execution_plan",
                content=plan_response,
                section="planning",
                role="system"
            )
            
            # Convert plan to Step objects
            steps = self.planning_agent.create_steps_from_plan(plan_response)
            
            if not steps:
                log.warning("PlanningAgent returned empty plan, falling back to standard workflow")
                return super()._create_plan()
            
            # Validate that all required agents are available
            missing_agents = []
            for step in steps:
                if step.agent_key and step.agent_key not in self.agents:
                    missing_agents.append(step.agent_key)
            
            if missing_agents:
                log.warning(f"Missing agents for plan: {missing_agents}, falling back to standard workflow")
                return super()._create_plan()
            
            log.info(f"Successfully generated {len(steps)} execution steps")
            return steps
            
        except Exception as e:
            log.error(f"Error generating intelligent plan: {e}")
            log.info("Falling back to standard workflow")
            return super()._create_plan()
    
    def _prepare_initial_prompt(self) -> str:
        """
        Prepare enhanced initial prompt with planning context.
        
        Returns:
            Enhanced prompt string with plan and memory context
        """
        base_prompt = super()._prepare_initial_prompt()
        
        # Add planning context if available
        if self.generated_plan and self.plan_analysis:
            plan_context = self._format_plan_context()
            base_prompt += f"\n\n{plan_context}"
        
        # Add memory context if enabled
        if self.enable_memory:
            memory_context = self._get_memory_context()
            if memory_context:
                base_prompt += f"\n\n{memory_context}"
        
        return base_prompt
    
    def _format_plan_context(self) -> str:
        """Format planning context for agent prompts."""
        if not self.generated_plan:
            return ""
        
        analysis = self.plan_analysis or {}
        
        context_parts = [
            "=== EXECUTION PLAN CONTEXT ===",
            f"Task Complexity: {analysis.get('complexity', 'unknown').title()}",
            f"Project Type: {analysis.get('type', 'unknown').title()}",
        ]
        
        if analysis.get("components"):
            context_parts.append(f"Required Components: {', '.join(analysis['components'])}")
        
        if analysis.get("technologies"):
            context_parts.append(f"Technologies: {', '.join(analysis['technologies'])}")
        
        if analysis.get("risk_factors"):
            context_parts.append(f"Risk Factors: {', '.join(analysis['risk_factors'])}")
        
        success_criteria = self.generated_plan.get("success_criteria", [])
        if success_criteria:
            context_parts.append("Success Criteria:")
            for criterion in success_criteria:
                context_parts.append(f"  - {criterion}")
        
        context_parts.append("=== END PLAN CONTEXT ===")
        
        return "\n".join(context_parts)
    
    def _get_memory_context(self) -> str:
        """Get relevant memory context for the current task."""
        if not self.enable_memory or not hasattr(self, 'reminiscing_agent'):
            return ""
        
        try:
            # Use ReminiscingAgent to find relevant memories
            memory_response = self.reminiscing_agent.run(self.task)
            
            # Parse memory response to extract key insights
            if "No relevant memory traces" in memory_response:
                return ""
            
            return f"""
=== RELEVANT MEMORY CONTEXT ===
{memory_response}
=== END MEMORY CONTEXT ===
"""
        except Exception as e:
            log.warning(f"Error retrieving memory context: {e}")
            return ""
    
    def run(self) -> bool:
        """
        Execute the intelligent workflow with enhanced logging and monitoring.
        
        Returns:
            True if execution completed successfully, False otherwise
        """
        log.info("Starting intelligent Talk orchestration...")
        
        # Store session metadata with planning info
        self._store_session_metadata()
        
        # Execute the workflow using base implementation
        success = super().run()
        
        # Store final results with intelligent analysis
        self._store_final_analysis(success)
        
        return success
    
    def _store_session_metadata(self):
        """Store enhanced session metadata including planning information."""
        metadata = {
            "session_type": "intelligent_talk",
            "planning_enabled": self.enable_planning,
            "memory_enabled": self.enable_memory,
            "original_task": self.task,
            "model": getattr(self, 'model', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.generated_plan:
            metadata["generated_plan"] = {
                "plan_type": self.generated_plan.get("plan_type"),
                "complexity": self.generated_plan.get("estimated_complexity"),
                "total_steps": self.generated_plan.get("total_steps"),
                "research_required": self.generated_plan.get("research_required"),
                "testing_strategy": self.generated_plan.get("testing_strategy")
            }
        
        self.blackboard.add_sync(
            label="session_metadata",
            content=json.dumps(metadata, indent=2),
            section="system",
            role="system"
        )
    
    def _store_final_analysis(self, success: bool):
        """Store final execution analysis."""
        analysis = {
            "execution_success": success,
            "completion_time": datetime.now().isoformat(),
            "total_blackboard_entries": len(self.blackboard.entries)
        }
        
        if self.generated_plan:
            # Compare planned vs actual execution
            planned_steps = self.generated_plan.get("total_steps", 0)
            actual_entries = len([e for e in self.blackboard.entries if e.author == "system"])
            
            analysis["plan_execution"] = {
                "planned_steps": planned_steps,
                "actual_entries": actual_entries,
                "plan_followed": abs(planned_steps - actual_entries) <= 2  # Allow some variance
            }
            
            # Check if success criteria were met
            success_criteria = self.generated_plan.get("success_criteria", [])
            analysis["success_criteria_count"] = len(success_criteria)
            
            # TODO: Implement automated success criteria checking
            # For now, rely on overall execution success
            analysis["success_criteria_met"] = success
        
        self.blackboard.add_sync(
            label="final_analysis",
            content=json.dumps(analysis, indent=2),
            section="system",
            role="system"
        )
        
        log.info(f"Execution completed with success: {success}")
        if self.generated_plan:
            log.info(f"Plan execution analysis: {analysis.get('plan_execution', {})}")
    
    def get_plan_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the generated execution plan."""
        if not self.generated_plan:
            return None
        
        return {
            "plan_type": self.generated_plan.get("plan_type"),
            "complexity": self.generated_plan.get("estimated_complexity"),
            "total_steps": self.generated_plan.get("total_steps"),
            "research_required": self.generated_plan.get("research_required"),
            "testing_strategy": self.generated_plan.get("testing_strategy"),
            "success_criteria": self.generated_plan.get("success_criteria", []),
            "risk_mitigation": self.generated_plan.get("risk_mitigation", [])
        }
    
    def print_plan_summary(self):
        """Print a human-readable summary of the execution plan."""
        summary = self.get_plan_summary()
        if not summary:
            print("No intelligent plan was generated.")
            return
        
        print("\n=== INTELLIGENT EXECUTION PLAN ===")
        print(f"Plan Type: {summary['plan_type']}")
        print(f"Complexity: {summary['complexity'].title()}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Research Required: {'Yes' if summary['research_required'] else 'No'}")
        print(f"Testing Strategy: {summary['testing_strategy'].title()}")
        
        if summary['success_criteria']:
            print("\nSuccess Criteria:")
            for i, criterion in enumerate(summary['success_criteria'], 1):
                print(f"  {i}. {criterion}")
        
        if summary['risk_mitigation']:
            print("\nRisk Mitigation:")
            for i, mitigation in enumerate(summary['risk_mitigation'], 1):
                print(f"  {i}. {mitigation}")
        
        print("=" * 35)


def create_intelligent_talk(
    task: str,
    working_dir: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    **kwargs
) -> IntelligentTalkOrchestrator:
    """
    Factory function to create an IntelligentTalkOrchestrator.
    
    Args:
        task: The development task to execute
        working_dir: Working directory for the project
        model: LLM model to use
        **kwargs: Additional arguments for the orchestrator
        
    Returns:
        Configured IntelligentTalkOrchestrator instance
    """
    return IntelligentTalkOrchestrator(
        task=task,
        working_dir=working_dir,
        model=model,
        **kwargs
    )