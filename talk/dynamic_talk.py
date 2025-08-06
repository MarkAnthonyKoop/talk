#!/usr/bin/env python3
"""
Dynamic Talk Orchestrator - The beast Talk was meant to be.

This enhanced orchestrator uses intelligent task assessment and dynamic
workflow generation to create optimal execution plans for any task.
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from plan_runner.blackboard import Blackboard
from agent.output_manager import OutputManager
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import all specialized agents
from special_agents.assessor_agent import AssessorAgent, TaskComplexity
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.web_search_agent import WebSearchAgent

# Import new orchestration components
from orchestration.workflow_selector import WorkflowSelector

log = logging.getLogger(__name__)


class DynamicTalkOrchestrator:
    """
    Enhanced Talk orchestrator with dynamic workflow generation.
    
    This orchestrator analyzes tasks intelligently and creates optimal
    execution workflows that leverage Talk's full agent ecosystem.
    """
    
    def __init__(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout_minutes: int = 60,
        interactive: bool = False,
        auto_approve: bool = False
    ):
        """
        Initialize the dynamic orchestrator.
        
        Args:
            task: The task description
            working_dir: Directory for code changes
            model: LLM model to use
            timeout_minutes: Maximum runtime
            interactive: Whether to prompt for approval
            auto_approve: Auto-approve all operations
        """
        self.task = task
        self.model = model
        self.timeout_minutes = timeout_minutes
        self.interactive = interactive
        self.auto_approve = auto_approve
        self.start_time = time.time()
        
        # Set model globally
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output management
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Initialize blackboard for shared state
        self.blackboard = Blackboard()
        
        # Initialize workflow selector
        self.workflow_selector = WorkflowSelector()
        
        # This will be populated dynamically
        self.agents = {}
        self.plan = []
        
        # Setup logging
        self._setup_logging()
        
        log.info(f"Dynamic Talk Orchestrator initialized")
        log.info(f"Task: {self.task}")
        log.info(f"Session: {self.session_dir}")
    
    def run(self) -> int:
        """
        Run the orchestrator with dynamic workflow.
        
        Returns:
            Exit code (0 for success)
        """
        try:
            print(f"\n🚀 [DYNAMIC TALK] Initializing intelligent orchestration...")
            print(f"📋 [TASK] {self.task}")
            print(f"📁 [SESSION] {self.session_dir}")
            print(f"🏗️  [WORKSPACE] {self.working_dir}")
            
            # Phase 1: Task Assessment
            print(f"\n🧠 [ASSESSMENT] Analyzing task complexity...")
            assessment = self._assess_task()
            
            # Phase 2: Workflow Generation
            print(f"\n🔧 [PLANNING] Generating optimal workflow...")
            self.plan = self._generate_workflow(assessment)
            
            # Phase 3: Agent Initialization
            print(f"\n🤖 [AGENTS] Initializing {len(self._get_required_agents())} specialized agents...")
            self._initialize_agents()
            
            # Phase 4: Show execution plan
            if not self.auto_approve:
                self._show_execution_plan(assessment)
                
                if self.interactive:
                    response = input("\n🤔 Proceed with this plan? (y/n): ")
                    if response.lower() != 'y':
                        print("❌ Execution cancelled by user")
                        return 1
            
            # Phase 5: Execute workflow
            print(f"\n⚡ [EXECUTION] Starting dynamic workflow execution...")
            result = self._execute_workflow()
            
            # Phase 6: Summary
            self._show_summary()
            
            print(f"\n✅ [SUCCESS] Task completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            print("\n\n⚠️  [INTERRUPTED] Execution cancelled by user")
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n❌ [ERROR] {str(e)}")
            return 1
        finally:
            self._save_session()
    
    def _assess_task(self) -> Dict:
        """Assess task complexity using AssessorAgent."""
        assessor = AssessorAgent(
            name="TaskAssessor",
            overrides={"provider": {"anthropic": {"model_name": self.model}}}
        )
        
        assessment = assessor.assess_task(self.task)
        
        # Log assessment results
        log.info(f"Task complexity: {assessment['complexity'].value}")
        log.info(f"Domains: {[d.value for d in assessment['domains']]}")
        log.info(f"Estimated steps: {assessment['estimated_steps']}")
        
        # Store in blackboard
        self.blackboard.add("task_assessment", json.dumps({
            "complexity": assessment["complexity"].value,
            "domains": [d.value for d in assessment["domains"]],
            "estimated_steps": assessment["estimated_steps"],
            "requires_research": assessment["requires_research"],
            "requires_planning": assessment["requires_planning"]
        }))
        
        return assessment
    
    def _generate_workflow(self, assessment: Dict) -> List[Step]:
        """Generate dynamic workflow based on assessment."""
        workflow = self.workflow_selector.select_workflow(assessment)
        
        log.info(f"Generated workflow with {len(workflow)} steps")
        for step in workflow:
            log.info(f"  - {step.label}: {step.agent_key}")
        
        return workflow
    
    def _get_required_agents(self) -> List[str]:
        """Get list of required agent keys from plan."""
        agents = set()
        for step in self.plan:
            agents.add(step.agent_key)
            # Add agents from parallel steps
            for parallel_step in step.parallel_steps:
                agents.add(parallel_step.agent_key)
        return list(agents)
    
    def _initialize_agents(self):
        """Initialize only the agents needed for the workflow."""
        agent_classes = {
            "assessor": AssessorAgent,
            "coder": CodeAgent,
            "file": FileAgent,
            "tester": TestAgent,
            "shell": None,  # ShellAgent has import issues
            "researcher": WebSearchAgent,
            # Add more agent mappings as they're created
            "critic": None,  # Placeholder
            "planner": None,  # Placeholder
            "architect": None,  # Placeholder
            "documenter": None,  # Placeholder
            "verifier": None,  # Placeholder
            "analyzer": None,  # Placeholder
            "refiner": None,  # Placeholder
            "checker": None,  # Placeholder
        }
        
        required_agents = self._get_required_agents()
        
        for agent_key in required_agents:
            agent_class = agent_classes.get(agent_key)
            if agent_class:
                # Initialize agent with appropriate configuration
                if agent_key in ["file", "tester", "shell"]:
                    # File-based agents need working directory
                    agent = agent_class(
                        base_dir=str(self.working_dir),
                        name=agent_key.title() + "Agent"
                    )
                else:
                    # LLM-based agents
                    agent = agent_class(
                        name=agent_key.title() + "Agent",
                        overrides={"provider": {"anthropic": {"model_name": self.model}}}
                    )
                
                self.agents[agent_key] = agent
                log.info(f"Initialized {agent_key} agent")
            else:
                # Use a generic agent for placeholders
                log.warning(f"Agent class not found for {agent_key}, using generic")
                from agent.agent import Agent
                self.agents[agent_key] = Agent(
                    roles=[f"You are a {agent_key} agent."],
                    name=agent_key.title() + "Agent"
                )
    
    def _show_execution_plan(self, assessment: Dict):
        """Display the execution plan to the user."""
        print(f"\n📊 [ANALYSIS] Task Assessment:")
        print(f"   Complexity: {assessment['complexity'].value.upper()}")
        print(f"   Domains: {', '.join(d.value for d in assessment['domains'])}")
        print(f"   Estimated steps: {assessment['estimated_steps']}")
        print(f"   Requires research: {'Yes' if assessment['requires_research'] else 'No'}")
        print(f"   Requires planning: {'Yes' if assessment['requires_planning'] else 'No'}")
        
        print(f"\n📝 [WORKFLOW] Execution Plan:")
        for i, step in enumerate(self.plan, 1):
            print(f"   {i}. {step.label} ({step.agent_key})")
            for parallel in step.parallel_steps:
                print(f"      ├─ {parallel.label} ({parallel.agent_key})")
            if step.on_success:
                print(f"      └─> {step.on_success}")
    
    def _execute_workflow(self) -> str:
        """Execute the dynamically generated workflow."""
        runner = PlanRunner(self.plan, self.agents, self.blackboard)
        
        # Create initial prompt based on task
        initial_prompt = f"Task: {self.task}\n\nPlease proceed with your specialized role."
        
        # Execute the plan
        result = runner.run(initial_prompt)
        
        return result
    
    def _show_summary(self):
        """Show execution summary."""
        print(f"\n📈 [SUMMARY] Execution Statistics:")
        
        # Count entries by agent
        agent_entries = {}
        for entry in self.blackboard.entries():
            author = entry.author
            if author not in agent_entries:
                agent_entries[author] = 0
            agent_entries[author] += 1
        
        print(f"   Total steps executed: {len(self.blackboard.entries())}")
        print(f"   Agents involved: {len(agent_entries)}")
        
        print(f"\n   Activity by agent:")
        for agent, count in sorted(agent_entries.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {agent}: {count} operations")
        
        # Show execution time
        duration = time.time() - self.start_time
        print(f"\n   Execution time: {duration:.1f} seconds")
    
    def _create_session(self, working_dir: Optional[str]) -> Tuple[Path, Path]:
        """Create session directories."""
        # Generate session name
        task_name = self.task[:50].lower()
        task_name = "".join(c if c.isalnum() or c.isspace() else "_" for c in task_name)
        task_name = "_".join(task_name.split())
        
        session_dir = self.output_manager.create_session_dir("dynamic_talk", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Setup session logging."""
        log_file = self.session_dir / "logs" / "orchestrator.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(handler)
    
    def _save_session(self):
        """Save session data."""
        session_file = self.session_dir / "session.json"
        
        session_data = {
            "task": self.task,
            "model": self.model,
            "working_dir": str(self.working_dir),
            "session_dir": str(self.session_dir),
            "duration": time.time() - self.start_time,
            "blackboard_entries": len(self.blackboard.entries()),
            "workflow_steps": [
                {
                    "label": step.label,
                    "agent": step.agent_key,
                    "parallel": [p.label for p in step.parallel_steps]
                }
                for step in self.plan
            ]
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save blackboard
        blackboard_file = self.session_dir / "blackboard.json"
        entries = []
        for entry in self.blackboard.entries():
            entries.append({
                "id": str(entry.id),
                "label": entry.label,
                "author": entry.author,
                "content": entry.content,
                "timestamp": str(entry.ts)
            })
        
        with open(blackboard_file, 'w') as f:
            json.dump({"entries": entries}, f, indent=2)


def main():
    """CLI entry point for Dynamic Talk."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dynamic Talk - Intelligent task orchestration"
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Task description"
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use"
    )
    parser.add_argument(
        "--dir", "-d",
        help="Working directory for changes"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode with approval prompts"
    )
    parser.add_argument(
        "--auto-approve", "-y",
        action="store_true",
        help="Auto-approve all operations"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in minutes"
    )
    
    args = parser.parse_args()
    
    # Join task words
    task = " ".join(args.task)
    
    # Create and run orchestrator
    orchestrator = DynamicTalkOrchestrator(
        task=task,
        working_dir=args.dir,
        model=args.model,
        timeout_minutes=args.timeout,
        interactive=args.interactive,
        auto_approve=args.auto_approve
    )
    
    exit_code = orchestrator.run()
    exit(exit_code)


if __name__ == "__main__":
    main()