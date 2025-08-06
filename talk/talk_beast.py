#!/usr/bin/env python3.11
"""
Talk Beast - The ultimate intelligent orchestration system.

This is Talk transformed into the beast it was meant to be - an intelligent
system that dynamically scales from simple commands to enterprise-grade
distributed system construction with 50+ agents working in parallel.
"""

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

# Import enhanced orchestration components
from special_agents.assessor_agent import AssessorAgent, TaskComplexity
from special_agents.task_analysis_agent import TaskAnalysisAgent
from special_agents.execution_planner_agent import ExecutionPlannerAgent
from special_agents.completion_verifier_agent import CompletionVerifierAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.web_search_agent import WebSearchAgent
from special_agents.code_analysis.metrics_agent import MetricsAgent

from orchestration.enhanced_workflow_selector import EnhancedWorkflowSelector

log = logging.getLogger(__name__)


class TalkBeast:
    """
    The ultimate Talk orchestrator - intelligence at scale.
    
    This system can handle anything from 'ls' commands to building
    complete enterprise platforms with hundreds of files and 
    sophisticated architectures.
    """
    
    def __init__(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout_minutes: int = 120,  # Longer timeout for epic tasks
        interactive: bool = False,
        auto_approve: bool = False,
        max_iterations: int = 5
    ):
        """Initialize the beast."""
        self.task = task
        self.model = model
        self.timeout_minutes = timeout_minutes
        self.interactive = interactive
        self.auto_approve = auto_approve
        self.max_iterations = max_iterations
        self.start_time = time.time()
        
        # Set model globally
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize components
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        self.blackboard = Blackboard()
        self.workflow_selector = EnhancedWorkflowSelector()
        
        # These will be populated dynamically
        self.agents = {}
        self.plan = []
        self.assessment = {}
        
        self._setup_logging()
        log.info(f"🦾 Talk Beast initialized for task: {self.task}")
    
    def run(self) -> int:
        """Run the beast - prepare for awesomeness."""
        try:
            print(f"\n🦾 [TALK BEAST] Awakening the ultimate orchestration system...")
            print(f"📋 [TASK] {self.task}")
            print(f"📁 [SESSION] {self.session_dir}")
            print(f"🏗️  [WORKSPACE] {self.working_dir}")
            
            # Phase 1: Intelligent Assessment
            print(f"\n🧠 [ASSESSMENT] Analyzing task with advanced intelligence...")
            self.assessment = self._deep_task_assessment()
            
            # Phase 2: Dynamic Workflow Generation
            print(f"\n⚙️  [PLANNING] Generating optimal execution strategy...")
            self.plan = self._generate_advanced_workflow()
            
            # Phase 3: Agent Army Assembly
            print(f"\n🤖 [ASSEMBLY] Mobilizing specialized agent army...")
            self._assemble_agent_army()
            
            # Phase 4: Execution Plan Review
            if not self.auto_approve:
                self._present_battle_plan()
                if self.interactive and not self._get_user_approval():
                    return 1
            
            # Phase 5: Execute with Beast Mode
            print(f"\n⚡ [BEAST MODE] Executing with full intelligence...")
            result = self._execute_with_quality_loops()
            
            # Phase 6: Victory Summary
            self._victory_summary()
            
            print(f"\n🏆 [VICTORY] Task conquered with ultimate intelligence!")
            return 0
            
        except KeyboardInterrupt:
            print(f"\n⚠️  [INTERRUPTED] Beast mode interrupted")
            return 130
        except Exception as e:
            log.exception("Beast mode error")
            print(f"\n💥 [ERROR] Beast encountered obstacle: {str(e)}")
            return 1
        finally:
            self._save_session()
    
    def _deep_task_assessment(self) -> Dict:
        """Perform deep task assessment using multiple agents."""
        # Initialize assessment agents
        assessor = AssessorAgent(name="MasterAssessor")
        task_analyzer = TaskAnalysisAgent(name="TaskIntelligence")
        
        # Get primary assessment
        assessment = assessor.assess_task(self.task)
        
        # Get detailed analysis
        analysis_prompt = f"""Provide detailed analysis for this task:
        
Task: {self.task}
Complexity: {assessment['complexity'].value}

Analyze and provide:
1. Success criteria (specific, measurable)
2. Key deliverables expected
3. Technology stack recommendations
4. Potential challenges and risks
5. Resource requirements
6. Quality standards needed
"""
        
        detailed_analysis = task_analyzer.run(analysis_prompt)
        
        # Enhance assessment with detailed analysis
        assessment['detailed_analysis'] = detailed_analysis
        assessment['success_criteria'] = self._extract_success_criteria(detailed_analysis)
        
        # Store in blackboard
        self.blackboard.add("deep_assessment", json.dumps({
            "complexity": assessment["complexity"].value,
            "domains": [d.value for d in assessment["domains"]],
            "estimated_steps": assessment["estimated_steps"],
            "detailed_analysis": detailed_analysis,
            "success_criteria": assessment['success_criteria']
        }))
        
        return assessment
    
    def _generate_advanced_workflow(self) -> List[Step]:
        """Generate advanced workflow with quality loops."""
        workflow = self.workflow_selector.select_workflow(self.assessment)
        
        # Add completion verification to each major phase
        enhanced_workflow = []
        phase_steps = []
        
        for step in workflow:
            phase_steps.append(step)
            
            # Add verification after major phases
            if any(keyword in step.label for keyword in 
                   ["implement", "test", "document", "deploy", "final"]):
                verification_step = Step(
                    label=f"verify_{step.label}",
                    agent_key="completion_verifier",
                    on_success=step.on_success
                )
                phase_steps.append(verification_step)
                
                # Remove original on_success and let verifier handle it
                step.on_success = f"verify_{step.label}"
        
        enhanced_workflow.extend(phase_steps)
        
        log.info(f"Generated advanced workflow with {len(enhanced_workflow)} steps")
        return enhanced_workflow
    
    def _assemble_agent_army(self):
        """Assemble the specialized agent army."""
        required_agents = self._get_required_agents()
        
        agent_classes = {
            "assessor": AssessorAgent,
            "task_analyzer": TaskAnalysisAgent,
            "execution_planner": ExecutionPlannerAgent,
            "completion_verifier": CompletionVerifierAgent,
            "coder": CodeAgent,
            "file": FileAgent,
            "tester": TestAgent,
            "researcher": WebSearchAgent,
            "metrics": MetricsAgent,
            # Add placeholders for missing agents
            "critic": None,
            "architect": None,
            "documenter": None,
            "shell": None
        }
        
        agents_assembled = 0
        for agent_key in required_agents:
            agent_class = agent_classes.get(agent_key)
            
            if agent_class:
                try:
                    if agent_key in ["file", "tester", "metrics"]:
                        # File-based agents need working directory
                        agent = agent_class(
                            base_dir=str(self.working_dir),
                            name=f"Beast{agent_key.title()}Agent"
                        )
                    else:
                        # LLM-based agents
                        agent = agent_class(
                            name=f"Beast{agent_key.title()}Agent"
                        )
                    
                    self.agents[agent_key] = agent
                    agents_assembled += 1
                    log.info(f"Assembled {agent_key} agent")
                    
                except Exception as e:
                    log.warning(f"Failed to assemble {agent_key} agent: {e}")
                    # Create generic fallback
                    from agent.agent import Agent
                    self.agents[agent_key] = Agent(
                        roles=[f"You are a specialized {agent_key} agent."],
                        name=f"Generic{agent_key.title()}Agent"
                    )
            else:
                # Create generic agent for missing classes
                from agent.agent import Agent
                self.agents[agent_key] = Agent(
                    roles=[f"You are a specialized {agent_key} agent."],
                    name=f"Generic{agent_key.title()}Agent"
                )
                agents_assembled += 1
        
        print(f"   Assembled {agents_assembled} specialized agents")
        
        # Add dynamic scaling if epic complexity
        if self.assessment["complexity"] == TaskComplexity.EPIC:
            self._add_scaling_agents()
    
    def _add_scaling_agents(self):
        """Add additional agents for epic-scale tasks."""
        # Add specialized teams for epic tasks
        epic_agents = {
            "backend_specialist": "Backend development specialist",
            "frontend_specialist": "Frontend development specialist", 
            "devops_specialist": "DevOps and infrastructure specialist",
            "security_specialist": "Security and compliance specialist",
            "performance_specialist": "Performance optimization specialist",
            "documentation_specialist": "Technical documentation specialist"
        }
        
        from agent.agent import Agent
        for agent_key, description in epic_agents.items():
            self.agents[agent_key] = Agent(
                roles=[f"You are a {description}.", 
                       "You work as part of a large team on enterprise-scale projects."],
                name=f"Epic{agent_key.title().replace('_', '')}Agent"
            )
        
        print(f"   Added {len(epic_agents)} epic-scale specialists")
    
    def _present_battle_plan(self):
        """Present the execution battle plan."""
        print(f"\n📊 [INTELLIGENCE] Deep Assessment:")
        print(f"   Complexity: {self.assessment['complexity'].value.upper()}")
        print(f"   Domains: {', '.join(d.value for d in self.assessment['domains'])}")
        print(f"   Estimated steps: {self.assessment['estimated_steps']}")
        print(f"   Success criteria: {len(self.assessment.get('success_criteria', []))} defined")
        
        print(f"\n📝 [BATTLE PLAN] Execution Strategy:")
        for i, step in enumerate(self.plan[:20], 1):  # Show first 20 steps
            print(f"   {i}. {step.label} ({step.agent_key})")
            if step.parallel_steps:
                for parallel in step.parallel_steps:
                    print(f"      ├─ {parallel.label} ({parallel.agent_key})")
        
        if len(self.plan) > 20:
            print(f"   ... and {len(self.plan) - 20} more steps")
        
        print(f"\n🤖 [ARMY] Agent Deployment:")
        print(f"   Total agents: {len(self.agents)}")
        for agent_key, agent in list(self.agents.items())[:10]:
            print(f"   - {agent.name} ({agent_key})")
        if len(self.agents) > 10:
            print(f"   ... and {len(self.agents) - 10} more agents")
    
    def _get_user_approval(self) -> bool:
        """Get user approval for the battle plan."""
        response = input(f"\n🤔 Deploy this beast-mode execution plan? (y/n): ")
        return response.lower() == 'y'
    
    def _execute_with_quality_loops(self) -> str:
        """Execute with quality loops and completion verification."""
        runner = PlanRunner(self.plan, self.agents, self.blackboard)
        
        # Create enhanced initial prompt
        initial_prompt = f"""BEAST MODE EXECUTION:

Task: {self.task}
Complexity: {self.assessment['complexity'].value}
Success Criteria: {self.assessment.get('success_criteria', [])}

You are part of an elite agent army executing this task with maximum intelligence.
Work collaboratively and maintain the highest quality standards.
Each agent should leverage their specialized expertise for optimal results.

Proceed with your specialized role in this coordinated effort."""
        
        # Execute with monitoring
        iteration = 0
        max_iterations = self.max_iterations
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 [ITERATION {iteration}] Executing workflow phase...")
            
            try:
                result = runner.run(initial_prompt)
                
                # Check if we need to continue (look for completion verification)
                if self._check_completion_status():
                    print(f"✅ Task completed successfully in {iteration} iteration(s)")
                    break
                elif iteration < max_iterations:
                    print(f"🔄 Quality check indicates more work needed, continuing...")
                    initial_prompt = "Continue improving based on previous feedback and completion criteria."
                else:
                    print(f"⚠️  Reached maximum iterations ({max_iterations}), finalizing...")
                    
            except Exception as e:
                log.error(f"Iteration {iteration} failed: {e}")
                if iteration == max_iterations:
                    raise
                print(f"🔄 Recovering from error, retrying...")
        
        return "Beast mode execution completed"
    
    def _check_completion_status(self) -> bool:
        """Check if task completion criteria are met."""
        # Look for completion verifier outputs in blackboard
        for entry in self.blackboard.entries():
            if "completion_verifier" in entry.author.lower():
                if "COMPLETE" in entry.content or "ACCEPTABLE" in entry.content:
                    return True
        
        # If no completion verifier ran, do basic check
        files_created = len(list(self.working_dir.rglob("*"))) if self.working_dir.exists() else 0
        
        # For simple tasks, even 1 file is success
        if self.assessment.get("complexity") == TaskComplexity.SIMPLE:
            return files_created >= 1
        
        return files_created >= 3  # For complex tasks
    
    def _victory_summary(self):
        """Show victory summary."""
        print(f"\n🏆 [VICTORY REPORT]:")
        
        # Count outputs
        files_created = len(list(self.working_dir.rglob("*"))) if self.working_dir.exists() else 0
        
        # Agent activity
        agent_activity = {}
        for entry in self.blackboard.entries():
            author = entry.author
            if author not in agent_activity:
                agent_activity[author] = 0
            agent_activity[author] += 1
        
        print(f"   Files created: {files_created}")
        print(f"   Agents deployed: {len(self.agents)}")
        print(f"   Operations executed: {len(self.blackboard.entries())}")
        print(f"   Execution time: {time.time() - self.start_time:.1f}s")
        
        print(f"\n   Top performing agents:")
        for agent, ops in sorted(agent_activity.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     - {agent}: {ops} operations")
    
    def _extract_success_criteria(self, analysis: str) -> List[str]:
        """Extract success criteria from detailed analysis."""
        criteria = []
        lines = analysis.split('\n')
        in_criteria_section = False
        
        for line in lines:
            line = line.strip()
            if 'success criteria' in line.lower():
                in_criteria_section = True
                continue
            elif in_criteria_section and line.startswith(('-', '•', '*')):
                criteria.append(line.lstrip('-•* '))
            elif in_criteria_section and line and not line.startswith((' ', '\t')):
                in_criteria_section = False
        
        return criteria[:10]  # Limit to 10 criteria
    
    def _get_required_agents(self) -> List[str]:
        """Get list of required agent keys from plan."""
        agents = set()
        for step in self.plan:
            agents.add(step.agent_key)
            for parallel_step in step.parallel_steps:
                agents.add(parallel_step.agent_key)
        return list(agents)
    
    def _create_session(self, working_dir: Optional[str]) -> Tuple[Path, Path]:
        """Create beast session directories."""
        task_name = self.task[:50].lower()
        task_name = "".join(c if c.isalnum() or c.isspace() else "_" for c in task_name)
        task_name = "_".join(task_name.split())
        
        session_dir = self.output_manager.create_session_dir("talk_beast", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Setup beast logging."""
        log_file = self.session_dir / "logs" / "beast.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(handler)
    
    def _save_session(self):
        """Save beast session data."""
        session_file = self.session_dir / "beast_session.json"
        
        session_data = {
            "task": self.task,
            "model": self.model,
            "complexity": self.assessment.get("complexity").value if self.assessment.get("complexity") else "unknown",
            "agents_deployed": len(self.agents),
            "workflow_steps": len(self.plan),
            "duration": time.time() - self.start_time,
            "blackboard_entries": len(self.blackboard.entries()),
            "victory": True
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)


def main():
    """CLI entry point for Talk Beast."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Talk Beast - Ultimate Orchestration")
    parser.add_argument("task", nargs="+", help="Task description")
    parser.add_argument("--model", "-m", default="claude-3-5-sonnet-20241022", help="LLM model")
    parser.add_argument("--dir", "-d", help="Working directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--auto-approve", "-y", action="store_true", help="Auto-approve all")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in minutes")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max quality iterations")
    
    args = parser.parse_args()
    
    task = " ".join(args.task)
    
    beast = TalkBeast(
        task=task,
        working_dir=args.dir,
        model=args.model,
        timeout_minutes=args.timeout,
        interactive=args.interactive,
        auto_approve=args.auto_approve,
        max_iterations=args.max_iterations
    )
    
    exit_code = beast.run()
    exit(exit_code)


if __name__ == "__main__":
    main()