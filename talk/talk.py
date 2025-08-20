#!/usr/bin/env python3.11
"""
Talk v2 - Enhanced orchestration with RefinementAgent and BranchingAgent.

This version uses:
- Planning agents for task assessment
- RefinementAgent for iterative development cycles
- BranchingAgent for control flow decisions
- Simplified Step syntax (agent key only for linear flow)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import agent framework
from agent.agent import Agent
from agent.settings import Settings
from agent.output_manager import OutputManager

# Import runtime components
from plan_runner.blackboard import Blackboard
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import specialized agents
from special_agents.assessor_agent import AssessorAgent
from special_agents.execution_planning_agent import ExecutionPlanningAgent
from special_agents.refinement_agent import RefinementAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.file_agent import FileAgent
from special_agents.research_agents.web_search_agent import WebSearchAgent

log = logging.getLogger("talk_v2")


class TalkOrchestratorV2:
    """Enhanced Talk orchestrator with iterative refinement."""
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 interactive: bool = False,
                 enable_web_search: bool = True):
        """Initialize Talk v2."""
        self.task = task
        self.interactive = interactive
        self.enable_web_search = enable_web_search
        self.start_time = time.time()
        
        # Set model globally
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # Create session directory
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Setup logging
        self._setup_logging()
        
        log.info(f"Session directory: {self.session_dir}")
        log.info(f"Working directory: {self.working_dir}")
        
        # Initialize blackboard
        self.blackboard = Blackboard()
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        # Create execution plan
        self.plan = self._create_plan()
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        # Generate clean task name
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        # Create session directory
        session_dir = self.output_manager.create_session_dir("talk_v2", task_name)
        
        # Determine working directory
        if working_dir:
            work_dir = Path(working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = session_dir / "workspace"
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Setup session logging."""
        log_file = self.output_manager.get_logs_dir(self.session_dir) / "talk_v2.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            force=True
        )
    
    def _create_agents(self, model: str) -> Dict[str, Agent]:
        """Create all agents for v2 workflow."""
        # Get provider config
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif "claude" in model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": model}}}
        else:
            provider_config = {"provider": {"google": {"model_name": model}}}
        
        # Convert working directory path
        working_dir_str = str(self.working_dir).replace('\\\\wsl.localhost\\Ubuntu', '')
        if working_dir_str.startswith('\\'):
            working_dir_str = working_dir_str.replace('\\', '/')
        
        agents = {
            # Planning agents
            "assessor": AssessorAgent(
                overrides=provider_config,
                name="TaskAssessor"
            ),
            "planner": ExecutionPlanningAgent(
                overrides=provider_config,
                name="ExecutionPlanner"
            ),
            
            # Core development agent
            "refinement": RefinementAgent(
                base_dir=working_dir_str,
                max_iterations=5,
                overrides=provider_config,
                name="RefinementOrchestrator"
            ),
            
            # Note: BranchingAgent will be created in _create_plan() after steps exist
            
            # Utility agents
            "file": FileAgent(
                base_dir=working_dir_str,
                overrides=provider_config,
                name="FinalFileAgent"
            )
        }
        
        # Add optional research agent
        if self.enable_web_search:
            agents["researcher"] = WebSearchAgent(
                overrides=provider_config,
                name="ResearchAgent"
            )
        
        return agents
    
    def _create_plan(self) -> List[Step]:
        """Create v2 execution plan with simplified syntax."""
        steps = []
        
        # Phase 1: Assessment and Planning
        steps.append(Step(agent_key="assessor"))     # Assess task complexity
        steps.append(Step(agent_key="planner"))      # Generate execution plan
        
        # Optional research step (determined dynamically)
        if self.enable_web_search:
            steps.append(Step(
                agent_key="researcher",
                label="research_phase",  # Label for potential skip
                on_success="development_cycle"
            ))
        
        # Phase 2: Iterative Development (with label for looping)
        steps.append(Step(
            agent_key="refinement",
            label="development_cycle"
        ))
        
        # Phase 3: Branching Decision
        steps.append(Step(
            agent_key="branching",
            label="decision_point",
            # Branching logic handled by agent's output
        ))
        
        # Phase 4: Final File Application
        steps.append(Step(
            agent_key="file",
            label="final_apply"
        ))
        
        # Now create BranchingAgent with the complete plan
        branch_step = next((s for s in steps if s.label == "decision_point"), None)
        if branch_step:
            # Get provider config (same as other agents)
            model = os.environ.get("TALK_FORCE_MODEL", "gemini-2.0-flash")
            if "gpt" in model.lower():
                provider_config = {"provider": {"openai": {"model_name": model}}}
            elif "claude" in model.lower():
                provider_config = {"provider": {"anthropic": {"model_name": model}}}
            else:
                provider_config = {"provider": {"google": {"model_name": model}}}
            
            # Create BranchingAgent with step and plan references
            self.agents["branching"] = BranchingAgent(
                step=branch_step,
                plan=steps,
                overrides=provider_config,
                name="FlowController"
            )
        
        return steps
    
    def _handle_branching(self, decision_output: str) -> Optional[str]:
        """Process branching agent decision and return next step label."""
        try:
            decision = json.loads(decision_output)
            decision_type = decision.get("decision", "continue")
            target = decision.get("target")
            
            if decision_type == "complete":
                return None  # End workflow
            elif decision_type == "loop_refinement":
                return target or "development_cycle"
            elif decision_type == "restart":
                return target or "assessor"
            elif decision_type == "escalate":
                print(f"\n[ESCALATION] {decision.get('reason', 'Human review needed')}")
                if self.interactive:
                    input("Press Enter to continue...")
                return None
            else:  # continue
                return None  # Continue to next step
                
        except json.JSONDecodeError:
            log.warning("Failed to parse branching decision")
            return None
    
    def run(self) -> int:
        """Run the v2 orchestrator."""
        print(f"\n[TASK] {self.task}")
        print(f"[SESSION] {self.session_dir}")
        print(f"[WORKSPACE] {self.working_dir}")
        print("[TALK v2] Starting enhanced orchestration...\n")
        
        try:
            # Custom execution with branching support
            current_step_idx = 0
            last_output = self.task
            
            while current_step_idx < len(self.plan):
                step = self.plan[current_step_idx]
                
                print(f"[RUNNING] {step.label or f'step_{current_step_idx}'} ({step.agent_key})")
                
                # Execute step
                agent = self.agents[step.agent_key]
                output = agent.run(last_output)
                self.blackboard.add_sync(step.label or f"step_{current_step_idx}", output)
                
                # Handle branching if this is the branching agent
                if step.agent_key == "branching":
                    next_label = self._handle_branching(output)
                    if next_label:
                        # Find step with target label
                        for i, s in enumerate(self.plan):
                            if s.label == next_label:
                                current_step_idx = i
                                print(f"[BRANCHING] Jumping to {next_label}")
                                break
                        else:
                            log.warning(f"Branch target '{next_label}' not found")
                            current_step_idx += 1
                    else:
                        # Check if branching said complete
                        try:
                            decision = json.loads(output)
                            if decision.get("decision") == "complete":
                                print("[COMPLETE] Task completed successfully!")
                                break
                        except:
                            pass
                        current_step_idx += 1
                else:
                    # Normal flow or check on_success
                    if step.on_success:
                        # Find target step
                        for i, s in enumerate(self.plan):
                            if s.label == step.on_success:
                                current_step_idx = i
                                break
                        else:
                            current_step_idx += 1
                    else:
                        current_step_idx += 1
                
                last_output = output
            
            print("\n[DONE] Workflow completed!")
            print(f"Session saved in: {self.session_dir}")
            print(f"Workspace: {self.working_dir}")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Execution stopped by user")
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] {str(e)}")
            return 1
        finally:
            # Save blackboard state
            self._save_blackboard()
    
    def _save_blackboard(self):
        """Save blackboard state to file."""
        try:
            blackboard_file = self.session_dir / "blackboard.json"
            entries = [
                {
                    "label": entry.label,
                    "author": entry.author,
                    "content": entry.content,
                    "timestamp": str(entry.ts)
                }
                for entry in self.blackboard.entries()
            ]
            
            with open(blackboard_file, "w") as f:
                json.dump({
                    "task": self.task,
                    "session_dir": str(self.session_dir),
                    "working_dir": str(self.working_dir),
                    "entries": entries
                }, f, indent=2)
                
            print(f"Blackboard saved to: {blackboard_file}")
        except Exception as e:
            log.warning(f"Failed to save blackboard: {e}")


def main():
    """CLI entry point for Talk v2."""
    parser = argparse.ArgumentParser(
        description="Talk v2 - Enhanced multi-agent orchestration"
    )
    parser.add_argument("task", help="Task description")
    parser.add_argument("--dir", "-d", help="Working directory")
    parser.add_argument("--model", "-m", default="claude-3-5-sonnet-20241022",
                       help="LLM model to use")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--no-research", action="store_true",
                       help="Disable research phase")
    
    args = parser.parse_args()
    
    orchestrator = TalkOrchestratorV2(
        task=args.task,
        working_dir=args.dir,
        model=args.model,
        interactive=args.interactive,
        enable_web_search=not args.no_research
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())