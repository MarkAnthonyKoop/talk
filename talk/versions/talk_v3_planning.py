#!/usr/bin/env python3.11
"""
Talk v3 - Planning-driven orchestration with intelligent branching.

This version features:
- PlanningAgent with hierarchical todo tracking for strategic decisions
- BranchingAgent with LLM-based Step label selection
- DependencyAgent for automatic package management
- Clean separation of concerns between agents
- Central planning/branching control flow

Usage:
    python3 talk_v3_planning.py --task "Create a web scraper"
    python3 talk_v3_planning.py --interactive
    python3 talk_v3_planning.py --enhance "Add new feature X"
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import agent framework
from agent.agent import Agent
from agent.settings import Settings
from agent.output_manager import OutputManager

# Import runtime components
from plan_runner.blackboard import Blackboard, BlackboardEntry
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import specialized agents
from special_agents.planning_agent import PlanningAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.execution_planning_agent import ExecutionPlanningAgent
from special_agents.research_agents.web_search_agent import WebSearchAgent

# Configure logging
log = logging.getLogger("talk_v3")


class TalkOrchestratorV3:
    """
    Talk v3 orchestrator with planning-driven architecture.
    
    Key features:
    - PlanningAgent maintains hierarchical todos and makes strategic decisions
    - BranchingAgent selects specific Step labels based on planning recommendations
    - All agents have single, focused responsibilities
    - Central control flow: work → plan → branch → work
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gpt-4o",
                 timeout_minutes: int = 30,
                 interactive: bool = False,
                 resume_session: Optional[str] = None,
                 enable_web_search: bool = True):
        """
        Initialize Talk v3 orchestrator.
        
        Args:
            task: The task description
            working_dir: Directory for code changes
            model: LLM model to use
            timeout_minutes: Maximum runtime
            interactive: Whether to run interactively
            resume_session: Path to resume from
            enable_web_search: Whether to enable web search
        """
        self.task = task
        self.timeout_minutes = timeout_minutes
        self.interactive = interactive
        self.start_time = time.time()
        self.resume_session = resume_session
        self.enable_web_search = enable_web_search
        
        # Set model globally
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # Create or resume session
        if resume_session:
            self.session_dir, self.working_dir = self._resume_session(resume_session, working_dir)
        else:
            self.session_dir, self.working_dir = self._create_new_session(working_dir)
        
        # Setup logging
        self._setup_session_logging()
        
        log.info(f"Session directory: {self.session_dir}")
        log.info(f"Working directory: {self.working_dir}")
        
        # Initialize blackboard
        self.blackboard = self._initialize_blackboard()
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        # Create the execution plan with central planning/branching
        self.plan = self._create_plan_v3()
        
        # Setup timeout handler (Unix/Linux only)
        try:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(timeout_minutes * 60))
        except AttributeError:
            log.warning("Timeout not supported on this platform")
    
    def _create_new_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create a new session with directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("talk_v3", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = session_dir / "workspace"
        
        # Save session info
        session_info = {
            "task": self.task,
            "working_directory": str(work_dir),
            "model": os.environ.get("TALK_FORCE_MODEL", "gpt-4o"),
            "created": datetime.now().isoformat()
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        return session_dir, work_dir
    
    def _resume_session(self, resume_path: str, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Resume from a previous session."""
        session_dir = Path(resume_path).resolve()
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        # Load session info
        session_info_file = session_dir / "session_info.json"
        if session_info_file.exists():
            with open(session_info_file) as f:
                session_info = json.load(f)
            
            if not self.task:
                self.task = session_info.get("task", "Resumed session")
        
        # Determine working directory
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        log.info(f"Resuming session from: {session_dir}")
        return session_dir, work_dir
    
    def _setup_session_logging(self):
        """Configure session-specific logging."""
        log_file = self.output_manager.get_logs_dir(self.session_dir) / "talk_v3.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            force=True
        )
    
    def _initialize_blackboard(self) -> Blackboard:
        """Initialize or resume blackboard."""
        blackboard = Blackboard()
        
        # Add initial task
        blackboard.add_sync(
            label="task_description",
            content=self.task,
            section="input",
            role="user"
        )
        
        if self.resume_session:
            # Try to load previous blackboard
            blackboard_file = self.session_dir / "blackboard.json"
            if blackboard_file.exists():
                try:
                    with open(blackboard_file) as f:
                        data = json.load(f)
                    
                    for entry_data in data.get("entries", []):
                        blackboard.add_sync(
                            label=entry_data["label"],
                            content=entry_data["content"],
                            section=entry_data.get("section", "default"),
                            role=entry_data.get("author", "system")
                        )
                    
                    log.info(f"Resumed blackboard with {len(data.get('entries', []))} entries")
                except Exception as e:
                    log.warning(f"Failed to load blackboard: {e}")
        
        return blackboard
    
    def _create_agents(self, model: str) -> Dict[str, Agent]:
        """Create all agents for v3 workflow."""
        # Get provider config
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif "claude" in model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": model}}}
        else:
            provider_config = {"provider": {"google": {"model_name": model}}}
        
        # Clean working directory path
        working_dir_str = str(self.working_dir).replace('\\\\wsl.localhost\\Ubuntu', '')
        if working_dir_str.startswith('\\'):
            working_dir_str = working_dir_str.replace('\\', '/')
        
        # Create work steps first (will be updated by _create_plan_v3)
        work_steps = []
        
        agents = {
            # Strategic planning agent
            "planning": PlanningAgent(
                base_dir=working_dir_str,
                overrides=provider_config,
                name="StrategicPlanner"
            ),
            
            # Flow control agent (initialized later with Step reference)
            # Will be set in _create_plan_v3
            "branching": None,
            
            # Core work agents
            "code": CodeAgent(
                overrides=provider_config,
                name="CodeGenerator"
            ),
            
            "file": FileAgent(
                base_dir=working_dir_str,
                overrides=provider_config,
                name="FileOperator"
            ),
            
            "test": TestAgent(
                base_dir=working_dir_str,
                name="TestRunner"
            ),
            
            # Optional: Initial planning agent for complex tasks
            "initial_planner": ExecutionPlanningAgent(
                overrides=provider_config,
                name="InitialPlanner"
            )
        }
        
        # Add web search if enabled
        if self.enable_web_search:
            agents["researcher"] = WebSearchAgent(
                overrides=provider_config,
                name="WebResearcher"
            )
        
        return agents
    
    def _create_plan_v3(self) -> List[Step]:
        """
        Create the v3 execution plan with central planning/branching.
        
        The flow is:
        1. Initial analysis (optional)
        2. Main loop: plan → branch → work → plan → branch → work...
        3. Work steps are selected dynamically by BranchingAgent
        """
        steps = []
        
        # Step 1: Initial task analysis (optional for complex tasks)
        if self._should_do_initial_planning():
            steps.append(Step(
                label="initial_analysis",
                agent_key="initial_planner",
                on_success="plan_next",
                on_fail="plan_next"
            ))
        
        # Step 2: Planning checkpoint - happens after every action
        plan_step = Step(
            label="plan_next",
            agent_key="planning",
            on_success="select_action",
            on_fail="error_recovery"
        )
        steps.append(plan_step)
        
        # Step 3: Branch selection - picks next action based on planning
        branch_step = Step(
            label="select_action",
            agent_key="branching",
            on_success="dynamic",  # Will be set dynamically by BranchingAgent
            on_fail="error_recovery"
        )
        steps.append(branch_step)
        
        # Work steps - these are the actual actions
        work_steps = [
            Step(
                label="generate_code",
                agent_key="code",
                on_success="plan_next",
                on_fail="plan_next"
            ),
            
            Step(
                label="apply_files",
                agent_key="file",
                on_success="plan_next",
                on_fail="plan_next"
            ),
            
            Step(
                label="run_tests",
                agent_key="test",
                on_success="plan_next",
                on_fail="plan_next"
            )
        ]
        
        # Add research step if enabled
        if self.enable_web_search:
            work_steps.append(Step(
                label="research",
                agent_key="researcher",
                on_success="plan_next",
                on_fail="plan_next"
            ))
        
        # Terminal steps
        work_steps.extend([
            Step(
                label="complete",
                agent_key=None,  # Terminal state
                on_success=None,
                on_fail=None
            ),
            
            Step(
                label="error_recovery",
                agent_key="planning",  # Let planner handle errors
                on_success="select_action",
                on_fail="manual_intervention"
            ),
            
            Step(
                label="manual_intervention",
                agent_key=None,  # Terminal state
                on_success=None,
                on_fail=None
            )
        ])
        
        steps.extend(work_steps)
        
        # Now create the BranchingAgent with the complete plan
        self.agents["branching"] = BranchingAgent(
            step=branch_step,  # Reference to its own step
            plan=steps,  # Complete plan to choose from
            agents=self.agents,  # Pass agents dict for descriptions
            overrides={"provider": {"openai": {"model_name": "gpt-4o"}}},
            name="FlowController"
        )
        
        return steps
    
    def _should_do_initial_planning(self) -> bool:
        """Determine if initial planning is needed."""
        # Simple heuristic: do initial planning for complex-sounding tasks
        complex_keywords = ["system", "framework", "architecture", "multiple", 
                          "integrate", "refactor", "optimize", "analyze"]
        
        task_lower = self.task.lower()
        return any(keyword in task_lower for keyword in complex_keywords)
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        elapsed = (time.time() - self.start_time) / 60
        log.error(f"Execution timed out after {elapsed:.1f} minutes")
        self._save_blackboard()
        sys.exit(1)
    
    def _save_blackboard(self):
        """Save blackboard state to file."""
        try:
            blackboard_file = self.session_dir / "blackboard.json"
            entries = []
            
            for entry in self.blackboard.entries():
                entries.append({
                    "label": entry.label,
                    "content": entry.content,
                    "section": entry.section,
                    "author": entry.author,
                    "timestamp": entry.ts if isinstance(entry.ts, str) else 
                               datetime.fromtimestamp(entry.ts).isoformat()
                })
            
            with open(blackboard_file, "w") as f:
                json.dump({
                    "task": self.task,
                    "entries": entries
                }, f, indent=2)
            
            log.info(f"Blackboard saved to: {blackboard_file}")
        except Exception as e:
            log.warning(f"Failed to save blackboard: {e}")
    
    def run(self) -> int:
        """Run the v3 orchestrator."""
        try:
            print(f"\n[TASK] {self.task}")
            print(f"[SESSION] {self.session_dir}")
            print(f"[WORKSPACE] {self.working_dir}")
            print("[TALK v3] Starting planning-driven orchestration...\n")
            
            # Create plan runner
            runner = PlanRunner(self.plan, self.agents, self.blackboard)
            
            if self.interactive:
                return self._run_interactive(runner)
            else:
                return self._run_automatic(runner)
                
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Execution stopped by user")
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] {str(e)}")
            return 1
        finally:
            self._save_blackboard()
            # Cancel timeout
            try:
                signal.alarm(0)
            except AttributeError:
                pass
    
    def _run_automatic(self, runner: PlanRunner) -> int:
        """Run in automatic mode."""
        try:
            # Start with first step
            initial_step = self.plan[0]
            
            # Prepare initial input based on first step
            if initial_step.label == "initial_analysis":
                initial_input = self.task
            else:
                # Go straight to planning
                initial_input = json.dumps({
                    "task_description": self.task,
                    "blackboard_state": {}
                })
            
            # Run the workflow
            result = runner.run(initial_input)
            
            print("\n[COMPLETE] Workflow finished successfully")
            print(f"Final result: {result[:200]}..." if len(result) > 200 else f"Final result: {result}")
            
            return 0
            
        except Exception as e:
            log.error(f"Workflow failed: {e}")
            print(f"\n[FAILED] Workflow error: {e}")
            return 1
    
    def _run_interactive(self, runner: PlanRunner) -> int:
        """Run in interactive mode."""
        print("[INTERACTIVE] User confirmation required for each step\n")
        
        current_step = self.plan[0]
        last_result = ""
        
        # Prepare initial input
        if current_step.label == "initial_analysis":
            current_input = self.task
        else:
            current_input = json.dumps({
                "task_description": self.task,
                "blackboard_state": {}
            })
        
        while current_step:
            # Show user what's happening
            print(f"\n[NEXT] {current_step.label} (agent: {current_step.agent_key})")
            
            if last_result:
                print(f"[PREVIOUS OUTPUT] {last_result[:200]}..." if len(last_result) > 200 else f"[PREVIOUS OUTPUT] {last_result}")
            
            proceed = input("Continue? (y/n): ").lower()
            if proceed != 'y':
                print("Workflow paused by user")
                break
            
            # Execute the step
            print(f"[RUNNING] {current_step.label}...")
            
            try:
                agent = self.agents.get(current_step.agent_key)
                if agent:
                    last_result = agent.run(current_input)
                    self.blackboard.add_sync(current_step.label, last_result)
                    print(f"[OK] Completed {current_step.label}")
                    
                    # Prepare input for next step
                    if "plan_next" in (current_step.on_success or ""):
                        # Prepare planning input
                        current_input = self._prepare_planning_input()
                    elif "select_action" in (current_step.on_success or ""):
                        # Prepare branching input
                        current_input = last_result  # Planning output
                    else:
                        # Use last result as input
                        current_input = last_result
                    
                    # Move to next step
                    next_label = current_step.on_success
                else:
                    # Terminal state
                    print(f"[TERMINAL] Reached {current_step.label}")
                    break
                
            except Exception as e:
                print(f"[ERROR] Step failed: {e}")
                next_label = current_step.on_fail
                current_input = str(e)
            
            # Find next step
            current_step = None
            if next_label:
                for step in self.plan:
                    if step.label == next_label:
                        current_step = step
                        break
        
        print("\n[DONE] Interactive workflow complete")
        return 0
    
    def _prepare_planning_input(self) -> str:
        """Prepare input for the planning agent."""
        # Gather blackboard state
        blackboard_state = {}
        for entry in self.blackboard.entries():
            blackboard_state[entry.label] = {
                "content": entry.content[:500] if len(entry.content) > 500 else entry.content,
                "author": entry.author
            }
        
        return json.dumps({
            "task_description": self.task,
            "blackboard_state": blackboard_state,
            "last_action": self.blackboard.entries()[-1].label if self.blackboard.entries() else None,
            "last_result": self.blackboard.entries()[-1].content[:500] if self.blackboard.entries() else None
        })


def main():
    """Parse arguments and run Talk v3."""
    # Get default model from settings
    settings = Settings.resolve()
    provider_settings = settings.get_provider_settings()
    default_model = provider_settings.model_name
    
    parser = argparse.ArgumentParser(
        description="Talk v3 - Planning-driven multi-agent orchestration"
    )
    parser.add_argument(
        "--task", "-t",
        help="Task description",
        default=None
    )
    parser.add_argument(
        "--dir", "-d",
        help="Working directory for code changes",
        default=None
    )
    parser.add_argument(
        "--model", "-m",
        help=f"LLM model to use (default: {default_model})",
        default=default_model
    )
    parser.add_argument(
        "--timeout",
        help="Maximum runtime in minutes",
        type=int,
        default=30
    )
    parser.add_argument(
        "--interactive", "-i",
        help="Run in interactive mode",
        action="store_true"
    )
    parser.add_argument(
        "--resume", "-r",
        help="Resume from previous session",
        default=None
    )
    parser.add_argument(
        "--no-web-search",
        help="Disable web search",
        action="store_true"
    )
    parser.add_argument(
        "--enhance",
        help="Enhancement mode for Talk framework",
        action="store_true"
    )
    parser.add_argument(
        "words",
        nargs="*",
        help="Task as positional arguments"
    )
    
    args = parser.parse_args()
    
    # Handle enhancement mode
    if args.enhance:
        enhancement_request = " ".join(args.words) if args.words else ""
        if not enhancement_request and not args.resume:
            enhancement_request = input("Enter enhancement description: ")
        
        if enhancement_request:
            task = f"""Enhance the Talk framework by: {enhancement_request}

Focus on improving the Talk multi-agent system located in /home/xx/code/. This includes:
- Adding new specialized agents in the special_agents/ directory
- Improving existing agents and their capabilities
- Enhancing the core framework in agent/ and talk/ directories
- Adding new features and integrations
- Improving the orchestration and communication between agents

Analyze the existing codebase and implement the requested enhancement."""
        elif args.resume:
            task = None
        else:
            print("Error: Enhancement description required")
            return 1
    else:
        # Normal mode
        task = args.task
        if not task and args.words:
            task = " ".join(args.words)
        if not task and not args.resume:
            task = input("Enter task description: ")
            if not task:
                print("Error: Task description required")
                return 1
    
    # Set working directory for enhancement mode
    working_dir = args.dir
    if args.enhance and not working_dir and not args.resume:
        working_dir = "/home/xx/code"
    
    # Create and run orchestrator
    orchestrator = TalkOrchestratorV3(
        task=task,
        working_dir=working_dir,
        model=args.model,
        timeout_minutes=args.timeout,
        interactive=args.interactive,
        resume_session=args.resume,
        enable_web_search=not args.no_web_search
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())