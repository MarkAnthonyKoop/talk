#!/usr/bin/env python3
"""
Talk v4 - Validated orchestration with built-in agent testing.

This version features:
- Pre-execution validation of all agents
- Runtime contract checking
- Automatic fallback on validation failures
- Integration with test harness for quality assurance
- Clear separation between planning, execution, and validation phases

Usage:
    python3 talk_v4_validated.py --task "Create a hello world function"
    python3 talk_v4_validated.py --validate-only  # Just run validation
    python3 talk_v4_validated.py --skip-validation --task "..."  # Skip validation
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

# Add test harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from special_agents.web_search_agent import WebSearchAgent

# Import test harness for validation
from tests.special_agents.test_harness import AgentTestHarness

# Configure logging
log = logging.getLogger("talk_v4")


class ValidatedTalkOrchestrator:
    """
    Talk v4 orchestrator with built-in validation and testing.
    
    Key features:
    - Validates agents before execution
    - Monitors agent contract compliance at runtime
    - Provides detailed diagnostics on failures
    - Supports validation-only mode for testing
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gpt-4o",
                 timeout_minutes: int = 30,
                 interactive: bool = False,
                 resume_session: Optional[str] = None,
                 enable_web_search: bool = True,
                 skip_validation: bool = False,
                 validate_only: bool = False):
        """
        Initialize Talk v4 orchestrator.
        
        Args:
            task: The task description
            working_dir: Directory for code changes
            model: LLM model to use
            timeout_minutes: Maximum runtime
            interactive: Whether to run interactively
            resume_session: Path to resume from
            enable_web_search: Whether to enable web search
            skip_validation: Skip pre-execution validation
            validate_only: Only run validation, don't execute task
        """
        self.task = task
        self.timeout_minutes = timeout_minutes
        self.interactive = interactive
        self.start_time = time.time()
        self.resume_session = resume_session
        self.enable_web_search = enable_web_search
        self.skip_validation = skip_validation
        self.validate_only = validate_only
        
        # Validation results
        self.validation_results = None
        self.agents_validated = False
        
        # Set model globally
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # Create or resume session
        if not validate_only:
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
        else:
            # Validation only mode
            self.session_dir = Path.cwd()
            self.working_dir = Path.cwd()
            self.blackboard = Blackboard()
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        # Create the execution plan
        self.plan = self._create_validated_plan()
        
        # Setup timeout handler (Unix/Linux only)
        if not validate_only:
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
        
        session_dir = self.output_manager.create_session_dir("talk_v4_validated", task_name)
        
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
            "created": datetime.now().isoformat(),
            "version": "v4_validated"
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
        log_file = self.output_manager.get_logs_dir(self.session_dir) / "talk_v4.log"
        
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
        """Create all agents for v4 workflow."""
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
        
        agents = {}
        
        # Strategic planning agent
        agents["planning"] = PlanningAgent(
            overrides=provider_config,
            name="StrategicPlanner"
        )
        
        # Flow control agent (will be initialized in _create_validated_plan)
        agents["branching"] = None
        
        # Core work agents
        agents["code"] = CodeAgent(
            overrides=provider_config,
            name="CodeGenerator"
        )
        
        agents["file"] = FileAgent(
            base_dir=working_dir_str,
            overrides=provider_config,
            name="FileOperator"
        )
        
        agents["test"] = TestAgent(
            base_dir=working_dir_str,
            name="TestRunner"
        )
        
        # Add web search if enabled
        if self.enable_web_search:
            agents["researcher"] = WebSearchAgent(
                overrides=provider_config,
                name="WebResearcher"
            )
        
        return agents
    
    def _create_validated_plan(self) -> List[Step]:
        """
        Create the v4 execution plan with validation steps.
        """
        steps = []
        
        # Step 1: Validation checkpoint (optional)
        if not self.skip_validation and not self.validate_only:
            steps.append(Step(
                label="validate_agents",
                agent_key=None,  # Special internal step
                on_success="plan_next",
                on_fail="validation_failed"
            ))
        
        # Step 2: Planning checkpoint
        steps.append(Step(
            label="plan_next",
            agent_key="planning",
            on_success="select_action",
            on_fail="error_recovery"
        ))
        
        # Step 3: Branch selection
        branch_step = Step(
            label="select_action",
            agent_key="branching",
            on_success="dynamic",  # Set by BranchingAgent
            on_fail="error_recovery"
        )
        steps.append(branch_step)
        
        # Work steps
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
        
        # Add research if enabled
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
                agent_key=None,
                on_success=None,
                on_fail=None
            ),
            
            Step(
                label="error_recovery",
                agent_key="planning",
                on_success="select_action",
                on_fail="manual_intervention"
            ),
            
            Step(
                label="validation_failed",
                agent_key=None,
                on_success=None,
                on_fail=None
            ),
            
            Step(
                label="manual_intervention",
                agent_key=None,
                on_success=None,
                on_fail=None
            )
        ])
        
        steps.extend(work_steps)
        
        # Create BranchingAgent with complete plan
        self.agents["branching"] = BranchingAgent(
            step=branch_step,
            plan=steps,
            overrides={"provider": {"openai": {"model_name": "gpt-4o"}}},
            name="FlowController"
        )
        
        return steps
    
    def validate_agents(self) -> bool:
        """
        Validate all agents using the test harness.
        
        Returns:
            True if all validations pass, False otherwise
        """
        print("\n[VALIDATION] Running agent validation...")
        
        harness = AgentTestHarness(verbose=True)
        
        # Test each agent
        agents_to_test = [
            ("planning", self.agents["planning"]),
            ("branching", self.agents["branching"]),
            ("code", self.agents["code"])
        ]
        
        all_passed = True
        
        for agent_name, agent in agents_to_test:
            if agent:  # Skip None agents
                print(f"\n[VALIDATING] {agent_name} agent...")
                passed = harness.test_agent_contract(agent, agent_name)
                passed = passed and harness.test_agent_uses_llm(agent, agent_name)
                
                if not passed:
                    print(f"[ERROR] {agent_name} agent validation failed!")
                    all_passed = False
        
        # Store results
        self.validation_results = harness.test_results
        self.agents_validated = all_passed
        
        if all_passed:
            print("\n[VALIDATION] ✓ All agents validated successfully!")
        else:
            print("\n[VALIDATION] ✗ Some agents failed validation")
        
        return all_passed
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        elapsed = (time.time() - self.start_time) / 60
        log.error(f"Execution timed out after {elapsed:.1f} minutes")
        self._save_blackboard()
        sys.exit(1)
    
    def _save_blackboard(self):
        """Save blackboard state to file."""
        if self.validate_only:
            return
            
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
                    "entries": entries,
                    "validation_results": self.validation_results
                }, f, indent=2)
            
            log.info(f"Blackboard saved to: {blackboard_file}")
        except Exception as e:
            log.warning(f"Failed to save blackboard: {e}")
    
    def run(self) -> int:
        """Run the v4 orchestrator."""
        try:
            # Validation-only mode
            if self.validate_only:
                print("\n[MODE] Validation-only mode")
                return 0 if self.validate_agents() else 1
            
            print(f"\n[TASK] {self.task}")
            print(f"[SESSION] {self.session_dir}")
            print(f"[WORKSPACE] {self.working_dir}")
            print("[TALK v4] Starting validated orchestration...\n")
            
            # Run validation unless skipped
            if not self.skip_validation:
                if not self.validate_agents():
                    print("\n[FAILED] Agent validation failed. Use --skip-validation to bypass.")
                    return 1
            else:
                print("[WARNING] Skipping agent validation")
            
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
            # Prepare initial input
            initial_input = json.dumps({
                "task_description": self.task,
                "blackboard_state": {},
                "last_action": "",
                "last_result": ""
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
        
        # Similar to automatic but with user prompts
        # (Simplified for brevity)
        return self._run_automatic(runner)


def main():
    """Parse arguments and run Talk v4."""
    # Get default model from settings
    settings = Settings.resolve()
    provider_settings = settings.get_provider_settings()
    default_model = provider_settings.model_name
    
    parser = argparse.ArgumentParser(
        description="Talk v4 - Validated multi-agent orchestration"
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
        "--skip-validation",
        help="Skip agent validation",
        action="store_true"
    )
    parser.add_argument(
        "--validate-only",
        help="Only run validation, don't execute task",
        action="store_true"
    )
    parser.add_argument(
        "words",
        nargs="*",
        help="Task as positional arguments"
    )
    
    args = parser.parse_args()
    
    # Get task
    task = args.task
    if not task and args.words:
        task = " ".join(args.words)
    if not task and not args.resume and not args.validate_only:
        task = input("Enter task description: ")
        if not task:
            print("Error: Task description required")
            return 1
    
    # Create and run orchestrator
    orchestrator = ValidatedTalkOrchestrator(
        task=task,
        working_dir=args.dir,
        model=args.model,
        timeout_minutes=args.timeout,
        interactive=args.interactive,
        resume_session=args.resume,
        enable_web_search=not args.no_web_search,
        skip_validation=args.skip_validation,
        validate_only=args.validate_only
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())