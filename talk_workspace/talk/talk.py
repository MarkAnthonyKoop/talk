#!/usr/bin/env python3

"""
Talk - Multi-agent orchestration system for autonomous code generation.

Usage:
    python3 talk.py --task "Implement a function to calculate Fibonacci numbers"
    python3 talk.py --interactive  # Start in interactive mode
    python3 talk.py --dir my_project  # Specify working directory
    python3 talk.py --model gemini-1.5-pro  # Specify model for CodeAgent

Talk creates a multi-agent workflow that:
1. Analyzes the task and generates code changes
2. Applies changes to the filesystem
3. Runs tests to validate the changes
4. Iterates on failures until the code works correctly
"""

import argparse
import asyncio
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

# Import runtime components (renamed package: plan_runner)
from plan_runner.blackboard import Blackboard, BlackboardEntry
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import specialized agents (moved to special_agents)
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.web_search_agent import WebSearchAgent, WebSearchAgentIntegration

# Configure logging (will be updated per session)
log = logging.getLogger("talk")

class TalkOrchestrator:
    """
    Orchestrates a multi-agent workflow for autonomous code generation.
    
    This class sets up the blackboard, specialized agents, and execution plan,
    then runs the workflow to generate, apply, and test code changes.
    """
    def __init__(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: str = "gpt-4o",
        timeout_minutes: int = 30,
        interactive: bool = False,
        resume_session: Optional[str] = None,
        enable_web_search: bool = True
    ):
        """
        Initialize the Talk orchestrator.
        
        Args:
            task: The code generation task description
            working_dir: Directory where code changes will be applied (overrides session workspace)
            model: LLM model to use for code generation
            timeout_minutes: Maximum runtime in minutes
            interactive: Whether to run in interactive mode
            resume_session: Path to previous session directory to resume
            enable_web_search: Whether to enable web search for research tasks
        """
        self.task = task
        self.timeout_minutes = timeout_minutes
        self.interactive = interactive
        self.start_time = time.time()
        self.resume_session = resume_session
        self.enable_web_search = enable_web_search
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # Create or resume session directory
        if resume_session:
            self.session_dir, self.working_dir = self._resume_session(resume_session, working_dir)
        else:
            self.session_dir, self.working_dir = self._create_new_session(working_dir)
        
        # Setup logging for this session
        self._setup_session_logging()
        
        log.info(f"Session directory: {self.session_dir}")
        log.info(f"Working directory: {self.working_dir}")
        
        # Initialize or resume the blackboard
        self.blackboard = self._initialize_blackboard()
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        # Define the execution plan
        self.plan = self._create_plan()
        
        # ------------------------------------------------------------------
        # Timeout support – only available on Unix platforms.  Windows lacks
        # SIGALRM, so we guard the setup with a try/except and simply warn
        # when the feature is unavailable.
        # ------------------------------------------------------------------
        try:
            # Set up timeout handler (Unix / Linux / macOS)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            # Convert minutes to seconds for the alarm
            signal.alarm(int(timeout_minutes * 60))
        except AttributeError:
            # Windows doesn't support SIGALRM
            log.warning("SIGALRM not supported on this platform. "
                        "Timeout functionality disabled.")
    
    def _create_new_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Create a new Talk session with proper output management.
        
        Args:
            working_dir: Optional custom working directory
            
        Returns:
            Tuple of (session_directory, working_directory)
        """
        # Generate a clean task name for the session
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]  # Limit length
        
        # Create session directory using OutputManager
        session_dir = self.output_manager.create_session_dir("talk", task_name)
        
        # Determine working directory
        if working_dir:
            # Use custom working directory but ensure it exists
            work_dir = Path(working_dir).resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use session workspace as working directory
            work_dir = session_dir / "workspace"
        
        # Update session info with additional task details
        session_info_file = session_dir / "session_info.json"
        with open(session_info_file, "r+") as f:
            session_info = json.load(f)
            session_info.update({
                "task": self.task,
                "working_directory": str(work_dir),
                "model": "gpt-4o-mini"  # Will be updated by caller
            })
            f.seek(0)
            json.dump(session_info, f, indent=2)
            f.truncate()
        
        return session_dir, work_dir
    
    def _resume_session(self, resume_path: str, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Resume a previous Talk session.
        
        Args:
            resume_path: Path to previous session directory
            working_dir: Optional override for working directory
            
        Returns:
            Tuple of (session_directory, working_directory)
        """
        session_dir = Path(resume_path).resolve()
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        
        # Load previous session info
        session_info_file = session_dir / "session_info.json"
        if not session_info_file.exists():
            raise FileNotFoundError(f"Session info not found: {session_info_file}")
        
        with open(session_info_file) as f:
            session_info = json.load(f)
        
        # Update task from session info if not provided
        if not self.task:
            self.task = session_info.get("task", "Resumed session")
        
        # Determine working directory
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            # Use working directory from session info or default to workspace
            prev_work_dir = session_info.get("working_directory")
            if prev_work_dir and Path(prev_work_dir).exists():
                work_dir = Path(prev_work_dir)
            else:
                work_dir = session_dir / "workspace"
        
        log.info(f"Resuming session from: {session_dir}")
        return session_dir, work_dir
    
    def _setup_session_logging(self):
        """Configure logging for this session."""
        # Create session-specific log file
        log_file = self.output_manager.get_logs_dir(self.session_dir) / "talk.log"
        
        # Configure root logger for this session
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            force=True  # Override any existing configuration
        )
    
    def _initialize_blackboard(self) -> Blackboard:
        """Initialize or resume blackboard from previous session."""
        blackboard = Blackboard()
        
        if self.resume_session:
            # Try to load previous blackboard state
            blackboard_file = self.session_dir / "blackboard.json"
            if blackboard_file.exists():
                try:
                    with open(blackboard_file) as f:
                        data = json.load(f)
                    
                    # Restore blackboard entries
                    for entry_data in data.get("entries", []):
                        blackboard.add_sync(
                            label=entry_data["label"],
                            content=entry_data["content"],
                            section=entry_data.get("section", "default"),
                            role=entry_data.get("author", "system")
                        )
                    
                    log.info(f"Resumed blackboard with {len(data.get('entries', []))} entries")
                except Exception as e:
                    log.warning(f"Failed to load previous blackboard: {e}")
        
        return blackboard
    
    def _create_agents(self, model: str) -> Dict[str, Agent]:
        """
        Create specialized agents for the workflow.
        
        Args:
            model: LLM model to use for the CodeAgent
            
        Returns:
            Dictionary of agent instances
        """
        # Create the CodeAgent with the specified model
        # --------------------------------------------------------------
        # Choose the provider dynamically based on the model name.
        #  * OpenAI models typically contain "gpt"
        #  * Google Gemini models usually start with "gemini"
        #  * Fallback: default to Google configuration
        # --------------------------------------------------------------
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif model.lower().startswith("gemini"):
            provider_config = {"provider": {"google": {"model_name": model}}}
        else:
            provider_config = {"provider": {"google": {"model_name": model}}}

        code_agent = CodeAgent(
            overrides=provider_config,
            name="CodeAgent"
        )
        
        # Convert paths to strings, handling Windows/WSL path issues
        working_dir_str = str(self.working_dir).replace('\\\\wsl.localhost\\Ubuntu', '')
        if working_dir_str.startswith('\\'):
            working_dir_str = working_dir_str.replace('\\', '/')
        
        # Create the FileAgent with our working directory and same provider
        file_agent = FileAgent(
            base_dir=working_dir_str,
            overrides=provider_config,
            name="FileAgent"
        )
        
        # Create the TestAgent with our working directory and same provider
        test_agent = TestAgent(
            base_dir=working_dir_str,
            overrides=provider_config,
            name="TestAgent"
        )
        
        # Create agents dictionary
        agents = {
            "coder": code_agent,
            "file": file_agent,
            "tester": test_agent
        }
        
        # Add WebSearchAgent if enabled
        if self.enable_web_search:
            web_search_agent = WebSearchAgent(
                overrides=provider_config,
                name="WebSearchAgent"
            )
            agents["researcher"] = web_search_agent
        
        return agents
    
    def _create_plan(self) -> List[Step]:
        """
        Define the execution plan for the code generation workflow.
        
        Returns:
            List of Step objects defining the workflow
        """
        # Determine if we should start with research using LLM intelligence
        should_research = False
        if self.enable_web_search and "researcher" in self.agents:
            # Use the CodeAgent to make the research decision
            coder_agent = self.agents.get("coder")
            should_research = WebSearchAgentIntegration.should_use_web_search(
                self.task, 
                llm_agent=coder_agent
            )
        
        # Create the workflow steps
        steps = []
        
        if should_research:
            # Optional Step 0: Research if the task would benefit from it
            research_step = Step(
                label="research_task",
                agent_key="researcher",
                on_success="generate_code"
            )
            steps.append(research_step)
        
        # Step 1: Generate code changes
        generate_code = Step(
            label="generate_code",
            agent_key="coder",
            on_success="apply_changes"
        )
        steps.append(generate_code)
        
        # Step 2: Apply code changes to filesystem
        apply_changes = Step(
            label="apply_changes",
            agent_key="file",
            on_success="run_tests"
        )
        steps.append(apply_changes)
        
        # Step 3: Run tests to validate changes
        run_tests = Step(
            label="run_tests",
            agent_key="tester",
            on_success="check_results"
        )
        steps.append(run_tests)
        
        # Step 4: Check test results and decide next action
        check_results = Step(
            label="check_results",
            agent_key="coder",  # CodeAgent analyzes test results
            on_success=None  # End of workflow
        )
        steps.append(check_results)
        
        return steps
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout by logging and exiting gracefully."""
        elapsed_minutes = (time.time() - self.start_time) / 60
        log.error(f"Execution timed out after {elapsed_minutes:.1f} minutes")
        print(f"\n[WARNING] Execution timed out after {elapsed_minutes:.1f} minutes")
        
        # Record timeout in blackboard
        self.blackboard.add_sync(
            label="timeout",
            content=f"Execution timed out after {elapsed_minutes:.1f} minutes",
            section="system",
            role="system"
        )
        
        # Exit with error code
        sys.exit(1)
    
    def _prepare_initial_prompt(self) -> str:
        """
        Prepare the initial prompt for the first agent (researcher or coder).
        
        Returns:
            Formatted prompt string
        """
        # Determine if we're starting with research using LLM intelligence
        should_research = False
        if self.enable_web_search and "researcher" in self.agents:
            coder_agent = self.agents.get("coder")
            should_research = WebSearchAgentIntegration.should_use_web_search(
                self.task, 
                llm_agent=coder_agent
            )
        
        if should_research:
            # Create research prompt
            research_query = WebSearchAgentIntegration.create_search_query(self.task)
            return research_query
        else:
            # Create coding prompt
            return self._prepare_coding_prompt()
    
    def _prepare_coding_prompt(self) -> str:
        """
        Prepare the initial prompt for the CodeAgent.
        
        Returns:
            Formatted prompt string
        """
        # Get a list of files in the working directory
        file_agent = self.agents["file"]
        files = file_agent.list_files()
        
        # Format the initial prompt with task and file list
        prompt = f"Task: {self.task}\n\n"
        
        # Check if we have research results from previous step
        research_results = self._get_research_results()
        if research_results:
            prompt += f"Research Information:\n{research_results}\n\n"
        
        if files:
            prompt += "Existing files in the project:\n"
            for file in files:
                prompt += f"- {file}\n"
        else:
            prompt += "This is a new project with no existing files.\n"
        
        prompt += "\nPlease generate code changes as unified diffs to implement the task."
        return prompt
    
    def _get_research_results(self) -> Optional[str]:
        """Get research results from blackboard if available."""
        try:
            for entry in self.blackboard.entries():
                if entry.label == "research_task":
                    return entry.content
        except:
            pass
        return None
    
    def _interactive_mode(self):
        """Run Talk in interactive mode with user feedback."""
        print(f"\n[TASK] {self.task}")
        print(f"[SESSION] {self.session_dir}")
        print(f"[WORKSPACE] {self.working_dir}")
        print("[ROBOT] Starting Talk in interactive mode...\n")
        
        # Get initial user confirmation
        input("Press Enter to start the workflow...")
        
        # Create and run the plan runner
        runner = PlanRunner(self.plan, self.agents, self.blackboard)
        initial_prompt = self._prepare_initial_prompt()
        
        # Execute the first step
        current_step = self.plan[0]
        print(f"\n[RUNNING] step: {current_step.label}")
        result = self.agents[current_step.agent_key].run(initial_prompt)
        self.blackboard.add_sync(current_step.label, result)
        print(f"[OK] Completed step: {current_step.label}")
        
        # Ask user for confirmation before each step
        current_step = self._get_next_step(current_step)
        while current_step:
            print(f"\n[OUTPUT] Previous output:\n{result}\n")
            proceed = input(f"Continue to step '{current_step.label}'? (y/n): ").lower()
            
            if proceed != 'y':
                print("Workflow paused. You can resume by running the same command.")
                break
            
            print(f"\n[RUNNING] step: {current_step.label}")
            
            # Prepare input for the current step
            step_input = result
            if current_step.label == "generate_code" and current_step != self.plan[0]:
                # If this is the code generation step and not the first step,
                # use the coding prompt with research results
                step_input = self._prepare_coding_prompt()
            
            result = self.agents[current_step.agent_key].run(step_input)
            self.blackboard.add_sync(current_step.label, result)
            print(f"[OK] Completed step: {current_step.label}")
            
            current_step = self._get_next_step(current_step)
        
        print("\n[DONE] Workflow completed!")
        print(f"Session saved in: {self.session_dir}")
        print(f"Workspace: {self.working_dir}")
    
    def _get_next_step(self, current_step: Step) -> Optional[Step]:
        """Get the next step in the plan based on on_success field."""
        if current_step.on_success:
            for step in self.plan:
                if step.label == current_step.on_success:
                    return step
        return None
    
    def _non_interactive_mode(self):
        """Run Talk in non-interactive mode without user intervention."""
        print(f"\n[TASK] {self.task}")
        print(f"[SESSION] {self.session_dir}")
        print(f"[WORKSPACE] {self.working_dir}")
        print("[ROBOT] Starting Talk in automatic mode...\n")
        
        # Create and run the plan runner
        runner = PlanRunner(self.plan, self.agents, self.blackboard)
        initial_prompt = self._prepare_initial_prompt()
        
        try:
            # Execute the full plan
            result = runner.run(initial_prompt)
            
            # Print final result
            print("\n[DONE] Workflow completed!")
            print(f"Session saved in: {self.session_dir}")
            print(f"Workspace: {self.working_dir}")
            
            # Print summary
            print("\n[SUMMARY]")
            for entry in self.blackboard.entries():
                print(f"- {entry.label}: {entry.author}")
            
        except Exception as e:
            log.error(f"Error during execution: {str(e)}")
            print(f"\n[ERROR] Error during execution: {str(e)}")
            return 1
        
        return 0
    
    def run(self) -> int:
        """
        Run the Talk orchestrator.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if self.interactive:
                return self._interactive_mode()
            else:
                return self._non_interactive_mode()
        except KeyboardInterrupt:
            print("\n\n[WARNING] Execution interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] Unhandled exception: {str(e)}")
            return 1
        finally:
            # ----------------------------------------------------------
            # Persist blackboard state so that downstream tooling and
            # users can inspect the full conversation even when the run
            # terminates due to errors / interrupts.
            # ----------------------------------------------------------
            try:
                blackboard_file = self.session_dir / "blackboard.json"
                with open(blackboard_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": self.task,
                            "session_dir": str(self.session_dir),
                            "working_dir": str(self.working_dir),
                            "entries": [
                                {
                                    "id": str(entry.id),
                                    "section": entry.section,
                                    "label": entry.label,
                                    "author": entry.author,
                                    "content": entry.content,
                                    # entry.ts may be a datetime, float (unix
                                    # epoch) or already a string depending on
                                    # where the BlackboardEntry originated.
                                    "timestamp": (
                                        entry.ts
                                        if isinstance(entry.ts, str)
                                        else datetime.fromtimestamp(entry.ts).isoformat()
                                        if isinstance(entry.ts, (int, float))
                                        else str(entry.ts)
                                    ),
                                }
                                for entry in self.blackboard.entries()
                            ],
                        },
                        f,
                        indent=2,
                    )
                print(f"Blackboard state saved to: {blackboard_file}")
            except Exception as e:  # pylint: disable=broad-except
                log.warning("Failed to save blackboard state: %s", e)

            # Cancel the alarm (Unix/Linux only)
            try:
                signal.alarm(0)
            except AttributeError:
                # signal.alarm is unavailable on Windows
                pass

def main():
    """Parse command line arguments and run the Talk orchestrator."""
    # Get default model from settings
    settings = Settings.resolve()
    default_model = settings.provider.google.model_name
    
    parser = argparse.ArgumentParser(
        description="Talk - Multi-agent orchestration system for autonomous code generation"
    )
    parser.add_argument(
        "--task", "-t",
        help="Task description for code generation",
        default=None
    )
    parser.add_argument(
        "--dir", "-d",
        help="Working directory for code changes",
        default=None
    )
    parser.add_argument(
        "--model", "-m",
        help=f"LLM model to use for code generation (default: {default_model})",
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
        help="Run in interactive mode with user confirmation",
        action="store_true"
    )
    parser.add_argument(
        "--resume", "-r",
        help="Resume from a previous session directory",
        default=None
    )
    parser.add_argument(
        "--no-web-search",
        help="Disable web search for research tasks",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Get task from arguments or prompt the user (unless resuming)
    task = args.task
    if not task and not args.resume:
        task = input("Enter task description: ")
        if not task:
            print("Error: Task description is required (unless resuming)")
            return 1
    
    # Create and run the orchestrator
    orchestrator = TalkOrchestrator(
        task=task,
        working_dir=args.dir,
        model=args.model,
        timeout_minutes=args.timeout,
        interactive=args.interactive,
        resume_session=args.resume,
        enable_web_search=not args.no_web_search
    )
    
    return orchestrator.run()

if __name__ == "__main__":
    sys.exit(main())
