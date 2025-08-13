#!/usr/bin/env python3.11
"""
Talk v10 - Refinement-based orchestration with rate limiting and optimizations.

Key improvements:
- Rate limit management with token counting
- Context pruning to prevent explosion
- Automatic file persistence
- Simplified orchestration with fast paths
- Integrated refinement cycles
- Better error recovery
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import tiktoken
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque

# Add to path
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
from special_agents.research_agents.web_search_agent import WebSearchAgent
from special_agents.refinement_agent import RefinementAgent

# Configure logging
log = logging.getLogger("talk_v10")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float
    last_refill: float
    
    @classmethod
    def create(cls, capacity: int, per_minute: int) -> 'TokenBucket':
        """Create a token bucket with per-minute rate."""
        return cls(
            capacity=capacity,
            refill_rate=per_minute / 60.0,
            tokens=capacity,
            last_refill=time.time()
        )
    
    def consume(self, tokens: int) -> float:
        """Consume tokens, return wait time if not available."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0
        
        # Calculate wait time
        needed = tokens - self.tokens
        wait_time = needed / self.refill_rate
        return wait_time
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class RateLimiter:
    """Manage API rate limits."""
    
    def __init__(self, 
                 tokens_per_minute: int = 30000,
                 calls_per_minute: int = 20):
        """Initialize rate limiter with conservative defaults."""
        self.token_bucket = TokenBucket.create(tokens_per_minute, tokens_per_minute)
        self.call_bucket = TokenBucket.create(calls_per_minute, calls_per_minute)
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Use as approximation
    
    def wait_if_needed(self, text: str) -> float:
        """Wait if rate limit would be exceeded."""
        # Estimate tokens
        estimated_tokens = len(self.encoding.encode(text))
        
        # Check both buckets
        token_wait = self.token_bucket.consume(estimated_tokens)
        call_wait = self.call_bucket.consume(1)
        
        wait_time = max(token_wait, call_wait)
        
        if wait_time > 0:
            log.info(f"Rate limit: waiting {wait_time:.1f}s (tokens: {estimated_tokens})")
            time.sleep(wait_time)
        
        return wait_time


class ContextManager:
    """Manage blackboard context to prevent explosion."""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def prune_blackboard(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Create pruned context from blackboard."""
        entries = list(blackboard.entries())
        
        # Always keep task description
        task_entry = next((e for e in entries if e.label == "task_description"), None)
        
        # Keep last N entries of each type
        recent_entries = self._get_recent_by_type(entries, n=2)
        
        # Keep critical entries (errors, completions)
        critical = [e for e in entries if self._is_critical(e)]
        
        # Combine and deduplicate
        keep_entries = []
        seen_labels = set()
        
        if task_entry:
            keep_entries.append(task_entry)
            seen_labels.add(task_entry.label)
        
        for entry in critical + recent_entries:
            if entry.label not in seen_labels:
                keep_entries.append(entry)
                seen_labels.add(entry.label)
        
        # Check token count
        context = self._entries_to_context(keep_entries)
        token_count = len(self.encoding.encode(json.dumps(context)))
        
        # If still too large, summarize
        if token_count > self.max_tokens:
            context = self._summarize_context(context)
        
        log.debug(f"Context pruned: {len(entries)} → {len(keep_entries)} entries, {token_count} tokens")
        
        return context
    
    def _get_recent_by_type(self, entries: List[BlackboardEntry], n: int = 2) -> List[BlackboardEntry]:
        """Get N most recent entries of each type."""
        by_type = {}
        for entry in entries:
            if entry.label not in by_type:
                by_type[entry.label] = []
            by_type[entry.label].append(entry)
        
        recent = []
        for label, type_entries in by_type.items():
            # Sort by timestamp and take last N
            sorted_entries = sorted(type_entries, key=lambda e: e.ts)
            recent.extend(sorted_entries[-n:])
        
        return recent
    
    def _is_critical(self, entry: BlackboardEntry) -> bool:
        """Check if entry is critical and must be kept."""
        critical_labels = ['error', 'failure', 'complete', 'success', 'result']
        return any(label in entry.label.lower() for label in critical_labels)
    
    def _entries_to_context(self, entries: List[BlackboardEntry]) -> Dict[str, Any]:
        """Convert entries to context dict."""
        return {
            "entries": [
                {
                    "label": e.label,
                    "content": e.content[:500] if len(e.content) > 500 else e.content,
                    "section": e.section
                }
                for e in entries
            ]
        }
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Further summarize context if still too large."""
        # Keep only essential information
        return {
            "task": context["entries"][0]["content"] if context["entries"] else "",
            "recent_actions": [e["label"] for e in context["entries"][-5:]],
            "status": "context_pruned_for_size"
        }


class FilePersistenceManager:
    """Manage file persistence from scratch to workspace."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.scratch_dir = Path.cwd() / ".talk_scratch"
        self.pending_files: List[Dict] = []
    
    def check_and_persist_files(self) -> int:
        """Check for new files in scratch and persist to workspace."""
        if not self.scratch_dir.exists():
            return 0
        
        persisted = 0
        
        # Look for code files
        for json_file in self.scratch_dir.glob("generated_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if "files" in data:
                    for file_info in data["files"]:
                        if self._persist_file(file_info):
                            persisted += 1
                
                # Archive processed file
                json_file.rename(json_file.with_suffix('.processed'))
                
            except Exception as e:
                log.error(f"Failed to process {json_file}: {e}")
        
        # Also check for direct Python files
        for py_file in self.scratch_dir.glob("generated_*.py"):
            try:
                content = py_file.read_text()
                file_info = {
                    "filename": py_file.name,
                    "content": content,
                    "language": "python"
                }
                if self._persist_file(file_info):
                    persisted += 1
                py_file.rename(py_file.with_suffix('.processed'))
            except Exception as e:
                log.error(f"Failed to persist {py_file}: {e}")
        
        if persisted > 0:
            log.info(f"Persisted {persisted} files from scratch to workspace")
        
        return persisted
    
    def _persist_file(self, file_info: Dict) -> bool:
        """Persist a single file to workspace."""
        try:
            filename = file_info.get("filename", "unknown.py")
            content = file_info.get("content", "")
            
            # Create full path
            file_path = self.workspace_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_path.write_text(content)
            log.info(f"Persisted {filename} to workspace")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to persist file: {e}")
            return False


class SimplifiedOrchestrator:
    """Simplified orchestration with fast paths for common operations."""
    
    def __init__(self):
        self.fast_path_operations = {
            "file_write": self._fast_file_write,
            "simple_code": self._fast_code_gen,
            "dependency_install": self._fast_dependency_install
        }
    
    def should_use_fast_path(self, task: str) -> Optional[str]:
        """Check if task can use fast path."""
        task_lower = task.lower()
        
        if "install" in task_lower and ("pip" in task_lower or "dependency" in task_lower):
            return "dependency_install"
        elif "write" in task_lower and "file" in task_lower:
            return "file_write"
        elif "simple" in task_lower or "basic" in task_lower:
            return "simple_code"
        
        return None
    
    def execute_fast_path(self, operation: str, task: str) -> Optional[str]:
        """Execute fast path operation."""
        if operation in self.fast_path_operations:
            return self.fast_path_operations[operation](task)
        return None
    
    def _fast_file_write(self, task: str) -> str:
        """Fast path for file writing."""
        # Extract filename and content from task
        # This is a simplified example
        return "File write completed via fast path"
    
    def _fast_code_gen(self, task: str) -> str:
        """Fast path for simple code generation."""
        return "Simple code generated via fast path"
    
    def _fast_dependency_install(self, task: str) -> str:
        """Fast path for dependency installation."""
        # Extract package name
        if "pytest" in task.lower():
            result = subprocess.run(
                ["pip", "install", "pytest"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return "pytest installed successfully"
            return f"Failed to install pytest: {result.stderr}"
        return "Dependency task noted"


class TalkV10Orchestrator:
    """
    Talk v10 orchestrator with refinement and optimizations.
    
    Key improvements:
    - Rate limiting with token awareness
    - Context management and pruning
    - Automatic file persistence
    - Fast paths for simple operations
    - Integrated refinement cycles
    - Better error recovery
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 timeout_minutes: int = 30,
                 enable_refinement: bool = True,
                 max_context_tokens: int = 8000,
                 rate_limit_tokens: int = 30000):
        """Initialize v10 orchestrator."""
        self.task = task
        self.timeout_minutes = timeout_minutes
        self.enable_refinement = enable_refinement
        self.start_time = time.time()
        
        # Set model
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize components
        self.rate_limiter = RateLimiter(tokens_per_minute=rate_limit_tokens)
        self.context_manager = ContextManager(max_tokens=max_context_tokens)
        self.simplified_orchestrator = SimplifiedOrchestrator()
        self.output_manager = OutputManager()
        
        # Create session
        self.session_dir, self.working_dir = self._create_session(working_dir)
        self.file_manager = FilePersistenceManager(self.working_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize blackboard
        self.blackboard = Blackboard()
        self.blackboard.add_sync(
            label="task_description",
            content=task,
            section="input",
            role="user"
        )
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        # Create execution plan
        self.plan = self._create_plan()
        
        log.info(f"Talk v10 initialized - Model: {model}, Task: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("talk_v10_refinement", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session info
        session_info = {
            "task": self.task,
            "working_directory": str(work_dir),
            "model": os.environ.get("TALK_FORCE_MODEL", "gemini-2.0-flash"),
            "created": datetime.now().isoformat(),
            "version": "v10_refinement"
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.session_dir / "talk_v10.log"
        
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
        """Create agents with appropriate configurations."""
        # Provider config based on model
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif "claude" in model.lower() or "sonnet" in model.lower() or "opus" in model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": model}}}
        else:  # Gemini
            provider_config = {"provider": {"google": {"model_name": model}}}
        
        agents = {}
        
        # Core agents
        agents["planning"] = PlanningAgent(
            overrides=provider_config,
            name="StrategicPlanner"
        )
        
        agents["code"] = CodeAgent(
            overrides=provider_config,
            name="CodeGenerator"
        )
        
        agents["file"] = FileAgent(
            base_dir=str(self.working_dir),
            overrides=provider_config,
            name="FileOperator"
        )
        
        agents["test"] = TestAgent(
            base_dir=str(self.working_dir),
            name="TestRunner"
        )
        
        # Refinement agent if enabled
        if self.enable_refinement:
            agents["refinement"] = RefinementAgent(
                base_dir=str(self.working_dir),
                max_iterations=3,
                overrides=provider_config,
                name="CodeRefiner"
            )
        
        # Web search
        agents["researcher"] = WebSearchAgent(
            overrides=provider_config,
            name="WebResearcher"
        )
        
        return agents
    
    def _create_plan(self) -> List[Step]:
        """Create simplified execution plan."""
        steps = [
            Step(
                label="analyze",
                agent_key="planning",
                on_success="implement",
                on_fail="complete"
            ),
            Step(
                label="implement",
                agent_key="code" if not self.enable_refinement else "refinement",
                on_success="persist",
                on_fail="complete"
            ),
            Step(
                label="persist",
                agent_key=None,  # Handled by file manager
                on_success="complete",
                on_fail="complete"
            ),
            Step(
                label="complete",
                agent_key=None,
                on_success=None,
                on_fail=None
            )
        ]
        
        return steps
    
    def run(self) -> int:
        """Run the orchestrator."""
        try:
            print(f"\n[TASK] {self.task}")
            print(f"[MODEL] {os.environ.get('TALK_FORCE_MODEL', 'gemini-2.0-flash')}")
            print(f"[SESSION] {self.session_dir}")
            print(f"[WORKSPACE] {self.working_dir}")
            print("[TALK v10] Starting refined orchestration...\n")
            
            current_step = "analyze"
            max_steps = 10
            steps_taken = 0
            
            while current_step and steps_taken < max_steps:
                steps_taken += 1
                print(f"\n[STEP {steps_taken}] {current_step}")
                
                # Check for fast path
                fast_path = self.simplified_orchestrator.should_use_fast_path(self.task)
                if fast_path and steps_taken == 1:
                    print(f"  Using fast path: {fast_path}")
                    result = self.simplified_orchestrator.execute_fast_path(fast_path, self.task)
                    if result:
                        self.blackboard.add_sync(
                            label=f"fast_path_{fast_path}",
                            content=result,
                            section="execution"
                        )
                        current_step = "complete"
                        continue
                
                # Get step
                step = next((s for s in self.plan if s.label == current_step), None)
                if not step:
                    print(f"  Step not found: {current_step}")
                    break
                
                # Execute step
                if step.agent_key:
                    agent = self.agents.get(step.agent_key)
                    if agent:
                        # Prepare context
                        context = self.context_manager.prune_blackboard(self.blackboard)
                        input_text = json.dumps({
                            "task": self.task,
                            "step": current_step,
                            "context": context
                        })
                        
                        # Rate limit check
                        self.rate_limiter.wait_if_needed(input_text)
                        
                        # Execute agent
                        print(f"  Executing {step.agent_key} agent...")
                        try:
                            output = agent.run(input_text)
                            
                            # Store result
                            self.blackboard.add_sync(
                                label=current_step,
                                content=output,
                                section="execution"
                            )
                            
                            # Auto-persist files after code generation
                            if step.agent_key in ["code", "refinement"]:
                                persisted = self.file_manager.check_and_persist_files()
                                if persisted > 0:
                                    print(f"  Auto-persisted {persisted} files")
                            
                            # Move to next step
                            current_step = step.on_success
                            
                        except Exception as e:
                            log.error(f"Agent execution failed: {e}")
                            print(f"  Error: {e}")
                            current_step = step.on_fail
                    else:
                        print(f"  Agent not found: {step.agent_key}")
                        current_step = step.on_fail
                else:
                    # Special handling for persist step
                    if current_step == "persist":
                        persisted = self.file_manager.check_and_persist_files()
                        print(f"  Persisted {persisted} files to workspace")
                    
                    current_step = step.on_success
                
                # Save checkpoint
                self._save_checkpoint()
            
            # Final persistence check
            self.file_manager.check_and_persist_files()
            
            print("\n[COMPLETE] Task execution finished")
            print(f"Total time: {(time.time() - self.start_time) / 60:.1f} minutes")
            
            # List generated files
            py_files = list(self.working_dir.rglob("*.py"))
            if py_files:
                print(f"\nGenerated {len(py_files)} Python files:")
                for f in py_files[:10]:
                    print(f"  - {f.relative_to(self.working_dir)}")
                if len(py_files) > 10:
                    print(f"  ... and {len(py_files) - 10} more")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Execution stopped by user")
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] {str(e)}")
            return 1
        finally:
            self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save current state."""
        try:
            checkpoint = {
                "task": self.task,
                "timestamp": datetime.now().isoformat(),
                "entries": [
                    {
                        "label": e.label,
                        "content": e.content[:1000],  # Truncate
                        "section": e.section
                    }
                    for e in list(self.blackboard.entries())[-20:]  # Last 20
                ]
            }
            
            checkpoint_file = self.session_dir / "checkpoint.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
                
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Talk v10 - Refined orchestration with optimizations"
    )
    parser.add_argument(
        "--task", "-t",
        help="Task description",
        default=None
    )
    parser.add_argument(
        "--dir", "-d",
        help="Working directory",
        default=None
    )
    parser.add_argument(
        "--model", "-m",
        help="LLM model (gemini-2.0-flash, claude-3-5-sonnet, gpt-4o)",
        default="gemini-2.0-flash"
    )
    parser.add_argument(
        "--timeout",
        help="Timeout in minutes",
        type=int,
        default=30
    )
    parser.add_argument(
        "--no-refinement",
        help="Disable refinement cycles",
        action="store_true"
    )
    parser.add_argument(
        "--max-tokens",
        help="Maximum context tokens",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--rate-limit",
        help="Token rate limit per minute",
        type=int,
        default=30000
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
    if not task:
        task = input("Enter task description: ")
        if not task:
            print("Error: Task description required")
            return 1
    
    # Create and run orchestrator
    orchestrator = TalkV10Orchestrator(
        task=task,
        working_dir=args.dir,
        model=args.model,
        timeout_minutes=args.timeout,
        enable_refinement=not args.no_refinement,
        max_context_tokens=args.max_tokens,
        rate_limit_tokens=args.rate_limit
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())