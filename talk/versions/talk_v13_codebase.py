#!/usr/bin/env python3
"""
Talk v13 - Codebase Generation with CodebaseAgent

This version of Talk leverages the CodebaseAgent for comprehensive,
multi-component codebase generation with intelligent looping and refinement.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.output_manager import OutputManager
from plan_runner.blackboard import Blackboard
from special_agents.codebase_agent import CodebaseAgent

log = logging.getLogger(__name__)


class TalkV13Orchestrator:
    """
    Talk v13 orchestrator that uses CodebaseAgent for comprehensive generation.
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_iterations: int = 30,
                 verbose: bool = True):
        """Initialize Talk v13 with CodebaseAgent."""
        self.task = task
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.start_time = time.time()
        
        # Set model environment variable
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize the CodebaseAgent
        self.codebase_agent = CodebaseAgent(
            task=task,
            working_dir=str(self.working_dir),
            model=model,
            max_iterations=max_iterations
        )
        
        log.info(f"Talk v13 initialized - Model: {model}, Task: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None):
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("talk_v13_codebase", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session info
        session_info = {
            "task": self.task,
            "working_directory": str(work_dir),
            "model": self.model,
            "created": datetime.now().isoformat(),
            "version": "v13_codebase",
            "max_iterations": self.max_iterations
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.session_dir / "talk_v13.log"
        
        # Configure based on verbosity
        level = logging.INFO if self.verbose else logging.WARNING
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler() if self.verbose else logging.NullHandler(),
                logging.FileHandler(log_file)
            ],
            force=True
        )
    
    def run(self) -> Dict[str, Any]:
        """
        Run Talk v13 with CodebaseAgent.
        
        Returns:
            Dictionary with results including files generated, lines of code, etc.
        """
        try:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"TALK v13 - Codebase Generation")
                print(f"{'='*60}")
                print(f"Task: {self.task}")
                print(f"Model: {self.model}")
                print(f"Session: {self.session_dir}")
                print(f"Workspace: {self.working_dir}")
                print(f"Max Iterations: {self.max_iterations}")
                print(f"{'='*60}\n")
            
            # Run the CodebaseAgent
            print("[TALK v13] Starting CodebaseAgent...")
            result_json = self.codebase_agent.run()
            
            # Parse result
            try:
                result = json.loads(result_json)
            except:
                result = {"status": "unknown", "output": result_json}
            
            # Collect metrics
            execution_time = (time.time() - self.start_time) / 60  # minutes
            
            # Count generated files and lines
            py_files = list(self.working_dir.rglob("*.py"))
            py_files = [f for f in py_files if ".talk_scratch" not in str(f)]
            
            total_lines = 0
            file_list = []
            for f in py_files:
                try:
                    content = f.read_text()
                    lines = content.count('\n')
                    total_lines += lines
                    file_list.append({
                        "path": str(f.relative_to(self.working_dir)),
                        "lines": lines,
                        "size": f.stat().st_size
                    })
                except:
                    pass
            
            # Create comprehensive result
            final_result = {
                "status": result.get("status", "complete"),
                "task": self.task,
                "model": self.model,
                "execution_time_minutes": round(execution_time, 2),
                "iterations": result.get("iterations", self.codebase_agent.state.iteration_count),
                "files_generated": len(py_files),
                "total_lines": total_lines,
                "components_planned": len(self.codebase_agent.state.components_planned),
                "components_completed": len(self.codebase_agent.state.components_completed),
                "files": file_list,
                "workspace": str(self.working_dir),
                "session": str(self.session_dir)
            }
            
            # Save result
            result_file = self.session_dir / "result.json"
            with open(result_file, "w") as f:
                json.dump(final_result, f, indent=2)
            
            # Print summary
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"TALK v13 - Generation Complete")
                print(f"{'='*60}")
                print(f"Status: {final_result['status']}")
                print(f"Time: {final_result['execution_time_minutes']} minutes")
                print(f"Iterations: {final_result['iterations']}")
                print(f"Files Generated: {final_result['files_generated']}")
                print(f"Total Lines: {final_result['total_lines']:,}")
                print(f"Components: {final_result['components_completed']}/{final_result['components_planned']}")
                
                if file_list:
                    print(f"\nGenerated Files:")
                    for f in file_list[:10]:
                        print(f"  - {f['path']} ({f['lines']} lines)")
                    if len(file_list) > 10:
                        print(f"  ... and {len(file_list) - 10} more")
                
                print(f"\nResults saved to: {result_file}")
                print(f"{'='*60}\n")
            
            return final_result
            
        except Exception as e:
            log.exception("Talk v13 execution failed")
            error_result = {
                "status": "error",
                "error": str(e),
                "task": self.task,
                "execution_time_minutes": round((time.time() - self.start_time) / 60, 2)
            }
            
            if self.verbose:
                print(f"\n[ERROR] Talk v13 failed: {e}")
            
            return error_result


def compare_with_claude_code(talk_result: Dict[str, Any], claude_code_path: str = "/home/xx/code/tests/talk/claude_code_results"):
    """
    Compare Talk v13 results with Claude Code output.
    
    Args:
        talk_result: Result dictionary from Talk v13
        claude_code_path: Path to Claude Code results
        
    Returns:
        Comparison report dictionary
    """
    claude_path = Path(claude_code_path)
    
    # Count Claude Code files and lines
    claude_files = list(claude_path.rglob("*.py"))
    claude_lines = 0
    claude_file_list = []
    
    for f in claude_files:
        try:
            content = f.read_text()
            lines = content.count('\n')
            claude_lines += lines
            claude_file_list.append({
                "path": str(f.relative_to(claude_path)),
                "lines": lines
            })
        except:
            pass
    
    # Create comparison
    comparison = {
        "task": talk_result.get("task", "Unknown"),
        "claude_code": {
            "files": len(claude_files),
            "total_lines": claude_lines,
            "file_list": claude_file_list
        },
        "talk_v13": {
            "files": talk_result.get("files_generated", 0),
            "total_lines": talk_result.get("total_lines", 0),
            "execution_time_minutes": talk_result.get("execution_time_minutes", 0),
            "iterations": talk_result.get("iterations", 0),
            "components_completed": talk_result.get("components_completed", 0)
        },
        "comparison": {
            "file_ratio": round(talk_result.get("files_generated", 0) / max(len(claude_files), 1), 2),
            "line_ratio": round(talk_result.get("total_lines", 0) / max(claude_lines, 1), 2),
            "lines_per_file_claude": round(claude_lines / max(len(claude_files), 1)),
            "lines_per_file_talk": round(talk_result.get("total_lines", 0) / max(talk_result.get("files_generated", 1), 1))
        }
    }
    
    return comparison


def main():
    """Run Talk v13 from command line."""
    parser = argparse.ArgumentParser(description="Talk v13 - Codebase Generation with CodebaseAgent")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum iterations")
    parser.add_argument("--compare", action="store_true", help="Compare with Claude Code output")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Run Talk v13
    orchestrator = TalkV13Orchestrator(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        max_iterations=args.max_iterations,
        verbose=not args.quiet
    )
    
    result = orchestrator.run()
    
    # Compare with Claude Code if requested
    if args.compare:
        print("\n" + "="*60)
        print("COMPARISON WITH CLAUDE CODE")
        print("="*60)
        
        comparison = compare_with_claude_code(result)
        
        print(f"\nClaude Code:")
        print(f"  Files: {comparison['claude_code']['files']}")
        print(f"  Lines: {comparison['claude_code']['total_lines']:,}")
        print(f"  Avg Lines/File: {comparison['comparison']['lines_per_file_claude']}")
        
        print(f"\nTalk v13:")
        print(f"  Files: {comparison['talk_v13']['files']}")
        print(f"  Lines: {comparison['talk_v13']['total_lines']:,}")
        print(f"  Avg Lines/File: {comparison['comparison']['lines_per_file_talk']}")
        print(f"  Time: {comparison['talk_v13']['execution_time_minutes']} minutes")
        print(f"  Iterations: {comparison['talk_v13']['iterations']}")
        
        print(f"\nRatios (Talk/Claude):")
        print(f"  Files: {comparison['comparison']['file_ratio']:.0%}")
        print(f"  Lines: {comparison['comparison']['line_ratio']:.0%}")
        
        # Save comparison
        comparison_file = Path(orchestrator.session_dir) / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")
    
    return 0 if result.get("status") != "error" else 1


if __name__ == "__main__":
    exit(main())