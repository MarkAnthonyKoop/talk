#!/usr/bin/env python3
"""
Talk v14 Enhanced - Production-grade code generation with hierarchical planning and quality assurance.

This version implements:
- Detailed hierarchical planning with todo tracking
- Quality evaluation and refinement loops
- Dependency management
- Supporting file generation (README, Dockerfile, etc.)
- Integration testing setup
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from special_agents.enhanced_codebase_agent import EnhancedCodebaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class TalkV14Orchestrator:
    """
    Talk v14 orchestrator using EnhancedCodebaseAgent.
    
    Features:
    - Hierarchical planning with todo tracking in .talk/talk_todos
    - Quality evaluation with strict standards (0.85 threshold)
    - Automatic refinement until quality met
    - Dependency management and validation
    - Supporting file generation
    """
    
    def __init__(self,
                 task: str,
                 model: str = "gemini-2.0-flash",
                 working_dir: Optional[str] = None,
                 quality_threshold: float = 0.85,
                 max_iterations: int = 50,
                 verbose: bool = True):
        """Initialize Talk v14."""
        self.task = task
        self.model = model
        self.working_dir = working_dir
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        log.info(f"Talk v14 Enhanced initialized")
        log.info(f"Task: {task}")
        log.info(f"Model: {model}")
        log.info(f"Quality Threshold: {quality_threshold}")
    
    def run(self) -> Dict[str, Any]:
        """Run the enhanced code generation."""
        try:
            start_time = time.time()
            
            if self.verbose:
                print("\n" + "="*70)
                print("TALK v14 ENHANCED - PRODUCTION-GRADE CODE GENERATION")
                print("="*70)
                print(f"Task: {self.task}")
                print(f"Quality Standards: {self.quality_threshold} (STRICT)")
                print(f"Model: {self.model}")
                print("="*70 + "\n")
            
            # Create enhanced agent
            agent = EnhancedCodebaseAgent(
                task=self.task,
                working_dir=self.working_dir,
                model=self.model,
                quality_threshold=self.quality_threshold,
                max_iterations=self.max_iterations
            )
            
            # Run generation
            result = agent.run()
            
            elapsed_time = time.time() - start_time
            
            # Enhance result with timing
            result["execution_time"] = f"{elapsed_time:.1f} seconds"
            result["model"] = self.model
            result["quality_threshold"] = self.quality_threshold
            
            if self.verbose:
                print("\n" + "="*70)
                print("GENERATION COMPLETE - SUMMARY")
                print("="*70)
                print(f"Status: {result.get('status', 'unknown')}")
                print(f"Files Generated: {result.get('files_generated', 0)}")
                print(f"Total Lines: {result.get('total_lines', 0):,}")
                print(f"Average Quality: {result.get('average_quality', 0):.2f}")
                print(f"Components: {result.get('components_completed', 0)}/{result.get('components_total', 0)}")
                print(f"Execution Time: {elapsed_time:.1f} seconds")
                print(f"Working Directory: {result.get('working_directory', 'N/A')}")
                print("="*70 + "\n")
            
            return result
            
        except Exception as e:
            log.exception("Talk v14 execution failed")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": f"{time.time() - start_time:.1f} seconds"
            }
    
    def compare_with_claude_code(self, claude_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compare Talk v14 output with Claude Code."""
        # Default Claude Code stats for comparison
        if not claude_stats:
            claude_stats = {
                "files": 10,
                "lines": 4132,
                "time": 120  # seconds
            }
        
        # Run Talk v14
        talk_result = self.run()
        
        if talk_result.get("status") == "success":
            comparison = {
                "task": self.task,
                "claude_code": claude_stats,
                "talk_v14": {
                    "files": talk_result.get("files_generated", 0),
                    "lines": talk_result.get("total_lines", 0),
                    "quality": talk_result.get("average_quality", 0),
                    "time": talk_result.get("execution_time", "unknown")
                },
                "performance": {
                    "files_ratio": talk_result.get("files_generated", 0) / claude_stats["files"],
                    "lines_ratio": talk_result.get("total_lines", 0) / claude_stats["lines"],
                    "quality_achieved": talk_result.get("average_quality", 0) >= self.quality_threshold
                }
            }
            
            print("\n" + "="*70)
            print("TALK v14 vs CLAUDE CODE COMPARISON")
            print("="*70)
            print(f"Task: {self.task}")
            print("\nClaude Code:")
            print(f"  Files: {claude_stats['files']}")
            print(f"  Lines: {claude_stats['lines']:,}")
            print(f"  Time: {claude_stats['time']}s")
            print("\nTalk v14 Enhanced:")
            print(f"  Files: {comparison['talk_v14']['files']}")
            print(f"  Lines: {comparison['talk_v14']['lines']:,}")
            print(f"  Quality: {comparison['talk_v14']['quality']:.2f}")
            print(f"  Time: {comparison['talk_v14']['time']}")
            print("\nPerformance:")
            print(f"  Files: {comparison['performance']['files_ratio']:.1%} of Claude Code")
            print(f"  Lines: {comparison['performance']['lines_ratio']:.1%} of Claude Code")
            print(f"  Quality Target: {'✓ MET' if comparison['performance']['quality_achieved'] else '✗ NOT MET'}")
            print("="*70 + "\n")
            
            return comparison
        else:
            return {
                "status": "error",
                "error": talk_result.get("error", "Unknown error")
            }


def main():
    """Test Talk v14 Enhanced."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Talk v14 Enhanced - Production-Grade Code Generation")
    parser.add_argument("task", nargs="?", default="build an agentic orchestration system", 
                       help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--quality", type=float, default=0.85, help="Quality threshold (0-1)")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum iterations")
    parser.add_argument("--compare", action="store_true", help="Compare with Claude Code")
    
    args = parser.parse_args()
    
    orchestrator = TalkV14Orchestrator(
        task=args.task,
        model=args.model,
        working_dir=args.working_dir,
        quality_threshold=args.quality,
        max_iterations=args.max_iterations
    )
    
    if args.compare:
        result = orchestrator.compare_with_claude_code()
    else:
        result = orchestrator.run()
    
    # Save result
    result_file = Path("talk_v14_result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to: {result_file}")
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    exit(main())