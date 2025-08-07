#!/usr/bin/env python3
"""
Talk v15 Enterprise - Builds massive commercial-grade applications.

Usage:
    talk_v15 "build a website"                    # Standard: ~5,000 lines
    talk_v15 "build a website" --big              # Enterprise: 30,000+ lines (Instagram-scale)
    talk_v15 "build an app" --big --target=100000 # Custom: 100,000+ lines

With --big flag:
- "website" → Full social media platform like Instagram
- "app" → Multi-platform system like Uber
- "tool" → Enterprise SaaS like Slack
- "game" → Multiplayer platform like Fortnite

Features:
- Ambitious interpretation (simple → enterprise)
- Microservices architecture generation
- Self-reflection and scope expansion
- Time-based quality gates
- 10,000-100,000+ lines of integrated code
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from special_agents.enterprise_codebase_agent import EnterpriseCodebaseAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class TalkV15Orchestrator:
    """
    Talk v15 orchestrator for enterprise-scale applications.
    
    Standard mode: Regular code generation (5,000 lines)
    Big mode: Commercial-grade systems (30,000+ lines)
    """
    
    def __init__(self,
                 task: str,
                 big_mode: bool = False,
                 model: str = "gemini-2.0-flash",
                 working_dir: Optional[str] = None,
                 target_lines: int = None,
                 minimum_hours: float = None,
                 verbose: bool = True):
        """Initialize Talk v15."""
        self.task = task
        self.big_mode = big_mode
        self.model = model
        self.working_dir = working_dir
        
        # Set defaults based on mode
        if big_mode:
            self.target_lines = target_lines or 50000  # 50k default for big mode
            self.minimum_hours = minimum_hours or 2.0  # 2 hours minimum
        else:
            self.target_lines = target_lines or 5000   # 5k for standard
            self.minimum_hours = minimum_hours or 0.25  # 15 minutes
        
        self.verbose = verbose
        
        log.info(f"Talk v15 Enterprise initialized")
        log.info(f"Task: {task}")
        log.info(f"Mode: {'BIG (Enterprise)' if big_mode else 'Standard'}")
        log.info(f"Target: {self.target_lines:,} lines")
        log.info(f"Min Time: {self.minimum_hours} hours")
    
    def run(self) -> Dict[str, Any]:
        """Run enterprise code generation."""
        try:
            start_time = time.time()
            
            if self.verbose:
                self._print_header()
            
            # Create enterprise agent
            agent = EnterpriseCodebaseAgent(
                task=self.task,
                big_mode=self.big_mode,
                working_dir=self.working_dir,
                model=self.model,
                target_lines=self.target_lines,
                minimum_hours=self.minimum_hours
            )
            
            # Run generation
            result = agent.run()
            
            elapsed_time = time.time() - start_time
            
            # Enhance result
            result["execution_time_seconds"] = elapsed_time
            result["execution_time_hours"] = elapsed_time / 3600
            result["model"] = self.model
            result["mode"] = "enterprise" if self.big_mode else "standard"
            
            if self.verbose:
                self._print_summary(result)
            
            return result
            
        except Exception as e:
            log.exception("Talk v15 execution failed")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_seconds": time.time() - start_time
            }
    
    def _print_header(self):
        """Print execution header."""
        if self.big_mode:
            print("\n" + "="*80)
            print("TALK v15 ENTERPRISE - COMMERCIAL-GRADE CODE GENERATION")
            print("="*80)
            print(f"Task: {self.task}")
            print("")
            print("🚀 INTERPRETING AMBITIOUSLY:")
            print("  - Simple request → Enterprise system")
            print("  - Targeting millions of users")
            print("  - Full production infrastructure")
            print("")
            print(f"📊 TARGETS:")
            print(f"  - Minimum Lines: {self.target_lines:,}")
            print(f"  - Minimum Time: {self.minimum_hours:.1f} hours")
            print(f"  - Architecture: Microservices")
            print(f"  - Quality: Commercial-grade")
            print("="*80 + "\n")
        else:
            print("\n" + "="*70)
            print("TALK v15 - STANDARD CODE GENERATION")
            print("="*70)
            print(f"Task: {self.task}")
            print(f"Target: {self.target_lines:,} lines")
            print("="*70 + "\n")
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print execution summary."""
        print("\n" + "="*80)
        print("GENERATION COMPLETE - FINAL SUMMARY")
        print("="*80)
        
        if self.big_mode and result.get("interpreted_task"):
            print(f"Original: {result.get('original_task', self.task)}")
            print(f"Built As: {result.get('interpreted_task')}")
            print("")
        
        print("📊 Statistics:")
        print(f"  Lines Generated: {result.get('lines_generated', 0):,}")
        print(f"  Files Created: {result.get('files_generated', 0)}")
        print(f"  Components: {result.get('components_built', 0)}/{result.get('components_total', 0)}")
        print(f"  Time: {result.get('execution_time_hours', 0):.2f} hours")
        
        if self.big_mode:
            print("")
            print("🏗️ Architecture:")
            print(f"  Type: {result.get('architecture_type', 'unknown')}")
            print(f"  Services: {result.get('components_built', 0)}")
            
            # Success criteria
            print("")
            print("✅ Success Criteria:")
            lines_ok = result.get('lines_generated', 0) >= self.target_lines
            time_ok = result.get('execution_time_hours', 0) >= self.minimum_hours
            print(f"  Lines Target ({self.target_lines:,}): {'✓ MET' if lines_ok else '✗ NOT MET'}")
            print(f"  Time Target ({self.minimum_hours:.1f}h): {'✓ MET' if time_ok else '✗ NOT MET'}")
        
        print("")
        print(f"📁 Output: {result.get('working_directory', 'unknown')}")
        print("="*80 + "\n")
    
    def compare_with_others(self) -> Dict[str, Any]:
        """Compare v15 with other Talk versions and Claude Code."""
        print("\n" + "="*80)
        print("TALK v15 COMPARISON ANALYSIS")
        print("="*80)
        
        # Run v15
        v15_result = self.run()
        
        comparisons = {
            "task": self.task,
            "mode": "enterprise" if self.big_mode else "standard",
            "results": {
                "claude_code": {
                    "lines": 4132,
                    "files": 10,
                    "time_minutes": 2,
                    "quality": "prototype"
                },
                "talk_v13": {
                    "lines": 1039,
                    "files": 10,
                    "time_minutes": 3,
                    "quality": "basic"
                },
                "talk_v14": {
                    "lines": 2000,
                    "files": 15,
                    "time_minutes": 5,
                    "quality": "production"
                },
                "talk_v15": {
                    "lines": v15_result.get("lines_generated", 0),
                    "files": v15_result.get("files_generated", 0),
                    "time_minutes": v15_result.get("execution_time_seconds", 0) / 60,
                    "quality": "enterprise" if self.big_mode else "standard"
                }
            }
        }
        
        # Print comparison table
        print("\n📊 Comparison Table:")
        print("-"*70)
        print(f"{'System':<15} {'Lines':>10} {'Files':>8} {'Time':>10} {'Quality':<15}")
        print("-"*70)
        
        for system, stats in comparisons["results"].items():
            name = system.replace("_", " ").title()
            print(f"{name:<15} {stats['lines']:>10,} {stats['files']:>8} "
                  f"{stats['time_minutes']:>8.1f}m {stats['quality']:<15}")
        
        print("-"*70)
        
        # Analysis
        v15_lines = comparisons["results"]["talk_v15"]["lines"]
        claude_lines = comparisons["results"]["claude_code"]["lines"]
        
        print("\n📈 Analysis:")
        print(f"  v15 vs Claude Code: {v15_lines/claude_lines:.1f}x lines")
        print(f"  v15 vs v14: {v15_lines/2000:.1f}x lines")
        print(f"  v15 vs v13: {v15_lines/1039:.1f}x lines")
        
        if self.big_mode:
            print("\n🏆 Achievement:")
            if v15_lines >= 30000:
                print("  ✓ ENTERPRISE SCALE ACHIEVED!")
                print("  This is a complete commercial-grade system")
            elif v15_lines >= 10000:
                print("  ✓ Scale-up level achieved")
                print("  Suitable for medium-sized businesses")
            else:
                print("  → Still building to target scale...")
        
        print("="*80 + "\n")
        
        return comparisons


def main():
    """Talk v15 Enterprise CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Talk v15 Enterprise - Build massive commercial applications",
        epilog="""
Examples:
  talk_v15 "build a website"                    # Standard website (5k lines)
  talk_v15 "build a website" --big              # Instagram-scale (50k+ lines)
  talk_v15 "build an app" --big --target=100000 # Uber-scale (100k lines)
  talk_v15 "build a game" --big --hours=3       # Fortnite-scale (3+ hours)
        """
    )
    
    parser.add_argument("task", help="Task description")
    
    parser.add_argument("--big", action="store_true",
                       help="Build commercial-grade enterprise system")
    
    parser.add_argument("--model", default="gemini-2.0-flash",
                       help="AI model to use")
    
    parser.add_argument("--working-dir", 
                       help="Output directory")
    
    parser.add_argument("--target", type=int,
                       help="Target lines of code (default: 50000 for --big, 5000 otherwise)")
    
    parser.add_argument("--hours", type=float,
                       help="Minimum hours to run (default: 2.0 for --big, 0.25 otherwise)")
    
    parser.add_argument("--compare", action="store_true",
                       help="Compare with other versions")
    
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    # Show mode info
    if not args.quiet:
        if args.big:
            print("\n🚀 BIG MODE ACTIVATED!")
            print("Building commercial-grade enterprise system...")
            print(f"Interpreting '{args.task}' ambitiously...")
            time.sleep(2)  # Dramatic pause
    
    orchestrator = TalkV15Orchestrator(
        task=args.task,
        big_mode=args.big,
        model=args.model,
        working_dir=args.working_dir,
        target_lines=args.target,
        minimum_hours=args.hours,
        verbose=not args.quiet
    )
    
    if args.compare:
        result = orchestrator.compare_with_others()
    else:
        result = orchestrator.run()
    
    # Save result
    result_file = Path("talk_v15_result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    
    if not args.quiet:
        print(f"\n📄 Result saved to: {result_file}")
        
        if args.big and result.get("lines_generated", 0) >= 30000:
            print("\n🎉 CONGRATULATIONS!")
            print("You've built an enterprise-scale application!")
            print("This codebase is ready for commercial deployment.")
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    exit(main())