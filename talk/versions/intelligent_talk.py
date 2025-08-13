#!/usr/bin/env python3
"""
Intelligent Talk - Enhanced multi-agent orchestration with dynamic planning.

This is an enhanced version of the Talk framework that includes:
- Intelligent task analysis and planning via PlanningAgent
- Memory-based contextual awareness via ReminiscingAgent  
- Dynamic execution plan generation
- Adaptive workflow management

Usage:
    python3 intelligent_talk.py --task "Create a REST API for user management"
    python3 intelligent_talk.py --task "Build a web dashboard" --model claude-3-5-sonnet
    python3 intelligent_talk.py --interactive
    python3 intelligent_talk.py --task "Debug login issues" --disable-planning
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from special_agents.intelligent_talk_orchestrator import IntelligentTalkOrchestrator

def main():
    """Main entry point for Intelligent Talk."""
    parser = argparse.ArgumentParser(
        description="Intelligent Talk - AI-powered software development with dynamic planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task "Create a FastAPI app with user authentication"
  %(prog)s --task "Build a React dashboard with charts" --model claude-3-5-sonnet
  %(prog)s --interactive
  %(prog)s --task "Fix database connection timeout" --disable-planning
  %(prog)s --task "Research best practices for microservices" --enable-memory
        """
    )
    
    # Task specification
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="The software development task to execute"
    )
    
    # Mode selection
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for step-by-step execution"
    )
    
    # Configuration options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
    )
    
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Working directory for the project"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Maximum execution time in minutes (default: 30)"
    )
    
    # Feature toggles
    parser.add_argument(
        "--disable-planning",
        action="store_true",
        help="Disable intelligent planning and use fixed workflow"
    )
    
    parser.add_argument(
        "--disable-memory",
        action="store_true", 
        help="Disable memory/reminiscing capabilities"
    )
    
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Disable web search for research tasks"
    )
    
    # Session management
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from previous session directory"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Show execution plan before running"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.task and not args.interactive and not args.resume:
        parser.error("Must specify --task, --interactive, or --resume")
    
    # Handle interactive mode
    if args.interactive:
        task = input("Enter your development task: ").strip()
        if not task:
            print("No task specified. Exiting.")
            return 1
        args.task = task
    
    # Create and configure the orchestrator
    try:
        orchestrator = IntelligentTalkOrchestrator(
            task=args.task,
            working_dir=args.dir,
            model=args.model,
            timeout_minutes=args.timeout,
            interactive=args.interactive,
            resume_session=args.resume,
            enable_web_search=not args.disable_web_search,
            enable_planning=not args.disable_planning,
            enable_memory=not args.disable_memory
        )
        
        # Show configuration
        print(f"\n{'=' * 60}")
        print("INTELLIGENT TALK - AI-Powered Development")
        print(f"{'=' * 60}")
        print(f"Task: {args.task}")
        print(f"Model: {args.model}")
        print(f"Planning: {'Enabled' if not args.disable_planning else 'Disabled'}")
        print(f"Memory: {'Enabled' if not args.disable_memory else 'Disabled'}")
        print(f"Web Search: {'Enabled' if not args.disable_web_search else 'Disabled'}")
        print(f"Session: {orchestrator.session_dir}")
        print(f"Workspace: {orchestrator.working_dir}")
        
        # Show execution plan if requested
        if args.show_plan and not args.disable_planning:
            print(f"\n{'=' * 60}")
            print("Analyzing task and generating execution plan...")
            # Pre-generate plan to show it
            orchestrator._create_plan()
            orchestrator.print_plan_summary()
            
            proceed = input("\nProceed with this plan? [Y/n]: ").strip().lower()
            if proceed and proceed != 'y' and proceed != 'yes':
                print("Execution cancelled.")
                return 0
        
        print(f"\n{'=' * 60}")
        print("Starting execution...")
        print(f"{'=' * 60}")
        
        # Execute the workflow
        success = orchestrator.run()
        
        # Show results
        print(f"\n{'=' * 60}")
        if success:
            print("[SUCCESS] EXECUTION COMPLETED SUCCESSFULLY!")
        else:
            print("[FAILED] EXECUTION FAILED!")
        
        print(f"Session saved: {orchestrator.session_dir}")
        print(f"Workspace: {orchestrator.working_dir}")
        
        # Show plan summary if intelligent planning was used
        if not args.disable_planning:
            summary = orchestrator.get_plan_summary()
            if summary:
                print(f"\nPlan Summary:")
                print(f"  Type: {summary['plan_type']}")
                print(f"  Complexity: {summary['complexity']}")
                print(f"  Steps: {summary['total_steps']}")
        
        print(f"{'=' * 60}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())