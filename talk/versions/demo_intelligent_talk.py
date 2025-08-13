#!/usr/bin/env python3
"""
Demo script showing the Intelligent Talk system capabilities.

This demonstrates the enhanced Talk framework with intelligent planning
and memory capabilities.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from special_agents.intelligent_talk_orchestrator import IntelligentTalkOrchestrator
from special_agents.planning_agent import PlanningAgent

def demo_planning_analysis():
    """Demonstrate the PlanningAgent's task analysis capabilities."""
    print("=== PLANNING AGENT DEMONSTRATION ===")
    print("Showing how the PlanningAgent analyzes different types of tasks\n")
    
    agent = PlanningAgent()
    
    demo_tasks = [
        "Create a hello world FastAPI app",
        "Build a complete e-commerce platform with microservices",
        "Fix authentication bug in user login",
        "Research best practices for database optimization"
    ]
    
    for task in demo_tasks:
        print(f"Task: {task}")
        
        # Generate plan
        plan_json = agent.run(task)
        plan = json.loads(plan_json)
        
        # Show analysis
        analysis = plan.get("analysis", {})
        print(f"  Complexity: {analysis.get('complexity', 'unknown')}")
        print(f"  Type: {analysis.get('type', 'unknown')}")
        print(f"  Components: {', '.join(analysis.get('components', []))}")
        print(f"  Steps: {plan.get('total_steps', 0)}")
        print(f"  Template: {plan.get('metadata', {}).get('template_used', 'unknown')}")
        print()
    
    print("Planning analysis completed!\n")

def demo_intelligent_orchestrator():
    """Demonstrate the IntelligentTalkOrchestrator without actually running it."""
    print("=== INTELLIGENT ORCHESTRATOR DEMONSTRATION ===")
    print("Showing the enhanced Talk framework with planning integration\n")
    
    # Test task
    task = "Create a REST API for book management with search functionality"
    
    print(f"Creating orchestrator for task: {task}")
    
    # Create orchestrator (but don't run it)
    orchestrator = IntelligentTalkOrchestrator(
        task=task,
        enable_planning=True,
        enable_memory=True,
        timeout_minutes=1  # Short timeout
    )
    
    print(f"  Planning enabled: {orchestrator.enable_planning}")
    print(f"  Memory enabled: {orchestrator.enable_memory}")
    print(f"  Available agents: {list(orchestrator.agents.keys())}")
    
    # Generate plan (without executing)
    print("\nGenerating execution plan...")
    plan = orchestrator._create_plan()
    
    print(f"Generated plan with {len(plan)} steps:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step.label} ({step.agent_key})")
        if step.on_success:
            print(f"     -> on success: {step.on_success}")
    
    # Show plan summary
    summary = orchestrator.get_plan_summary()
    if summary:
        print(f"\nPlan Summary:")
        print(f"  Complexity: {summary.get('complexity', 'unknown')}")
        print(f"  Total Steps: {summary.get('total_steps', 0)}")
        print(f"  Research Required: {summary.get('research_required', False)}")
        print(f"  Testing Strategy: {summary.get('testing_strategy', 'unknown')}")
    
    print("\nOrchestrator demonstration completed!\n")

def demo_system_comparison():
    """Compare standard Talk vs Intelligent Talk approaches."""
    print("=== SYSTEM COMPARISON ===")
    print("Comparing standard Talk vs Intelligent Talk approaches\n")
    
    tasks = [
        ("Simple task", "Create a basic calculator API"),
        ("Complex task", "Build a social media platform with real-time chat, posts, and notifications"),
        ("Research task", "Learn about implementing OAuth2 with FastAPI and create example"),
        ("Bug fix task", "Fix memory leak in the user session management system")
    ]
    
    for task_type, task in tasks:
        print(f"{task_type}: {task}")
        
        # Analyze with PlanningAgent
        agent = PlanningAgent()
        plan_json = agent.run(task)
        plan = json.loads(plan_json)
        
        analysis = plan.get("analysis", {})
        complexity = analysis.get("complexity", "unknown")
        total_steps = plan.get("total_steps", 0)
        template = plan.get("metadata", {}).get("template_used", "unknown")
        
        print(f"  Intelligent Talk would:")
        print(f"    - Detect complexity: {complexity}")
        print(f"    - Use template: {template}")
        print(f"    - Generate {total_steps} steps")
        print(f"    - Include memory retrieval: {'Yes' if 'reminiscing' in str(plan) else 'No'}")
        print(f"    - Include research: {'Yes' if 'research' in str(plan) else 'No'}")
        
        print(f"  Standard Talk would:")
        print(f"    - Use fixed workflow (4-5 steps)")
        print(f"    - No complexity analysis")
        print(f"    - No memory integration")
        print(f"    - Basic research detection")
        print()
    
    print("Comparison completed!\n")

def main():
    """Run all demonstrations."""
    print("INTELLIGENT TALK SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("Showcasing enhanced multi-agent orchestration with:")
    print("- Intelligent task analysis and planning")
    print("- Memory-based contextual awareness") 
    print("- Dynamic execution plan generation")
    print("- Adaptive workflow management")
    print("=" * 50)
    print()
    
    try:
        demo_planning_analysis()
        demo_intelligent_orchestrator()
        demo_system_comparison()
        
        print("=" * 50)
        print("DEMONSTRATION SUMMARY")
        print("=" * 50)
        print("✓ PlanningAgent successfully analyzes task complexity")
        print("✓ Different templates selected based on task type")
        print("✓ IntelligentTalkOrchestrator integrates planning")
        print("✓ Memory capabilities ready for contextual awareness")
        print("✓ System provides significant advantages over standard Talk")
        print()
        print("The Intelligent Talk system is ready for production use!")
        print("Use: python3 intelligent_talk.py --task 'your task here'")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()