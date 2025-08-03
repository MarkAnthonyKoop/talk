#!/usr/bin/env python3
"""
Test script for ExecutionPlannerAgent integration with TalkOrchestrator.

This script tests the new dynamic plan generation capabilities and validates
that the ExecutionPlannerAgent properly creates Step class instances that
can be used by the TalkOrchestrator.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from special_agents.execution_planner_agent import ExecutionPlannerAgent
from plan_runner.step import Step

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def test_execution_planner_basic():
    """Test basic ExecutionPlannerAgent functionality."""
    print("=== Testing ExecutionPlannerAgent Basic Functionality ===")
    
    try:
        # Create ExecutionPlannerAgent
        planner = ExecutionPlannerAgent(name="TestPlanner")
        
        # Test simple task
        simple_task = "Create a simple Hello World Python script"
        print(f"\nTesting simple task: {simple_task}")
        
        steps = planner.run(simple_task)
        
        if isinstance(steps, list) and len(steps) > 0:
            print(f"[OK] Generated {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step.label} [{step.agent_key}]")
                if step.on_success:
                    print(f"     -> on_success: {step.on_success}")
                if step.on_fail:
                    print(f"     -> on_fail: {step.on_fail}")
        else:
            print("[FAIL] Failed to generate steps")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing ExecutionPlannerAgent: {e}")
        return False

def test_execution_planner_complex():
    """Test ExecutionPlannerAgent with complex task."""
    print("\n=== Testing ExecutionPlannerAgent Complex Task ===")
    
    try:
        # Create ExecutionPlannerAgent
        planner = ExecutionPlannerAgent(name="TestPlanner")
        
        # Test complex task requiring research
        complex_task = "Create a comprehensive REST API for a task management system with user authentication, database integration, and real-time notifications"
        print(f"\nTesting complex task: {complex_task}")
        
        steps = planner.run(complex_task)
        
        if isinstance(steps, list) and len(steps) > 0:
            print(f"[OK] Generated {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step.label} [{step.agent_key}]")
                if step.on_success:
                    print(f"     -> on_success: {step.on_success}")
                if step.on_fail:
                    print(f"     -> on_fail: {step.on_fail}")
            
            # Check if research step is included
            research_steps = [s for s in steps if s.agent_key == "researcher"]
            if research_steps:
                print(f"[OK] Research step included for complex task")
            else:
                print("[INFO] No research step (may be expected depending on analysis)")
                
        else:
            print("[FAIL] Failed to generate steps")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing complex task: {e}")
        return False

def test_step_object_validity():
    """Test that generated Step objects are valid."""
    print("\n=== Testing Step Object Validity ===")
    
    try:
        # Create ExecutionPlannerAgent
        planner = ExecutionPlannerAgent(name="TestPlanner")
        
        # Generate steps
        task = "Build a simple calculator application"
        steps = planner.run(task)
        
        if not steps:
            print("[FAIL] No steps generated")
            return False
        
        # Validate each step
        for i, step in enumerate(steps):
            if not isinstance(step, Step):
                print(f"[FAIL] Step {i} is not a Step instance: {type(step)}")
                return False
            
            if not step.label:
                print(f"[FAIL] Step {i} has no label")
                return False
            
            if not step.agent_key:
                print(f"[FAIL] Step {i} has no agent_key")
                return False
            
            # Check agent_key is valid
            valid_agents = ["coder", "file", "tester", "researcher", "reminiscing"]
            if step.agent_key not in valid_agents:
                print(f"[FAIL] Step {i} has invalid agent_key: {step.agent_key}")
                return False
        
        print(f"[OK] All {len(steps)} steps are valid Step objects")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error validating Step objects: {e}")
        return False

def test_plan_summary():
    """Test the plan summary functionality."""
    print("\n=== Testing Plan Summary ===")
    
    try:
        # Create ExecutionPlannerAgent
        planner = ExecutionPlannerAgent(name="TestPlanner")
        
        # Generate steps
        task = "Create a web scraper for news articles"
        steps = planner.run(task)
        
        if not steps:
            print("[FAIL] No steps generated")
            return False
        
        # Get plan summary
        summary = planner.get_plan_summary(steps)
        print(f"Plan Summary:\n{summary}")
        
        if "Execution Plan" in summary and len(summary.split('\n')) > 1:
            print("[OK] Plan summary generated successfully")
            return True
        else:
            print("[FAIL] Invalid plan summary")
            return False
        
    except Exception as e:
        print(f"[FAIL] Error testing plan summary: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing ExecutionPlannerAgent Integration")
    print("=" * 50)
    
    tests = [
        test_execution_planner_basic,
        test_execution_planner_complex,
        test_step_object_validity,
        test_plan_summary
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())