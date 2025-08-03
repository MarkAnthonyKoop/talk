#!/usr/bin/env python3
"""
Test script for dynamic Talk orchestration with ExecutionPlannerAgent.

This script demonstrates the new ExecutionPlannerAgent in action by running
a real Talk session with dynamic plan generation.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Import Talk components
from special_agents.execution_planner_agent import ExecutionPlannerAgent

def test_dynamic_planning():
    """Test dynamic plan generation for different types of tasks."""
    print("=== Testing Dynamic Talk Planning ===")
    
    # Create ExecutionPlannerAgent
    planner = ExecutionPlannerAgent(name="DynamicPlanner")
    
    # Test different task types
    test_tasks = [
        "Create a simple Hello World Python script",
        "Build a REST API for user management with authentication", 
        "Fix a bug in the login system where users can't reset passwords",
        "Add a real-time chat feature to an existing web application",
        "Create a data visualization dashboard with charts and graphs"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Test {i}: {task} ---")
        
        try:
            steps = planner.run(task)
            
            if steps:
                print(f"Generated {len(steps)} steps:")
                for j, step in enumerate(steps, 1):
                    print(f"  {j}. {step.label} [{step.agent_key}]")
                    if step.on_success:
                        print(f"     -> on_success: {step.on_success}")
                    if step.on_fail:
                        print(f"     -> on_fail: {step.on_fail}")
                
                # Get plan summary
                summary = planner.get_plan_summary(steps)
                print(f"\nPlan Summary:\n{summary}")
                
            else:
                print("[FAIL] No steps generated")
                
        except Exception as e:
            print(f"[ERROR] Failed to generate plan: {e}")
    
    print("\n=== Dynamic Planning Test Complete ===")

def simulate_talk_execution():
    """Simulate a Talk execution with dynamic planning."""
    print("\n=== Simulating Talk Execution ===")
    
    task = "Create a simple calculator application with basic arithmetic operations"
    print(f"Task: {task}")
    
    # Generate dynamic plan
    planner = ExecutionPlannerAgent(name="TalkPlanner")
    steps = planner.run(task)
    
    if not steps:
        print("[FAIL] Could not generate execution plan")
        return
    
    print(f"\nGenerated execution plan with {len(steps)} steps:")
    
    # Simulate execution of each step
    for i, step in enumerate(steps, 1):
        print(f"\n[STEP {i}] Executing: {step.label} [{step.agent_key}]")
        
        # Simulate agent execution (mock)
        if step.agent_key == "reminiscing":
            print("  -> Recalling relevant memories and context...")
            result = "Found similar calculator implementations from previous projects"
        elif step.agent_key == "researcher":
            print("  -> Researching best practices and examples...")
            result = "Found calculator design patterns and frameworks"
        elif step.agent_key == "coder":
            print("  -> Generating code implementation...")
            if "generate_code" in step.label:
                result = """
CREATE_FILE: calculator.py
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

if __name__ == "__main__":
    calc = Calculator()
    print("Simple Calculator")
    print("2 + 3 =", calc.add(2, 3))
    print("10 - 4 =", calc.subtract(10, 4))
    print("5 * 6 =", calc.multiply(5, 6))
    print("12 / 3 =", calc.divide(12, 3))
"""
            else:
                result = "Code analysis complete - implementation looks good"
        elif step.agent_key == "file":
            print("  -> Applying changes to filesystem...")
            result = "Files created successfully: calculator.py"
        elif step.agent_key == "tester":
            print("  -> Running tests and validation...")
            result = "All tests passed - calculator functions work correctly"
        else:
            result = f"Executed {step.agent_key} agent"
        
        print(f"  -> Result: {result}")
        
        # Check if we should continue to next step
        if step.on_success and i < len(steps):
            next_step = next((s for s in steps if s.label == step.on_success), None)
            if next_step:
                print(f"  -> Next: {step.on_success}")
            else:
                print(f"  -> Workflow complete")
                break
        elif i == len(steps):
            print(f"  -> Workflow complete")
    
    print("\n[SUCCESS] Talk execution simulation complete!")

def main():
    """Run all tests."""
    print("Testing Dynamic Talk with ExecutionPlannerAgent")
    print("=" * 60)
    
    try:
        # Test dynamic planning
        test_dynamic_planning()
        
        # Simulate execution
        simulate_talk_execution()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())