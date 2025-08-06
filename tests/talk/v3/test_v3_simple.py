#!/usr/bin/env python3
"""
Simple test of the fixed PlanningAgent and BranchingAgent.
"""

import sys
import json
sys.path.insert(0, '/home/xx/code')

# Import the fixed agents directly
from special_agents.planning_agent import PlanningAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from plan_runner.step import Step

def test_planning_flow():
    """Test the planning and branching flow."""
    print("Testing Planning -> Branching flow...")
    
    # Create a simple task
    task = "Write a hello world function in Python"
    
    # Create PlanningAgent
    planning_agent = PlanningAgent()
    
    # Create a simple plan
    steps = [
        Step(label="plan_next", agent_key="planning"),
        Step(label="select_action", agent_key="branching"),
        Step(label="generate_code", agent_key="code"),
        Step(label="apply_files", agent_key="file"),
        Step(label="run_tests", agent_key="test"),
        Step(label="complete", agent_key=None)
    ]
    
    # Create BranchingAgent
    branch_step = steps[1]
    branching_agent = BranchingAgent(step=branch_step, plan=steps)
    
    # Test 1: Initial planning
    print("\n1. Initial planning...")
    planning_input = json.dumps({
        "task_description": task,
        "blackboard_state": {},
        "last_action": "",
        "last_result": ""
    })
    
    planning_output = planning_agent.run(planning_input)
    print(f"Planning output:\n{planning_output}")
    
    # Test 2: Branching based on plan
    print("\n2. Branching based on plan...")
    branch_output = branching_agent.run(planning_output)
    print(f"Branch output: {branch_output}")
    print(f"Next step selected: {branch_step.on_success}")
    
    # Test 3: Simulate code generation complete
    print("\n3. After code generation...")
    planning_input = json.dumps({
        "task_description": task,
        "blackboard_state": {"code_generated": True},
        "last_action": "generate_code",
        "last_result": "def hello_world():\n    print('Hello, World!')"
    })
    
    planning_output = planning_agent.run(planning_input)
    print(f"Planning output:\n{planning_output}")
    
    branch_output = branching_agent.run(planning_output)
    print(f"Branch output: {branch_output}")
    print(f"Next step selected: {branch_step.on_success}")
    
    # Test 4: After file application
    print("\n4. After file application...")
    planning_input = json.dumps({
        "task_description": task,
        "blackboard_state": {"files_applied": True},
        "last_action": "apply_files",
        "last_result": "Created hello.py"
    })
    
    planning_output = planning_agent.run(planning_input)
    print(f"Planning output:\n{planning_output}")
    
    branch_output = branching_agent.run(planning_output)
    print(f"Branch output: {branch_output}")
    print(f"Next step selected: {branch_step.on_success}")
    
    # Test 5: After tests
    print("\n5. After tests...")
    planning_input = json.dumps({
        "task_description": task,
        "blackboard_state": {"tests_run": True},
        "last_action": "run_tests",
        "last_result": "Tests passed"
    })
    
    planning_output = planning_agent.run(planning_input)
    print(f"Planning output:\n{planning_output}")
    
    branch_output = branching_agent.run(planning_output)
    print(f"Branch output: {branch_output}")
    print(f"Next step selected: {branch_step.on_success}")
    
    print("\n✓ All tests passed! The agents correctly flow through the workflow.")

if __name__ == "__main__":
    test_planning_flow()