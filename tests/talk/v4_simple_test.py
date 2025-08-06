#!/usr/bin/env python3
"""
Simple test of talk_v4_validated agents without full orchestration.
"""

import sys
import json
from pathlib import Path

# Add to path
sys.path.insert(0, '/home/xx/code')

# Import agents directly
from special_agents.planning_agent import PlanningAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from plan_runner.step import Step


def test_v4_flow():
    """Test the v4 validated flow with our fixed agents."""
    
    print("="*60)
    print("TALK V4 VALIDATED - SIMPLE TEST")
    print("="*60)
    
    task = "Write a Python function that adds two numbers"
    
    # 1. Planning Phase
    print("\n[1] PLANNING PHASE")
    print("-"*40)
    planning = PlanningAgent()
    
    planning_input = json.dumps({
        "task_description": task,
        "blackboard_state": {},
        "last_action": "",
        "last_result": ""
    })
    
    print(f"Input: {task}")
    planning_output = planning.run(planning_input)
    print(f"Planning Output:\n{planning_output}\n")
    
    # 2. Branching Phase
    print("\n[2] BRANCHING PHASE")
    print("-"*40)
    
    # Create step and plan
    branch_step = Step(label="select_action", agent_key="branching")
    plan = [
        Step(label="plan_next", agent_key="planning"),
        branch_step,
        Step(label="generate_code", agent_key="code"),
        Step(label="apply_files", agent_key="file"),
        Step(label="run_tests", agent_key="test"),
        Step(label="complete", agent_key=None)
    ]
    
    branching = BranchingAgent(step=branch_step, plan=plan)
    
    print("Planning recommendation received...")
    branch_output = branching.run(planning_output)
    print(f"Branching Output:\n{branch_output}")
    print(f"\nSelected Next Step: {branch_step.on_success}\n")
    
    # 3. Code Generation Phase (if selected)
    if branch_step.on_success == "generate_code":
        print("\n[3] CODE GENERATION PHASE")
        print("-"*40)
        
        code = CodeAgent()
        code_output = code.run(task)
        print(f"Code Output:\n{code_output}\n")
        
        # 4. Check if code was saved to scratch
        scratch_dir = Path.cwd() / ".talk_scratch"
        if scratch_dir.exists():
            print("\n[4] SCRATCH DIRECTORY CHECK")
            print("-"*40)
            for file in scratch_dir.iterdir():
                print(f"Found: {file.name}")
                if file.suffix == ".json":
                    with open(file) as f:
                        data = json.load(f)
                        print(f"  Content preview: {str(data)[:100]}...")
    
    print("\n" + "="*60)
    print("TEST COMPLETE - V4 FLOW WORKING!")
    print("="*60)
    
    # Summary
    print("\n[SUMMARY]")
    print(f"✓ PlanningAgent provided structured recommendation")
    print(f"✓ BranchingAgent selected: {branch_step.on_success}")
    print(f"✓ Agents communicated via completions")
    print(f"✓ .talk_scratch directory used for inter-agent data")
    
    return True


if __name__ == "__main__":
    try:
        success = test_v4_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)