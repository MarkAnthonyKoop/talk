#!/usr/bin/env python3
"""
Test ExecutionPlannerAgent to verify it returns List[Step] correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from special_agents.execution_planner_agent import ExecutionPlannerAgent
from plan_runner.step import Step


def test_execution_planner_returns_list_of_steps():
    """Test that ExecutionPlannerAgent returns List[Step]."""
    print("Testing ExecutionPlannerAgent return type...")
    
    # Create the agent
    planner = ExecutionPlannerAgent()
    
    # Test with a simple task
    task = "Create a simple hello world Python script"
    
    try:
        result = planner.run(task)
        
        # Verify it returns a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        print(f"✓ Returns list: {len(result)} items")
        
        # Verify list contains Step objects
        for i, item in enumerate(result):
            assert isinstance(item, Step), f"Item {i} is {type(item)}, expected Step"
        
        print(f"✓ All {len(result)} items are Step objects")
        
        # Print the generated plan
        print("\nGenerated execution plan:")
        for i, step in enumerate(result, 1):
            print(f"  {i}. {step.agent_key} - {step.label}")
        
        print("\n✓ ExecutionPlannerAgent test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ ExecutionPlannerAgent test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_execution_planner_returns_list_of_steps()
    sys.exit(0 if success else 1)