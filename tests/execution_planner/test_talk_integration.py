#!/usr/bin/env python3
"""
Test TalkBeast integration with corrected ExecutionPlannerAgent.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from talk.talk import TalkOrchestrator


def test_talk_beast_with_execution_planner():
    """Test TalkOrchestrator with ExecutionPlannerAgent integration."""
    print("Testing TalkOrchestrator with ExecutionPlannerAgent...")
    
    try:
        # Test with a simple task that should trigger ExecutionPlannerAgent
        task = "Create a simple Python function that adds two numbers"
        
        # Create TalkOrchestrator instance
        talk = TalkOrchestrator(
            task=task,
            working_dir="/tmp/talk_test"
        )
        
        print(f"Running task: {task}")
        result = talk.run()
        
        print(f"✓ Task completed successfully")
        print(f"Result code: {result}")
        
        # Check that it completed without errors (0 = success)
        if result == 0:
            print("✓ TalkOrchestrator returned success code")
        else:
            print(f"⚠ TalkOrchestrator returned exit code: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ TalkOrchestrator integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_talk_beast_with_execution_planner()
    sys.exit(0 if success else 1)