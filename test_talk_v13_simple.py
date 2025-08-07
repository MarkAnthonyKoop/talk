#!/usr/bin/env python3
import sys
import json
sys.path.insert(0, '/home/xx/code')

# Test just the planning agent first
from special_agents.codebase_agent import CodebaseState, CodebasePlanningAgent

state = CodebaseState(task_description="build a simple key-value store")

planner = CodebasePlanningAgent(
    state=state,
    overrides={"provider": {"google": {"model_name": "gemini-2.0-flash"}}}
)

print("Testing planning agent...")
input_json = json.dumps({
    "task": "build a simple key-value store",
    "iteration": 0,
    "completed": [],
    "in_progress": [],
    "current": None,
    "errors": []
})

output = planner.run(input_json)
print(f"\nPlanning output:\n{output[:500]}...")

# Try to parse it
try:
    plan = json.loads(output)
    print(f"\nSuccessfully parsed JSON!")
    print(f"Components: {len(plan.get('components', []))}")
    print(f"Next action: {plan.get('next_action', 'unknown')}")
except Exception as e:
    print(f"\nFailed to parse JSON: {e}")
    print("This is why the loop is stuck!")