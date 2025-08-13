#!/usr/bin/env python3
import sys
import json
sys.path.insert(0, '/home/xx/code')

from special_agents.codebase_agent import CodebaseState, CodebasePlanningAgent

# Test the planning agent directly
state = CodebaseState(task_description="build a simple REST API")

planner = CodebasePlanningAgent(
    state=state,
    overrides={"provider": {"google": {"model_name": "gemini-2.0-flash"}}}
)

context = {
    "task": "build a simple REST API",
    "iteration": 0,
    "completed": [],
    "in_progress": [],
    "current": None,
    "errors": []
}

prompt = f"""Current codebase generation state:
{json.dumps(context, indent=2)}

Analyze the task and current progress. If this is the first iteration, create a comprehensive plan
with 10-20 specific components. If we're mid-generation, assess progress and adjust.

Return JSON with:
{{
    "components": [
        {{
            "name": "core.storage_engine",
            "description": "Storage engine with file I/O and indexing",
            "estimated_lines": 400,
            "dependencies": [],
            "prompt": "Create a storage engine class that..."
        }},
        // ... more components
    ],
    "next_action": "generate_code|refine_code|run_tests|integrate|complete",
    "reasoning": "Explanation of decision",
    "is_complete": false,
    "confidence": 0.8
}}

Focus on generating a COMPLETE system with all necessary components."""

print("Sending prompt to planning agent...")
print(f"Prompt length: {len(prompt)} chars")

# Call the agent's run method
output = planner.run(json.dumps(context))

print(f"\nOutput type: {type(output)}")
print(f"Output length: {len(output)} chars")
print(f"Output starts with: {output[:100]}")

# Check if it's JSON
try:
    parsed = json.loads(output)
    print("\n✓ Successfully parsed as JSON!")
    print(f"  Keys: {list(parsed.keys())}")
except Exception as e:
    print(f"\n✗ Failed to parse as JSON: {e}")
    print(f"\nFull output:\n{output}")