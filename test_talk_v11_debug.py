#!/usr/bin/env python3
import sys
import json
sys.path.insert(0, '/home/xx/code')

from talk.talk import ComprehensivePlanningAgent, EnhancedCodeAgent
from pathlib import Path

# Create workspace
workspace = Path("/home/xx/code/tests/talk/v11_debug")
workspace.mkdir(parents=True, exist_ok=True)

print("Testing Talk v11 components...")

# Test planning agent
print("\n1. Testing ComprehensivePlanningAgent...")
planner = ComprehensivePlanningAgent(
    overrides={"provider": {"google": {"model_name": "gemini-2.0-flash"}}}
)

plan_input = json.dumps({
    "task": "build a simple key-value database",
    "max_prompts": 3
})

plan_output = planner.run(plan_input)
print(f"Plan output length: {len(plan_output)} chars")

# Try to parse it
import re
json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', plan_output, re.DOTALL)
if json_match:
    plan_json = json_match.group(1)
else:
    # Try direct JSON parsing if no markdown blocks
    plan_json = plan_output
    
try:
    plan = json.loads(plan_json)
    print(f"Successfully parsed plan with {len(plan.get('code_generation_prompts', []))} prompts")
    
    # Save plan for inspection
    with open(workspace / "plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    # Test code generation for first prompt
    if plan.get('code_generation_prompts'):
        print("\n2. Testing EnhancedCodeAgent...")
        code_agent = EnhancedCodeAgent(
            working_dir=workspace,
            overrides={"provider": {"google": {"model_name": "gemini-2.0-flash"}}}
        )
        
        first_prompt = plan['code_generation_prompts'][0]
        print(f"Generating code for: {first_prompt.get('component', 'unknown')}")
        
        code_output = code_agent.run(json.dumps(first_prompt))
        print(f"Code output length: {len(code_output)} chars")
        
        # Save raw output for inspection
        with open(workspace / "code_output.md", "w") as f:
            f.write(code_output)
        
        # Check what was saved to scratch
        scratch_dir = workspace / ".talk_scratch"
        if scratch_dir.exists():
            files = list(scratch_dir.rglob("*"))
            print(f"\nFiles in scratch: {len(files)}")
            for f in files:
                if f.is_file():
                    print(f"  - {f.relative_to(scratch_dir)} ({f.stat().st_size} bytes)")
except Exception as e:
    print(f"Failed to parse JSON: {e}")
    with open(workspace / "plan_raw.txt", "w") as f:
        f.write(plan_output)
    print(f"Saved raw plan to plan_raw.txt")

print("\nDebug output saved to:", workspace)