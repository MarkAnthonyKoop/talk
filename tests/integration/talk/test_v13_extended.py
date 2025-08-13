#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk_v13_codebase import TalkV13Orchestrator

# Test with higher iteration limit
orchestrator = TalkV13Orchestrator(
    task="build a simple REST API with user authentication",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v13_extended",
    max_iterations=20,  # Higher limit to complete all components
    verbose=True
)

print("[TEST] Running Talk v13 with extended iterations...")
result = orchestrator.run()

print(f"\n[TEST] Results:")
print(f"  Status: {result.get('status')}")
print(f"  Files: {result.get('files_generated')}")
print(f"  Lines: {result.get('total_lines')}")
print(f"  Iterations: {result.get('iterations')}")
print(f"  Components: {result.get('components_completed')}/{result.get('components_planned')}")