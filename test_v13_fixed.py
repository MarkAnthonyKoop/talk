#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk_v13_codebase import TalkV13Orchestrator

# Test with a simpler task first
orchestrator = TalkV13Orchestrator(
    task="build a simple REST API with user authentication",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v13_fixed",
    max_iterations=10,  # Smaller limit for testing
    verbose=True
)

print("[TEST] Running fixed Talk v13...")
result = orchestrator.run()

print(f"\n[TEST] Results:")
print(f"  Status: {result.get('status')}")
print(f"  Files: {result.get('files_generated')}")
print(f"  Lines: {result.get('total_lines')}")
print(f"  Iterations: {result.get('iterations')}")
print(f"  Components: {result.get('components_completed')}/{result.get('components_planned')}")