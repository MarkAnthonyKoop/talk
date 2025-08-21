#!/usr/bin/env python3
"""Test Talk v14 Enhanced with production-grade standards."""

import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk_v14_enhanced import TalkV14Orchestrator

# Test with a focused task to see quality in action
orchestrator = TalkV14Orchestrator(
    task="build a REST API with user authentication and rate limiting",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v14_enhanced",
    quality_threshold=0.85,  # High quality standard
    max_iterations=30,  # Enough for refinement loops
    verbose=True
)

print("\n[TEST] Running Talk v14 Enhanced with strict quality standards...")
print("[TEST] This will include:")
print("  - Hierarchical planning with todo tracking")
print("  - Quality evaluation and refinement loops")
print("  - Dependency management")
print("  - Supporting files (README, Dockerfile, etc.)")
print("\n")

result = orchestrator.run()

if result.get("status") == "success":
    print("\n[TEST] ✓ Success!")
    print(f"[TEST] Check the generated code at: {result.get('working_directory')}")
    print(f"[TEST] Check the todos at: {result.get('working_directory')}/.talk/talk_todos/")
else:
    print(f"\n[TEST] ✗ Failed: {result.get('error')}")