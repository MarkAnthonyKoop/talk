#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk_v13_codebase import TalkV13Orchestrator, compare_with_claude_code

# Test Talk v13 with the same task Claude Code completed
orchestrator = TalkV13Orchestrator(
    task="build an agentic orchestration system",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v13_orchestrator",
    max_iterations=15,  # Limit for testing
    verbose=True
)

print("[TEST] Running Talk v13 with orchestration task...")
result = orchestrator.run()

# Compare with Claude Code
print("\n[TEST] Comparing with Claude Code output...")
comparison = compare_with_claude_code(
    result, 
    "/home/xx/code/tests/talk/claude_code_results/orchestrator"
)

# Print comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"Claude Code: {comparison['claude_code']['files']} files, {comparison['claude_code']['total_lines']:,} lines")
print(f"Talk v13: {comparison['talk_v13']['files']} files, {comparison['talk_v13']['total_lines']:,} lines")
print(f"Generation Ratio: {comparison['comparison']['line_ratio']:.0%} of Claude's output")
print(f"Time: {comparison['talk_v13']['execution_time_minutes']} minutes")
print("="*60)