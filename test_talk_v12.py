#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk_v12_tracked import TalkV12Orchestrator

# Test Talk v12 with tracking
orchestrator = TalkV12Orchestrator(
    task="build a key-value database with storage engine, caching, and API",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v12_database",
    max_prompts=5  # Start with 5 components for testing
)

result = orchestrator.run()
print(f"\nTest completed with result: {result}")