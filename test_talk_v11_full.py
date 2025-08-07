#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk import TalkV11Orchestrator

# Test Talk v11 with more prompts
orchestrator = TalkV11Orchestrator(
    task="build a comprehensive SQL database engine with query parser, storage engine, indexing, transactions, and client",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/talk/v11_database",
    max_prompts=10  # Generate 10 components
)

result = orchestrator.run()
print(f"\nFinal Result: {result}")