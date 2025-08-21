#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from talk.talk import TalkV11Orchestrator

# Test Talk v11
orchestrator = TalkV11Orchestrator(
    task="build a SQL database engine with query parser and storage",
    model="gemini-2.0-flash",
    max_prompts=5
)

result = orchestrator.run()
print(f"Result: {result}")