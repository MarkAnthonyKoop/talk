#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/xx/code')

from special_agents.codebase_agent import CodebaseAgent

# Test the CodebaseAgent
agent = CodebaseAgent(
    task="build a REST API server with database models, authentication, and CRUD operations",
    model="gemini-2.0-flash",
    working_dir="/home/xx/code/tests/codebase_agent/api_server",
    max_iterations=20
)

result = agent.run()
print(f"\n[TEST] Final result: {result}")