#!/usr/bin/env python3
"""
Test basic agent initialization to understand proper usage.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_basic_agent():
    """Test basic Agent initialization."""
    print("Testing basic Agent initialization...")
    
    from agent.agent import Agent
    
    # Try creating a basic agent
    print("Creating basic agent...")
    start = time.time()
    
    agent = Agent(
        roles=["You are a helpful assistant"],
        overrides={"llm": {"provider": "google"}}  # Use a specific provider
    )
    
    elapsed = time.time() - start
    print(f"✓ Agent created in {elapsed:.2f}s")
    print(f"  Name: {agent.name}")
    print(f"  ID: {agent.id}")
    
    # Test run method
    print("\nTesting agent.run()...")
    start = time.time()
    
    result = agent.run("Say hello in 5 words or less")
    
    elapsed = time.time() - start
    print(f"✓ Response received in {elapsed:.2f}s")
    print(f"  Response: {result[:100]}")
    
    return True

if __name__ == "__main__":
    test_basic_agent()