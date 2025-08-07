#!/usr/bin/env python3

"""
Simple test for YoutubeAgent.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from youtube_agent import YoutubeAgent

def main():
    """Simple test of YoutubeAgent."""
    
    # Path to takeout file
    takeout_path = "./takeout_20250806T082512Z_1_001.zip"
    
    print("Testing YoutubeAgent...")
    
    # Test with mock mode if no API keys available
    try:
        agent = YoutubeAgent(
            takeout_path=takeout_path,
            overrides={"debug": {"mock_mode": True}}
        )
        print("✓ Agent created successfully")
        
        # Test the run method
        result = agent.run("Analyze my YouTube watch history")
        print(f"✓ Agent run completed")
        print(f"Result: {result[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    print("\n✓ YoutubeAgent is working!")
    return True

if __name__ == "__main__":
    main()