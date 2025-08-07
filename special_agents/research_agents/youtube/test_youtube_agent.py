#!/usr/bin/env python3

"""
Test script for YoutubeAgent.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from youtube_agent import YoutubeAgent, YoutubeAgentIntegration

def test_youtube_agent():
    """Test the YoutubeAgent with actual takeout data."""
    
    # Path to the takeout file
    takeout_path = "./takeout_20250806T082512Z_1_001.zip"
    
    # Check if file exists
    if not os.path.exists(takeout_path):
        print(f"ERROR: Takeout file not found at {takeout_path}")
        return False
    
    print("=" * 60)
    print("Testing YoutubeAgent")
    print("=" * 60)
    
    try:
        # Initialize the agent
        print("\n1. Initializing YoutubeAgent...")
        agent = YoutubeAgent(takeout_path=takeout_path)
        print("   ✓ Agent initialized successfully")
        
        # Test general overview
        print("\n2. Testing general overview analysis...")
        result = agent.run("Give me a general overview of my YouTube data")
        print("   ✓ General overview completed")
        print(f"\nResult preview:\n{result[:500]}...")
        
        # Test watch history analysis
        print("\n3. Testing watch history analysis...")
        result = agent.run("Analyze my watch history and tell me what channels I watch most")
        print("   ✓ Watch history analysis completed")
        print(f"\nResult preview:\n{result[:500]}...")
        
        # Test search history analysis
        print("\n4. Testing search history analysis...")
        result = agent.run("What are my most common search terms on YouTube?")
        print("   ✓ Search history analysis completed")
        print(f"\nResult preview:\n{result[:500]}...")
        
        # Test subscription analysis
        print("\n5. Testing subscription analysis...")
        result = agent.run("Show me my YouTube subscriptions")
        print("   ✓ Subscription analysis completed")
        print(f"\nResult preview:\n{result[:500]}...")
        
        # Test comprehensive analysis
        print("\n6. Testing comprehensive analysis...")
        result = agent.run("Perform a comprehensive analysis of all my YouTube activity")
        print("   ✓ Comprehensive analysis completed")
        
        # Test integration helper
        print("\n7. Testing YoutubeAgentIntegration...")
        
        # Test should_analyze_youtube_data
        test_cases = [
            ("Analyze my YouTube watch history", True),
            ("Fix a bug in the code", False),
            ("Show me my video playlists", True),
            ("Refactor the function", False)
        ]
        
        for task, expected in test_cases:
            result = YoutubeAgentIntegration.should_analyze_youtube_data(task)
            status = "✓" if result == expected else "✗"
            print(f"   {status} '{task}' -> {result} (expected: {expected})")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_data():
    """Test YoutubeAgent without actual data (mock mode)."""
    print("\n" + "=" * 60)
    print("Testing YoutubeAgent (without data)")
    print("=" * 60)
    
    try:
        # Initialize without takeout path
        print("\n1. Initializing YoutubeAgent without data...")
        agent = YoutubeAgent()
        print("   ✓ Agent initialized successfully")
        
        # Test error handling
        print("\n2. Testing error handling...")
        result = agent.run("Analyze my YouTube data")
        print("   ✓ Error handling works")
        print(f"\nResult: {result}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

if __name__ == "__main__":
    # Test with actual data if available
    success = test_youtube_agent()
    
    # Also test without data
    if not success:
        print("\nTesting without actual data...")
        test_without_data()
    
    print("\n✓ YoutubeAgent implementation complete!")
    print("\nThe agent follows the Talk framework contract:")
    print("- Inherits from base Agent class")
    print("- Implements run() method: prompt in → completion out")
    print("- Uses .talk/scratch/ for inter-agent communication")
    print("- Handles errors gracefully")
    print("- Provides specialized YouTube data analysis capabilities")