#!/usr/bin/env python3

"""
Quick test for YoutubeAgent using mock mode.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent, YoutubeAgentIntegration


def test_basic_functionality():
    """Test basic YoutubeAgent functionality."""
    print("Testing YoutubeAgent Basic Functionality")
    print("=" * 50)
    
    # Test 1: Initialization
    print("\n1. Testing initialization...")
    agent = YoutubeAgent(
        overrides={"debug": {"mock_mode": True}}  # Use mock mode to avoid LLM calls
    )
    print("   ✓ Agent initialized")
    
    # Test 2: Analysis type determination
    print("\n2. Testing analysis type determination...")
    test_cases = [
        ("Show my watch history", "watch_history"),
        ("What did I search for?", "search_history"),
        ("List my subscriptions", "subscriptions"),
        ("Show playlists", "playlists"),
        ("General overview", "general"),
    ]
    
    for input_text, expected in test_cases:
        result = agent._determine_analysis_type(input_text)
        if result == expected:
            print(f"   ✓ '{input_text}' -> {result}")
        else:
            print(f"   ✗ '{input_text}' -> {result} (expected: {expected})")
    
    # Test 3: Mock data analysis
    print("\n3. Testing with mock data...")
    
    # Add mock data
    agent.data_cache = {
        'watch_history': [
            {'title': 'Python Tutorial', 'channel': 'CodeChannel', 'timestamp': '2025-01-01, 10:00 AM'},
            {'title': 'Git Basics', 'channel': 'CodeChannel', 'timestamp': '2025-01-01, 11:00 AM'},
            {'title': 'Docker Guide', 'channel': 'DevOps Pro', 'timestamp': '2025-01-01, 02:00 PM'},
        ],
        'search_history': [
            {'query': 'python debugging', 'timestamp': '2025-01-01, 09:00 AM'},
            {'query': 'git merge conflicts', 'timestamp': '2025-01-01, 10:30 AM'},
        ],
        'subscriptions': [
            {'Channel Title': 'CodeChannel'},
            {'Channel Title': 'DevOps Pro'},
        ],
        'playlists': {
            'Watch later': [{'Video Title': 'Advanced Python'}],
            'Tutorials': [{'Video Title': 'React Basics'}, {'Video Title': 'Vue.js Guide'}]
        }
    }
    
    # Test watch history analysis
    result = agent._analyze_watch_history("analyze")
    print("   ✓ Watch history analysis completed")
    print(f"     Found {len(agent.data_cache['watch_history'])} videos")
    
    # Test search history analysis
    result = agent._analyze_search_history("analyze")
    print("   ✓ Search history analysis completed")
    print(f"     Found {len(agent.data_cache['search_history'])} searches")
    
    # Test subscription analysis
    result = agent._analyze_subscriptions("analyze")
    print("   ✓ Subscription analysis completed")
    print(f"     Found {len(agent.data_cache['subscriptions'])} subscriptions")
    
    # Test playlist analysis
    result = agent._analyze_playlists("analyze")
    print("   ✓ Playlist analysis completed")
    print(f"     Found {len(agent.data_cache['playlists'])} playlists")
    
    # Test 4: Integration helper
    print("\n4. Testing YoutubeAgentIntegration...")
    
    test_tasks = [
        ("Analyze my YouTube watch history", True),
        ("Fix a bug in the code", False),
        ("Show my video playlists", True),
        ("Refactor this function", False),
    ]
    
    for task, expected in test_tasks:
        result = YoutubeAgentIntegration.should_analyze_youtube_data(task)
        if result == expected:
            print(f"   ✓ '{task[:30]}...' -> {result}")
        else:
            print(f"   ✗ '{task[:30]}...' -> {result} (expected: {expected})")
    
    # Test 5: Results saving to scratch
    print("\n5. Testing scratch directory saving...")
    scratch_dir = Path(".talk/scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    # Count files before
    files_before = list(scratch_dir.glob("youtube_analysis_*.json"))
    
    # Save results
    agent._save_analysis_results("Test analysis", "Test insights")
    
    # Count files after  
    files_after = list(scratch_dir.glob("youtube_analysis_*.json"))
    
    if len(files_after) > len(files_before):
        print("   ✓ Results saved to scratch directory")
        newest = max(files_after, key=lambda f: f.stat().st_mtime)
        with open(newest, 'r') as f:
            data = json.load(f)
            if all(k in data for k in ['timestamp', 'analysis_result', 'insights']):
                print("   ✓ Saved data has correct structure")
    else:
        print("   ✗ Failed to save results")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    return True


def test_with_real_takeout():
    """Test with real takeout data if available."""
    takeout_path = Path(__file__).parent.parent.parent.parent.parent / "special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip"
    
    if not takeout_path.exists():
        print(f"Takeout file not found at {takeout_path}")
        return False
    
    print("\nTesting with Real Takeout Data")
    print("=" * 50)
    
    try:
        # Use mock mode to avoid LLM timeouts
        agent = YoutubeAgent(
            takeout_path=str(takeout_path),
            overrides={"debug": {"mock_mode": True}}
        )
        
        print("1. Loading takeout data...")
        agent._load_takeout_data()
        print(f"   ✓ Loaded {len(agent.data_cache)} data types")
        
        for key, value in agent.data_cache.items():
            if isinstance(value, list):
                print(f"     - {key}: {len(value)} items")
            elif isinstance(value, dict):
                total = sum(len(v) if isinstance(v, list) else 1 for v in value.values())
                print(f"     - {key}: {total} items in {len(value)} categories")
        
        print("\n2. Running analyses...")
        
        # Test each analysis type
        if 'watch_history' in agent.data_cache:
            result = agent._analyze_watch_history("test")
            print("   ✓ Watch history analysis completed")
        
        if 'search_history' in agent.data_cache:
            result = agent._analyze_search_history("test")
            print("   ✓ Search history analysis completed")
        
        if 'subscriptions' in agent.data_cache:
            result = agent._analyze_subscriptions("test")
            print("   ✓ Subscription analysis completed")
        
        if 'playlists' in agent.data_cache:
            result = agent._analyze_playlists("test")
            print("   ✓ Playlist analysis completed")
        
        print("\n✓ Real data testing complete!")
        return True
        
    except Exception as e:
        print(f"Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run basic tests
    success = test_basic_functionality()
    
    # Try with real data if available
    if success:
        print("\n" + "=" * 50)
        test_with_real_takeout()
    
    print("\n✅ All tests completed!")