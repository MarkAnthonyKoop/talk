#!/usr/bin/env python3
"""
Test YouTubeAgent with proper initialization.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_youtube_agent():
    """Test YouTubeAgent initialization and basic usage."""
    print("Testing YouTubeAgent...")
    print("=" * 50)
    
    from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent
    
    # Path to takeout
    takeout_path = Path("/home/xx/code/special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip")
    
    if not takeout_path.exists():
        print(f"✗ Takeout file not found at {takeout_path}")
        return False
    
    print(f"✓ Found takeout: {takeout_path.name}")
    
    # Create agent with specific provider settings
    print("\nCreating YouTubeAgent...")
    start = time.time()
    
    agent = YoutubeAgent(
        takeout_path=str(takeout_path),
        overrides={"llm": {"provider": "google"}}
    )
    
    elapsed = time.time() - start
    print(f"✓ Agent created in {elapsed:.2f}s")
    
    # Load the takeout data
    print("\nLoading takeout data...")
    start = time.time()
    
    agent._load_takeout_data()
    
    elapsed = time.time() - start
    print(f"✓ Data loaded in {elapsed:.2f}s")
    print(f"  Data types: {list(agent.data_cache.keys())}")
    
    # Test a simple analysis
    print("\nRunning simple analysis...")
    start = time.time()
    
    result = agent.run("How many YouTube videos have I watched in total?")
    
    elapsed = time.time() - start
    print(f"✓ Analysis completed in {elapsed:.2f}s")
    print(f"\nResult preview:")
    print("-" * 40)
    print(result[:500])
    print("-" * 40)
    
    return True

def test_background_processing():
    """Test running analysis in background with progress updates."""
    print("\n\nTesting Background Processing")
    print("=" * 50)
    
    import threading
    import queue
    
    from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent
    
    # Create a queue for progress updates
    progress_queue = queue.Queue()
    
    def analyze_in_background(q):
        """Run analysis in background thread."""
        try:
            q.put("Starting analysis...")
            
            takeout_path = "/home/xx/code/special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip"
            
            q.put("Creating agent...")
            agent = YoutubeAgent(
                takeout_path=takeout_path,
                overrides={"llm": {"provider": "google"}}
            )
            
            q.put("Loading data...")
            agent._load_takeout_data()
            
            q.put("Analyzing watch history...")
            result = agent._analyze_watch_history("analyze")
            
            q.put("Analysis complete!")
            q.put(("RESULT", result[:500]))
            
        except Exception as e:
            q.put(("ERROR", str(e)))
    
    # Start background thread
    thread = threading.Thread(target=analyze_in_background, args=(progress_queue,))
    thread.start()
    
    # Monitor progress with sleep loop
    print("Processing in background...")
    while thread.is_alive():
        # Check for updates
        while not progress_queue.empty():
            msg = progress_queue.get()
            if isinstance(msg, tuple):
                if msg[0] == "RESULT":
                    print(f"\n✓ Result: {msg[1][:200]}...")
                elif msg[0] == "ERROR":
                    print(f"\n✗ Error: {msg[1]}")
            else:
                print(f"  {msg}")
        
        # Sleep for 1 second
        time.sleep(1)
        print(".", end="", flush=True)
    
    # Get final messages
    while not progress_queue.empty():
        msg = progress_queue.get()
        if isinstance(msg, tuple):
            print(f"\nFinal: {msg[0]}")
        else:
            print(f"  {msg}")
    
    print("\n✓ Background processing complete")
    return True

if __name__ == "__main__":
    # Test direct usage
    success = test_youtube_agent()
    
    # Test background processing
    if success:
        test_background_processing()