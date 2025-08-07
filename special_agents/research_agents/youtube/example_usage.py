#!/usr/bin/env python3

"""
Example usage of YoutubeAgent.

This demonstrates how to use the YoutubeAgent to analyze YouTube takeout data.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent

def main():
    """Example usage of YoutubeAgent."""
    
    # Path to your YouTube takeout zip file
    takeout_path = "special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip"
    
    # Check if running from project root
    if not os.path.exists(takeout_path):
        # Try from current directory
        takeout_path = "./takeout_20250806T082512Z_1_001.zip"
    
    print("YouTube Agent Example Usage")
    print("=" * 50)
    print()
    
    # Initialize the agent
    print("Initializing YoutubeAgent...")
    agent = YoutubeAgent(takeout_path=takeout_path)
    
    # Example 1: General overview
    print("\nExample 1: General Overview")
    print("-" * 30)
    prompt = "Give me a general overview of my YouTube data"
    print(f"Prompt: {prompt}")
    print("\nProcessing...")
    # In production, you would call: result = agent.run(prompt)
    print("(Would analyze: watch history, search history, subscriptions, playlists)")
    
    # Example 2: Watch history analysis
    print("\nExample 2: Watch History Analysis")
    print("-" * 30)
    prompt = "What are my top 10 most watched YouTube channels?"
    print(f"Prompt: {prompt}")
    print("\nProcessing...")
    print("(Would analyze: channel frequency in watch history)")
    
    # Example 3: Search patterns
    print("\nExample 3: Search Pattern Analysis")
    print("-" * 30)
    prompt = "What topics do I search for most on YouTube?"
    print(f"Prompt: {prompt}")
    print("\nProcessing...")
    print("(Would analyze: search history keywords and frequencies)")
    
    # Example 4: Content preferences
    print("\nExample 4: Content Preferences")
    print("-" * 30)
    prompt = "Based on my watch and search history, what are my main interests?"
    print(f"Prompt: {prompt}")
    print("\nProcessing...")
    print("(Would analyze: cross-reference watch history with search terms)")
    
    # Example 5: Comprehensive analysis
    print("\nExample 5: Comprehensive Analysis")
    print("-" * 30)
    prompt = "Perform a comprehensive analysis of all my YouTube activity and provide insights"
    print(f"Prompt: {prompt}")
    print("\nProcessing...")
    print("(Would analyze: all available data and generate insights)")
    
    print("\n" + "=" * 50)
    print("YoutubeAgent Capabilities:")
    print("- Parse YouTube takeout zip files")
    print("- Analyze watch history patterns")
    print("- Extract search history insights")
    print("- Process subscription lists")
    print("- Examine playlist contents")
    print("- Generate comprehensive usage reports")
    print("- Save results to .talk/scratch/ for other agents")
    print("\nThe agent follows the Talk framework contract:")
    print("- Input: Natural language prompt")
    print("- Output: Analysis results and insights")
    print("- Side effects: Saves to .talk/scratch/ for inter-agent communication")

if __name__ == "__main__":
    main()