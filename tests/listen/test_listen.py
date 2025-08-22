#!/usr/bin/env python3
"""
Test script for Listen v1 functionality.

This script tests the Listen application without requiring actual audio input.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from listen.relevance_agent import RelevanceAgent
from listen.audio_listener_agent import AudioListenerAgent


def test_relevance_agent():
    """Test the RelevanceAgent with sample transcriptions."""
    print("\n=== Testing RelevanceAgent ===\n")
    
    # Create agent with a task
    task = "Create a REST API for a custom GPT that can interact with bash commands"
    agent = RelevanceAgent(task_description=task, relevance_threshold=0.3)
    
    # Sample transcriptions to test
    test_transcriptions = [
        {
            "text": "I need you to create a REST API that can execute bash commands",
            "timestamp": "2025-08-21T10:00:00"
        },
        {
            "text": "The weather is nice today",
            "timestamp": "2025-08-21T10:00:05"
        },
        {
            "text": "Let's add endpoints for running commands and getting output",
            "timestamp": "2025-08-21T10:00:10"
        },
        {
            "text": "Make sure to implement proper authentication for the API",
            "timestamp": "2025-08-21T10:00:15"
        },
        {
            "text": "I had lunch at a great restaurant",
            "timestamp": "2025-08-21T10:00:20"
        },
        {
            "text": "Can you also add support for streaming command output?",
            "timestamp": "2025-08-21T10:00:25"
        }
    ]
    
    # Filter transcriptions
    relevant = agent.filter_transcriptions(test_transcriptions)
    
    print(f"Task: {task}\n")
    print(f"Total transcriptions: {len(test_transcriptions)}")
    print(f"Relevant transcriptions: {len(relevant)}\n")
    
    # Display results
    for trans in relevant:
        print(f"Relevance: {trans['overall_score']:.2f} - {trans['text']}")
        if trans['matched_keywords']:
            print(f"  Matched keywords: {', '.join(trans['matched_keywords'][:5])}")
        if trans['actionable_items']:
            print(f"  Actionable: {trans['actionable_items'][0]}")
        print()
    
    # Get action summary
    summary = agent.get_action_summary()
    print("\nAction Summary:")
    print(f"  Total action items: {summary['total_action_items']}")
    print(f"  Average relevance: {summary['average_relevance']:.2f}")
    
    if summary['unique_actions']:
        print("  Unique actions:")
        for action in summary['unique_actions'][:3]:
            print(f"    - {action}")


def test_single_evaluation():
    """Test single content evaluation."""
    print("\n=== Testing Single Content Evaluation ===\n")
    
    task = "Build a web scraper for news articles"
    agent = RelevanceAgent(task_description=task)
    
    test_content = "We need to build a Python web scraper that can extract news articles from multiple websites and store them in a database."
    
    result = agent.evaluate_relevance(test_content)
    
    print(f"Task: {task}")
    print(f"Content: {test_content}\n")
    print(f"Overall relevance: {result['overall_score']:.2f}")
    print(f"Keyword score: {result['keyword_score']:.2f}")
    print(f"Concept score: {result['concept_score']:.2f}")
    print(f"Action score: {result['action_score']:.2f}")
    print(f"Is relevant: {result['is_relevant']}")
    
    if result['matched_keywords']:
        print(f"\nMatched keywords: {', '.join(result['matched_keywords'][:5])}")
    
    if result['matched_concepts']:
        print(f"Matched concepts: {', '.join(result['matched_concepts'])}")
    
    if result['actionable_items']:
        print(f"\nActionable items:")
        for item in result['actionable_items']:
            print(f"  - {item}")


def test_audio_agent_mock():
    """Test AudioListenerAgent with mock transcriptions."""
    print("\n=== Testing AudioListenerAgent Mock ===\n")
    
    task = "Implement a chat application with real-time messaging"
    
    # Note: This would normally require audio hardware
    # For testing, we'll just verify the agent initializes correctly
    try:
        agent = AudioListenerAgent(task_description=task, continuous=False)
        
        print(f"AudioListenerAgent initialized successfully")
        print(f"Task: {task}")
        print(f"Keywords extracted: {agent.relevance_keywords[:5]}")
        
        # Test relevance calculation
        test_text = "Let's implement a chat application with WebSocket support"
        score = agent._calculate_relevance(test_text)
        print(f"\nTest relevance calculation:")
        print(f"  Text: {test_text}")
        print(f"  Score: {score:.2f}")
        
    except Exception as e:
        print(f"Note: Audio agent initialization failed (expected without audio hardware)")
        print(f"Error: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LISTEN v1 - Component Tests")
    print("=" * 60)
    
    test_relevance_agent()
    test_single_evaluation()
    test_audio_agent_mock()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()