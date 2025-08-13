#!/usr/bin/env python3
"""
Demonstration of the ReminiscingAgent capabilities.

This script showcases how the ReminiscingAgent can categorize different types
of programming tasks and provide relevant memory traces to inform decision-making.
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to be less verbose for demo
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

def demo_reminiscing_agent():
    """Demonstrate the ReminiscingAgent with various scenarios."""
    
    print("=== ReminiscingAgent Demonstration ===")
    print("Showcasing human-like memory traces for programming tasks\n")
    
    # Initialize the agent
    agent = ReminiscingAgent()
    
    # Add some sample memories to the agent
    print("Populating the agent with sample memories...")
    agent.memory_trace_agent.populate_sample_memories()
    print("Memory populated with 8 sample traces\n")
    
    # Test scenarios representing different types of programming contexts
    scenarios = [
        {
            "context": "I need to implement user authentication with JWT tokens",
            "expected_category": "implementation",
            "description": "Code implementation task"
        },
        {
            "context": "There's a memory leak in our React application causing performance issues",
            "expected_category": "debugging", 
            "description": "Debugging task"
        },
        {
            "context": "How should I design the microservice architecture for this e-commerce platform?",
            "expected_category": "architectural",
            "description": "Architecture design task"
        },
        {
            "context": "What are the best practices for error handling in REST APIs?",
            "expected_category": "research",
            "description": "Research/learning task"
        },
        {
            "context": "The database connection keeps timing out in production",
            "expected_category": "debugging",
            "description": "Production issue"
        }
    ]
    
    print("Testing different types of programming contexts:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"--- Scenario {i}: {scenario['description']} ---")
        print(f"Context: {scenario['context']}")
        print()
        
        # Get the agent's response
        result = agent.run(scenario['context'])
        
        # Extract key information from the response
        lines = result.split('\n')
        category = "unknown"
        strategy = "unknown"
        confidence = "unknown"
        
        # Look for the actual format used by the agent
        for line in lines:
            if "CATEGORY:" in line:
                category = line.split("CATEGORY:", 1)[1].strip()
            elif "STRATEGY:" in line:
                strategy = line.split("STRATEGY:", 1)[1].strip()
            elif "CONFIDENCE:" in line:
                confidence = line.split("CONFIDENCE:", 1)[1].strip()
            elif line.startswith("Context Category:"):
                category = line.split(":", 1)[1].strip()
            elif line.startswith("Search Strategy:"):
                strategy = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                confidence = line.split(":", 1)[1].strip()
        
        print(f"Category: {category}")
        print(f"Strategy: {strategy}")
        print(f"Confidence: {confidence}")
        
        # Show if the categorization matches expectations
        expected = scenario['expected_category']
        match_status = "[MATCH]" if expected.lower() in category.lower() else "[DIFFERENT]"
        print(f"Expected: {expected} {match_status}")
        print()
        
        # Show a snippet of memory traces found
        if "No relevant memory traces" not in result:
            print("Memory traces found - system can provide contextual guidance")
        else:
            print("No specific memory traces - system would use general knowledge")
        
        print("-" * 60)
        print()

def demo_memory_storage():
    """Demonstrate how the agent stores and retrieves conversation memories."""
    
    print("=== Memory Storage Demonstration ===")
    print("Showing how the agent stores and retrieves conversation context\n")
    
    agent = ReminiscingAgent()
    
    # Store some sample conversation data
    conversations = [
        {
            "task": "Implement OAuth2 authentication",
            "messages": [
                "We need to add OAuth2 support to our API",
                "I'll use the passport.js library for this",
                "The implementation should support Google and GitHub providers"
            ],
            "outcome": "Successfully implemented OAuth2 with multiple providers"
        },
        {
            "task": "Fix database performance issue",
            "messages": [
                "The user queries are taking too long",
                "Added indexes on the user_id and created_at columns",
                "Query time improved from 2s to 50ms"
            ],
            "outcome": "Database performance optimized"
        }
    ]
    
    print("Storing conversation memories...")
    for conv in conversations:
        try:
            memory_id = agent.store_conversation(conv)
            if memory_id:
                print(f"Stored conversation: {conv['task']} (ID: {memory_id[:8]}...)")
            else:
                print(f"Stored conversation: {conv['task']} (no ID returned)")
        except Exception as e:
            print(f"Error storing conversation {conv['task']}: {e}")
    
    print("\nMemory storage statistics:")
    stats = agent.vector_store.get_stats()
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Total code contexts: {stats['total_code_contexts']}")
    print(f"Total embeddings: {stats['total_embeddings']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run the demonstrations
    demo_reminiscing_agent()
    demo_memory_storage()
    
    print("\n=== Summary ===")
    print("The ReminiscingAgent demonstrates:")
    print("1. Intelligent context categorization")
    print("2. Strategy-based memory search")
    print("3. Spreading activation networks for associative memory")
    print("4. Conversation and code context storage")
    print("5. Human-like memory traces for better decision making")
    print("\nThe system is ready for integration with the Talk framework!")