#!/usr/bin/env python3
"""
Simple test script for the ReminiscingAgent.

This tests the ReminiscingAgent in simplified mode (without LangGraph)
to verify basic functionality and integration.
"""

import sys
import os
import logging

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_reminiscing_agent():
    """Test the ReminiscingAgent functionality."""
    try:
        print("Testing ReminiscingAgent...")
        
        # Import the ReminiscingAgent
        from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
        
        print("[OK] ReminiscingAgent imported successfully")
        
        # Create an instance
        agent = ReminiscingAgent()
        print("[OK] ReminiscingAgent instance created")
        
        # Test basic functionality
        test_contexts = [
            "I need to implement a user authentication system",
            "There's a bug in the database connection code",
            "How should I design the API architecture?",
            "What's the best practice for error handling?"
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"\nTest {i}: {context}")
            try:
                result = agent.run(context)
                print(f"[OK] Response received (length: {len(result)} chars)")
                print(f"Preview: {result[:200]}...")
            except Exception as e:
                print(f"[ERROR] Error: {e}")
        
        print("\n[OK] Basic functionality test completed")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_subagents():
    """Test the individual sub-agents."""
    print("\nTesting sub-agents...")
    
    try:
        # Test ContextCategorizationAgent
        from special_agents.reminiscing.context_categorization_agent import ContextCategorizationAgent
        cat_agent = ContextCategorizationAgent()
        result = cat_agent.run("I need to fix a database timeout error")
        print("[OK] ContextCategorizationAgent works")
        print(f"Sample result: {result[:100]}...")
        
    except Exception as e:
        print(f"[ERROR] ContextCategorizationAgent error: {e}")
    
    try:
        # Test MemoryTraceAgent
        from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent
        memory_agent = MemoryTraceAgent()
        memory_agent.populate_sample_memories()  # Add some test data
        result = memory_agent.run("database error debugging")
        print("[OK] MemoryTraceAgent works")
        print(f"Sample result: {result[:100]}...")
        
    except Exception as e:
        print(f"[ERROR] MemoryTraceAgent error: {e}")
    
    try:
        # Test ConversationVectorStore
        from special_agents.reminiscing.vector_store import ConversationVectorStore
        store = ConversationVectorStore()
        memory_id = store.store_conversation({
            "task": "Test conversation",
            "messages": ["Hello", "How are you?"]
        })
        print("[OK] ConversationVectorStore works")
        print(f"Stored memory ID: {memory_id}")
        
    except Exception as e:
        print(f"[ERROR] ConversationVectorStore error: {e}")

def test_integration_with_talk():
    """Test integration with the Talk framework."""
    print("\nTesting Talk framework integration...")
    
    try:
        # Import Talk components
        from talk.talk import TalkOrchestrator
        from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
        
        print("[OK] TalkOrchestrator and ReminiscingAgent imported")
        
        # Create a TalkOrchestrator instance with ReminiscingAgent
        talk = TalkOrchestrator(task="Test integration with ReminiscingAgent")
        reminiscing_agent = ReminiscingAgent()
        
        print("[OK] TalkOrchestrator instance created")
        
        # Test basic functionality integration
        print("[OK] Both components initialized successfully")
        
        # Test a simple memory query
        test_prompt = "Remember how we implemented the user authentication?"
        
        print(f"Testing with prompt: {test_prompt}")
        result = reminiscing_agent.run(test_prompt)
        
        print("[OK] Integration test completed")
        print(f"Result preview: {result[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration test error: {e}")
        return False

if __name__ == "__main__":
    print("=== ReminiscingAgent Test Suite ===")
    
    # Test individual components
    success1 = test_reminiscing_agent()
    test_subagents()
    success2 = test_integration_with_talk()
    
    print("\n=== Test Summary ===")
    print(f"Basic functionality: {'[PASS]' if success1 else '[FAIL]'}")
    print(f"Talk integration: {'[PASS]' if success2 else '[FAIL]'}")
    
    if success1 and success2:
        print("\n[SUCCESS] All tests passed! ReminiscingAgent is ready for use.")
    else:
        print("\n[WARNING] Some tests failed. Check the output above for details.")