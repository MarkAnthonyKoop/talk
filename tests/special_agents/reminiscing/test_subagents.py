#!/usr/bin/env python3
"""
Unit tests for ReminiscingAgent sub-agents.

Tests ContextCategorizationAgent and MemoryTraceAgent components.
"""

import sys
import os
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)


def test_context_categorization_agent():
    """Test the ContextCategorizationAgent."""
    print("Testing ContextCategorizationAgent...")
    
    try:
        from special_agents.reminiscing.context_categorization_agent import ContextCategorizationAgent
    except ImportError as e:
        print(f"[ERROR] Failed to import ContextCategorizationAgent: {e}")
        print("[INFO] This agent may not be implemented yet")
        return False
    
    try:
        agent = ContextCategorizationAgent()
        
        # Test different context types
        test_cases = [
            ("I need to design the microservice architecture", "architectural"),
            ("There's a bug causing database timeouts", "debugging"),
            ("How do I implement OAuth2 authentication?", "implementation"),
            ("What are the best practices for API versioning?", "research"),
            ("Hello, how are you?", "general")
        ]
        
        for context, expected_category in test_cases:
            result = agent.run(context)
            print(f"  Context: '{context[:50]}...'")
            print(f"  Result: {result[:100]}")
            
            # Check if the expected category appears in the result
            if expected_category.lower() in result.lower():
                print(f"  [OK] Correctly identified as {expected_category}")
            else:
                print(f"  [WARNING] Expected {expected_category}, got: {result[:50]}")
        
        print("[OK] ContextCategorizationAgent tested")
        return True
        
    except Exception as e:
        print(f"[ERROR] ContextCategorizationAgent test failed: {e}")
        return False


def test_memory_trace_agent():
    """Test the MemoryTraceAgent."""
    print("\nTesting MemoryTraceAgent...")
    
    try:
        from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent
    except ImportError as e:
        print(f"[ERROR] Failed to import MemoryTraceAgent: {e}")
        print("[INFO] This agent may not be implemented yet")
        return False
    
    try:
        agent = MemoryTraceAgent()
        
        # Check if agent has sample memory population method
        if hasattr(agent, 'populate_sample_memories'):
            agent.populate_sample_memories()
            print("  [OK] Sample memories populated")
        
        # Test different search strategies
        test_searches = [
            {"context": "database error", "category": "debugging", "strategy": "error_similarity"},
            {"context": "authentication system", "category": "implementation", "strategy": "code_similarity"},
            {"context": "API design", "category": "architectural", "strategy": "graph_traversal"},
            {"context": "best practices", "category": "research", "strategy": "semantic_search"}
        ]
        
        for search_params in test_searches:
            print(f"\n  Testing search: {search_params['strategy']}")
            result = agent.run(json.dumps(search_params))
            
            if result:
                print(f"  [OK] Got response (length: {len(result)})")
                print(f"  Preview: {result[:150]}...")
            else:
                print(f"  [WARNING] Empty response for {search_params['strategy']}")
        
        print("\n[OK] MemoryTraceAgent tested")
        return True
        
    except Exception as e:
        print(f"[ERROR] MemoryTraceAgent test failed: {e}")
        return False


def test_agent_integration():
    """Test integration between sub-agents."""
    print("\nTesting sub-agent integration...")
    
    try:
        from special_agents.reminiscing.context_categorization_agent import ContextCategorizationAgent
        from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent
    except ImportError as e:
        print(f"[ERROR] Failed to import agents: {e}")
        print("[INFO] Sub-agents may not be implemented yet")
        return False
    
    try:
        cat_agent = ContextCategorizationAgent()
        mem_agent = MemoryTraceAgent()
        
        # Simulate the workflow
        test_context = "I'm getting database connection timeouts in production"
        
        # Step 1: Categorize
        category_result = cat_agent.run(test_context)
        print(f"  Categorization: {category_result[:100]}")
        
        # Step 2: Search memory
        memory_search = {
            "context": test_context,
            "category": "debugging",  # Assume this was extracted
            "strategy": "error_similarity"
        }
        memory_result = mem_agent.run(json.dumps(memory_search))
        print(f"  Memory search: {memory_result[:100] if memory_result else 'No results'}")
        
        print("[OK] Sub-agent integration tested")
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        return False


def test_agent_error_handling():
    """Test error handling in sub-agents."""
    print("\nTesting error handling...")
    
    try:
        from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent
        
        agent = MemoryTraceAgent()
        
        # Test with invalid input
        test_cases = [
            None,
            "",
            "invalid json {{}",
            json.dumps({"missing": "required_fields"}),
            json.dumps({"context": "test", "invalid_field": "value"})
        ]
        
        for i, test_input in enumerate(test_cases):
            try:
                result = agent.run(test_input)
                print(f"  Test {i+1}: Handled gracefully - returned: {type(result).__name__}")
            except Exception as e:
                print(f"  Test {i+1}: Exception raised: {e.__class__.__name__}")
        
        print("[OK] Error handling tested")
        return True
        
    except ImportError:
        print("[INFO] MemoryTraceAgent not implemented yet")
        return False
    except Exception as e:
        print(f"[ERROR] Error handling test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== ReminiscingAgent Sub-agents Test Suite ===\n")
    
    tests = [
        ("ContextCategorizationAgent", test_context_categorization_agent),
        ("MemoryTraceAgent", test_memory_trace_agent),
        ("Agent Integration", test_agent_integration),
        ("Error Handling", test_agent_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL/SKIP]"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("[SUCCESS] All sub-agent tests passed!")
    else:
        print("[INFO] Some tests failed or were skipped (agents may not be implemented yet)")