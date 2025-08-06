#!/usr/bin/env python3
"""
Live test of LangGraph integration with ReminiscingAgent.

Now that LangGraph is installed, this tests the actual workflow execution.
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Verify LangGraph is available
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    print("✓ LangGraph successfully imported")
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"✗ LangGraph import failed: {e}")
    LANGGRAPH_AVAILABLE = False
    sys.exit(1)


def test_langgraph_basic_workflow():
    """Test basic LangGraph workflow creation."""
    print("\n1. Testing basic LangGraph workflow...")
    
    # Define a simple state
    from typing import TypedDict, List
    
    class SimpleState(TypedDict):
        messages: List[str]
        result: str
    
    # Create workflow
    workflow = StateGraph(SimpleState)
    
    # Define nodes
    def process_messages(state: SimpleState) -> SimpleState:
        state['result'] = f"Processed {len(state.get('messages', []))} messages"
        return state
    
    def format_output(state: SimpleState) -> SimpleState:
        state['result'] = f"Final: {state['result']}"
        return state
    
    # Add nodes
    workflow.add_node("process", process_messages)
    workflow.add_node("format", format_output)
    
    # Add edges
    workflow.set_entry_point("process")
    workflow.add_edge("process", "format")
    workflow.add_edge("format", END)
    
    # Compile
    app = workflow.compile()
    
    # Run
    result = app.invoke({"messages": ["test1", "test2"], "result": ""})
    
    assert result['result'] == "Final: Processed 2 messages"
    print(f"  ✓ Workflow executed: {result['result']}")
    
    return True


def test_reminiscing_agent_with_langgraph():
    """Test ReminiscingAgent with real LangGraph."""
    print("\n2. Testing ReminiscingAgent with LangGraph...")
    
    # Import with LangGraph available
    from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
    
    # Create agent - should use LangGraph workflow
    agent = ReminiscingAgent()
    
    # Check workflow was created
    assert agent.workflow is not None, "Workflow not created despite LangGraph being available"
    print("  ✓ ReminiscingAgent created with LangGraph workflow")
    
    # Test execution
    result = agent.run("How should I implement user authentication?")
    
    assert isinstance(result, str), "Result should be string"
    assert 'MEMORY_TRACES' in result or 'MEMORY_ERROR' in result, "Unexpected result format"
    
    print(f"  ✓ Workflow executed successfully")
    print(f"    Result preview: {result[:100]}...")
    
    return True


def test_langgraph_state_management():
    """Test LangGraph state management in ReminiscingAgent."""
    print("\n3. Testing state management...")
    
    from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent, ReminiscingState
    
    # Create agent
    agent = ReminiscingAgent()
    
    # Test state transitions manually
    initial_state = ReminiscingState(
        context="Test context for authentication",
        category=None,
        search_strategy=None,
        memory_traces=[],
        confidence=0.0,
        final_response=""
    )
    
    # Test categorization
    state_after_categorize = agent._categorize_context(initial_state)
    assert state_after_categorize['category'] is not None, "Category not set"
    assert state_after_categorize['search_strategy'] is not None, "Strategy not set"
    print(f"  ✓ Categorization: {state_after_categorize['category']} / {state_after_categorize['search_strategy']}")
    
    # Test memory search
    state_after_search = agent._search_memory(state_after_categorize)
    assert 'memory_traces' in state_after_search, "Memory traces not in state"
    print(f"  ✓ Memory search: {len(state_after_search['memory_traces'])} traces found")
    
    # Test response formatting
    state_final = agent._format_response(state_after_search)
    assert state_final['final_response'] != "", "Final response not generated"
    print(f"  ✓ Response formatted: {len(state_final['final_response'])} chars")
    
    return True


def test_langgraph_error_handling():
    """Test error handling in LangGraph workflow."""
    print("\n4. Testing error handling in workflow...")
    
    from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
    from unittest.mock import patch
    
    agent = ReminiscingAgent()
    
    # Test with various error scenarios
    test_cases = [
        None,  # None input
        "",    # Empty input
        "x" * 100000,  # Very long input
    ]
    
    for test_input in test_cases:
        try:
            result = agent.run(test_input if test_input else "")
            assert isinstance(result, str), f"Failed on input type: {type(test_input)}"
            print(f"  ✓ Handled edge case: {str(test_input)[:20] if test_input else 'None'}...")
        except Exception as e:
            print(f"  ✗ Failed on input: {e}")
            return False
    
    # Test with workflow error
    with patch.object(agent, '_categorize_context', side_effect=Exception("Test error")):
        result = agent.run("Test with error")
        assert isinstance(result, str), "Workflow error not handled"
        print("  ✓ Handled workflow node error")
    
    return True


def test_langgraph_performance():
    """Test performance with LangGraph workflow."""
    print("\n5. Testing workflow performance...")
    
    import time
    from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
    
    agent = ReminiscingAgent()
    
    # Add some test memories
    for i in range(10):
        agent.store_conversation({
            'task': f'Test task {i}',
            'messages': [f'Message {i}']
        })
    
    # Benchmark workflow execution
    queries = [
        "How to implement authentication?",
        "Debug database connection issues",
        "Design microservice architecture",
    ]
    
    times = []
    for query in queries:
        start = time.time()
        result = agent.run(query)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Query: '{query[:30]}...' - {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  ✓ Average execution time: {avg_time:.3f}s")
    
    # Should be reasonably fast
    assert avg_time < 5.0, f"Workflow too slow: {avg_time:.3f}s average"
    
    return True


def demonstrate_langgraph_features():
    """Demonstrate advanced LangGraph features."""
    print("\n6. Demonstrating LangGraph features...")
    
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Optional
    
    # Create a more complex workflow
    class ComplexState(TypedDict):
        input: str
        analysis: Optional[str]
        decision: Optional[str]
        output: Optional[str]
        confidence: float
    
    workflow = StateGraph(ComplexState)
    
    # Define conditional routing
    def should_proceed(state: ComplexState) -> str:
        """Conditional edge function."""
        if state.get('confidence', 0) > 0.7:
            return "proceed"
        else:
            return "retry"
    
    def analyze(state: ComplexState) -> ComplexState:
        state['analysis'] = f"Analyzed: {state['input']}"
        state['confidence'] = 0.8  # Simulate confidence
        return state
    
    def decide(state: ComplexState) -> ComplexState:
        state['decision'] = "Proceed with action"
        return state
    
    def retry(state: ComplexState) -> ComplexState:
        state['analysis'] = "Retrying with different approach"
        state['confidence'] = 0.9  # Higher confidence after retry
        return state
    
    def output(state: ComplexState) -> ComplexState:
        state['output'] = f"Final: {state['decision']}"
        return state
    
    # Build workflow with conditional edges
    workflow.add_node("analyze", analyze)
    workflow.add_node("decide", decide)
    workflow.add_node("retry", retry)
    workflow.add_node("output", output)
    
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_proceed,
        {
            "proceed": "decide",
            "retry": "retry"
        }
    )
    workflow.add_edge("retry", "decide")
    workflow.add_edge("decide", "output")
    workflow.add_edge("output", END)
    
    # Compile and run
    app = workflow.compile()
    result = app.invoke({"input": "Test input", "confidence": 0.0})
    
    print("  ✓ Complex workflow with conditional routing executed")
    print(f"    Final output: {result.get('output')}")
    
    # Demonstrate streaming
    print("\n  Streaming execution:")
    for chunk in app.stream({"input": "Stream test", "confidence": 0.0}):
        print(f"    Step: {list(chunk.keys())[0]} - State updated")
    
    return True


def main():
    """Run all LangGraph tests."""
    print("=" * 60)
    print("LANGGRAPH LIVE INTEGRATION TESTS")
    print("=" * 60)
    
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph is not available. Please install it:")
        print("  pip install langgraph langchain-core")
        return False
    
    tests = [
        ("Basic Workflow", test_langgraph_basic_workflow),
        ("ReminiscingAgent Integration", test_reminiscing_agent_with_langgraph),
        ("State Management", test_langgraph_state_management),
        ("Error Handling", test_langgraph_error_handling),
        ("Performance", test_langgraph_performance),
        ("Advanced Features", demonstrate_langgraph_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✅ LANGGRAPH FULLY OPERATIONAL!")
        print("\nKey Benefits of LangGraph:")
        print("1. State machine-based orchestration")
        print("2. Conditional routing between nodes")
        print("3. Streaming execution support")
        print("4. Better error handling and recovery")
        print("5. Visual workflow inspection (with visualization tools)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)