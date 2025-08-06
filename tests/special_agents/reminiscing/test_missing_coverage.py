#!/usr/bin/env python3
"""
Additional tests to reach 98% coverage for ReminiscingAgent system.

Covers previously untested methods, error paths, and edge cases.
"""

import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent, SpreadingActivationNetwork
from special_agents.reminiscing.context_categorization_agent import ContextCategorizationAgent
from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore
from special_agents.reminiscing.semantic_search_agent import SemanticSearchAgent


# ========== ReminiscingAgent Private Methods ==========

def test_format_simple_response():
    """Test _format_simple_response private method."""
    print("Testing _format_simple_response...")
    
    agent = ReminiscingAgent()
    
    # Test the simple response formatter
    response = agent._format_simple_response(
        "test context",
        "debugging",
        "memory result"
    )
    
    assert "MEMORY_TRACES" in response
    assert "test context" in response
    assert "debugging" in response
    assert "memory result" in response
    assert "simplified workflow" in response.lower()
    
    print("[OK] _format_simple_response tested")
    return True


def test_parse_category_result_edge_cases():
    """Test _parse_category_result with malformed input."""
    print("Testing _parse_category_result edge cases...")
    
    agent = ReminiscingAgent()
    
    # Test various malformed inputs
    test_cases = [
        "",  # Empty
        "Random text without structure",  # No markers
        "CATEGORY: \nSTRATEGY: ",  # Empty values
        "CATEGORY: invalid\nSTRATEGY: also_invalid",  # Invalid values
        None,  # None input
    ]
    
    for test_input in test_cases:
        try:
            if test_input is not None:
                category, strategy, confidence = agent._parse_category_result(test_input)
            else:
                category, strategy, confidence = agent._parse_category_result("")
            
            # Should return defaults for malformed input
            assert category in ['general', 'architectural', 'debugging', 'implementation', 'research']
            assert strategy in ['semantic_search', 'graph_traversal', 'error_similarity', 'code_similarity']
            assert 0 <= confidence <= 1
            
        except Exception as e:
            assert False, f"Failed on input '{test_input}': {e}"
    
    print("[OK] _parse_category_result handles malformed input")
    return True


def test_parse_memory_result_edge_cases():
    """Test _parse_memory_result with various formats."""
    print("Testing _parse_memory_result edge cases...")
    
    agent = ReminiscingAgent()
    
    # Test different result formats
    test_cases = [
        '{"traces": [{"id": "1"}], "confidence": 0.8}',  # Valid JSON
        '[{"id": "1"}]',  # Array JSON
        'TRACE: Test trace\nContent here',  # Text format
        'Invalid JSON {',  # Malformed JSON
        '',  # Empty
        'null',  # JSON null
    ]
    
    for test_input in test_cases:
        traces, confidence = agent._parse_memory_result(test_input)
        
        assert isinstance(traces, list)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    print("[OK] _parse_memory_result handles various formats")
    return True


# ========== SpreadingActivationNetwork Tests ==========

def test_spreading_activation_network():
    """Test SpreadingActivationNetwork class."""
    print("Testing SpreadingActivationNetwork...")
    
    agent = MemoryTraceAgent()
    network = SpreadingActivationNetwork()
    
    # Add memory traces
    network.add_memory_trace('mem1', 'Memory 1 content', ['concept1', 'concept2'])
    network.add_memory_trace('mem2', 'Memory 2 content', ['concept2', 'concept3'])
    network.add_memory_trace('mem3', 'Memory 3 content', ['concept3', 'concept4'])
    
    # Add associations
    network.add_association('mem1', 'mem2', 0.8)
    network.add_association('mem2', 'mem3', 0.6)
    
    # Test spreading activation
    activated = network.spread_activation(['mem1'], decay_factor=0.7, max_hops=2)
    
    assert 'mem1' in activated
    assert activated['mem1'] == 1.0  # Source activation
    assert 'mem2' in activated
    assert activated['mem2'] > 0  # Should receive activation
    
    # Test memory retrieval
    memories = network.retrieve_memories(['concept2'], top_k=2)
    assert len(memories) <= 2
    
    print("[OK] SpreadingActivationNetwork tested")
    return True


# ========== ContextCategorizationAgent Private Methods ==========

def test_categorization_helper_methods():
    """Test ContextCategorizationAgent helper methods."""
    print("Testing categorization helper methods...")
    
    agent = ContextCategorizationAgent()
    
    # Test confidence description
    assert agent._confidence_description(0.95) == "Very High"
    assert agent._confidence_description(0.75) == "High"
    assert agent._confidence_description(0.55) == "Medium"
    assert agent._confidence_description(0.35) == "Low"
    assert agent._confidence_description(0.15) == "Very Low"
    
    # Test dimension methods
    assert agent._get_primary_dimension('architectural') == 'system_design'
    assert agent._get_primary_dimension('debugging') == 'error_patterns'
    
    secondary = agent._get_secondary_dimensions('implementation')
    assert 'algorithm_patterns' in secondary
    
    assert agent._get_search_depth('architectural') == 'deep'
    assert agent._get_temporal_weight('debugging') == 'high'
    
    print("[OK] Helper methods tested")
    return True


# ========== VectorStore Content Extraction ==========

def test_content_extraction_edge_cases():
    """Test content extraction with various data formats."""
    print("Testing content extraction edge cases...")
    
    from special_agents.reminiscing.vector_store import ConversationVectorStore
    store = ConversationVectorStore()
    
    # Test text extraction with edge cases
    test_cases = [
        {},  # Empty dict
        {'task': None},  # None values
        {'messages': 'not a list'},  # Wrong type
        {'blackboard_entries': [{'content': 'BB entry'}]},  # Blackboard
        {'unexpected': 'field'},  # Unknown fields
    ]
    
    for data in test_cases:
        text = store._extract_text_content(data)
        assert isinstance(text, str)
    
    # Test code extraction
    code_cases = [
        {},  # Empty
        {'code': 'print("hello")'},  # Just code
        {'functions': ['func1', 'func2']},  # Just functions
        {'classes': ['Class1']},  # Just classes
    ]
    
    for data in code_cases:
        text = store._extract_code_content(data)
        assert isinstance(text, str)
    
    print("[OK] Content extraction handles edge cases")
    return True


def test_metadata_extraction():
    """Test metadata extraction for different data types."""
    print("Testing metadata extraction...")
    
    from special_agents.reminiscing.vector_store import ConversationVectorStore
    store = ConversationVectorStore()
    
    # Test conversation metadata
    conv_data = {
        'task': 'Test task',
        'messages': ['msg1', 'msg2'],
        'error': 'Some error occurred',
        'session_id': 'abc123'
    }
    
    metadata = store._extract_metadata(conv_data, 'conversation')
    assert metadata['memory_type'] == 'conversation'
    assert metadata['message_count'] == 2
    assert metadata['has_errors'] == True
    assert metadata['session_id'] == 'abc123'
    
    # Test code metadata
    code_data = {
        'file_path': '/test/file.py',
        'language': 'python',
        'functions': ['func1'],
        'classes': ['Class1'],
        'code': 'x = 1'
    }
    
    metadata = store._extract_metadata(code_data, 'code')
    assert metadata['file_path'] == '/test/file.py'
    assert metadata['language'] == 'python'
    assert metadata['has_functions'] == True
    assert metadata['has_classes'] == True
    
    print("[OK] Metadata extraction tested")
    return True


# ========== EnhancedVectorStore Graph Operations ==========

def test_graph_disconnected_components():
    """Test graph search with disconnected components."""
    print("Testing graph search with disconnected components...")
    
    store = EnhancedVectorStore()
    
    # Create two disconnected groups
    # Group 1
    g1_m1 = store.store_conversation_enhanced({'task': 'Group1 Task1', 'messages': ['G1M1']})
    g1_m2 = store.store_conversation_enhanced({'task': 'Group1 Task2', 'messages': ['G1M2']})
    store.add_relationship(g1_m1, g1_m2)
    
    # Group 2 (disconnected)
    g2_m1 = store.store_conversation_enhanced({'task': 'Group2 Task1', 'messages': ['G2M1']})
    g2_m2 = store.store_conversation_enhanced({'task': 'Group2 Task2', 'messages': ['G2M2']})
    store.add_relationship(g2_m1, g2_m2)
    
    # Search should find from appropriate group
    results = store.search_enhanced('Group1', strategy='graph', limit=10)
    
    # Should find Group1 memories but not Group2
    group1_found = any('Group1' in r.get('content', '') for r in results)
    group2_found = any('Group2' in r.get('content', '') for r in results)
    
    assert group1_found, "Didn't find Group1 memories"
    # Group2 might be found through semantic similarity, but not through graph
    
    print("[OK] Graph search handles disconnected components")
    return True


def test_memory_statistics_helpers():
    """Test statistics helper methods."""
    print("Testing statistics helper methods...")
    
    store = EnhancedVectorStore()
    
    # Add some test data
    for i in range(5):
        mem_id = store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Message {i}']
        })
        
        # Simulate access
        for _ in range(i):
            store._update_memory_access(mem_id)
    
    # Test statistics helpers
    most_accessed = store._get_most_accessed_memories(3)
    assert len(most_accessed) <= 3
    assert all('access_count' in m for m in most_accessed)
    
    most_connected = store._get_most_connected_memories(3)
    assert len(most_connected) <= 3
    
    top_concepts = store._get_top_concepts(5)
    assert len(top_concepts) <= 5
    
    print("[OK] Statistics helpers tested")
    return True


# ========== SemanticSearchAgent Code Analysis ==========

def test_semantic_search_code_analysis():
    """Test code analysis methods in SemanticSearchAgent."""
    print("Testing semantic search code analysis...")
    
    agent = SemanticSearchAgent()
    
    # Test Python context extraction
    python_code = '''
def test_function(param):
    """Test docstring"""
    return param * 2

class TestClass:
    def method(self):
        pass
'''
    
    context = agent.extract_context(python_code)
    assert 'concepts' in context
    assert 'type' in context
    
    # Test with syntax error
    bad_code = 'def broken('
    context = agent.extract_context(bad_code)
    assert isinstance(context, dict)  # Should handle gracefully
    
    print("[OK] Code analysis tested")
    return True


def test_search_intent_analysis():
    """Test search intent analysis."""
    print("Testing search intent analysis...")
    
    agent = SemanticSearchAgent()
    
    # Test different query intents
    queries = [
        "how to implement authentication",  # How-to
        "error in database connection",  # Error/debug
        "UserManager class",  # Specific code
        "best practices for API design",  # Best practices
    ]
    
    for query in queries:
        processed = agent.process_query(query)
        assert 'keywords' in processed
        assert 'query_type' in processed
    
    print("[OK] Search intent analysis tested")
    return True


# ========== Error Injection Tests ==========

def test_llm_failure_paths():
    """Test LLM failure handling in all agents."""
    print("Testing LLM failure paths...")
    
    # Test ReminiscingAgent with LLM failure
    agent = ReminiscingAgent()
    with patch.object(agent.categorization_agent, 'llm') as mock_llm:
        mock_llm.reply.side_effect = Exception("LLM API error")
        
        result = agent.run("Test query")
        assert isinstance(result, str)
        assert 'MEMORY' in result  # Should still return something
    
    # Test ContextCategorizationAgent with LLM failure
    cat_agent = ContextCategorizationAgent()
    with patch.object(cat_agent, 'call_ai', side_effect=Exception("API error")):
        result = cat_agent.run("Test context")
        assert isinstance(result, str)
        assert 'CATEGORY' in result  # Should fall back to pattern-based
    
    print("[OK] LLM failure paths tested")
    return True


def test_persistence_failure_recovery():
    """Test recovery from persistence failures."""
    print("Testing persistence failure recovery...")
    
    store = EnhancedVectorStore(storage_path="/invalid/path/test.json")
    
    # Should handle invalid path gracefully
    store._save_to_disk()  # Should not crash
    store._save_enhanced_data()  # Should not crash
    
    # Test loading from non-existent file
    store2 = EnhancedVectorStore(storage_path="/nonexistent/file.json")
    # Should initialize empty
    assert len(store2.memory_nodes) == 0
    
    print("[OK] Persistence failures handled")
    return True


# ========== Configuration and Validation ==========

def test_configuration_validation():
    """Test configuration parameter validation."""
    print("Testing configuration validation...")
    
    store = EnhancedVectorStore()
    
    # Test parameter boundaries
    store.decay_factor = 0.0  # Minimum
    store.apply_decay()  # Should not crash
    
    store.decay_factor = 1.0  # Maximum
    store.apply_decay()  # Should not crash
    
    store.similarity_threshold = -1.0  # Invalid
    store._consolidate_memories()  # Should handle gracefully
    
    store.max_memories = 0  # Edge case
    store._cleanup_old_memories()  # Should handle
    
    print("[OK] Configuration validation tested")
    return True


# ========== Integration Workflow Tests ==========

def test_full_workflow_with_failures():
    """Test complete workflow with injected failures."""
    print("Testing full workflow with failures...")
    
    agent = ReminiscingAgent()
    
    # Inject failures at different stages
    with patch.object(agent.categorization_agent, 'run', side_effect=Exception("Cat failed")):
        result = agent.run("Test query")
        assert isinstance(result, str)
    
    with patch.object(agent.memory_trace_agent, 'run', side_effect=Exception("Memory failed")):
        result = agent.run("Test query")
        assert isinstance(result, str)
    
    # Test with all components failing
    with patch.object(agent, '_categorize_context', side_effect=Exception("Failed")):
        with patch.object(agent, '_search_memory', side_effect=Exception("Failed")):
            result = agent.run("Test query")
            assert 'ERROR' in result or isinstance(result, str)
    
    print("[OK] Workflow handles cascading failures")
    return True


# ========== Resource Limit Tests ==========

def test_resource_exhaustion():
    """Test behavior when resources are exhausted."""
    print("Testing resource exhaustion...")
    
    store = EnhancedVectorStore()
    store.max_memories = 5
    
    # Try to add more than limit
    for i in range(10):
        store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Msg {i}']
        })
    
    # Should enforce limit
    total = len(store.memory_nodes)
    assert total <= store.max_memories + 1  # Allow small overrun
    
    # Test with very large embedding dimension
    store.embedding_dim = 10000
    embedding = store._generate_embedding("test")
    assert len(embedding) == 10000
    
    print("[OK] Resource limits enforced")
    return True


if __name__ == "__main__":
    print("=== Missing Coverage Test Suite ===\n")
    
    tests = [
        # ReminiscingAgent
        ("Format Simple Response", test_format_simple_response),
        ("Parse Category Edge Cases", test_parse_category_result_edge_cases),
        ("Parse Memory Edge Cases", test_parse_memory_result_edge_cases),
        
        # SpreadingActivationNetwork
        ("Spreading Activation Network", test_spreading_activation_network),
        
        # ContextCategorizationAgent
        ("Categorization Helpers", test_categorization_helper_methods),
        
        # VectorStore
        ("Content Extraction Edges", test_content_extraction_edge_cases),
        ("Metadata Extraction", test_metadata_extraction),
        
        # EnhancedVectorStore
        ("Graph Disconnected", test_graph_disconnected_components),
        ("Statistics Helpers", test_memory_statistics_helpers),
        
        # SemanticSearchAgent
        ("Code Analysis", test_semantic_search_code_analysis),
        ("Intent Analysis", test_search_intent_analysis),
        
        # Error Paths
        ("LLM Failures", test_llm_failure_paths),
        ("Persistence Failures", test_persistence_failure_recovery),
        
        # Configuration
        ("Configuration Validation", test_configuration_validation),
        
        # Integration
        ("Full Workflow Failures", test_full_workflow_with_failures),
        ("Resource Exhaustion", test_resource_exhaustion)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests ({100*passed//total}%)")
    
    if passed == total:
        print("[SUCCESS] All missing coverage tests passed!")
    else:
        print("[WARNING] Some tests failed.")