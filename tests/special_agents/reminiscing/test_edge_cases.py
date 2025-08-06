#!/usr/bin/env python3
"""
Edge case tests for ReminiscingAgent system.

Tests unusual but valid scenarios and boundary conditions:
- Empty/single item operations
- Extreme values
- Unicode and special characters
- Very large/small inputs
- Boundary conditions
"""

import sys
import os
import random
import string
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent


def test_empty_string_operations():
    """Test operations with empty strings."""
    print("Testing empty string operations...")
    
    store = EnhancedVectorStore()
    agent = ReminiscingAgent()
    
    # Store empty content
    memory_id = store.store_conversation({'task': '', 'messages': ['']})
    assert memory_id is not None, "Failed to store empty content"
    
    # Search with empty query
    results = store.search_enhanced('', limit=5)
    assert isinstance(results, list), "Empty search failed"
    
    # Agent with empty input
    result = agent.run('')
    assert isinstance(result, str), "Agent failed with empty input"
    
    print("[OK] Empty string operations handled")
    return True


def test_single_memory_operations():
    """Test operations with single memory."""
    print("\nTesting single memory operations...")
    
    store = EnhancedVectorStore()
    
    # Store single memory
    memory_id = store.store_conversation_enhanced({
        'task': 'Only task',
        'messages': ['Only message']
    })
    
    # All search strategies should work with single memory
    strategies = ['semantic', 'graph', 'concept', 'temporal', 'hybrid']
    
    for strategy in strategies:
        results = store.search_enhanced('task', strategy=strategy, limit=10)
        assert len(results) <= 1, f"Got multiple results for single memory with {strategy}"
        print(f"  {strategy}: {len(results)} result(s)")
    
    # Consolidation with single memory
    store._consolidate_memories()
    assert len(store.memory_nodes) == 1, "Single memory was removed during consolidation"
    
    print("[OK] Single memory operations work correctly")
    return True


def test_unicode_and_special_characters():
    """Test handling of Unicode and special characters."""
    print("\nTesting Unicode and special characters...")
    
    store = EnhancedVectorStore()
    
    # Test various Unicode content
    unicode_tests = [
        {'task': '测试中文', 'messages': ['中文内容测试']},  # Chinese
        {'task': 'テスト', 'messages': ['日本語のテスト']},  # Japanese
        {'task': 'Тест', 'messages': ['Русский тест']},  # Russian
        {'task': '🚀 Emoji test 🎉', 'messages': ['Emojis: 😀 🌟 💻']},
        {'task': 'Math: ∑∫∂', 'messages': ['√π ≈ 1.77']},
        {'task': 'Symbols: @#$%^&*()', 'messages': ['<>{}[]|\\']},
    ]
    
    memory_ids = []
    for test_data in unicode_tests:
        try:
            memory_id = store.store_conversation_enhanced(test_data)
            memory_ids.append(memory_id)
            assert memory_id is not None, f"Failed to store: {test_data['task']}"
        except Exception as e:
            assert False, f"Error storing Unicode: {e}"
    
    # Search with Unicode
    results = store.search_enhanced('测试', limit=5)
    assert isinstance(results, list), "Unicode search failed"
    
    # Emoji search
    results = store.search_enhanced('🚀', limit=5)
    assert isinstance(results, list), "Emoji search failed"
    
    print(f"[OK] Stored and searched {len(memory_ids)} Unicode memories")
    return True


def test_extreme_lengths():
    """Test with extremely long and short content."""
    print("\nTesting extreme content lengths...")
    
    store = EnhancedVectorStore()
    
    # Very short content
    short_id = store.store_conversation_enhanced({
        'task': 'a',
        'messages': ['b']
    })
    assert short_id is not None, "Failed to store short content"
    
    # Very long content
    long_content = 'x' * 10000  # 10KB of text
    long_id = store.store_conversation_enhanced({
        'task': 'Long task',
        'messages': [long_content]
    })
    assert long_id is not None, "Failed to store long content"
    
    # Search should handle both
    results = store.search_enhanced('a', limit=10)
    assert any(r['memory_id'] == short_id for r in results), "Short content not found"
    
    # Very long search query
    long_query = 'test ' * 1000
    results = store.search_enhanced(long_query, limit=5)
    assert isinstance(results, list), "Long query search failed"
    
    print("[OK] Extreme lengths handled correctly")
    return True


def test_maximum_relationships():
    """Test memory with maximum number of relationships."""
    print("\nTesting maximum relationships...")
    
    store = EnhancedVectorStore()
    
    # Create hub memory with many connections
    hub_id = store.store_conversation_enhanced({
        'task': 'Hub memory',
        'messages': ['Central node']
    })
    
    # Create many connected memories
    num_connections = 100
    connected_ids = []
    
    for i in range(num_connections):
        connected_id = store.store_conversation_enhanced({
            'task': f'Connected {i}',
            'messages': [f'Node {i}']
        })
        connected_ids.append(connected_id)
        store.add_relationship(hub_id, connected_id, random.random())
    
    # Check relationships were created
    assert len(store.memory_graph.get(hub_id, set())) == num_connections, \
        "Not all relationships created"
    
    # Graph search should handle high connectivity
    results = store.search_enhanced('Hub', strategy='graph', limit=20)
    assert len(results) > 0, "Graph search failed with high connectivity"
    
    print(f"[OK] Handled {num_connections} relationships")
    return True


def test_zero_similarity_scores():
    """Test handling of zero and negative similarity scores."""
    print("\nTesting zero similarity scores...")
    
    store = EnhancedVectorStore()
    
    # Create orthogonal content (should have ~0 similarity)
    mem1 = store.store_conversation_enhanced({
        'task': 'AAA BBB CCC',
        'messages': ['Unique content one']
    })
    
    mem2 = store.store_conversation_enhanced({
        'task': 'XXX YYY ZZZ',
        'messages': ['Completely different two']
    })
    
    # Search for unrelated content
    results = store.search_enhanced('QQQ RRR SSS', limit=10)
    
    # Should still return results even with low similarity
    assert isinstance(results, list), "Search failed with low similarity"
    
    # All scores should be non-negative
    for result in results:
        assert result.get('score', 0) >= 0, f"Negative score: {result.get('score')}"
    
    print("[OK] Zero similarity handled correctly")
    return True


def test_timestamp_edge_cases():
    """Test edge cases with timestamps."""
    print("\nTesting timestamp edge cases...")
    
    store = EnhancedVectorStore()
    now = datetime.now()
    
    # Future timestamp
    future_id = store.store_conversation_enhanced({
        'task': 'Future task',
        'messages': ['From the future']
    })
    if future_id in store.memory_nodes:
        store.memory_nodes[future_id].timestamp = now + timedelta(days=1)
    
    # Very old timestamp
    old_id = store.store_conversation_enhanced({
        'task': 'Ancient task',
        'messages': ['From the past']
    })
    if old_id in store.memory_nodes:
        store.memory_nodes[old_id].timestamp = datetime(1970, 1, 1)
    
    # Current timestamp
    current_id = store.store_conversation_enhanced({
        'task': 'Current task',
        'messages': ['Right now']
    })
    
    # Temporal search should handle all cases
    results = store.search_enhanced('task', strategy='temporal', limit=10)
    assert len(results) >= 2, "Temporal search failed with edge timestamps"
    
    # Recent memories filter should handle future dates
    recent = store.get_recent_memories(hours=24)
    assert isinstance(recent, list), "Recent memories failed with edge timestamps"
    
    print("[OK] Timestamp edge cases handled")
    return True


def test_duplicate_memories():
    """Test handling of duplicate memories."""
    print("\nTesting duplicate memory handling...")
    
    store = EnhancedVectorStore()
    store.similarity_threshold = 0.99  # Very high threshold
    
    # Store identical memories
    identical_content = {'task': 'Duplicate task', 'messages': ['Same message']}
    
    ids = []
    for _ in range(5):
        memory_id = store.store_conversation_enhanced(identical_content.copy())
        ids.append(memory_id)
    
    # All should be stored initially
    assert len(set(ids)) == 5, "Duplicates not stored"
    
    initial_count = len(store.memory_nodes)
    
    # Consolidation should merge duplicates
    store._consolidate_memories()
    
    final_count = len(store.memory_nodes)
    assert final_count < initial_count, "Duplicates not consolidated"
    
    print(f"[OK] Consolidated {initial_count - final_count} duplicates")
    return True


def test_special_search_queries():
    """Test special characters in search queries."""
    print("\nTesting special search queries...")
    
    store = EnhancedVectorStore()
    agent = ReminiscingAgent()
    
    # Add content with special patterns
    store.store_conversation_enhanced({
        'task': 'Implement user@domain.com validation',
        'messages': ['Email regex: ^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$']
    })
    
    # Special character queries
    special_queries = [
        '@',
        '.*',
        '\\n',
        '${}',
        '[]',
        '()',
        '#!',
        'C++',
        '.NET',
        'Node.js'
    ]
    
    for query in special_queries:
        try:
            results = store.search_enhanced(query, limit=5)
            assert isinstance(results, list), f"Search failed for: {query}"
            
            # Agent should also handle
            agent_result = agent.run(query)
            assert isinstance(agent_result, str), f"Agent failed for: {query}"
            
        except Exception as e:
            assert False, f"Failed on special query '{query}': {e}"
    
    print(f"[OK] Handled {len(special_queries)} special queries")
    return True


def test_concept_extraction_edge_cases():
    """Test concept extraction with edge cases."""
    print("\nTesting concept extraction edge cases...")
    
    store = EnhancedVectorStore()
    
    edge_cases = [
        '',  # Empty
        ' ',  # Whitespace only
        '123456789',  # Numbers only
        '!!!###$$$',  # Symbols only
        'a',  # Single character
        'CamelCaseOnlyNoSpaces',  # No word boundaries
        'snake_case_only_underscores',  # All underscores
        'ALLCAPSNOBREAKS',  # All caps
        '这是中文文本',  # Non-Latin script
        'Mix123Of456Everything!!!',  # Mixed content
    ]
    
    for text in edge_cases:
        concepts = store._extract_concepts(text)
        assert isinstance(concepts, set), f"Concept extraction failed for: {text}"
        print(f"  '{text[:20]}...': {len(concepts)} concepts")
    
    print("[OK] Concept extraction handles edge cases")
    return True


def test_score_boundary_conditions():
    """Test score calculation boundary conditions."""
    print("\nTesting score boundary conditions...")
    
    store = EnhancedVectorStore()
    
    # Test with extreme importance scores
    mem_id = store.store_conversation_enhanced({
        'task': 'Test task',
        'messages': ['Test message']
    })
    
    if mem_id in store.memory_nodes:
        # Test with importance = 0
        store.memory_nodes[mem_id].importance_score = 0.0
        results = store.search_enhanced('test', strategy='semantic', limit=5)
        assert all(r.get('score', 0) >= 0 for r in results), "Negative scores with 0 importance"
        
        # Test with importance = 1
        store.memory_nodes[mem_id].importance_score = 1.0
        results = store.search_enhanced('test', strategy='semantic', limit=5)
        assert all(r.get('score', 0) <= 1.0 for r in results), "Scores exceed 1.0"
        
        # Test with very high access count
        store.memory_nodes[mem_id].access_count = 1000000
        store._update_memory_access(mem_id)
        assert store.memory_nodes[mem_id].importance_score <= 1.0, \
            "Importance exceeds 1.0 after many accesses"
    
    print("[OK] Score boundaries maintained")
    return True


def test_filter_edge_cases():
    """Test filtering with edge cases."""
    print("\nTesting filter edge cases...")
    
    store = EnhancedVectorStore()
    
    # Add test memories
    for i in range(10):
        store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Message {i}']
        })
    
    # Empty filters
    results = store.search_enhanced('task', filters={})
    assert len(results) > 0, "Empty filters broke search"
    
    # Invalid filter values
    results = store.search_enhanced('task', filters={'time_range': -1})
    assert isinstance(results, list), "Negative time range broke search"
    
    results = store.search_enhanced('task', filters={'time_range': 0})
    assert isinstance(results, list), "Zero time range broke search"
    
    # Non-existent filter keys
    results = store.search_enhanced('task', filters={'invalid_key': 'value'})
    assert isinstance(results, list), "Invalid filter key broke search"
    
    # Multiple filters
    results = store.search_enhanced('task', filters={
        'time_range': 24,
        'concepts': ['task'],
        'type': 'conversation'
    })
    assert isinstance(results, list), "Multiple filters broke search"
    
    print("[OK] Filter edge cases handled")
    return True


def test_graph_edge_cases():
    """Test graph operations with edge cases."""
    print("\nTesting graph edge cases...")
    
    store = EnhancedVectorStore()
    
    # Self-relationship
    mem_id = store.store_conversation_enhanced({
        'task': 'Self-referential',
        'messages': ['Points to itself']
    })
    store.add_relationship(mem_id, mem_id, 1.0)
    
    # Should not create self-loop
    assert mem_id not in store.memory_graph.get(mem_id, set()), \
        "Self-loop was created"
    
    # Isolated node (no relationships)
    isolated_id = store.store_conversation_enhanced({
        'task': 'Isolated',
        'messages': ['No connections']
    })
    
    # Graph search should still find isolated nodes through initial semantic search
    results = store.search_enhanced('Isolated', strategy='graph', limit=5)
    assert any(r['memory_id'] == isolated_id for r in results), \
        "Isolated node not found in graph search"
    
    # Fully connected subgraph
    subgraph_ids = []
    for i in range(5):
        sid = store.store_conversation_enhanced({
            'task': f'Subgraph {i}',
            'messages': [f'Node {i}']
        })
        subgraph_ids.append(sid)
    
    # Connect all to all
    for i, id1 in enumerate(subgraph_ids):
        for id2 in subgraph_ids[i+1:]:
            store.add_relationship(id1, id2, 0.8)
    
    # Graph search should handle fully connected subgraph
    results = store.search_enhanced('Subgraph', strategy='graph', limit=10)
    assert len(results) > 0, "Failed with fully connected subgraph"
    
    print("[OK] Graph edge cases handled")
    return True


if __name__ == "__main__":
    print("=== Edge Cases Test Suite ===\n")
    
    tests = [
        ("Empty Strings", test_empty_string_operations),
        ("Single Memory", test_single_memory_operations),
        ("Unicode/Special Chars", test_unicode_and_special_characters),
        ("Extreme Lengths", test_extreme_lengths),
        ("Maximum Relationships", test_maximum_relationships),
        ("Zero Similarity", test_zero_similarity_scores),
        ("Timestamp Edge Cases", test_timestamp_edge_cases),
        ("Duplicate Memories", test_duplicate_memories),
        ("Special Search Queries", test_special_search_queries),
        ("Concept Extraction Edges", test_concept_extraction_edge_cases),
        ("Score Boundaries", test_score_boundary_conditions),
        ("Filter Edge Cases", test_filter_edge_cases),
        ("Graph Edge Cases", test_graph_edge_cases)
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
        print("[SUCCESS] All edge case tests passed!")
    else:
        print("[WARNING] Some edge case tests failed.")