#!/usr/bin/env python3
"""
Test suite for EnhancedVectorStore functionality.

Tests advanced features including:
- Graph-based relationship tracking
- Multiple search strategies
- Concept indexing
- Memory consolidation
- Temporal decay and reinforcement
"""

import sys
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore, MemoryNode


def test_enhanced_storage():
    """Test enhanced storage with concept extraction."""
    print("Testing enhanced storage...")
    
    store = EnhancedVectorStore()
    
    # Store enhanced conversation
    conversation = {
        "task": "Implement OAuth2 authentication using passport.js",
        "messages": [
            "User needs secure authentication",
            "We'll use passport.js with Google and GitHub providers",
            "Successfully implemented OAuth2 flow"
        ]
    }
    
    memory_id = store.store_conversation_enhanced(conversation)
    
    # Check memory node was created
    assert memory_id in store.memory_nodes, "Memory node not created"
    node = store.memory_nodes[memory_id]
    
    # Check concepts were extracted
    assert len(node.concepts) > 0, "No concepts extracted"
    assert 'authentication' in node.concepts or 'oauth' in node.concepts, "Key concepts not found"
    
    # Check indexing
    found = False
    for concept in node.concepts:
        if memory_id in store.concept_index.get(concept, set()):
            found = True
            break
    assert found, "Memory not indexed by concepts"
    
    print(f"[OK] Stored memory with {len(node.concepts)} concepts")
    return True


def test_graph_relationships():
    """Test automatic relationship creation."""
    print("\nTesting graph relationships...")
    
    store = EnhancedVectorStore()
    
    # Store related memories
    memories = [
        {
            "task": "Design authentication system",
            "messages": ["Need OAuth2 and JWT support"]
        },
        {
            "task": "Implement OAuth2 with passport.js",
            "messages": ["Using passport-google-oauth20"]
        },
        {
            "task": "Add JWT token validation",
            "messages": ["Implementing JWT middleware"]
        },
        {
            "task": "Fix database connection timeout",
            "messages": ["Increased connection pool size"]
        }
    ]
    
    memory_ids = []
    for mem in memories:
        memory_id = store.store_conversation_enhanced(mem)
        memory_ids.append(memory_id)
    
    # Check relationships were created between related memories
    # First three should be related (authentication topics)
    auth_related = False
    for i in range(3):
        for j in range(i+1, 3):
            if memory_ids[j] in store.memory_graph.get(memory_ids[i], set()):
                auth_related = True
                break
    
    assert auth_related, "No relationships created between related memories"
    
    # Database memory should be less connected
    db_connections = len(store.memory_graph.get(memory_ids[3], set()))
    auth_connections = len(store.memory_graph.get(memory_ids[0], set()))
    
    print(f"[OK] Auth memory has {auth_connections} connections, DB memory has {db_connections}")
    return True


def test_search_strategies():
    """Test different search strategies."""
    print("\nTesting search strategies...")
    
    store = EnhancedVectorStore()
    
    # Populate with diverse memories
    test_memories = [
        {"task": "Implement user authentication", "messages": ["OAuth2", "JWT tokens"]},
        {"task": "Debug authentication errors", "messages": ["Token expired", "Invalid credentials"]},
        {"task": "Design API architecture", "messages": ["REST endpoints", "GraphQL schema"]},
        {"task": "Fix memory leak in event handlers", "messages": ["Memory profiling", "Event listener cleanup"]},
        {"task": "Optimize database queries", "messages": ["Query performance", "Index optimization"]}
    ]
    
    for mem in test_memories:
        store.store_conversation_enhanced(mem)
    
    # Test semantic search
    semantic_results = store.search_enhanced("authentication problems", strategy='semantic', limit=3)
    assert len(semantic_results) > 0, "Semantic search returned no results"
    assert semantic_results[0]['strategy'] == 'semantic', "Wrong strategy in results"
    print(f"[OK] Semantic search found {len(semantic_results)} results")
    
    # Test concept search
    concept_results = store.search_enhanced("authentication oauth jwt", strategy='concept', limit=3)
    assert len(concept_results) > 0, "Concept search returned no results"
    assert 'matched_concepts' in concept_results[0], "Missing concept matches"
    print(f"[OK] Concept search found {len(concept_results)} results")
    
    # Test hybrid search
    hybrid_results = store.search_enhanced("debug authentication", strategy='hybrid', limit=5)
    assert len(hybrid_results) > 0, "Hybrid search returned no results"
    print(f"[OK] Hybrid search found {len(hybrid_results)} results")
    
    return True


def test_temporal_features():
    """Test temporal search and decay."""
    print("\nTesting temporal features...")
    
    store = EnhancedVectorStore()
    
    # Add memories with different timestamps
    now = datetime.now()
    
    # Store recent memory
    recent_id = store.store_conversation_enhanced({
        "task": "Recent task",
        "messages": ["Just happened"]
    })
    
    # Manually adjust timestamp for testing
    if recent_id in store.memory_nodes:
        store.memory_nodes[recent_id].timestamp = now - timedelta(hours=1)
    
    # Store old memory
    old_id = store.store_conversation_enhanced({
        "task": "Old task",
        "messages": ["Happened long ago"]
    })
    
    if old_id in store.memory_nodes:
        store.memory_nodes[old_id].timestamp = now - timedelta(days=7)
    
    # Test temporal search
    temporal_results = store.search_enhanced("task", strategy='temporal', limit=2)
    
    assert len(temporal_results) > 0, "Temporal search returned no results"
    
    # Recent memory should score higher
    if len(temporal_results) >= 2:
        recent_result = next((r for r in temporal_results if r['memory_id'] == recent_id), None)
        old_result = next((r for r in temporal_results if r['memory_id'] == old_id), None)
        
        if recent_result and old_result:
            assert recent_result['score'] > old_result['score'], "Recent memory should score higher"
    
    print("[OK] Temporal search working correctly")
    
    # Test decay
    initial_importance = store.memory_nodes[recent_id].importance_score
    store.apply_decay()
    # Importance should remain similar for recent memory
    assert store.memory_nodes[recent_id].importance_score >= initial_importance * 0.9
    
    print("[OK] Temporal decay applied correctly")
    return True


def test_memory_consolidation():
    """Test memory consolidation for similar memories."""
    print("\nTesting memory consolidation...")
    
    store = EnhancedVectorStore()
    store.similarity_threshold = 0.8  # Lower threshold for testing
    
    # Store very similar memories
    similar_memories = [
        {"task": "Implement user authentication", "messages": ["Using OAuth2"]},
        {"task": "Implement user authentication", "messages": ["Using OAuth2"]},  # Duplicate
        {"task": "Setup authentication system", "messages": ["OAuth2 implementation"]},  # Very similar
        {"task": "Fix database timeout", "messages": ["Connection pool issue"]}  # Different
    ]
    
    memory_ids = []
    for mem in similar_memories:
        memory_id = store.store_conversation_enhanced(mem)
        memory_ids.append(memory_id)
    
    initial_count = len(store.memory_nodes)
    
    # Trigger consolidation
    store._consolidate_memories()
    
    final_count = len(store.memory_nodes)
    
    # Should have fewer memories after consolidation
    assert final_count < initial_count, f"Consolidation didn't reduce memories: {initial_count} -> {final_count}"
    
    # Database memory should still exist (different from others)
    db_memory_exists = any(
        'database' in node.content.lower() 
        for node in store.memory_nodes.values()
    )
    assert db_memory_exists, "Unique memory was incorrectly consolidated"
    
    print(f"[OK] Consolidated {initial_count} -> {final_count} memories")
    return True


def test_filtering():
    """Test search result filtering."""
    print("\nTesting search filtering...")
    
    store = EnhancedVectorStore()
    
    # Add diverse memories
    memories = [
        {"task": "Code review", "messages": ["Review PR #123"]},
        {"task": "Bug fix", "messages": ["Fixed null pointer"]},
        {"task": "Feature implementation", "messages": ["Added new API endpoint"]}
    ]
    
    for i, mem in enumerate(memories):
        memory_id = store.store_conversation_enhanced(mem)
        # Set different timestamps for testing
        if memory_id in store.memory_nodes:
            store.memory_nodes[memory_id].timestamp = datetime.now() - timedelta(hours=i*12)
    
    # Test time range filter
    results = store.search_enhanced(
        "task",
        strategy='semantic',
        filters={'time_range': 24}  # Last 24 hours
    )
    
    # Should only get recent memories
    for result in results:
        node = store.memory_nodes.get(result['memory_id'])
        if node:
            age = (datetime.now() - node.timestamp).total_seconds() / 3600
            assert age <= 24, f"Got memory older than 24 hours: {age:.1f}h"
    
    print(f"[OK] Time filter working correctly")
    
    # Test concept filter
    results = store.search_enhanced(
        "work",
        strategy='semantic',
        filters={'concepts': ['bug', 'fix']}
    )
    
    if results:
        # Results should contain bug-related concepts
        has_bug_concept = any(
            'bug' in r.get('concepts', []) or 'fix' in r.get('concepts', [])
            for r in results
        )
        assert has_bug_concept, "Concept filter not working"
    
    print("[OK] Concept filter working correctly")
    return True


def test_memory_statistics():
    """Test statistics generation."""
    print("\nTesting memory statistics...")
    
    store = EnhancedVectorStore()
    
    # Populate store
    for i in range(5):
        store.store_conversation_enhanced({
            "task": f"Task {i}",
            "messages": [f"Message {i}"]
        })
    
    stats = store.get_memory_statistics()
    
    assert stats['total_nodes'] == 5, "Incorrect node count"
    assert stats['total_concepts'] > 0, "No concepts indexed"
    assert 'avg_concepts_per_memory' in stats, "Missing average concepts stat"
    assert 'top_concepts' in stats, "Missing top concepts"
    
    print("[OK] Statistics generated correctly")
    print(f"    Total nodes: {stats['total_nodes']}")
    print(f"    Total concepts: {stats['total_concepts']}")
    print(f"    Avg concepts/memory: {stats['avg_concepts_per_memory']:.2f}")
    
    return True


def test_persistence():
    """Test enhanced data persistence."""
    print("\nTesting enhanced persistence...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_enhanced.json"
        
        # Create and populate store
        store1 = EnhancedVectorStore(storage_path=str(storage_path))
        
        memory_id = store1.store_conversation_enhanced({
            "task": "Test persistence",
            "messages": ["Testing enhanced features"]
        })
        
        # Add a relationship
        memory_id2 = store1.store_conversation_enhanced({
            "task": "Related task",
            "messages": ["Related to first task"]
        })
        
        store1.add_relationship(memory_id, memory_id2, 0.8)
        
        # Save enhanced data
        store1._save_enhanced_data()
        
        # Load in new store
        store2 = EnhancedVectorStore(storage_path=str(storage_path))
        
        # Check enhanced data was loaded
        assert len(store2.memory_nodes) == 2, "Memory nodes not loaded"
        assert memory_id in store2.memory_nodes, "Specific memory not loaded"
        assert memory_id2 in store2.memory_graph.get(memory_id, set()), "Relationships not loaded"
        assert len(store2.concept_index) > 0, "Concept index not loaded"
        
        print("[OK] Enhanced data persisted and loaded correctly")
    
    return True


if __name__ == "__main__":
    print("=== EnhancedVectorStore Test Suite ===\n")
    
    tests = [
        ("Enhanced Storage", test_enhanced_storage),
        ("Graph Relationships", test_graph_relationships),
        ("Search Strategies", test_search_strategies),
        ("Temporal Features", test_temporal_features),
        ("Memory Consolidation", test_memory_consolidation),
        ("Filtering", test_filtering),
        ("Statistics", test_memory_statistics),
        ("Persistence", test_persistence)
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
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n[SUCCESS] All enhanced vector store tests passed!")
    else:
        print("\n[WARNING] Some tests failed. Check output above.")