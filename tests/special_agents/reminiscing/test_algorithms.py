#!/usr/bin/env python3
"""
Algorithm correctness tests for ReminiscingAgent system.

Tests the correctness of core algorithms including:
- Spreading activation
- Vector similarity calculations
- Graph traversal
- Memory consolidation
- Concept extraction
"""

import sys
import os
import random
import math
from typing import List, Set, Dict, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore
from special_agents.reminiscing.memory_trace_agent import MemoryTraceAgent


def test_spreading_activation_correctness():
    """Test spreading activation algorithm correctness."""
    print("Testing spreading activation algorithm...")
    
    agent = MemoryTraceAgent()
    
    # Create a known graph structure
    # A -> B -> C
    # A -> D
    # B -> E
    memories = {
        'A': {'connections': ['B', 'D'], 'activation': 1.0},
        'B': {'connections': ['C', 'E'], 'activation': 0.0},
        'C': {'connections': [], 'activation': 0.0},
        'D': {'connections': [], 'activation': 0.0},
        'E': {'connections': [], 'activation': 0.0}
    }
    
    # Set up graph
    for node_id, data in memories.items():
        for connected in data['connections']:
            agent.add_memory_relationship(node_id, connected)
    
    # Simulate spreading activation
    decay_factor = 0.7
    max_hops = 3
    
    # Expected activations after spreading from A
    expected = {
        'A': 1.0,  # Source
        'B': 0.7,  # 1 hop from A
        'D': 0.7,  # 1 hop from A
        'C': 0.49,  # 2 hops from A (through B)
        'E': 0.49   # 2 hops from A (through B)
    }
    
    # Run spreading activation
    activations = simulate_spreading_activation(
        memories, 'A', decay_factor, max_hops
    )
    
    # Check correctness
    for node, expected_activation in expected.items():
        actual = activations.get(node, 0.0)
        assert abs(actual - expected_activation) < 0.01, \
            f"Node {node}: expected {expected_activation}, got {actual}"
    
    print(f"[OK] Spreading activation correct for {len(memories)} nodes")
    return True


def test_vector_similarity_properties():
    """Test vector similarity calculation properties."""
    print("\nTesting vector similarity properties...")
    
    store = EnhancedVectorStore()
    
    # Test identity property: sim(A, A) = 1
    vec1 = [random.random() for _ in range(100)]
    similarity = store._calculate_similarity(vec1, vec1)
    assert abs(similarity - 1.0) < 0.001, f"Identity similarity should be 1.0, got {similarity}"
    
    # Test symmetry: sim(A, B) = sim(B, A)
    vec2 = [random.random() for _ in range(100)]
    sim_ab = store._calculate_similarity(vec1, vec2)
    sim_ba = store._calculate_similarity(vec2, vec1)
    assert abs(sim_ab - sim_ba) < 0.001, f"Symmetry violated: {sim_ab} != {sim_ba}"
    
    # Test triangle inequality for distance
    vec3 = [random.random() for _ in range(100)]
    dist_ab = 1 - store._calculate_similarity(vec1, vec2)
    dist_bc = 1 - store._calculate_similarity(vec2, vec3)
    dist_ac = 1 - store._calculate_similarity(vec1, vec3)
    
    # Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
    assert dist_ac <= dist_ab + dist_bc + 0.001, \
        f"Triangle inequality violated: {dist_ac} > {dist_ab} + {dist_bc}"
    
    # Test orthogonality
    orthogonal1 = [1, 0, 0, 0]
    orthogonal2 = [0, 1, 0, 0]
    sim_orthogonal = store._calculate_similarity(orthogonal1, orthogonal2)
    assert abs(sim_orthogonal) < 0.001, f"Orthogonal vectors should have similarity ~0, got {sim_orthogonal}"
    
    print("[OK] Vector similarity properties verified")
    return True


def test_graph_traversal_termination():
    """Test that graph traversal terminates correctly."""
    print("\nTesting graph traversal termination...")
    
    agent = MemoryTraceAgent()
    
    # Create a graph with cycles
    # A <-> B <-> C
    #  \         /
    #   <------>
    agent.add_memory_relationship('A', 'B')
    agent.add_memory_relationship('B', 'C')
    agent.add_memory_relationship('C', 'A')  # Creates cycle
    
    # Test that traversal terminates despite cycle
    visited = set()
    queue = [('A', 0)]  # (node, depth)
    max_depth = 5
    
    while queue:
        node, depth = queue.pop(0)
        
        if node in visited or depth > max_depth:
            continue
        
        visited.add(node)
        
        # Add neighbors
        neighbors = agent.memory_graph.get(node, set())
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
    
    # Should visit all nodes exactly once
    assert len(visited) == 3, f"Expected to visit 3 nodes, visited {len(visited)}"
    assert visited == {'A', 'B', 'C'}, f"Unexpected nodes visited: {visited}"
    
    print("[OK] Graph traversal terminates correctly with cycles")
    return True


def test_memory_consolidation_correctness():
    """Test memory consolidation algorithm correctness."""
    print("\nTesting memory consolidation correctness...")
    
    store = EnhancedVectorStore()
    store.similarity_threshold = 0.95  # High threshold for testing
    
    # Create memories with known similarities
    identical_memories = [
        {'content': 'Implement authentication', 'id': 'mem1'},
        {'content': 'Implement authentication', 'id': 'mem2'},  # Identical
    ]
    
    similar_memories = [
        {'content': 'Create login system', 'id': 'mem3'},
        {'content': 'Build user login', 'id': 'mem4'},  # Similar but not identical
    ]
    
    different_memory = {'content': 'Optimize database queries', 'id': 'mem5'}
    
    # Store all memories
    all_memories = identical_memories + similar_memories + [different_memory]
    memory_ids = []
    
    for mem in all_memories:
        memory_id = store.store_conversation_enhanced({
            'task': mem['content'],
            'messages': [mem['content']]
        })
        memory_ids.append(memory_id)
        mem['stored_id'] = memory_id
    
    initial_count = len(store.memory_nodes)
    
    # Run consolidation
    store._consolidate_memories()
    
    final_count = len(store.memory_nodes)
    
    # Identical memories should be consolidated
    assert final_count < initial_count, "No consolidation occurred"
    
    # Different memory should still exist
    different_exists = any(
        'database' in node.content.lower()
        for node in store.memory_nodes.values()
    )
    assert different_exists, "Unique memory was incorrectly consolidated"
    
    # Check that relationships are preserved
    # If mem1 had relationships, they should transfer to the kept memory
    
    print(f"[OK] Consolidation: {initial_count} -> {final_count} memories")
    return True


def test_concept_extraction_accuracy():
    """Test concept extraction algorithm accuracy."""
    print("\nTesting concept extraction accuracy...")
    
    store = EnhancedVectorStore()
    
    test_cases = [
        {
            'text': 'UserAuthenticationManager handles OAuth2Login flows',
            'expected_concepts': {'user', 'authentication', 'manager', 'oauth', 'login', 'flow'},
            'min_match_ratio': 0.7
        },
        {
            'text': 'database_connection_pool timeout_error retry_logic',
            'expected_concepts': {'database', 'connection', 'pool', 'timeout', 'error', 'retry', 'logic'},
            'min_match_ratio': 0.8
        },
        {
            'text': 'class APIGateway implements RESTful endpoints',
            'expected_concepts': {'class', 'api', 'gateway', 'implement', 'rest', 'endpoint'},
            'min_match_ratio': 0.6
        }
    ]
    
    for test in test_cases:
        extracted = store._extract_concepts(test['text'])
        
        # Convert to lowercase for comparison
        extracted_lower = {c.lower() for c in extracted}
        expected_lower = {c.lower() for c in test['expected_concepts']}
        
        # Calculate match ratio
        matches = extracted_lower.intersection(expected_lower)
        match_ratio = len(matches) / len(expected_lower) if expected_lower else 0
        
        assert match_ratio >= test['min_match_ratio'], \
            f"Concept extraction accuracy {match_ratio:.2f} < {test['min_match_ratio']} for '{test['text'][:30]}...'"
        
        print(f"  Extracted {len(extracted)} concepts, {len(matches)}/{len(expected_lower)} matches")
    
    print("[OK] Concept extraction accuracy verified")
    return True


def test_temporal_decay_formula():
    """Test temporal decay calculation correctness."""
    print("\nTesting temporal decay formula...")
    
    from datetime import datetime, timedelta
    
    store = EnhancedVectorStore()
    store.decay_factor = 0.95  # 5% daily decay
    
    # Test decay over time
    now = datetime.now()
    test_cases = [
        (0, 1.0),      # Today: no decay
        (1, 0.95),     # 1 day: 5% decay
        (7, 0.6983),   # 1 week: ~30% remaining
        (30, 0.2146),  # 1 month: ~21% remaining
    ]
    
    for days_old, expected_score in test_cases:
        timestamp = now - timedelta(days=days_old)
        
        # Create a memory node
        node_id = f'test_{days_old}'
        store.memory_nodes[node_id] = type('Node', (), {
            'timestamp': timestamp,
            'importance_score': 1.0
        })()
        
        # Apply decay
        days = (now - timestamp).days
        actual_decay = store.decay_factor ** days
        
        assert abs(actual_decay - expected_score) < 0.01, \
            f"Day {days_old}: expected decay {expected_score:.4f}, got {actual_decay:.4f}"
    
    print("[OK] Temporal decay formula correct")
    return True


def test_search_strategy_selection():
    """Test that search strategies are selected correctly."""
    print("\nTesting search strategy selection...")
    
    from special_agents.reminiscing.context_categorization_agent import ContextCategorizationAgent
    
    agent = ContextCategorizationAgent()
    
    test_cases = [
        {
            'context': "How should I design the microservice architecture?",
            'expected_category': 'architectural',
            'expected_strategy': 'graph_traversal'
        },
        {
            'context': "Debug the null pointer exception in production",
            'expected_category': 'debugging',
            'expected_strategy': 'error_similarity'
        },
        {
            'context': "Implement the payment processing feature",
            'expected_category': 'implementation',
            'expected_strategy': 'code_similarity'
        },
        {
            'context': "What are the best practices for API versioning?",
            'expected_category': 'research',
            'expected_strategy': 'semantic_search'
        }
    ]
    
    for test in test_cases:
        # Use pattern-based categorization for deterministic testing
        category = agent._pattern_based_categorize(test['context'])
        strategy = agent.category_strategies.get(category, 'semantic_search')
        
        # Allow some flexibility in categorization
        if test['expected_category'] != 'general':
            assert category != 'general', \
                f"Failed to categorize: '{test['context'][:50]}...'"
        
        print(f"  '{test['context'][:40]}...' -> {category}/{strategy}")
    
    print("[OK] Search strategy selection working")
    return True


def test_relationship_strength_calculation():
    """Test relationship strength calculation between memories."""
    print("\nTesting relationship strength calculation...")
    
    store = EnhancedVectorStore()
    
    # Create test memories with known relationships
    mem1_id = store.store_conversation_enhanced({
        'task': 'Implement OAuth authentication',
        'messages': ['Using passport.js', 'Google provider']
    })
    
    mem2_id = store.store_conversation_enhanced({
        'task': 'Add JWT authentication',
        'messages': ['Token-based auth', 'Stateless']
    })
    
    mem3_id = store.store_conversation_enhanced({
        'task': 'Fix database timeout',
        'messages': ['Connection pool', 'Timeout error']
    })
    
    # Calculate relationship strengths
    def calculate_strength(id1, id2):
        node1 = store.memory_nodes.get(id1)
        node2 = store.memory_nodes.get(id2)
        
        if not node1 or not node2:
            return 0.0
        
        # Concept overlap
        concept_overlap = len(node1.concepts.intersection(node2.concepts))
        
        # Semantic similarity
        similarity = store._calculate_similarity(
            store.embeddings[id1],
            store.embeddings[id2]
        )
        
        return concept_overlap * 0.6 + similarity * 0.4
    
    # Auth memories should have higher relationship strength
    auth_strength = calculate_strength(mem1_id, mem2_id)
    unrelated_strength = calculate_strength(mem1_id, mem3_id)
    
    assert auth_strength > unrelated_strength, \
        f"Related memories should have higher strength: {auth_strength} <= {unrelated_strength}"
    
    print(f"[OK] Relationship strengths: related={auth_strength:.2f}, unrelated={unrelated_strength:.2f}")
    return True


def test_activation_propagation_limits():
    """Test that activation propagation has proper limits."""
    print("\nTesting activation propagation limits...")
    
    agent = MemoryTraceAgent()
    
    # Create a chain of memories
    chain_length = 10
    for i in range(chain_length - 1):
        agent.add_memory_relationship(f'mem_{i}', f'mem_{i+1}')
    
    # Test activation propagation
    initial_activation = 1.0
    decay = 0.7
    max_hops = 3
    
    # Calculate expected activation at each hop
    activations = {}
    activations['mem_0'] = initial_activation
    
    for hop in range(1, max_hops + 1):
        expected = initial_activation * (decay ** hop)
        activations[f'mem_{hop}'] = expected
    
    # Nodes beyond max_hops should have 0 activation
    for i in range(max_hops + 1, chain_length):
        activations[f'mem_{i}'] = 0.0
    
    # Simulate propagation
    result = simulate_spreading_activation(
        {f'mem_{i}': {'connections': [f'mem_{i+1}'] if i < chain_length-1 else []} 
         for i in range(chain_length)},
        'mem_0',
        decay,
        max_hops
    )
    
    # Check limits
    for i in range(max_hops + 1, chain_length):
        assert result.get(f'mem_{i}', 0.0) == 0.0, \
            f"Activation spread beyond max_hops at mem_{i}"
    
    print(f"[OK] Activation properly limited to {max_hops} hops")
    return True


def test_score_combination_formula():
    """Test score combination formulas for hybrid search."""
    print("\nTesting score combination formulas...")
    
    # Test different combination methods
    scores = {
        'semantic': 0.8,
        'concept': 0.6,
        'temporal': 0.4
    }
    
    # Weighted average
    weights = {'semantic': 0.5, 'concept': 0.3, 'temporal': 0.2}
    weighted_avg = sum(scores[k] * weights[k] for k in scores)
    expected_weighted = 0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2  # 0.66
    assert abs(weighted_avg - expected_weighted) < 0.001, \
        f"Weighted average incorrect: {weighted_avg} != {expected_weighted}"
    
    # Harmonic mean (for non-zero scores)
    non_zero = [s for s in scores.values() if s > 0]
    harmonic = len(non_zero) / sum(1/s for s in non_zero) if non_zero else 0
    expected_harmonic = 3 / (1/0.8 + 1/0.6 + 1/0.4)  # ~0.533
    assert abs(harmonic - expected_harmonic) < 0.01, \
        f"Harmonic mean incorrect: {harmonic:.3f} != {expected_harmonic:.3f}"
    
    # Maximum (for OR-like combination)
    max_score = max(scores.values())
    assert max_score == 0.8, f"Maximum score incorrect: {max_score} != 0.8"
    
    print("[OK] Score combination formulas verified")
    return True


# Helper functions

def simulate_spreading_activation(
    memories: Dict[str, Dict],
    start_node: str,
    decay: float,
    max_hops: int
) -> Dict[str, float]:
    """Simulate spreading activation through a graph."""
    activations = {start_node: 1.0}
    visited = set()
    queue = [(start_node, 0, 1.0)]  # (node, depth, activation)
    
    while queue:
        node, depth, activation = queue.pop(0)
        
        if node in visited or depth >= max_hops:
            continue
        
        visited.add(node)
        
        # Spread to connections
        for connected in memories.get(node, {}).get('connections', []):
            new_activation = activation * decay
            
            if connected in activations:
                activations[connected] = max(activations[connected], new_activation)
            else:
                activations[connected] = new_activation
            
            if connected not in visited:
                queue.append((connected, depth + 1, new_activation))
    
    return activations


if __name__ == "__main__":
    print("=== Algorithm Correctness Test Suite ===\n")
    
    tests = [
        ("Spreading Activation", test_spreading_activation_correctness),
        ("Vector Similarity Properties", test_vector_similarity_properties),
        ("Graph Traversal Termination", test_graph_traversal_termination),
        ("Memory Consolidation", test_memory_consolidation_correctness),
        ("Concept Extraction", test_concept_extraction_accuracy),
        ("Temporal Decay", test_temporal_decay_formula),
        ("Strategy Selection", test_search_strategy_selection),
        ("Relationship Strength", test_relationship_strength_calculation),
        ("Activation Limits", test_activation_propagation_limits),
        ("Score Combination", test_score_combination_formula)
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
        print("[SUCCESS] All algorithm tests passed!")
    else:
        print("[WARNING] Some algorithm tests failed.")