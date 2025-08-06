#!/usr/bin/env python3
"""
Test collision prevention in memory ID generation.

Verifies that the UUID-based ID generation prevents collisions.
"""

import sys
import os
import time
import hashlib
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.vector_store import ConversationVectorStore
from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore


def test_uuid_based_id_generation():
    """Test that new UUID-based ID generation prevents collisions."""
    print("Testing UUID-based ID generation...")
    
    store = ConversationVectorStore()
    
    # Generate many IDs for identical content
    ids = set()
    identical_data = {"task": "Same task", "messages": ["Same message"]}
    
    for i in range(1000):
        memory_id = store._generate_memory_id(identical_data.copy())
        
        # All IDs should be unique even with identical content
        assert memory_id not in ids, f"Collision detected at iteration {i}: {memory_id}"
        ids.add(memory_id)
    
    print(f"  ✓ Generated {len(ids)} unique IDs for identical content")
    
    # Verify ID format (8 chars content hash + 8 chars UUID)
    sample_id = next(iter(ids))
    assert len(sample_id) == 16, f"ID length should be 16, got {len(sample_id)}"
    
    # First 8 chars should be same for identical content (content hash)
    content_hashes = {id[:8] for id in ids}
    assert len(content_hashes) == 1, "Content hash should be consistent for same content"
    
    # Last 8 chars should all be different (UUID part)
    unique_parts = {id[8:] for id in ids}
    assert len(unique_parts) == 1000, "UUID parts should all be unique"
    
    print(f"  ✓ ID format correct: {sample_id[:8]}(content)|{sample_id[8:]}(uuid)")
    
    return True


def test_collision_detection_in_store():
    """Test that store methods detect and handle collisions."""
    print("\nTesting collision detection in store methods...")
    
    store = ConversationVectorStore()
    
    # Store many similar memories rapidly
    memory_ids = []
    for i in range(100):
        memory_id = store.store_conversation({
            'task': 'Rapid insertion test',
            'messages': [f'Message {i % 10}']  # Some duplicates
        })
        memory_ids.append(memory_id)
    
    # All IDs should be unique
    assert len(set(memory_ids)) == 100, f"Collisions detected: {100 - len(set(memory_ids))}"
    
    print(f"  ✓ Stored {len(memory_ids)} memories with no collisions")
    
    # Verify all memories are actually stored
    assert len(store.conversations) == 100, "Some memories were lost"
    
    print("  ✓ All memories preserved (no overwrites)")
    
    return True


def test_concurrent_id_generation_safety():
    """Test thread-safe ID generation under concurrent load."""
    print("\nTesting concurrent ID generation safety...")
    
    store = ConversationVectorStore()
    all_ids = set()
    errors = []
    
    def generate_batch(worker_id):
        """Generate IDs concurrently."""
        local_ids = []
        try:
            for i in range(50):
                data = {
                    'task': f'Worker {worker_id} Task {i}',
                    'messages': [f'Concurrent test {worker_id}-{i}']
                }
                memory_id = store._generate_memory_id(data)
                local_ids.append(memory_id)
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")
        return local_ids
    
    # Run concurrent ID generation
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(generate_batch, i) for i in range(20)]
        
        for future in as_completed(futures):
            ids = future.result()
            all_ids.update(ids)
    
    expected_count = 20 * 50  # 20 workers * 50 IDs each
    unique_count = len(all_ids)
    
    assert unique_count == expected_count, \
        f"Collisions in concurrent generation: {expected_count - unique_count}"
    assert len(errors) == 0, f"Errors during concurrent generation: {errors}"
    
    print(f"  ✓ Generated {unique_count} unique IDs concurrently (20 threads)")
    
    return True


def test_enhanced_store_collision_prevention():
    """Test collision prevention in EnhancedVectorStore."""
    print("\nTesting EnhancedVectorStore collision prevention...")
    
    store = EnhancedVectorStore()
    
    # Test with many similar memories
    memory_ids = []
    for i in range(200):
        memory_id = store.store_conversation_enhanced({
            'task': f'Enhanced test {i % 20}',  # Create some similar content
            'messages': ['Testing collision prevention']
        })
        memory_ids.append(memory_id)
    
    # Check uniqueness
    unique_ids = set(memory_ids)
    assert len(unique_ids) == 200, \
        f"Collisions detected: {200 - len(unique_ids)} duplicate IDs"
    
    print(f"  ✓ Enhanced store: {len(unique_ids)} unique IDs")
    
    # Verify memory integrity
    assert len(store.memory_nodes) >= 190, "Memory loss detected (after consolidation)"
    
    print("  ✓ Memory integrity maintained")
    
    return True


def test_collision_probability_eliminated():
    """Verify that collision probability is effectively zero."""
    print("\nTesting collision probability elimination...")
    
    store = ConversationVectorStore()
    
    # Generate a large number of IDs
    num_tests = 10000
    ids = set()
    
    # Test with varied content
    for i in range(num_tests):
        data = {
            'task': f'Task {i % 100}',
            'messages': [f'Message {i}', f'Detail {i % 50}']
        }
        memory_id = store._generate_memory_id(data)
        ids.add(memory_id)
    
    collision_rate = (num_tests - len(ids)) / num_tests
    
    assert collision_rate == 0, f"Collision rate: {collision_rate*100:.4f}%"
    
    print(f"  ✓ {num_tests} IDs generated with 0% collision rate")
    
    # Test with identical content (worst case)
    identical_ids = set()
    identical_data = {"task": "Identical", "messages": ["Same"]}
    
    for i in range(1000):
        memory_id = store._generate_memory_id(identical_data.copy())
        identical_ids.add(memory_id)
    
    assert len(identical_ids) == 1000, \
        f"Collisions with identical content: {1000 - len(identical_ids)}"
    
    print(f"  ✓ 1000 identical content IDs all unique")
    
    return True


def test_backwards_compatibility():
    """Test that existing memories can still be accessed."""
    print("\nTesting backwards compatibility...")
    
    store = ConversationVectorStore()
    
    # Simulate old-style ID (16 hex chars)
    old_style_id = hashlib.md5(b"test").hexdigest()[:16]
    
    # Manually add a memory with old-style ID
    old_memory = {
        'memory_id': old_style_id,
        'timestamp': datetime.now().isoformat(),
        'content': 'Old memory',
        'original_data': {'task': 'Old task'},
        'memory_type': 'conversation'
    }
    store.conversations.append(old_memory)
    store.embeddings[old_style_id] = [0.0] * store.embedding_dim
    store.metadata[old_style_id] = {'memory_type': 'conversation'}
    
    # Verify old memory can be searched
    results = store.search_memories('Old memory', limit=10)
    
    found_old = any(r.get('memory_id') == old_style_id for r in results)
    assert found_old, "Old-style memory not found in search"
    
    print(f"  ✓ Old-style ID '{old_style_id}' still searchable")
    
    # Add new memory and verify both work
    new_id = store.store_conversation({
        'task': 'New task',
        'messages': ['New memory']
    })
    
    # Search should find both
    all_results = store.search_memories('memory', limit=10)
    assert len(all_results) >= 2, "Should find both old and new memories"
    
    print(f"  ✓ Mixed old and new IDs work together")
    
    return True


if __name__ == "__main__":
    print("=== Collision Prevention Test Suite ===\n")
    
    tests = [
        ("UUID-based ID Generation", test_uuid_based_id_generation),
        ("Collision Detection in Store", test_collision_detection_in_store),
        ("Concurrent ID Generation Safety", test_concurrent_id_generation_safety),
        ("Enhanced Store Prevention", test_enhanced_store_collision_prevention),
        ("Collision Probability Eliminated", test_collision_probability_eliminated),
        ("Backwards Compatibility", test_backwards_compatibility),
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
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✅ COLLISION PREVENTION WORKING PERFECTLY!")
        print("The memory ID collision issue has been completely resolved.")
    else:
        print("\n⚠️ Some tests failed - check implementation")