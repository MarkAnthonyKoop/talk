#!/usr/bin/env python3
"""
Performance and load testing for ReminiscingAgent system.

Tests system performance under various load conditions:
- Large-scale memory storage
- Search performance with many memories
- Memory leak detection
- Concurrent access patterns
- Resource usage monitoring
"""

import sys
import os
import time
import random
import psutil
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent


def test_large_scale_storage():
    """Test storage performance with large number of memories."""
    print("Testing large-scale storage performance...")
    
    store = EnhancedVectorStore()
    
    # Test different scales
    scales = [100, 500, 1000, 5000]
    times = []
    
    for scale in scales:
        memories = [
            {
                'task': f'Task {i}',
                'messages': [f'Message {i}', f'Details for task {i}'],
                'metadata': {'index': i, 'timestamp': datetime.now()}
            }
            for i in range(scale)
        ]
        
        start = time.time()
        
        for memory in memories:
            store.store_conversation_enhanced(memory)
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        rate = scale / elapsed
        print(f"  {scale} memories: {elapsed:.2f}s ({rate:.0f} memories/sec)")
        
        # Check memory wasn't corrupted
        assert len(store.memory_nodes) >= scale * 0.9, "Memory loss detected"
    
    # Check that time grows sub-linearly (good scaling)
    if len(times) >= 2:
        scaling_factor = times[-1] / times[0]
        scale_ratio = scales[-1] / scales[0]
        
        # Should scale better than O(n^2)
        assert scaling_factor < scale_ratio * scale_ratio, \
            f"Poor scaling: {scaling_factor:.1f}x time for {scale_ratio}x data"
    
    print("[OK] Storage scales appropriately")
    return True


def test_search_performance():
    """Test search performance with varying dataset sizes."""
    print("\nTesting search performance...")
    
    store = EnhancedVectorStore()
    
    # Populate with memories
    num_memories = 1000
    for i in range(num_memories):
        store.store_conversation_enhanced({
            'task': f'Task {i % 100}',  # Create some duplicates
            'messages': [f'Implementation of feature {i}']
        })
    
    # Test search performance
    search_configs = [
        ('semantic', 10),
        ('semantic', 100),
        ('graph', 10),
        ('concept', 10),
        ('hybrid', 10)
    ]
    
    for strategy, limit in search_configs:
        query = "Implementation of feature 42"
        
        start = time.time()
        results = store.search_enhanced(query, strategy=strategy, limit=limit)
        elapsed = time.time() - start
        
        assert len(results) <= limit, f"Got {len(results)} results, expected <= {limit}"
        assert elapsed < 1.0, f"Search too slow: {elapsed:.2f}s for {strategy}"
        
        print(f"  {strategy} (limit={limit}): {elapsed*1000:.1f}ms")
    
    print("[OK] Search performance acceptable")
    return True


def test_memory_leak_detection():
    """Test for memory leaks during extended operation."""
    print("\nTesting for memory leaks...")
    
    process = psutil.Process(os.getpid())
    
    # Get initial memory usage
    gc.collect()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    store = EnhancedVectorStore()
    
    # Perform many operations
    for cycle in range(5):
        # Add memories
        for i in range(100):
            store.store_conversation_enhanced({
                'task': f'Cycle {cycle} Task {i}',
                'messages': ['x' * 1000]  # Some content
            })
        
        # Search
        for i in range(50):
            store.search_enhanced(f"Task {i}", limit=10)
        
        # Delete some memories (simulate cleanup)
        if len(store.memory_nodes) > 200:
            to_remove = list(store.memory_nodes.keys())[:50]
            for mem_id in to_remove:
                store._remove_memory(mem_id)
    
    # Force garbage collection
    gc.collect()
    
    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_memory - initial_memory
    
    print(f"  Initial memory: {initial_memory:.1f} MB")
    print(f"  Final memory: {final_memory:.1f} MB")
    print(f"  Growth: {memory_growth:.1f} MB")
    
    # Allow some growth but flag excessive usage
    assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f} MB"
    
    print("[OK] No significant memory leaks detected")
    return True


def test_concurrent_access():
    """Test concurrent access to the vector store."""
    print("\nTesting concurrent access...")
    
    store = EnhancedVectorStore()
    errors = []
    
    def worker_store(worker_id: int, count: int):
        """Worker function for storing memories."""
        try:
            for i in range(count):
                store.store_conversation_enhanced({
                    'task': f'Worker {worker_id} Task {i}',
                    'messages': [f'Concurrent test {worker_id}-{i}']
                })
        except Exception as e:
            errors.append(f"Store worker {worker_id}: {e}")
    
    def worker_search(worker_id: int, count: int):
        """Worker function for searching memories."""
        try:
            for i in range(count):
                results = store.search_enhanced(
                    f"Task {i}",
                    strategy='semantic',
                    limit=5
                )
        except Exception as e:
            errors.append(f"Search worker {worker_id}: {e}")
    
    # Run concurrent operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        # Submit store workers
        for i in range(5):
            futures.append(executor.submit(worker_store, i, 20))
        
        # Submit search workers
        for i in range(5):
            futures.append(executor.submit(worker_search, i, 10))
        
        # Wait for completion
        for future in as_completed(futures):
            future.result()
    
    # Check for errors
    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    # Verify data integrity
    assert len(store.memory_nodes) == 100, \
        f"Expected 100 memories, got {len(store.memory_nodes)}"
    
    print("[OK] Concurrent access handled correctly")
    return True


def test_graph_traversal_performance():
    """Test performance of graph traversal with complex relationships."""
    print("\nTesting graph traversal performance...")
    
    store = EnhancedVectorStore()
    
    # Create a complex graph structure
    num_nodes = 500
    memories = []
    
    for i in range(num_nodes):
        memory_id = store.store_conversation_enhanced({
            'task': f'Node {i}',
            'messages': [f'Graph test node {i}']
        })
        memories.append(memory_id)
    
    # Create relationships (each node connected to 2-5 others)
    for i, memory_id in enumerate(memories):
        num_connections = random.randint(2, 5)
        for _ in range(num_connections):
            target = random.choice(memories)
            if target != memory_id:
                store.add_relationship(memory_id, target, random.random())
    
    # Test graph search performance
    start = time.time()
    results = store.search_enhanced(
        "Graph test",
        strategy='graph',
        limit=20
    )
    elapsed = time.time() - start
    
    assert len(results) <= 20, f"Got {len(results)} results"
    assert elapsed < 2.0, f"Graph search too slow: {elapsed:.2f}s"
    
    print(f"  Graph search on {num_nodes} nodes: {elapsed*1000:.1f}ms")
    print(f"  Found {len(results)} results")
    
    print("[OK] Graph traversal performance acceptable")
    return True


def test_consolidation_performance():
    """Test memory consolidation performance."""
    print("\nTesting consolidation performance...")
    
    store = EnhancedVectorStore()
    store.similarity_threshold = 0.9
    
    # Add many similar memories
    for i in range(200):
        # Create groups of similar memories
        group = i // 10
        store.store_conversation_enhanced({
            'task': f'Group {group} Task variation {i % 10}',
            'messages': [f'Similar content for group {group}']
        })
    
    initial_count = len(store.memory_nodes)
    
    start = time.time()
    store._consolidate_memories()
    elapsed = time.time() - start
    
    final_count = len(store.memory_nodes)
    consolidated = initial_count - final_count
    
    print(f"  Consolidated {consolidated} memories in {elapsed:.2f}s")
    print(f"  {initial_count} -> {final_count} memories")
    
    assert elapsed < 5.0, f"Consolidation too slow: {elapsed:.2f}s"
    assert consolidated > 0, "No memories consolidated"
    
    print("[OK] Consolidation performance acceptable")
    return True


def test_temporal_search_performance():
    """Test temporal search performance with time-based queries."""
    print("\nTesting temporal search performance...")
    
    store = EnhancedVectorStore()
    
    # Add memories with different timestamps
    now = datetime.now()
    for i in range(1000):
        memory_id = store.store_conversation_enhanced({
            'task': f'Temporal task {i}',
            'messages': [f'Task at time {i}']
        })
        
        # Manually set timestamps for testing
        if memory_id in store.memory_nodes:
            hours_ago = random.randint(0, 168)  # Up to 1 week
            store.memory_nodes[memory_id].timestamp = now - timedelta(hours=hours_ago)
    
    # Rebuild temporal index
    store.temporal_index = [
        (node.timestamp, memory_id)
        for memory_id, node in store.memory_nodes.items()
    ]
    store.temporal_index.sort(reverse=True)
    
    # Test temporal search
    start = time.time()
    results = store.search_enhanced(
        "Temporal task",
        strategy='temporal',
        limit=50
    )
    elapsed = time.time() - start
    
    assert len(results) <= 50, f"Got {len(results)} results"
    assert elapsed < 1.0, f"Temporal search too slow: {elapsed:.2f}s"
    
    # Verify results are temporally ordered
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i].get('temporal_score', 0) >= results[i+1].get('temporal_score', 0), \
                "Results not properly ordered by time"
    
    print(f"  Temporal search: {elapsed*1000:.1f}ms for {len(results)} results")
    print("[OK] Temporal search performance acceptable")
    return True


def test_batch_operation_performance():
    """Test performance of batch operations."""
    print("\nTesting batch operation performance...")
    
    store = EnhancedVectorStore()
    
    # Batch store
    batch_size = 1000
    memories = [
        {
            'task': f'Batch task {i}',
            'messages': [f'Batch content {i}']
        }
        for i in range(batch_size)
    ]
    
    start = time.time()
    memory_ids = []
    for memory in memories:
        memory_id = store.store_conversation_enhanced(memory)
        memory_ids.append(memory_id)
    batch_store_time = time.time() - start
    
    # Batch search
    queries = [f"Batch task {i}" for i in range(0, batch_size, 100)]
    
    start = time.time()
    for query in queries:
        results = store.search_enhanced(query, limit=5)
    batch_search_time = time.time() - start
    
    print(f"  Batch store ({batch_size} items): {batch_store_time:.2f}s")
    print(f"  Batch search ({len(queries)} queries): {batch_search_time:.2f}s")
    
    store_rate = batch_size / batch_store_time
    search_rate = len(queries) / batch_search_time
    
    print(f"  Store rate: {store_rate:.0f} items/sec")
    print(f"  Search rate: {search_rate:.0f} queries/sec")
    
    assert store_rate > 100, f"Store rate too low: {store_rate:.0f} items/sec"
    assert search_rate > 10, f"Search rate too low: {search_rate:.0f} queries/sec"
    
    print("[OK] Batch operations perform well")
    return True


def test_resource_usage():
    """Monitor resource usage during operations."""
    print("\nTesting resource usage...")
    
    process = psutil.Process(os.getpid())
    
    # Monitor CPU and memory during operations
    store = EnhancedVectorStore()
    
    cpu_samples = []
    memory_samples = []
    
    # Start monitoring
    for _ in range(5):
        # Perform operations
        for i in range(100):
            store.store_conversation_enhanced({
                'task': f'Resource test {i}',
                'messages': ['x' * 100]
            })
        
        # Sample resources
        cpu_samples.append(process.cpu_percent(interval=0.1))
        memory_samples.append(process.memory_info().rss / 1024 / 1024)
        
        # Search operations
        for i in range(50):
            store.search_enhanced(f"test {i}", limit=5)
    
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    avg_memory = sum(memory_samples) / len(memory_samples)
    peak_memory = max(memory_samples)
    
    print(f"  Average CPU: {avg_cpu:.1f}%")
    print(f"  Average Memory: {avg_memory:.1f} MB")
    print(f"  Peak Memory: {peak_memory:.1f} MB")
    
    # Check resource usage is reasonable
    assert avg_cpu < 80, f"Excessive CPU usage: {avg_cpu:.1f}%"
    assert peak_memory < 500, f"Excessive memory usage: {peak_memory:.1f} MB"
    
    print("[OK] Resource usage within acceptable limits")
    return True


def test_scaling_characteristics():
    """Test how the system scales with increasing load."""
    print("\nTesting scaling characteristics...")
    
    results = []
    
    for size in [100, 200, 400, 800]:
        store = EnhancedVectorStore()
        
        # Add memories
        start = time.time()
        for i in range(size):
            store.store_conversation_enhanced({
                'task': f'Scale test {i}',
                'messages': [f'Content {i}']
            })
        store_time = time.time() - start
        
        # Search
        start = time.time()
        for i in range(min(50, size // 2)):
            store.search_enhanced(f"test {i}", limit=10)
        search_time = time.time() - start
        
        results.append({
            'size': size,
            'store_time': store_time,
            'search_time': search_time,
            'store_rate': size / store_time,
            'search_rate': min(50, size // 2) / search_time
        })
        
        print(f"  Size {size}: store={store_time:.2f}s, search={search_time:.2f}s")
    
    # Analyze scaling
    if len(results) >= 2:
        # Check that performance doesn't degrade too badly
        first_store_rate = results[0]['store_rate']
        last_store_rate = results[-1]['store_rate']
        
        degradation = (first_store_rate - last_store_rate) / first_store_rate
        
        assert degradation < 0.5, \
            f"Severe performance degradation: {degradation*100:.1f}% slower"
    
    print("[OK] System scales acceptably")
    return True


if __name__ == "__main__":
    print("=== Performance Test Suite ===")
    print("Note: These tests may take several seconds each\n")
    
    tests = [
        ("Large-scale Storage", test_large_scale_storage),
        ("Search Performance", test_search_performance),
        ("Memory Leak Detection", test_memory_leak_detection),
        ("Concurrent Access", test_concurrent_access),
        ("Graph Traversal", test_graph_traversal_performance),
        ("Consolidation Performance", test_consolidation_performance),
        ("Temporal Search", test_temporal_search_performance),
        ("Batch Operations", test_batch_operation_performance),
        ("Resource Usage", test_resource_usage),
        ("Scaling Characteristics", test_scaling_characteristics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n--- {test_name} ---")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n=== Performance Test Summary ===")
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests ({100*passed//total}%)")
    
    if passed == total:
        print("[SUCCESS] All performance tests passed!")
    else:
        print("[WARNING] Some performance tests failed.")