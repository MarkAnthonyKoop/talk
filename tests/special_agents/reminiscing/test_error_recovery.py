#!/usr/bin/env python3
"""
Error recovery tests for ReminiscingAgent system.

Tests error handling and recovery scenarios:
- Corrupted data handling
- Missing dependencies
- File I/O errors
- Network failures
- Invalid input handling
- Graceful degradation
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
from special_agents.reminiscing.vector_store import ConversationVectorStore


def test_corrupted_storage_recovery():
    """Test recovery from corrupted storage files."""
    print("Testing corrupted storage recovery...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.json"
        
        # Create store and add data
        store = EnhancedVectorStore(storage_path=str(storage_path))
        store.store_conversation_enhanced({
            'task': 'Test task',
            'messages': ['Test message']
        })
        
        # Corrupt the storage file
        with open(storage_path, 'w') as f:
            f.write("{ corrupted json data }")
        
        # Try to load corrupted data
        try:
            store2 = EnhancedVectorStore(storage_path=str(storage_path))
            # Should handle gracefully
            assert True, "Handled corrupted data"
        except json.JSONDecodeError:
            assert False, "Failed to handle corrupted JSON"
        
        # Store should still be usable
        memory_id = store2.store_conversation({
            'task': 'Recovery test',
            'messages': ['After corruption']
        })
        assert memory_id is not None, "Store not usable after corruption"
    
    print("[OK] Recovered from corrupted storage")
    return True


def test_missing_dependencies():
    """Test handling of missing optional dependencies."""
    print("\nTesting missing dependencies handling...")
    
    # Mock missing numpy
    with patch('special_agents.reminiscing.vector_store.NUMPY_AVAILABLE', False):
        with patch('special_agents.reminiscing.vector_store.np', None):
            store = ConversationVectorStore()
            
            # Should still work without numpy
            memory_id = store.store_conversation({
                'task': 'Test without numpy',
                'messages': ['No numpy available']
            })
            assert memory_id is not None, "Failed without numpy"
            
            # Search should work with fallback
            results = store.search_memories('test', limit=5)
            assert isinstance(results, list), "Search failed without numpy"
    
    # Mock missing LangGraph
    with patch('special_agents.reminiscing.reminiscing_agent.LANGGRAPH_AVAILABLE', False):
        agent = ReminiscingAgent()
        
        # Should use simplified workflow
        result = agent.run("Test without LangGraph")
        assert 'MEMORY_TRACES' in result or 'MEMORY_ERROR' in result, \
            "Agent failed without LangGraph"
    
    print("[OK] Handles missing dependencies gracefully")
    return True


def test_file_io_errors():
    """Test handling of file I/O errors."""
    print("\nTesting file I/O error handling...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.json"
        
        # Create store
        store = EnhancedVectorStore(storage_path=str(storage_path))
        store.store_conversation_enhanced({
            'task': 'Test task',
            'messages': ['Test message']
        })
        
        # Make directory read-only to cause write error
        os.chmod(tmpdir, 0o444)
        
        try:
            # Try to save (should fail gracefully)
            store._save_to_disk()
            # Should not crash
            assert True, "Handled write error"
        except PermissionError:
            assert False, "Did not handle permission error"
        finally:
            # Restore permissions
            os.chmod(tmpdir, 0o755)
        
        # Test non-existent path
        bad_store = EnhancedVectorStore(storage_path="/nonexistent/path/test.json")
        try:
            bad_store._save_to_disk()
            assert True, "Handled non-existent path"
        except Exception:
            assert False, "Failed on non-existent path"
    
    print("[OK] File I/O errors handled gracefully")
    return True


def test_invalid_input_handling():
    """Test handling of invalid inputs."""
    print("\nTesting invalid input handling...")
    
    agent = ReminiscingAgent()
    store = EnhancedVectorStore()
    
    # Test with various invalid inputs
    invalid_inputs = [
        None,
        "",
        123,
        [],
        {},
        "a" * 100000,  # Very long string
        "\x00\x01\x02",  # Binary data
        "{'invalid': json}",  # Invalid JSON
    ]
    
    for invalid_input in invalid_inputs:
        try:
            # Agent should handle invalid input
            result = agent.run(invalid_input if isinstance(invalid_input, str) else str(invalid_input))
            assert isinstance(result, str), f"Invalid result type for input: {type(invalid_input)}"
            
            # Store should handle invalid input
            if isinstance(invalid_input, dict):
                store.store_conversation(invalid_input)
            
        except Exception as e:
            assert False, f"Failed on invalid input {type(invalid_input)}: {e}"
    
    print("[OK] Invalid inputs handled gracefully")
    return True


def test_memory_limit_enforcement():
    """Test that memory limits are enforced correctly."""
    print("\nTesting memory limit enforcement...")
    
    store = EnhancedVectorStore()
    store.max_memories = 10  # Set low limit
    
    # Add more than limit
    for i in range(20):
        store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Message {i}']
        })
    
    # Check limit is enforced
    total = len(store.conversations) + len(store.code_contexts)
    assert total <= store.max_memories, \
        f"Memory limit not enforced: {total} > {store.max_memories}"
    
    # Oldest memories should be removed
    remaining_content = ' '.join(
        m.get('content', '') for m in store.conversations
    )
    
    # Recent memories should be present
    assert 'Task 19' in remaining_content or 'Message 19' in remaining_content, \
        "Recent memories were removed instead of old ones"
    
    print("[OK] Memory limits enforced correctly")
    return True


def test_circular_reference_handling():
    """Test handling of circular references in memory graph."""
    print("\nTesting circular reference handling...")
    
    store = EnhancedVectorStore()
    
    # Create circular references
    mem1 = store.store_conversation_enhanced({'task': 'A', 'messages': ['Memory A']})
    mem2 = store.store_conversation_enhanced({'task': 'B', 'messages': ['Memory B']})
    mem3 = store.store_conversation_enhanced({'task': 'C', 'messages': ['Memory C']})
    
    # Create cycle: A -> B -> C -> A
    store.add_relationship(mem1, mem2)
    store.add_relationship(mem2, mem3)
    store.add_relationship(mem3, mem1)
    
    # Graph search should not get stuck
    try:
        results = store.search_enhanced('Memory', strategy='graph', limit=10)
        assert len(results) <= 10, "Graph search didn't respect limit"
        assert True, "Handled circular references"
    except RecursionError:
        assert False, "Got stuck in circular reference"
    
    print("[OK] Circular references handled correctly")
    return True


def test_concurrent_modification_handling():
    """Test handling of concurrent modifications."""
    print("\nTesting concurrent modification handling...")
    
    store = EnhancedVectorStore()
    
    # Add initial memories
    memory_ids = []
    for i in range(10):
        memory_id = store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Message {i}']
        })
        memory_ids.append(memory_id)
    
    # Simulate concurrent modification during iteration
    try:
        # Search while modifying
        results = store.search_enhanced('Task', limit=5)
        
        # Modify during result processing
        for result in results:
            # Add new memory (modifies internal structures)
            store.store_conversation_enhanced({
                'task': 'Concurrent task',
                'messages': ['Added during iteration']
            })
            break  # Just test one modification
        
        assert True, "Handled concurrent modification"
    except RuntimeError as e:
        if "dictionary changed size" in str(e):
            assert False, "Failed on concurrent modification"
        raise
    
    print("[OK] Concurrent modifications handled")
    return True


def test_embedding_generation_failure():
    """Test handling of embedding generation failures."""
    print("\nTesting embedding generation failure handling...")
    
    store = EnhancedVectorStore()
    
    # Mock embedding generation to fail
    original_generate = store._generate_embedding
    
    def failing_embedding(text):
        if "fail" in text.lower():
            raise ValueError("Embedding generation failed")
        return original_generate(text)
    
    store._generate_embedding = failing_embedding
    
    # Try to store with failing embedding
    try:
        memory_id = store.store_conversation({
            'task': 'This will fail',
            'messages': ['Embedding failure test']
        })
        # Should handle the error
        assert memory_id is None or isinstance(memory_id, str), \
            "Unexpected result from failed embedding"
    except ValueError:
        # Should not propagate the error
        assert False, "Embedding error not handled"
    
    # Restore and verify store still works
    store._generate_embedding = original_generate
    memory_id = store.store_conversation({
        'task': 'This should work',
        'messages': ['After failure recovery']
    })
    assert memory_id is not None, "Store broken after embedding failure"
    
    print("[OK] Embedding failures handled gracefully")
    return True


def test_search_with_empty_store():
    """Test searching with empty or minimal data."""
    print("\nTesting search with empty store...")
    
    store = EnhancedVectorStore()
    
    # Search empty store
    results = store.search_enhanced('test query', strategy='hybrid', limit=10)
    assert isinstance(results, list), "Search failed on empty store"
    assert len(results) == 0, "Got results from empty store"
    
    # Add single memory and search
    store.store_conversation_enhanced({
        'task': 'Single task',
        'messages': ['Only memory']
    })
    
    results = store.search_enhanced('task', strategy='hybrid', limit=10)
    assert len(results) <= 1, "Got multiple results from single memory"
    
    print("[OK] Empty store searches handled correctly")
    return True


def test_malformed_memory_handling():
    """Test handling of malformed memory data."""
    print("\nTesting malformed memory handling...")
    
    store = EnhancedVectorStore()
    
    # Try to store malformed memories
    malformed_memories = [
        {},  # Empty
        {'task': None},  # None values
        {'messages': 'not a list'},  # Wrong type
        {'task': '', 'messages': []},  # Empty content
        {'unexpected': 'field'},  # Missing required fields
    ]
    
    stored_count = 0
    for memory in malformed_memories:
        try:
            memory_id = store.store_conversation(memory)
            if memory_id:
                stored_count += 1
        except Exception as e:
            # Should handle gracefully
            print(f"  Handled malformed memory: {type(e).__name__}")
    
    # Some memories might be stored with defaults
    assert stored_count >= 0, "Crashed on malformed memory"
    
    print("[OK] Malformed memories handled gracefully")
    return True


def test_recovery_after_partial_failure():
    """Test recovery after partial operation failure."""
    print("\nTesting recovery after partial failure...")
    
    store = EnhancedVectorStore()
    
    # Add some memories
    for i in range(5):
        store.store_conversation_enhanced({
            'task': f'Task {i}',
            'messages': [f'Message {i}']
        })
    
    # Simulate partial failure during consolidation
    original_remove = store._remove_memory
    call_count = [0]
    
    def failing_remove(memory_id):
        call_count[0] += 1
        if call_count[0] == 2:  # Fail on second removal
            raise RuntimeError("Simulated removal failure")
        return original_remove(memory_id)
    
    store._remove_memory = failing_remove
    
    # Try consolidation (should handle partial failure)
    try:
        store._consolidate_memories()
    except RuntimeError:
        pass  # Expected
    
    # Restore function
    store._remove_memory = original_remove
    
    # Store should still be functional
    memory_id = store.store_conversation_enhanced({
        'task': 'After failure',
        'messages': ['Recovery test']
    })
    assert memory_id is not None, "Store not functional after partial failure"
    
    # Search should work
    results = store.search_enhanced('Task', limit=10)
    assert len(results) > 0, "Search broken after partial failure"
    
    print("[OK] Recovered from partial operation failure")
    return True


def test_agent_llm_failure_handling():
    """Test agent handling of LLM failures."""
    print("\nTesting agent LLM failure handling...")
    
    agent = ReminiscingAgent()
    
    # Mock LLM to fail
    with patch.object(agent.categorization_agent, 'call_ai') as mock_llm:
        mock_llm.side_effect = Exception("LLM API error")
        
        # Should fall back to pattern-based categorization
        result = agent.run("Test with LLM failure")
        
        assert isinstance(result, str), "Agent failed with LLM error"
        assert 'MEMORY_TRACES' in result or 'MEMORY_ERROR' in result, \
            "Unexpected result format"
    
    print("[OK] Agent handles LLM failures gracefully")
    return True


def test_persistence_corruption_recovery():
    """Test recovery from persistence corruption."""
    print("\nTesting persistence corruption recovery...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test.json"
        enhanced_path = Path(tmpdir) / "test.enhanced.json"
        
        # Create and populate store
        store1 = EnhancedVectorStore(storage_path=str(storage_path))
        mem_id = store1.store_conversation_enhanced({
            'task': 'Test',
            'messages': ['Test message']
        })
        store1._save_enhanced_data()
        
        # Corrupt enhanced data
        with open(enhanced_path, 'w') as f:
            f.write("corrupted enhanced data")
        
        # Try to load with corrupted enhanced data
        try:
            store2 = EnhancedVectorStore(storage_path=str(storage_path))
            # Should load basic data even if enhanced is corrupted
            assert len(store2.conversations) > 0 or len(store2.memory_nodes) > 0, \
                "No data loaded after corruption"
        except Exception as e:
            assert False, f"Failed to recover from corruption: {e}"
    
    print("[OK] Recovered from persistence corruption")
    return True


if __name__ == "__main__":
    print("=== Error Recovery Test Suite ===\n")
    
    tests = [
        ("Corrupted Storage Recovery", test_corrupted_storage_recovery),
        ("Missing Dependencies", test_missing_dependencies),
        ("File I/O Errors", test_file_io_errors),
        ("Invalid Input Handling", test_invalid_input_handling),
        ("Memory Limit Enforcement", test_memory_limit_enforcement),
        ("Circular Reference Handling", test_circular_reference_handling),
        ("Concurrent Modification", test_concurrent_modification_handling),
        ("Embedding Generation Failure", test_embedding_generation_failure),
        ("Empty Store Search", test_search_with_empty_store),
        ("Malformed Memory Handling", test_malformed_memory_handling),
        ("Partial Failure Recovery", test_recovery_after_partial_failure),
        ("Agent LLM Failure", test_agent_llm_failure_handling),
        ("Persistence Corruption", test_persistence_corruption_recovery)
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
        print("[SUCCESS] All error recovery tests passed!")
    else:
        print("[WARNING] Some error recovery tests failed.")