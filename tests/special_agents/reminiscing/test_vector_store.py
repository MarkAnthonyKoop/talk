#!/usr/bin/env python3
"""
Unit tests for ConversationVectorStore.

Tests storage, retrieval, and search functionality of the vector store.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.vector_store import ConversationVectorStore


def test_conversation_storage():
    """Test storing and retrieving conversations."""
    print("Testing conversation storage...")
    
    store = ConversationVectorStore()
    
    # Store a conversation
    conversation_data = {
        "task": "Implement user authentication",
        "messages": [
            "User wants OAuth2 support",
            "Using passport.js library",
            "Successfully integrated Google and GitHub"
        ],
        "outcome": "Authentication system completed"
    }
    
    memory_id = store.store_conversation(conversation_data)
    assert memory_id is not None, "Failed to generate memory ID"
    print(f"[OK] Stored conversation with ID: {memory_id}")
    
    # Check storage
    assert len(store.conversations) == 1, "Conversation not stored"
    assert memory_id in store.embeddings, "Embedding not generated"
    assert memory_id in store.metadata, "Metadata not stored"
    print("[OK] Conversation storage verified")
    
    return True


def test_code_context_storage():
    """Test storing and retrieving code contexts."""
    print("\nTesting code context storage...")
    
    store = ConversationVectorStore()
    
    # Store code context
    code_data = {
        "file_path": "auth/oauth.js",
        "code": "function authenticateOAuth(provider) { /* OAuth logic */ }",
        "functions": ["authenticateOAuth", "validateToken"],
        "classes": ["OAuthProvider"],
        "description": "OAuth2 authentication implementation"
    }
    
    memory_id = store.store_code_context(code_data)
    assert memory_id is not None, "Failed to generate memory ID"
    print(f"[OK] Stored code context with ID: {memory_id}")
    
    # Check storage
    assert len(store.code_contexts) == 1, "Code context not stored"
    assert memory_id in store.embeddings, "Embedding not generated"
    
    metadata = store.metadata[memory_id]
    assert metadata['memory_type'] == 'code', "Incorrect memory type"
    assert metadata['file_path'] == 'auth/oauth.js', "File path not stored"
    print("[OK] Code context storage verified")
    
    return True


def test_memory_search():
    """Test searching memories by similarity."""
    print("\nTesting memory search...")
    
    store = ConversationVectorStore()
    
    # Add multiple memories
    memories = [
        {"task": "Implement authentication", "messages": ["OAuth2", "passport.js"]},
        {"task": "Fix database bug", "messages": ["timeout error", "connection pool"]},
        {"task": "Design API", "messages": ["REST endpoints", "versioning"]},
        {"task": "Authentication error", "messages": ["OAuth failure", "token expired"]}
    ]
    
    for mem in memories:
        store.store_conversation(mem)
    
    # Search for authentication-related memories
    results = store.search_memories("authentication OAuth", limit=2)
    
    assert len(results) > 0, "No search results returned"
    assert len(results) <= 2, "Limit not respected"
    print(f"[OK] Found {len(results)} relevant memories")
    
    # Check that results have similarity scores
    for result in results:
        assert 'similarity_score' in result, "Missing similarity score"
        assert -1.0 <= result['similarity_score'] <= 1.0, "Invalid similarity score"
    
    print("[OK] Memory search working correctly")
    return True


def test_recent_memories():
    """Test retrieving recent memories."""
    print("\nTesting recent memory retrieval...")
    
    store = ConversationVectorStore()
    
    # Add some memories
    store.store_conversation({"task": "Task 1", "messages": ["Message 1"]})
    store.store_conversation({"task": "Task 2", "messages": ["Message 2"]})
    store.store_code_context({"file_path": "test.py", "code": "print('test')"})
    
    # Get recent memories
    recent = store.get_recent_memories(hours=1)
    
    assert len(recent) == 3, f"Expected 3 recent memories, got {len(recent)}"
    print(f"[OK] Retrieved {len(recent)} recent memories")
    
    # Test filtering by type
    recent_conv = store.get_recent_memories(hours=1, memory_type='conversation')
    assert len(recent_conv) == 2, "Conversation filter not working"
    
    recent_code = store.get_recent_memories(hours=1, memory_type='code')
    assert len(recent_code) == 1, "Code filter not working"
    
    print("[OK] Recent memory filtering works")
    return True


def test_persistence():
    """Test saving and loading memories from disk."""
    print("\nTesting persistence...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_memories.json"
        
        # Create store and add memories
        store1 = ConversationVectorStore(storage_path=str(storage_path))
        store1.store_conversation({"task": "Test task", "messages": ["Test message"]})
        store1.store_code_context({"file_path": "test.py", "code": "test code"})
        
        assert storage_path.exists(), "Memory file not created"
        print("[OK] Memories saved to disk")
        
        # Load in new store
        store2 = ConversationVectorStore(storage_path=str(storage_path))
        
        assert len(store2.conversations) == 1, "Conversations not loaded"
        assert len(store2.code_contexts) == 1, "Code contexts not loaded"
        assert len(store2.embeddings) == 2, "Embeddings not loaded"
        
        print("[OK] Memories loaded from disk")
    
    return True


def test_memory_cleanup():
    """Test automatic cleanup of old memories."""
    print("\nTesting memory cleanup...")
    
    store = ConversationVectorStore()
    store.max_memories = 5  # Set low limit for testing
    
    # Add more memories than the limit
    for i in range(10):
        store.store_conversation({"task": f"Task {i}", "messages": [f"Message {i}"]})
    
    total_memories = len(store.conversations) + len(store.code_contexts)
    assert total_memories <= store.max_memories, f"Cleanup failed: {total_memories} > {store.max_memories}"
    
    print(f"[OK] Memory cleanup working (kept {total_memories}/{store.max_memories} memories)")
    return True


def test_stats():
    """Test statistics reporting."""
    print("\nTesting statistics...")
    
    store = ConversationVectorStore()
    
    # Add some data
    store.store_conversation({"task": "Test", "messages": ["test"]})
    store.store_code_context({"file_path": "test.py", "code": "test"})
    
    stats = store.get_stats()
    
    assert stats['total_conversations'] == 1, "Incorrect conversation count"
    assert stats['total_code_contexts'] == 1, "Incorrect code context count"
    assert stats['total_embeddings'] == 2, "Incorrect embedding count"
    assert stats['embedding_dim'] == 384, "Incorrect embedding dimension"
    
    print("[OK] Statistics reporting correctly")
    print(f"    Stats: {stats}")
    
    return True


if __name__ == "__main__":
    print("=== ConversationVectorStore Test Suite ===\n")
    
    tests = [
        ("Conversation Storage", test_conversation_storage),
        ("Code Context Storage", test_code_context_storage),
        ("Memory Search", test_memory_search),
        ("Recent Memories", test_recent_memories),
        ("Persistence", test_persistence),
        ("Memory Cleanup", test_memory_cleanup),
        ("Statistics", test_stats)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n[SUCCESS] All vector store tests passed!")
    else:
        print("\n[WARNING] Some tests failed. Check output above.")