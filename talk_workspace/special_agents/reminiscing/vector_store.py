#!/usr/bin/env python3
"""
ConversationVectorStore - Vector storage and retrieval for conversation memories.

This module provides the storage layer for the reminiscing system, handling:
- Conversation history storage with vector embeddings
- Code context storage and retrieval
- Multi-dimensional search capabilities
- Memory persistence and loading

Currently implements an in-memory store with optional persistence to JSON.
Can be extended to use external vector databases like Qdrant, Pinecone, etc.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

log = logging.getLogger(__name__)

class ConversationVectorStore:
    """
    Vector storage for conversation and code memories.
    
    This class provides the storage and retrieval functionality for the
    reminiscing system. It uses vector embeddings for semantic search
    and maintains metadata for contextual filtering.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        # In-memory storage
        self.conversations = []  # List of conversation entries
        self.code_contexts = []  # List of code context entries
        self.embeddings = {}     # memory_id -> embedding vector
        self.metadata = {}       # memory_id -> metadata dict
        
        # Configuration
        self.max_memories = 10000  # Limit memory usage
        self.embedding_dim = 384   # Dimension for sentence transformers
        
        # Load existing data if storage path exists
        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()
    
    def store_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """
        Store a conversation in the vector store.
        
        Args:
            conversation_data: Dictionary containing conversation details
            
        Returns:
            Unique memory ID for the stored conversation
        """
        try:
            # Generate unique ID
            memory_id = self._generate_memory_id(conversation_data)
            
            # Extract text content for embedding
            text_content = self._extract_text_content(conversation_data)
            
            # Generate embedding (placeholder for now)
            embedding = self._generate_embedding(text_content)
            
            # Store conversation
            conversation_entry = {
                'memory_id': memory_id,
                'timestamp': datetime.now().isoformat(),
                'content': text_content,
                'original_data': conversation_data,
                'memory_type': 'conversation'
            }
            
            self.conversations.append(conversation_entry)
            self.embeddings[memory_id] = embedding
            self.metadata[memory_id] = self._extract_metadata(conversation_data, 'conversation')
            
            # Cleanup if we exceed max memories
            self._cleanup_old_memories()
            
            # Persist if storage path is configured
            if self.storage_path:
                self._save_to_disk()
            
            log.info(f"Stored conversation memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            log.error(f"Error storing conversation: {e}")
            raise
    
    def store_code_context(self, code_data: Dict[str, Any]) -> str:
        """
        Store code context in the vector store.
        
        Args:
            code_data: Dictionary containing code context details
            
        Returns:
            Unique memory ID for the stored code context
        """
        try:
            # Generate unique ID
            memory_id = self._generate_memory_id(code_data)
            
            # Extract text content for embedding
            text_content = self._extract_code_content(code_data)
            
            # Generate embedding
            embedding = self._generate_embedding(text_content)
            
            # Store code context
            code_entry = {
                'memory_id': memory_id,
                'timestamp': datetime.now().isoformat(),
                'content': text_content,
                'original_data': code_data,
                'memory_type': 'code'
            }
            
            self.code_contexts.append(code_entry)
            self.embeddings[memory_id] = embedding
            self.metadata[memory_id] = self._extract_metadata(code_data, 'code')
            
            # Cleanup if we exceed max memories
            self._cleanup_old_memories()
            
            # Persist if storage path is configured
            if self.storage_path:
                self._save_to_disk()
            
            log.info(f"Stored code context memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            log.error(f"Error storing code context: {e}")
            raise
    
    def search_memories(self, query: str, memory_type: Optional[str] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories using vector similarity.
        
        Args:
            query: Search query string
            memory_type: Optional filter by memory type ('conversation', 'code')
            limit: Maximum number of results to return
            
        Returns:
            List of memory entries with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Get all memories to search
            all_memories = []
            if not memory_type or memory_type == 'conversation':
                all_memories.extend(self.conversations)
            if not memory_type or memory_type == 'code':
                all_memories.extend(self.code_contexts)
            
            # Calculate similarities
            results = []
            for memory in all_memories:
                memory_id = memory['memory_id']
                if memory_id in self.embeddings:
                    similarity = self._calculate_similarity(
                        query_embedding, 
                        self.embeddings[memory_id]
                    )
                    
                    result = memory.copy()
                    result['similarity_score'] = similarity
                    result['metadata'] = self.metadata.get(memory_id, {})
                    results.append(result)
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            log.error(f"Error searching memories: {e}")
            return []
    
    def get_recent_memories(self, hours: int = 24, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent memories within the specified time window.
        
        Args:
            hours: Number of hours to look back
            memory_type: Optional filter by memory type
            
        Returns:
            List of recent memory entries
        """
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            all_memories = []
            if not memory_type or memory_type == 'conversation':
                all_memories.extend(self.conversations)
            if not memory_type or memory_type == 'code':
                all_memories.extend(self.code_contexts)
            
            recent_memories = []
            for memory in all_memories:
                memory_timestamp = datetime.fromisoformat(memory['timestamp']).timestamp()
                if memory_timestamp >= cutoff_time:
                    memory_with_metadata = memory.copy()
                    memory_with_metadata['metadata'] = self.metadata.get(memory['memory_id'], {})
                    recent_memories.append(memory_with_metadata)
            
            # Sort by timestamp (most recent first)
            recent_memories.sort(key=lambda x: x['timestamp'], reverse=True)
            return recent_memories
            
        except Exception as e:
            log.error(f"Error getting recent memories: {e}")
            return []
    
    def _generate_memory_id(self, data: Dict[str, Any]) -> str:
        """Generate a unique ID for a memory entry."""
        # Create hash based on content and timestamp
        content_str = json.dumps(data, sort_keys=True, default=str)
        timestamp_str = str(datetime.now().timestamp())
        combined = f"{content_str}_{timestamp_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _extract_text_content(self, conversation_data: Dict[str, Any]) -> str:
        """Extract text content from conversation data."""
        text_parts = []
        
        # Extract task/prompt
        if 'task' in conversation_data:
            text_parts.append(f"Task: {conversation_data['task']}")
        
        # Extract messages
        if 'messages' in conversation_data:
            for msg in conversation_data['messages']:
                if isinstance(msg, dict) and 'content' in msg:
                    text_parts.append(msg['content'])
                elif isinstance(msg, str):
                    text_parts.append(msg)
        
        # Extract blackboard entries
        if 'blackboard_entries' in conversation_data:
            for entry in conversation_data['blackboard_entries']:
                if isinstance(entry, dict) and 'content' in entry:
                    text_parts.append(entry['content'])
        
        # Fallback: stringify the entire data
        if not text_parts:
            text_parts.append(str(conversation_data))
        
        return " ".join(text_parts)
    
    def _extract_code_content(self, code_data: Dict[str, Any]) -> str:
        """Extract text content from code data."""
        text_parts = []
        
        # Extract code content
        if 'code' in code_data:
            text_parts.append(code_data['code'])
        
        # Extract file paths
        if 'file_path' in code_data:
            text_parts.append(f"File: {code_data['file_path']}")
        
        # Extract comments/descriptions
        if 'description' in code_data:
            text_parts.append(code_data['description'])
        
        # Extract function/class names
        if 'functions' in code_data:
            text_parts.extend(code_data['functions'])
        if 'classes' in code_data:
            text_parts.extend(code_data['classes'])
        
        # Fallback
        if not text_parts:
            text_parts.append(str(code_data))
        
        return " ".join(text_parts)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Currently uses a simple hash-based approach.
        In production, would use proper embedding models like:
        - sentence-transformers
        - OpenAI embeddings
        - Cohere embeddings
        """
        if not text:
            return [0.0] * self.embedding_dim
        
        # Simple hash-based embedding for prototype
        # This is NOT a proper embedding, just for testing
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to normalized vector
        embedding = []
        for i in range(min(self.embedding_dim, len(hash_bytes))):
            embedding.append((hash_bytes[i] / 255.0) * 2.0 - 1.0)  # Normalize to [-1, 1]
        
        # Pad with zeros if needed
        while len(embedding) < self.embedding_dim:
            embedding.append(0.0)
        
        return embedding[:self.embedding_dim]
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not NUMPY_AVAILABLE:
            # Simple dot product fallback
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        else:
            # Use numpy for better performance
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(cosine_sim)
    
    def _extract_metadata(self, data: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        """Extract metadata from memory data."""
        metadata = {
            'memory_type': memory_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract relevant metadata based on type
        if memory_type == 'conversation':
            metadata.update({
                'has_task': 'task' in data,
                'message_count': len(data.get('messages', [])),
                'has_errors': 'error' in str(data).lower(),
                'session_id': data.get('session_id', 'unknown')
            })
        elif memory_type == 'code':
            metadata.update({
                'file_path': data.get('file_path', ''),
                'language': data.get('language', 'unknown'),
                'has_functions': bool(data.get('functions', [])),
                'has_classes': bool(data.get('classes', [])),
                'code_length': len(data.get('code', ''))
            })
        
        return metadata
    
    def _cleanup_old_memories(self):
        """Remove old memories if we exceed the maximum."""
        total_memories = len(self.conversations) + len(self.code_contexts)
        
        if total_memories <= self.max_memories:
            return
        
        # Sort all memories by timestamp and remove oldest
        all_memories = []
        all_memories.extend([(m, 'conversation') for m in self.conversations])
        all_memories.extend([(m, 'code') for m in self.code_contexts])
        
        all_memories.sort(key=lambda x: x[0]['timestamp'])
        
        # Remove oldest memories
        to_remove = total_memories - self.max_memories
        for i in range(to_remove):
            memory, memory_type = all_memories[i]
            memory_id = memory['memory_id']
            
            # Remove from storage
            if memory_type == 'conversation':
                self.conversations = [m for m in self.conversations if m['memory_id'] != memory_id]
            else:
                self.code_contexts = [m for m in self.code_contexts if m['memory_id'] != memory_id]
            
            # Remove embeddings and metadata
            self.embeddings.pop(memory_id, None)
            self.metadata.pop(memory_id, None)
        
        log.info(f"Cleaned up {to_remove} old memories")
    
    def _save_to_disk(self):
        """Save memories to disk for persistence."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'conversations': self.conversations,
                'code_contexts': self.code_contexts,
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            log.debug(f"Saved memories to {self.storage_path}")
            
        except Exception as e:
            log.error(f"Error saving memories to disk: {e}")
    
    def _load_from_disk(self):
        """Load memories from disk."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversations = data.get('conversations', [])
            self.code_contexts = data.get('code_contexts', [])
            self.embeddings = data.get('embeddings', {})
            self.metadata = data.get('metadata', {})
            
            total_loaded = len(self.conversations) + len(self.code_contexts)
            log.info(f"Loaded {total_loaded} memories from {self.storage_path}")
            
        except Exception as e:
            log.error(f"Error loading memories from disk: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            'total_conversations': len(self.conversations),
            'total_code_contexts': len(self.code_contexts),
            'total_embeddings': len(self.embeddings),
            'total_metadata': len(self.metadata),
            'storage_path': str(self.storage_path) if self.storage_path else None,
            'max_memories': self.max_memories,
            'embedding_dim': self.embedding_dim
        }