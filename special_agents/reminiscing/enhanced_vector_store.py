#!/usr/bin/env python3
"""
EnhancedVectorStore - Advanced vector storage with graph relationships and better search.

This module extends the basic ConversationVectorStore with:
- Graph-based relationship tracking between memories
- Multiple search strategies (semantic, temporal, structural)
- Concept indexing for faster retrieval
- Memory consolidation and compression
- Advanced filtering and ranking
"""

from __future__ import annotations

import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .vector_store import ConversationVectorStore

log = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """Represents a memory node in the graph."""
    memory_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    memory_type: str
    concepts: Set[str] = field(default_factory=set)
    relationships: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance_score: float = 0.5


class EnhancedVectorStore(ConversationVectorStore):
    """
    Enhanced vector storage with graph relationships and advanced search.
    
    This class extends the basic vector store with:
    - Memory graph for relationship tracking
    - Concept indexing for fast retrieval
    - Multiple search strategies
    - Memory importance scoring
    - Temporal decay and reinforcement
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize enhanced vector store."""
        super().__init__(storage_path)
        
        # Graph structures
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.memory_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Indexing structures
        self.concept_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[Tuple[datetime, str]] = []
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Advanced configuration
        self.decay_factor = 0.95  # Daily decay for memory importance
        self.reinforcement_factor = 1.2  # Boost for accessed memories
        self.relationship_weight = 0.3  # Weight for graph relationships
        self.concept_weight = 0.4  # Weight for concept matching
        
        # Memory consolidation
        self.consolidation_threshold = 100  # Consolidate after N memories
        self.similarity_threshold = 0.85  # Threshold for merging similar memories
        
        # Load enhanced data if exists
        if self.storage_path:
            self._load_enhanced_data()
    
    def store_conversation_enhanced(self, conversation_data: Dict[str, Any]) -> str:
        """
        Store conversation with enhanced metadata and graph relationships.
        
        Args:
            conversation_data: Conversation data with optional relationship info
            
        Returns:
            Memory ID of stored conversation
        """
        # Store using parent method
        memory_id = self.store_conversation(conversation_data)
        
        # Create memory node
        content = self._extract_text_content(conversation_data)
        concepts = self._extract_concepts(content)
        
        node = MemoryNode(
            memory_id=memory_id,
            content=content,
            embedding=self.embeddings[memory_id],
            metadata=self.metadata[memory_id],
            timestamp=datetime.now(),
            memory_type='conversation',
            concepts=concepts
        )
        
        self.memory_nodes[memory_id] = node
        
        # Index concepts
        for concept in concepts:
            self.concept_index[concept].add(memory_id)
        
        # Index by type
        self.type_index['conversation'].add(memory_id)
        
        # Add to temporal index
        self.temporal_index.append((node.timestamp, memory_id))
        self.temporal_index.sort(reverse=True)  # Keep sorted by recency
        
        # Find and create relationships
        self._create_automatic_relationships(memory_id, content, concepts)
        
        # Check for consolidation
        if len(self.memory_nodes) % self.consolidation_threshold == 0:
            self._consolidate_memories()
        
        log.info(f"Stored enhanced conversation: {memory_id} with {len(concepts)} concepts")
        return memory_id
    
    def search_enhanced(self, query: str, strategy: str = 'hybrid', 
                       limit: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Enhanced search with multiple strategies.
        
        Args:
            query: Search query
            strategy: Search strategy (semantic, graph, concept, temporal, hybrid)
            limit: Maximum results
            filters: Optional filters (type, time_range, concepts)
            
        Returns:
            List of memory results with scores
        """
        results = []
        
        if strategy == 'semantic':
            results = self._semantic_search_enhanced(query, limit * 2)
        elif strategy == 'graph':
            results = self._graph_search(query, limit * 2)
        elif strategy == 'concept':
            results = self._concept_search(query, limit * 2)
        elif strategy == 'temporal':
            results = self._temporal_search_enhanced(query, limit * 2)
        elif strategy == 'hybrid':
            # Combine multiple strategies
            semantic_results = self._semantic_search_enhanced(query, limit)
            graph_results = self._graph_search(query, limit // 2)
            concept_results = self._concept_search(query, limit // 2)
            
            # Merge and deduplicate
            results = self._merge_results([semantic_results, graph_results, concept_results])
        else:
            results = self._semantic_search_enhanced(query, limit)
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Update access counts and importance
        for result in results[:limit]:
            memory_id = result['memory_id']
            if memory_id in self.memory_nodes:
                self._update_memory_access(memory_id)
        
        # Sort by final score and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _semantic_search_enhanced(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Enhanced semantic search with importance scoring."""
        # Get base semantic results
        base_results = self.search_memories(query, limit=limit * 2)
        
        enhanced_results = []
        for result in base_results:
            memory_id = result['memory_id']
            
            # Get memory node for enhanced scoring
            node = self.memory_nodes.get(memory_id)
            if not node:
                # Fallback for memories without nodes
                enhanced_results.append({
                    'memory_id': memory_id,
                    'content': result.get('content', ''),
                    'score': result.get('similarity_score', 0),
                    'strategy': 'semantic',
                    'metadata': result.get('metadata', {})
                })
                continue
            
            # Calculate enhanced score
            semantic_score = result.get('similarity_score', 0)
            importance_score = node.importance_score
            recency_score = self._calculate_recency_score(node.timestamp)
            
            # Weighted combination
            final_score = (
                semantic_score * 0.5 +
                importance_score * 0.3 +
                recency_score * 0.2
            )
            
            enhanced_results.append({
                'memory_id': memory_id,
                'content': node.content,
                'score': final_score,
                'semantic_score': semantic_score,
                'importance_score': importance_score,
                'recency_score': recency_score,
                'strategy': 'semantic',
                'concepts': list(node.concepts),
                'metadata': self.metadata.get(memory_id, {})
            })
        
        return enhanced_results
    
    def _graph_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Graph-based search using spreading activation.
        
        Starts from seed nodes and traverses relationships.
        """
        # Get seed nodes from semantic search
        seed_results = self._semantic_search_enhanced(query, 5)
        
        results = []
        visited = set()
        activation_queue = []
        
        # Initialize activation for seeds
        for seed in seed_results:
            memory_id = seed['memory_id']
            activation = seed['score']
            activation_queue.append((activation, memory_id, 0))  # (activation, id, depth)
        
        # Spread activation through graph
        while activation_queue and len(results) < limit:
            activation_queue.sort(reverse=True)  # Sort by activation
            current_activation, current_id, depth = activation_queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Add to results if activation is significant
            if current_activation > 0.3:
                node = self.memory_nodes.get(current_id)
                if node:
                    results.append({
                        'memory_id': current_id,
                        'content': node.content,
                        'score': current_activation,
                        'graph_depth': depth,
                        'strategy': 'graph',
                        'concepts': list(node.concepts),
                        'metadata': self.metadata.get(current_id, {})
                    })
            
            # Spread to neighbors if not too deep
            if depth < 3:
                neighbors = self.memory_graph.get(current_id, set())
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        # Decay activation
                        neighbor_activation = current_activation * 0.7
                        activation_queue.append((neighbor_activation, neighbor_id, depth + 1))
        
        return results
    
    def _concept_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search based on concept matching."""
        # Extract concepts from query
        query_concepts = self._extract_concepts(query)
        
        # Find memories with matching concepts
        memory_scores = Counter()
        
        for concept in query_concepts:
            memory_ids = self.concept_index.get(concept, set())
            for memory_id in memory_ids:
                memory_scores[memory_id] += 1
        
        # Convert to results
        results = []
        for memory_id, concept_count in memory_scores.most_common(limit):
            node = self.memory_nodes.get(memory_id)
            if not node:
                continue
            
            # Calculate concept overlap score
            overlap = len(query_concepts.intersection(node.concepts))
            overlap_score = overlap / max(len(query_concepts), 1)
            
            results.append({
                'memory_id': memory_id,
                'content': node.content,
                'score': overlap_score,
                'concept_matches': concept_count,
                'matched_concepts': list(query_concepts.intersection(node.concepts)),
                'strategy': 'concept',
                'metadata': self.metadata.get(memory_id, {})
            })
        
        return results
    
    def _temporal_search_enhanced(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Enhanced temporal search with decay."""
        results = []
        
        # Get recent memories from temporal index
        now = datetime.now()
        
        for timestamp, memory_id in self.temporal_index[:limit * 2]:
            node = self.memory_nodes.get(memory_id)
            if not node:
                continue
            
            # Calculate temporal score
            time_diff = now - timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            # Exponential decay over time
            temporal_score = np.exp(-hours_ago / 24) if NUMPY_AVAILABLE else max(0, 1 - hours_ago / 24)
            
            # Combine with basic semantic similarity
            query_embedding = self._generate_embedding(query)
            similarity = self._calculate_similarity(query_embedding, node.embedding)
            
            final_score = temporal_score * 0.7 + similarity * 0.3
            
            results.append({
                'memory_id': memory_id,
                'content': node.content,
                'score': final_score,
                'temporal_score': temporal_score,
                'hours_ago': hours_ago,
                'strategy': 'temporal',
                'metadata': self.metadata.get(memory_id, {})
            })
        
        return results
    
    def _merge_results(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Merge multiple result sets, combining scores for duplicates."""
        merged = {}
        
        for results in result_sets:
            for result in results:
                memory_id = result['memory_id']
                
                if memory_id in merged:
                    # Combine scores
                    merged[memory_id]['score'] = max(
                        merged[memory_id]['score'],
                        result['score']
                    )
                    # Merge strategies
                    if 'strategies' not in merged[memory_id]:
                        merged[memory_id]['strategies'] = [merged[memory_id].get('strategy', 'unknown')]
                    merged[memory_id]['strategies'].append(result.get('strategy', 'unknown'))
                else:
                    merged[memory_id] = result
        
        return list(merged.values())
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results."""
        filtered = results
        
        # Type filter
        if 'type' in filters:
            required_type = filters['type']
            filtered = [r for r in filtered if r.get('metadata', {}).get('memory_type') == required_type]
        
        # Time range filter
        if 'time_range' in filters:
            hours = filters['time_range']
            cutoff = datetime.now() - timedelta(hours=hours)
            filtered = [
                r for r in filtered
                if self.memory_nodes.get(r['memory_id'], MemoryNode(
                    memory_id='', content='', embedding=[], metadata={},
                    timestamp=datetime.min, memory_type=''
                )).timestamp >= cutoff
            ]
        
        # Concept filter
        if 'concepts' in filters:
            required_concepts = set(filters['concepts'])
            filtered = [
                r for r in filtered
                if required_concepts.intersection(r.get('concepts', []))
            ]
        
        return filtered
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text."""
        concepts = set()
        
        # Technical terms
        tech_patterns = [
            r'\b(api|database|server|client|frontend|backend)\b',
            r'\b(function|method|class|object|interface)\b',
            r'\b(authentication|authorization|security)\b',
            r'\b(algorithm|structure|pattern|design)\b',
            r'\b(error|exception|bug|debug|fix)\b'
        ]
        
        text_lower = text.lower()
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.update(matches)
        
        # Extract CamelCase and snake_case identifiers
        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', text)
        snake_case = re.findall(r'[a-z]+(?:_[a-z]+)+', text)
        
        concepts.update(word.lower() for word in camel_case[:10])
        concepts.update(snake_case[:10])
        
        # Extract quoted strings (might be important)
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.update(q.lower() for q in quoted[:5] if len(q) < 30)
        
        return concepts
    
    def _create_automatic_relationships(self, memory_id: str, content: str, concepts: Set[str]):
        """Automatically create relationships based on similarity and concepts."""
        # Find related memories by concept overlap
        related_memories = set()
        
        for concept in concepts:
            concept_memories = self.concept_index.get(concept, set())
            related_memories.update(concept_memories)
        
        # Remove self
        related_memories.discard(memory_id)
        
        # Score and link top related memories
        if related_memories:
            scores = []
            
            for related_id in related_memories:
                related_node = self.memory_nodes.get(related_id)
                if not related_node:
                    continue
                
                # Calculate relationship strength
                concept_overlap = len(concepts.intersection(related_node.concepts))
                similarity = self._calculate_similarity(
                    self.embeddings[memory_id],
                    self.embeddings[related_id]
                )
                
                score = concept_overlap * 0.6 + similarity * 0.4
                scores.append((score, related_id))
            
            # Create relationships with top matches
            scores.sort(reverse=True)
            for score, related_id in scores[:5]:  # Top 5 relationships
                if score > 0.5:  # Threshold for relationship
                    self.add_relationship(memory_id, related_id, score)
    
    def add_relationship(self, memory_id1: str, memory_id2: str, strength: float = 1.0):
        """
        Add a bidirectional relationship between memories.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            strength: Relationship strength (0-1)
        """
        self.memory_graph[memory_id1].add(memory_id2)
        self.memory_graph[memory_id2].add(memory_id1)
        
        # Update nodes if they exist
        if memory_id1 in self.memory_nodes:
            self.memory_nodes[memory_id1].relationships.add(memory_id2)
        if memory_id2 in self.memory_nodes:
            self.memory_nodes[memory_id2].relationships.add(memory_id1)
        
        log.debug(f"Added relationship: {memory_id1} <-> {memory_id2} (strength: {strength:.2f})")
    
    def _update_memory_access(self, memory_id: str):
        """Update access count and importance for a memory."""
        if memory_id in self.memory_nodes:
            node = self.memory_nodes[memory_id]
            node.access_count += 1
            node.last_accessed = datetime.now()
            
            # Boost importance based on access
            node.importance_score = min(
                1.0,
                node.importance_score * self.reinforcement_factor
            )
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score for a timestamp."""
        time_diff = datetime.now() - timestamp
        hours = time_diff.total_seconds() / 3600
        
        # Exponential decay
        if NUMPY_AVAILABLE:
            return np.exp(-hours / 168)  # Decay over a week
        else:
            return max(0, 1 - hours / 168)
    
    def _consolidate_memories(self):
        """
        Consolidate similar memories to reduce redundancy.
        
        This merges very similar memories and updates relationships.
        """
        log.info("Starting memory consolidation...")
        
        # Find similar memory pairs
        memory_ids = list(self.memory_nodes.keys())
        similar_pairs = []
        
        for i, id1 in enumerate(memory_ids):
            for id2 in memory_ids[i+1:]:
                if id1 in self.embeddings and id2 in self.embeddings:
                    similarity = self._calculate_similarity(
                        self.embeddings[id1],
                        self.embeddings[id2]
                    )
                    
                    if similarity > self.similarity_threshold:
                        similar_pairs.append((similarity, id1, id2))
        
        # Sort by similarity
        similar_pairs.sort(reverse=True)
        
        # Merge similar memories
        merged_count = 0
        merged_ids = set()
        
        for similarity, id1, id2 in similar_pairs:
            if id1 in merged_ids or id2 in merged_ids:
                continue
            
            # Keep the more important/accessed memory
            node1 = self.memory_nodes[id1]
            node2 = self.memory_nodes[id2]
            
            if node1.importance_score >= node2.importance_score:
                keep_id, merge_id = id1, id2
            else:
                keep_id, merge_id = id2, id1
            
            # Merge concepts
            self.memory_nodes[keep_id].concepts.update(node2.concepts)
            
            # Merge relationships
            self.memory_graph[keep_id].update(self.memory_graph[merge_id])
            self.memory_graph[keep_id].discard(keep_id)  # Remove self-loop
            
            # Update relationship targets
            for related_id in self.memory_graph[merge_id]:
                if related_id != keep_id:
                    self.memory_graph[related_id].discard(merge_id)
                    self.memory_graph[related_id].add(keep_id)
            
            # Remove merged memory
            self._remove_memory(merge_id)
            merged_ids.add(merge_id)
            merged_count += 1
        
        log.info(f"Consolidation complete: merged {merged_count} similar memories")
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory from all indexes."""
        # Remove from nodes
        if memory_id in self.memory_nodes:
            node = self.memory_nodes[memory_id]
            
            # Remove from concept index
            for concept in node.concepts:
                self.concept_index[concept].discard(memory_id)
            
            # Remove from type index
            self.type_index[node.memory_type].discard(memory_id)
            
            del self.memory_nodes[memory_id]
        
        # Remove from graph
        if memory_id in self.memory_graph:
            del self.memory_graph[memory_id]
        
        # Remove from base store
        if memory_id in self.embeddings:
            del self.embeddings[memory_id]
        if memory_id in self.metadata:
            del self.metadata[memory_id]
        
        # Remove from conversation/code lists
        self.conversations = [c for c in self.conversations if c.get('memory_id') != memory_id]
        self.code_contexts = [c for c in self.code_contexts if c.get('memory_id') != memory_id]
    
    def apply_decay(self):
        """Apply temporal decay to memory importance scores."""
        for node in self.memory_nodes.values():
            # Decay importance over time
            days_old = (datetime.now() - node.timestamp).days
            decay = self.decay_factor ** days_old
            
            # Apply decay but maintain minimum importance
            node.importance_score = max(
                0.1,  # Minimum importance
                node.importance_score * decay
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the memory store."""
        stats = super().get_stats()
        
        # Add enhanced statistics
        stats.update({
            'total_nodes': len(self.memory_nodes),
            'total_relationships': sum(len(rels) for rels in self.memory_graph.values()) // 2,
            'total_concepts': len(self.concept_index),
            'avg_concepts_per_memory': (
                sum(len(node.concepts) for node in self.memory_nodes.values()) / 
                max(len(self.memory_nodes), 1)
            ),
            'avg_relationships_per_memory': (
                sum(len(rels) for rels in self.memory_graph.values()) / 
                max(len(self.memory_graph), 1)
            ),
            'most_accessed_memories': self._get_most_accessed_memories(5),
            'most_connected_memories': self._get_most_connected_memories(5),
            'top_concepts': self._get_top_concepts(10)
        })
        
        return stats
    
    def _get_most_accessed_memories(self, limit: int) -> List[Dict]:
        """Get the most frequently accessed memories."""
        sorted_nodes = sorted(
            self.memory_nodes.values(),
            key=lambda n: n.access_count,
            reverse=True
        )
        
        return [
            {
                'memory_id': node.memory_id,
                'access_count': node.access_count,
                'content_preview': node.content[:100]
            }
            for node in sorted_nodes[:limit]
        ]
    
    def _get_most_connected_memories(self, limit: int) -> List[Dict]:
        """Get memories with the most relationships."""
        connection_counts = [
            (memory_id, len(connections))
            for memory_id, connections in self.memory_graph.items()
        ]
        
        connection_counts.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'memory_id': memory_id,
                'connection_count': count,
                'content_preview': self.memory_nodes[memory_id].content[:100]
                if memory_id in self.memory_nodes else 'Unknown'
            }
            for memory_id, count in connection_counts[:limit]
        ]
    
    def _get_top_concepts(self, limit: int) -> List[Tuple[str, int]]:
        """Get the most common concepts."""
        concept_counts = [
            (concept, len(memory_ids))
            for concept, memory_ids in self.concept_index.items()
        ]
        
        concept_counts.sort(key=lambda x: x[1], reverse=True)
        return concept_counts[:limit]
    
    def _save_enhanced_data(self):
        """Save enhanced data structures to disk."""
        if not self.storage_path:
            return
        
        enhanced_path = Path(self.storage_path).with_suffix('.enhanced.json')
        
        try:
            # Prepare data for serialization
            nodes_data = {}
            for memory_id, node in self.memory_nodes.items():
                nodes_data[memory_id] = {
                    'content': node.content,
                    'metadata': node.metadata,
                    'timestamp': node.timestamp.isoformat(),
                    'memory_type': node.memory_type,
                    'concepts': list(node.concepts),
                    'relationships': list(node.relationships),
                    'access_count': node.access_count,
                    'last_accessed': node.last_accessed.isoformat() if node.last_accessed else None,
                    'importance_score': node.importance_score
                }
            
            enhanced_data = {
                'nodes': nodes_data,
                'graph': {k: list(v) for k, v in self.memory_graph.items()},
                'concept_index': {k: list(v) for k, v in self.concept_index.items()},
                'type_index': {k: list(v) for k, v in self.type_index.items()}
            }
            
            with open(enhanced_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, default=str)
            
            log.debug(f"Saved enhanced data to {enhanced_path}")
            
        except Exception as e:
            log.error(f"Error saving enhanced data: {e}")
    
    def _load_enhanced_data(self):
        """Load enhanced data structures from disk."""
        enhanced_path = Path(self.storage_path).with_suffix('.enhanced.json')
        
        if not enhanced_path.exists():
            return
        
        try:
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)
            
            # Reconstruct memory nodes
            for memory_id, node_data in enhanced_data.get('nodes', {}).items():
                node = MemoryNode(
                    memory_id=memory_id,
                    content=node_data['content'],
                    embedding=self.embeddings.get(memory_id, []),
                    metadata=node_data['metadata'],
                    timestamp=datetime.fromisoformat(node_data['timestamp']),
                    memory_type=node_data['memory_type'],
                    concepts=set(node_data['concepts']),
                    relationships=set(node_data['relationships']),
                    access_count=node_data['access_count'],
                    last_accessed=datetime.fromisoformat(node_data['last_accessed']) 
                                 if node_data['last_accessed'] else None,
                    importance_score=node_data['importance_score']
                )
                self.memory_nodes[memory_id] = node
            
            # Reconstruct indexes
            self.memory_graph = defaultdict(set)
            for k, v in enhanced_data.get('graph', {}).items():
                self.memory_graph[k] = set(v)
            
            self.concept_index = defaultdict(set)
            for k, v in enhanced_data.get('concept_index', {}).items():
                self.concept_index[k] = set(v)
            
            self.type_index = defaultdict(set)
            for k, v in enhanced_data.get('type_index', {}).items():
                self.type_index[k] = set(v)
            
            # Rebuild temporal index
            self.temporal_index = [
                (node.timestamp, memory_id)
                for memory_id, node in self.memory_nodes.items()
            ]
            self.temporal_index.sort(reverse=True)
            
            log.info(f"Loaded enhanced data with {len(self.memory_nodes)} nodes")
            
        except Exception as e:
            log.error(f"Error loading enhanced data: {e}")