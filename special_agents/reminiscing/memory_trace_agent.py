#!/usr/bin/env python3
"""
MemoryTraceAgent - Implements spreading activation search for memory retrieval.

This agent performs the core "memory trace" functionality, mimicking human 
associative memory through spreading activation networks. It searches through
conversation history and code context using multiple strategies:

1. Semantic similarity search through vector embeddings
2. Graph traversal for relationship-based retrieval  
3. Spreading activation for associative memory traces
4. Temporal weighting for recency effects
5. Multi-dimensional search combining different aspects

The goal is to find past contexts that "ring a bell" - relevant memories that
might inform the current task, even if not obviously related.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from agent.agent import Agent
from agent.messages import Message, Role

log = logging.getLogger(__name__)

@dataclass
class MemoryTrace:
    """Represents a single memory trace with activation level."""
    content: str
    context_type: str  # 'conversation', 'code', 'error', 'design'
    timestamp: datetime
    activation_level: float
    relevance_score: float
    associations: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'content': self.content,
            'context_type': self.context_type,
            'timestamp': self.timestamp.isoformat(),
            'activation_level': self.activation_level,
            'relevance_score': self.relevance_score,
            'associations': self.associations,
            'metadata': self.metadata
        }

class SpreadingActivationNetwork:
    """
    Implements spreading activation for associative memory retrieval.
    
    This mimics how human memory works - when we encounter a concept,
    activation spreads to related concepts, bringing up associated memories.
    """
    
    def __init__(self, decay_factor: float = 0.8, max_depth: int = 3):
        self.decay_factor = decay_factor
        self.max_depth = max_depth
        self.concept_graph = {}  # concept -> [related_concepts]
        self.concept_memories = {}  # concept -> [memory_traces]
    
    def add_memory_trace(self, trace: MemoryTrace, concepts: List[str]):
        """Add a memory trace and its associated concepts."""
        for concept in concepts:
            if concept not in self.concept_memories:
                self.concept_memories[concept] = []
            self.concept_memories[concept].append(trace)
    
    def add_association(self, concept1: str, concept2: str, strength: float = 1.0):
        """Add an association between two concepts."""
        if concept1 not in self.concept_graph:
            self.concept_graph[concept1] = []
        if concept2 not in self.concept_graph:
            self.concept_graph[concept2] = []
        
        self.concept_graph[concept1].append((concept2, strength))
        self.concept_graph[concept2].append((concept1, strength))
    
    def spread_activation(self, seed_concepts: List[str], initial_activation: float = 1.0) -> Dict[str, float]:
        """
        Spread activation from seed concepts through the network.
        
        Returns a dictionary of concept -> activation_level
        """
        activation = {concept: initial_activation for concept in seed_concepts}
        visited = set()
        
        # Spreading activation with breadth-first search
        queue = [(concept, initial_activation, 0) for concept in seed_concepts]
        
        while queue:
            current_concept, current_activation, depth = queue.pop(0)
            
            if current_concept in visited or depth >= self.max_depth:
                continue
            
            visited.add(current_concept)
            
            # Spread to connected concepts
            if current_concept in self.concept_graph:
                for neighbor, strength in self.concept_graph[current_concept]:
                    new_activation = current_activation * self.decay_factor * strength
                    
                    if neighbor not in activation:
                        activation[neighbor] = 0
                    activation[neighbor] = max(activation[neighbor], new_activation)
                    
                    if new_activation > 0.1:  # Only continue if activation is significant
                        queue.append((neighbor, new_activation, depth + 1))
        
        return activation
    
    def retrieve_memories(self, activation_levels: Dict[str, float], min_activation: float = 0.1) -> List[MemoryTrace]:
        """Retrieve memory traces based on activation levels."""
        retrieved_traces = []
        
        for concept, activation in activation_levels.items():
            if activation >= min_activation and concept in self.concept_memories:
                for trace in self.concept_memories[concept]:
                    # Boost the trace's activation based on spreading activation
                    trace.activation_level = max(trace.activation_level, activation)
                    retrieved_traces.append(trace)
        
        # Remove duplicates and sort by activation level
        unique_traces = {}
        for trace in retrieved_traces:
            trace_id = f"{trace.content[:50]}_{trace.timestamp}"
            if trace_id not in unique_traces or trace.activation_level > unique_traces[trace_id].activation_level:
                unique_traces[trace_id] = trace
        
        return sorted(unique_traces.values(), key=lambda t: t.activation_level, reverse=True)

class MemoryTraceAgent(Agent):
    """
    Performs memory retrieval using spreading activation and multi-dimensional search.
    
    This agent implements the core memory functionality that makes the system
    "remember" relevant past contexts when encountering new tasks.
    """
    
    def __init__(self, **kwargs):
        """Initialize with memory search capabilities."""
        super().__init__(roles=[
            "You are a memory retrieval specialist that finds relevant past contexts.",
            "You use spreading activation to find associatively related memories.",
            "You provide memory traces that inform better decision making."
        ], **kwargs)
        
        # Initialize spreading activation network
        self.activation_network = SpreadingActivationNetwork()
        
        # Memory storage (will be replaced with proper vector store)
        self.memory_traces = []
        
        # Concept extraction patterns for building associations
        self.concept_patterns = [
            r'\b(class|function|method|variable)\s+(\w+)',
            r'\b(error|exception|bug)\s+(\w+)',
            r'\b(pattern|algorithm|approach)\s+(\w+)',
            r'\b(framework|library|tool)\s+(\w+)',
            r'\b(feature|component|module)\s+(\w+)'
        ]
    
    def run(self, input_text: str) -> str:
        """
        Perform memory search and return relevant traces.
        
        Args:
            input_text: Search context, potentially including category and strategy
            
        Returns:
            JSON-formatted response with memory traces and metadata
        """
        try:
            # Parse input to extract search parameters
            search_params = self._parse_search_input(input_text)
            
            # Extract concepts from the context
            concepts = self._extract_concepts(search_params['context'])
            
            # Perform spreading activation search
            activation_levels = self.activation_network.spread_activation(concepts)
            
            # Retrieve activated memories
            activated_traces = self.activation_network.retrieve_memories(activation_levels)
            
            # Apply additional filtering based on strategy
            filtered_traces = self._apply_strategy_filter(
                activated_traces, 
                search_params['strategy'],
                search_params['category']
            )
            
            # Apply temporal weighting
            weighted_traces = self._apply_temporal_weighting(filtered_traces)
            
            # Limit results and format response
            final_traces = weighted_traces[:10]  # Top 10 traces
            
            return self._format_memory_response(final_traces, search_params)
            
        except Exception as e:
            log.error(f"Error in memory trace search: {e}")
            return json.dumps({
                "traces": [],
                "confidence": 0.0,
                "error": str(e),
                "search_type": "error_fallback"
            })
    
    def _parse_search_input(self, input_text: str) -> Dict[str, str]:
        """Parse the search input to extract parameters."""
        try:
            # Try parsing as JSON first
            if input_text.strip().startswith('{'):
                data = json.loads(input_text)
                return {
                    'context': data.get('context', input_text),
                    'category': data.get('category', 'general'),
                    'strategy': data.get('strategy', 'semantic_search')
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        context = input_text
        category = 'general'
        strategy = 'semantic_search'
        
        # Look for structured text format
        lines = input_text.split('\n')
        for line in lines:
            if line.startswith('Context:'):
                context = line.split(':', 1)[1].strip()
            elif line.startswith('Category:'):
                category = line.split(':', 1)[1].strip()
            elif line.startswith('Strategy:'):
                strategy = line.split(':', 1)[1].strip()
        
        return {
            'context': context,
            'category': category.lower(),
            'strategy': strategy.lower()
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for spreading activation."""
        import re
        
        concepts = []
        text_lower = text.lower()
        
        # Extract programming-related concepts
        programming_concepts = [
            'function', 'class', 'method', 'variable', 'algorithm',
            'pattern', 'framework', 'library', 'api', 'database',
            'error', 'exception', 'bug', 'test', 'debug',
            'architecture', 'design', 'component', 'module',
            'performance', 'security', 'optimization'
        ]
        
        for concept in programming_concepts:
            if concept in text_lower:
                concepts.append(concept)
        
        # Extract specific terms using patterns
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.extend([m.lower() for m in match if m])
                else:
                    concepts.append(match.lower())
        
        # Extract key words (longer than 3 characters, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'has', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        words = re.findall(r'\b\w{4,}\b', text_lower)
        for word in words:
            if word not in common_words and word not in concepts:
                concepts.append(word)
        
        return list(set(concepts))  # Remove duplicates
    
    def _apply_strategy_filter(self, traces: List[MemoryTrace], strategy: str, category: str) -> List[MemoryTrace]:
        """Apply strategy-specific filtering to memory traces."""
        if strategy == 'graph_traversal':
            # For architectural decisions, prioritize design-related traces
            return [t for t in traces if t.context_type in ['design', 'architecture', 'conversation']]
        
        elif strategy == 'code_similarity':
            # For implementation, prioritize code traces
            return [t for t in traces if t.context_type in ['code', 'implementation', 'function']]
        
        elif strategy == 'error_similarity':
            # For debugging, prioritize error traces
            return [t for t in traces if t.context_type in ['error', 'debug', 'exception']]
        
        elif strategy == 'temporal_search':
            # For temporal search, prioritize recent traces
            cutoff = datetime.now() - timedelta(days=7)
            return [t for t in traces if t.timestamp > cutoff]
        
        else:  # semantic_search or default
            # Return all traces for semantic search
            return traces
    
    def _apply_temporal_weighting(self, traces: List[MemoryTrace]) -> List[MemoryTrace]:
        """Apply temporal weighting to boost recent memories."""
        now = datetime.now()
        
        for trace in traces:
            # Calculate time decay (exponential decay over days)
            days_ago = (now - trace.timestamp).days
            temporal_factor = math.exp(-days_ago / 30.0)  # 30-day half-life
            
            # Boost activation level with temporal factor
            trace.activation_level *= (0.5 + 0.5 * temporal_factor)
        
        # Re-sort by activation level
        return sorted(traces, key=lambda t: t.activation_level, reverse=True)
    
    def _format_memory_response(self, traces: List[MemoryTrace], search_params: Dict[str, str]) -> str:
        """Format the memory traces into a structured response."""
        response = {
            "traces": [trace.to_dict() for trace in traces],
            "confidence": self._calculate_confidence(traces),
            "search_params": search_params,
            "search_type": "spreading_activation",
            "total_traces": len(traces),
            "summary": self._generate_summary(traces)
        }
        
        return json.dumps(response, indent=2)
    
    def _calculate_confidence(self, traces: List[MemoryTrace]) -> float:
        """Calculate overall confidence in the memory retrieval."""
        if not traces:
            return 0.0
        
        # Base confidence on number and quality of traces
        activation_sum = sum(trace.activation_level for trace in traces)
        relevance_sum = sum(trace.relevance_score for trace in traces)
        
        # Normalize and combine
        activation_confidence = min(activation_sum / len(traces), 1.0)
        relevance_confidence = min(relevance_sum / len(traces), 1.0)
        
        return (activation_confidence + relevance_confidence) / 2.0
    
    def _generate_summary(self, traces: List[MemoryTrace]) -> str:
        """Generate a summary of the retrieved memory traces."""
        if not traces:
            return "No relevant memory traces found."
        
        context_types = {}
        for trace in traces:
            context_types[trace.context_type] = context_types.get(trace.context_type, 0) + 1
        
        summary_parts = []
        summary_parts.append(f"Found {len(traces)} relevant memory traces:")
        
        for context_type, count in context_types.items():
            summary_parts.append(f"- {count} {context_type} traces")
        
        avg_activation = sum(t.activation_level for t in traces) / len(traces)
        summary_parts.append(f"Average activation level: {avg_activation:.2f}")
        
        return " ".join(summary_parts)
    
    def add_memory_trace(self, content: str, context_type: str, metadata: Optional[Dict] = None):
        """Add a new memory trace to the system."""
        try:
            # Extract concepts for spreading activation
            concepts = self._extract_concepts(content)
            
            # Create memory trace
            trace = MemoryTrace(
                content=content,
                context_type=context_type,
                timestamp=datetime.now(),
                activation_level=0.1,  # Base activation level
                relevance_score=0.5,   # Default relevance
                associations=concepts,
                metadata=metadata or {}
            )
            
            # Add to memory storage
            self.memory_traces.append(trace)
            
            # Add to spreading activation network
            self.activation_network.add_memory_trace(trace, concepts)
            
            # Build associations between concepts
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    self.activation_network.add_association(concept1, concept2, 0.5)
            
            log.info(f"Added memory trace: {context_type} with {len(concepts)} concepts")
            
        except Exception as e:
            log.error(f"Error adding memory trace: {e}")
    
    def populate_sample_memories(self):
        """Populate with sample memories for testing."""
        sample_memories = [
            ("Implemented Flask user authentication system", "code", {"language": "python", "framework": "flask"}),
            ("Debugged database connection timeout error", "error", {"error_type": "timeout", "component": "database"}),
            ("Designed microservice architecture for e-commerce", "design", {"pattern": "microservices", "domain": "ecommerce"}),
            ("Created REST API endpoints for user management", "implementation", {"api_type": "REST", "domain": "users"}),
            ("Fixed memory leak in JavaScript application", "debug", {"language": "javascript", "issue": "memory_leak"}),
            ("Researched best practices for React state management", "research", {"framework": "react", "topic": "state_management"}),
            ("Implemented caching layer with Redis", "implementation", {"technology": "redis", "pattern": "caching"}),
            ("Troubleshot CORS issues in web application", "debug", {"issue": "cors", "context": "web_api"}),
        ]
        
        for content, context_type, metadata in sample_memories:
            self.add_memory_trace(content, context_type, metadata)
        
        log.info(f"Populated {len(sample_memories)} sample memories")