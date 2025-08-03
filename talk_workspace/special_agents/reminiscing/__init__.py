#!/usr/bin/env python3
"""
Reminiscing Agent Package

This package implements a sophisticated memory system that mimics human memory traces
for code generation agents. It provides contextual memory retrieval through:

- Context categorization for determining memory search strategy
- Multi-dimensional vector search across conversations and code
- Spreading activation networks for associative memory retrieval
- Graph-based relationship modeling for deeper context understanding

The system is designed to provide "memory traces" - relevant past contexts that
"ring a bell" when considering new tasks, similar to human memory recall.
"""

from .reminiscing_agent import ReminiscingAgent
from .context_categorization_agent import ContextCategorizationAgent  
from .memory_trace_agent import MemoryTraceAgent
from .vector_store import ConversationVectorStore

__all__ = [
    'ReminiscingAgent',
    'ContextCategorizationAgent', 
    'MemoryTraceAgent',
    'ConversationVectorStore'
]