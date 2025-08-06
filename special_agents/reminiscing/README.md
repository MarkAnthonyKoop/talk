# ReminiscingAgent - Memory System for Talk Framework

## Overview

The ReminiscingAgent provides human-like contextual memory for the Talk framework, enabling intelligent retrieval of past experiences and knowledge to inform current tasks.

## Architecture

```
ReminiscingAgent (Orchestrator)
├── ContextCategorizationAgent
│   ├── LLM-based categorization
│   └── Pattern-based fallback
├── MemoryTraceAgent
│   ├── Semantic search
│   ├── Graph traversal
│   ├── Error similarity
│   └── Temporal search
├── ConversationVectorStore
│   ├── Basic storage/retrieval
│   └── Vector embeddings
└── EnhancedVectorStore
    ├── Graph relationships
    ├── Concept indexing
    ├── Memory consolidation
    └── Advanced search strategies
```

## Components

### 1. ReminiscingAgent (`reminiscing_agent.py`)
Main orchestrator that coordinates memory retrieval workflow:
- Categorizes incoming contexts
- Selects appropriate search strategy
- Returns formatted memory traces with insights

### 2. ContextCategorizationAgent (`context_categorization_agent.py`)
Analyzes prompts to determine task type and search strategy:
- **Categories**: architectural, debugging, implementation, research, general
- **Strategies**: graph_traversal, error_similarity, code_similarity, semantic_search
- Uses LLM with pattern-based fallback

### 3. MemoryTraceAgent (`memory_trace_agent.py`)
Performs sophisticated memory retrieval:
- **Spreading Activation**: Mimics human associative memory
- **Multiple Strategies**: Adapts search based on context type
- **Concept Extraction**: Identifies key concepts for indexing
- **Graph Traversal**: Follows memory relationships

### 4. ConversationVectorStore (`vector_store.py`)
Basic storage layer with vector embeddings:
- In-memory storage with JSON persistence
- Hash-based embeddings (upgradeable to semantic models)
- Similarity search capabilities
- Metadata tracking

### 5. EnhancedVectorStore (`enhanced_vector_store.py`)
Advanced storage with graph capabilities:
- **Graph Relationships**: Automatic relationship creation
- **Concept Indexing**: Fast concept-based retrieval
- **Memory Consolidation**: Merges similar memories
- **Temporal Features**: Recency scoring and decay
- **Hybrid Search**: Combines multiple strategies

## Usage

### Basic Usage

```python
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

# Create agent
agent = ReminiscingAgent()

# Find relevant memories
result = agent.run("I need to implement user authentication")
print(result)
```

### Enhanced Storage

```python
from special_agents.reminiscing.enhanced_vector_store import EnhancedVectorStore

# Create enhanced store
store = EnhancedVectorStore(storage_path="memories.json")

# Store with automatic relationship creation
memory_id = store.store_conversation_enhanced({
    "task": "Implement OAuth2",
    "messages": ["Using passport.js", "Google and GitHub providers"]
})

# Search with different strategies
results = store.search_enhanced(
    query="authentication issues",
    strategy='hybrid',  # semantic, graph, concept, temporal, hybrid
    limit=5,
    filters={'time_range': 24}  # Last 24 hours
)
```

### Integration with Talk

```python
from talk.talk import TalkOrchestrator
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

# Create Talk with memory
talk = TalkOrchestrator(task="Build web app")
memory = ReminiscingAgent()

# Use memory to inform decisions
context = "How should I handle authentication?"
memory_traces = memory.run(context)

# Memory provides relevant past experiences
print(memory_traces)
```

## Search Strategies

| Strategy | Use Case | Method |
|----------|----------|--------|
| **Semantic** | General similarity | Vector embeddings |
| **Graph** | Related concepts | Spreading activation |
| **Concept** | Specific topics | Concept matching |
| **Temporal** | Recent context | Time-weighted |
| **Error** | Debugging | Pattern matching |
| **Hybrid** | Comprehensive | Multi-strategy |

## Features

### Memory Graph
- Automatic relationship creation based on similarity
- Spreading activation for associative retrieval
- Graph traversal up to N hops

### Concept Indexing
- Automatic concept extraction from text
- Fast concept-based retrieval
- Technical term recognition

### Memory Consolidation
- Merges highly similar memories
- Reduces redundancy
- Preserves relationships

### Temporal Features
- Recency scoring
- Importance decay over time
- Access-based reinforcement

### Advanced Filtering
- Time range filtering
- Concept filtering
- Memory type filtering

## Configuration

### Storage Options
```python
# In-memory only
store = EnhancedVectorStore()

# With persistence
store = EnhancedVectorStore(storage_path="memories.json")
```

### Tuning Parameters
```python
store.decay_factor = 0.95           # Daily importance decay
store.reinforcement_factor = 1.2    # Access boost
store.relationship_weight = 0.3     # Graph weight in scoring
store.similarity_threshold = 0.85   # Consolidation threshold
store.max_memories = 10000         # Memory limit
```

## Testing

Run the comprehensive test suite:

```bash
# Basic tests
python3 tests/special_agents/reminiscing/test_reminiscing_agent.py
python3 tests/special_agents/reminiscing/test_vector_store.py

# Enhanced features
python3 tests/special_agents/reminiscing/test_enhanced_vector_store.py

# Sub-agent tests
python3 tests/special_agents/reminiscing/test_subagents.py
```

## Performance Considerations

### Current Implementation
- **Embeddings**: Hash-based (fast but not semantic)
- **Storage**: In-memory with JSON persistence
- **Scale**: Supports ~10,000 memories efficiently

### Production Upgrades
1. **Better Embeddings**: 
   - Sentence-transformers
   - OpenAI embeddings
   - Custom trained models

2. **Vector Databases**:
   - Qdrant
   - Pinecone
   - Weaviate
   - ChromaDB

3. **Graph Databases**:
   - Neo4j for complex relationships
   - NetworkX for in-memory graphs

## API Reference

### ReminiscingAgent

```python
agent = ReminiscingAgent()
result = agent.run(context: str) -> str
agent.store_conversation(data: dict)
agent.store_code_context(data: dict)
```

### EnhancedVectorStore

```python
store = EnhancedVectorStore(storage_path: str)
memory_id = store.store_conversation_enhanced(data: dict) -> str
results = store.search_enhanced(
    query: str,
    strategy: str,
    limit: int,
    filters: dict
) -> List[dict]
store.add_relationship(id1: str, id2: str, strength: float)
store.apply_decay()
stats = store.get_memory_statistics() -> dict
```

## Future Enhancements

1. **Semantic Embeddings**: Replace hash-based with transformer models
2. **External Storage**: Integration with vector/graph databases
3. **Learning**: Adaptive memory based on usage patterns
4. **Compression**: Smart consolidation strategies
5. **Multi-modal**: Support for code, images, diagrams
6. **Federated Memory**: Shared memory across team/organization

## Dependencies

- `numpy`: Vector operations (optional)
- `scipy`: Scientific computing (optional)
- `langgraph`: Workflow orchestration (optional)
- Base Talk framework

## License

Part of the Talk framework - see main project license.