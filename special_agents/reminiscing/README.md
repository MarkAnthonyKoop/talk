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
├── Vector Stores (Choose One)
│   ├── ConversationVectorStore (Basic)
│   │   ├── Hash-based embeddings
│   │   └── Simple similarity search
│   └── EnhancedVectorStoreV2 (Recommended)
│       ├── Real embeddings (sentence-transformers)
│       ├── Code structure extraction (AST)
│       ├── Smart similarity boosting
│       └── Performance metrics
└── Optional: SerenaAgent Integration
    ├── LSP-based code analysis
    ├── 13+ language support
    └── Semantic symbol search
```

## 🆕 Recent Enhancements

### Enhanced Vector Store with Real Embeddings
- **Sentence-Transformers Integration**: Uses `all-MiniLM-L6-v2` for real semantic embeddings
- **Code Structure Extraction**: AST parsing for Python, pattern matching for other languages
- **Smart Search**: Boosts results based on query intent (functions vs classes vs imports)
- **Performance**: 2-3 second response time with dramatically improved quality

### Optional Serena Integration
- **Off by Default**: Enable with `enable_semantic_search=True`
- **Smart Routing**: Automatically uses Serena only for complex code queries
- **Trade-offs**: 5-10 second response time but 90-95% accuracy for code queries
- **When to Use**: Complex codebases, cross-file dependencies, architectural analysis

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

## Usage Examples

### Basic Usage (Original)
```python
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

# Original implementation - lightweight, hash-based
agent = ReminiscingAgent(storage_path="~/.talk/memories")
result = agent.run("What did we discuss about authentication?")
```

### Enhanced Usage (Recommended)
```python
from special_agents.reminiscing.reminiscing_agent_enhanced import EnhancedReminiscingAgent

# Standard configuration - real embeddings, no Serena
agent = EnhancedReminiscingAgent.create_standard(
    storage_path="~/.talk/memories"
)

# Store memories
agent.store_conversation({
    "task": "Implement user authentication",
    "messages": ["Use JWT tokens", "Store in Redis"]
})

agent.store_code_context({
    "file_path": "auth.py",
    "code": "def authenticate(token): ...",
    "functions": ["authenticate"]
})

# Search with enhanced capabilities
result = agent.run("How should I implement authentication?")
```

### Advanced Usage (With Serena)
```python
# Full capabilities - real embeddings + Serena for code analysis
agent = EnhancedReminiscingAgent.create_enhanced(
    storage_path="~/.talk/memories"
)

# Or manually configure
agent = EnhancedReminiscingAgent(
    storage_path="~/.talk/memories",
    enable_semantic_search=True,      # Enable Serena
    use_enhanced_vector_store=True,   # Real embeddings
    auto_route_to_serena=True         # Smart routing
)

# Serena automatically activated for code queries
result = agent.run("Find the Agent class implementation")
```

## Migration Guide

### From Original ReminiscingAgent

**No changes required!** Existing code continues to work:
```python
# This still works exactly as before
agent = ReminiscingAgent(storage_path="~/.talk/memories")
```

### To Enhanced Version

**Option 1: Drop-in Replacement**
```python
# Change import
from special_agents.reminiscing.reminiscing_agent_enhanced import EnhancedReminiscingAgent

# Use factory method for standard config
agent = EnhancedReminiscingAgent.create_standard(storage_path="~/.talk/memories")
```

**Option 2: Gradual Migration**
```python
# Start with basic mode (no changes to behavior)
agent = EnhancedReminiscingAgent.create_basic(storage_path="~/.talk/memories")

# Later, enable real embeddings
agent = EnhancedReminiscingAgent.create_standard(storage_path="~/.talk/memories")

# Eventually, try Serena for code-heavy workflows
agent = EnhancedReminiscingAgent.create_enhanced(storage_path="~/.talk/memories")
```

## Performance Comparison

| Configuration | Response Time | Memory Usage | Quality | When to Use |
|--------------|---------------|--------------|---------|-------------|
| Basic (Original) | 1-2s | 50MB | 60% | Resource-constrained |
| Enhanced (No Serena) | 2-3s | 80MB | 80% | **Most use cases** |
| Enhanced + Serena | 5-10s | 150MB | 95% | Complex code analysis |

## Configuration

### Storage Options
```python
# Basic store (hash-based)
from special_agents.reminiscing.vector_store import ConversationVectorStore
store = ConversationVectorStore(storage_path="memories.json")

# Enhanced store (real embeddings)
from special_agents.reminiscing.enhanced_vector_store_v2 import EnhancedVectorStoreV2
store = EnhancedVectorStoreV2(
    storage_path="memories.json",
    embedding_model="all-MiniLM-L6-v2",  # Or any sentence-transformer model
    use_real_embeddings=True
)
```

### Tuning Parameters
```python
# Enhanced store parameters
store.decay_factor = 0.95           # Daily importance decay
store.reinforcement_factor = 1.2    # Access boost
store.relationship_weight = 0.3     # Graph weight in scoring
store.similarity_threshold = 0.85   # Consolidation threshold
store.max_memories = 10000         # Memory limit

# Serena routing (if enabled)
agent.auto_route_to_serena = True   # Auto-detect when to use Serena
```

## Testing

Run the comprehensive test suite:

```bash
# Basic tests
python3 tests/special_agents/reminiscing/test_reminiscing_agent.py
python3 tests/special_agents/reminiscing/test_vector_store.py

# Enhanced features
python3 tests/special_agents/reminiscing/test_integration_simple.py
python3 tests/special_agents/reminiscing/test_performance_comparison.py

# Serena integration (if available)
python3 tests/special_agents/reminiscing_agent/serena/test_serena_agent.py

# Sub-agent tests
python3 tests/special_agents/reminiscing/test_subagents.py
```

## Troubleshooting

### Common Issues

**"sentence-transformers not available"**
```bash
pip install sentence-transformers
# The system will fall back to hash-based embeddings if not installed
```

**"Serena not available" when trying to use semantic search**
```bash
# Serena requires UV and complex setup
# For most users, the enhanced vector store is sufficient
# Use enable_semantic_search=False (default)
```

**Slow first run with real embeddings**
```
# First run downloads the model (~80MB)
# Subsequent runs are much faster
# Use create_basic() to avoid model download
```

**High memory usage**
```python
# Reduce memory limit
agent.vector_store.max_memories = 5000  # Default is 10000

# Or use basic mode
agent = EnhancedReminiscingAgent.create_basic()
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

### Required
- Base Talk framework

### Optional (Automatically Installed)
- `numpy`: Vector operations
- `scipy`: Scientific computing  
- `langgraph`: Workflow orchestration

### Enhanced Features (Install Separately)
```bash
# For real embeddings (recommended)
pip install sentence-transformers

# For Serena integration (advanced users)
# See serena.md for setup instructions
```

## License

Part of the Talk framework - see main project license.