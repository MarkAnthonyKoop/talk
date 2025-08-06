# ReminiscingAgent Usage Guide

The ReminiscingAgent is a sophisticated memory system that provides human-like contextual memory for the Talk framework. It mimics how humans recall relevant past experiences when encountering new tasks.

## Quick Start

```python
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

# Create the agent
agent = ReminiscingAgent()

# Use it to find relevant memories for a task
result = agent.run("I need to implement user authentication")
print(result)
```

## Integration with Talk Framework

The ReminiscingAgent is designed to enhance the Talk framework with memory-based decision making:

```python
from talk.talk import TalkOrchestrator
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent

# Create Talk instance
talk = TalkOrchestrator(task="Build a web application")

# Create and integrate memory agent
memory_agent = ReminiscingAgent()

# Use memory agent to inform decisions
context = "How should I structure the authentication system?"
memory_traces = memory_agent.run(context)
print(memory_traces)
```

## Core Components

### 1. ContextCategorizationAgent
Analyzes incoming prompts and categorizes them:
- **architectural**: Design decisions, system architecture
- **implementation**: Coding tasks, feature development
- **debugging**: Error analysis, troubleshooting
- **research**: Learning, best practices, comparisons
- **general**: Unclear intent or general questions

### 2. MemoryTraceAgent
Performs memory retrieval using spreading activation:
- **Semantic similarity**: Vector-based content matching
- **Graph traversal**: Relationship-based retrieval
- **Error similarity**: Pattern matching for debugging
- **Temporal search**: Recent context progression
- **Code similarity**: Similar implementation examples

### 3. ConversationVectorStore
Stores and retrieves conversation history:
- Vector embeddings for semantic search
- Metadata filtering and categorization
- Temporal weighting for recency effects
- Conversation and code context storage

## Memory Search Strategies

The agent automatically selects the best strategy based on context:

| Category | Strategy | Use Case |
|----------|----------|----------|
| Architectural | Graph Traversal | Design decisions requiring relationship analysis |
| Implementation | Code Similarity | Finding similar code examples and patterns |
| Debugging | Error Similarity | Matching error patterns and solutions |
| Research | Semantic Search | General knowledge and best practices |
| General | Semantic Search | Default fallback strategy |

## Storing Memories

### Conversation Memory
```python
# Store conversation context
conversation_data = {
    "task": "Implement OAuth2 authentication",
    "messages": [
        "We need to add OAuth2 support",
        "I'll use passport.js for this",
        "Implementation supports Google and GitHub"
    ],
    "outcome": "Successfully implemented OAuth2"
}

memory_id = agent.store_conversation(conversation_data)
```

### Code Context Memory
```python
# Store code context
code_data = {
    "file_path": "auth/oauth.js",
    "code": "// OAuth2 implementation...",
    "functions": ["authenticateOAuth", "validateToken"],
    "description": "OAuth2 authentication system"
}

memory_id = agent.store_code_context(code_data)
```

## Advanced Usage

### Custom Memory Search
```python
# Direct search with specific parameters
search_params = {
    "context": "database connection issues",
    "category": "debugging",
    "strategy": "error_similarity"
}

import json
result = agent.memory_trace_agent.run(json.dumps(search_params))
```

### Memory Statistics
```python
# Get memory store statistics
stats = agent.vector_store.get_stats()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total code contexts: {stats['total_code_contexts']}")
```

## Configuration Options

### Vector Store Configuration
```python
# Custom storage path
agent = ReminiscingAgent()
agent.vector_store = ConversationVectorStore(storage_path="./memories.json")
```

### Memory Limits
```python
# Configure memory limits
agent.vector_store.max_memories = 5000  # Default: 10000
```

## Example Scenarios

### 1. Architecture Guidance
```python
context = "How should I design the microservice architecture?"
result = agent.run(context)
# Returns: architectural traces with design patterns and best practices
```

### 2. Debug Assistance
```python
context = "Getting database timeout errors in production"
result = agent.run(context)
# Returns: debugging traces with similar error patterns and solutions
```

### 3. Implementation Help
```python
context = "Need to implement real-time notifications"
result = agent.run(context)
# Returns: code similarity traces with relevant implementation examples
```

### 4. Research Support
```python
context = "What are the best practices for API rate limiting?"
result = agent.run(context)
# Returns: research traces with best practices and comparisons
```

## Performance Notes

- **Memory Efficiency**: Uses hash-based embeddings for fast prototyping
- **Upgrade Path**: Can be enhanced with proper embedding models (sentence-transformers, OpenAI, etc.)
- **Scalability**: Supports up to 10,000 memories by default
- **Persistence**: Automatically saves/loads memory state to JSON

## Dependencies

Required packages:
- `numpy>=1.24.0` - Vector operations
- `scipy>=1.10.0` - Scientific computing
- `langgraph>=0.0.30` - Workflow orchestration (optional)
- `langchain-core>=0.1.0` - Core framework components

## Testing

Run the test suite:
```bash
python3 test_reminiscing_agent.py
```

Run the demonstration:
```bash
python3 demo_reminiscing_agent.py
```

## Future Enhancements

Potential improvements:
1. **Better Embeddings**: Replace hash-based with transformer models
2. **External Vector DBs**: Integration with Qdrant, Pinecone, Weaviate
3. **Graph Networks**: Enhanced relationship modeling
4. **Learning**: Adaptive memory based on usage patterns
5. **Compression**: Smart memory consolidation over time

The ReminiscingAgent provides a foundation for human-like memory in AI systems, making the Talk framework more intelligent and context-aware.