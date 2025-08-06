# ReminiscingAgent Test Suite

This directory contains comprehensive tests for the ReminiscingAgent system and its components.

## Test Files

### `test_reminiscing_agent.py`
Main integration tests for the ReminiscingAgent:
- Basic functionality testing
- Sub-agent integration
- Talk framework integration
- End-to-end workflow testing

### `test_vector_store.py`
Unit tests for ConversationVectorStore:
- Conversation storage and retrieval
- Code context storage
- Vector similarity search
- Recent memory retrieval
- Persistence to disk
- Memory cleanup and limits
- Statistics reporting

### `test_subagents.py`
Unit tests for sub-agents:
- ContextCategorizationAgent testing
- MemoryTraceAgent testing
- Sub-agent integration
- Error handling

## Running Tests

### Run all tests:
```bash
# From project root
python3 tests/special_agents/reminiscing/test_reminiscing_agent.py
python3 tests/special_agents/reminiscing/test_vector_store.py
python3 tests/special_agents/reminiscing/test_subagents.py
```

### Run individual test:
```bash
cd /home/xx/code
python3 -m tests.special_agents.reminiscing.test_vector_store
```

## Test Coverage

The test suite covers:

1. **Core Functionality**
   - Agent initialization
   - Memory storage and retrieval
   - Search strategies
   - Workflow orchestration

2. **Vector Store**
   - In-memory storage
   - Embedding generation
   - Similarity calculations
   - Metadata management
   - Persistence

3. **Sub-agents**
   - Context categorization
   - Memory trace retrieval
   - Strategy selection
   - Error handling

4. **Integration**
   - Talk framework compatibility
   - Agent composition
   - Workflow execution

## Dependencies

Required for testing:
- `numpy` (optional, for vector operations)
- `scipy` (optional, for scientific computing)
- Base Talk framework

## Known Issues

1. **LangGraph Dependency**: Tests run in simplified mode without LangGraph
2. **Embedding Quality**: Currently uses hash-based embeddings (not semantic)
3. **Sub-agent Implementation**: Some sub-agents may not be fully implemented

## Future Improvements

1. Add performance benchmarks
2. Test with real embedding models
3. Add stress testing for large memory stores
4. Test external vector database integrations
5. Add mock-based unit tests for better isolation