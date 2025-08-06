#!/usr/bin/env python3
"""
Comprehensive test suite for SemanticSearchAgent.

Tests semantic search capabilities, codebase analysis, and context extraction.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from special_agents.reminiscing.semantic_search_agent import (
    SemanticSearchAgent, CodebaseIndex, ContextExtractor, 
    SemanticMatcher, QueryProcessor
)


def test_semantic_search_agent_initialization():
    """Test SemanticSearchAgent initialization."""
    print("Testing SemanticSearchAgent initialization...")
    
    agent = SemanticSearchAgent()
    
    assert agent is not None, "Agent not created"
    assert hasattr(agent, 'codebase_index'), "Missing codebase_index"
    assert hasattr(agent, 'context_extractor'), "Missing context_extractor"
    assert hasattr(agent, 'semantic_matcher'), "Missing semantic_matcher"
    assert hasattr(agent, 'query_processor'), "Missing query_processor"
    
    print("[OK] SemanticSearchAgent initialized correctly")
    return True


def test_codebase_indexing():
    """Test codebase indexing functionality."""
    print("\nTesting codebase indexing...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = {
            'main.py': 'def main():\n    print("Hello")\n    return 0',
            'utils.py': 'def helper(x):\n    return x * 2',
            'test.txt': 'This is not code',
            'nested/module.py': 'class MyClass:\n    pass'
        }
        
        for filepath, content in test_files.items():
            full_path = Path(tmpdir) / filepath
            full_path.parent.mkdir(exist_ok=True)
            full_path.write_text(content)
        
        # Index the codebase
        agent = SemanticSearchAgent()
        index = agent.index_codebase(tmpdir)
        
        assert 'files' in index, "Missing files in index"
        assert 'functions' in index, "Missing functions in index"
        assert 'classes' in index, "Missing classes in index"
        
        # Check indexed files
        assert len(index['files']) >= 2, "Not all Python files indexed"
        assert any('main.py' in f for f in index['files']), "main.py not indexed"
        
        # Check extracted functions
        assert 'main' in index['functions'], "main function not extracted"
        assert 'helper' in index['functions'], "helper function not extracted"
        
        # Check extracted classes
        assert 'MyClass' in index['classes'], "MyClass not extracted"
        
        print(f"[OK] Indexed {len(index['files'])} files, {len(index['functions'])} functions, {len(index['classes'])} classes")
    
    return True


def test_semantic_similarity():
    """Test semantic similarity calculations."""
    print("\nTesting semantic similarity...")
    
    agent = SemanticSearchAgent()
    
    # Test similar phrases
    similarity_tests = [
        ("user authentication", "login system", 0.5, 1.0),  # Should be similar
        ("database connection", "DB link", 0.5, 1.0),  # Should be similar
        ("error handling", "exception management", 0.5, 1.0),  # Should be similar
        ("authentication", "pizza recipe", 0.0, 0.3),  # Should be dissimilar
    ]
    
    for phrase1, phrase2, min_sim, max_sim in similarity_tests:
        similarity = agent.calculate_semantic_similarity(phrase1, phrase2)
        assert min_sim <= similarity <= max_sim, \
            f"Similarity between '{phrase1}' and '{phrase2}' = {similarity}, expected {min_sim}-{max_sim}"
        print(f"  '{phrase1}' <-> '{phrase2}': {similarity:.2f}")
    
    print("[OK] Semantic similarity working correctly")
    return True


def test_context_extraction():
    """Test context extraction from code."""
    print("\nTesting context extraction...")
    
    agent = SemanticSearchAgent()
    
    code_samples = [
        {
            'code': '''
def authenticate_user(username, password):
    """Authenticate a user with username and password."""
    # Check credentials against database
    user = db.find_user(username)
    if user and user.check_password(password):
        return create_token(user)
    return None
''',
            'expected_concepts': ['authenticate', 'user', 'password', 'database', 'token'],
            'expected_type': 'authentication'
        },
        {
            'code': '''
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def connect(self):
        self.connection = psycopg2.connect(host=self.host, port=self.port)
''',
            'expected_concepts': ['database', 'connection', 'host', 'port'],
            'expected_type': 'database'
        }
    ]
    
    for sample in code_samples:
        context = agent.extract_context(sample['code'])
        
        assert 'concepts' in context, "Missing concepts in context"
        assert 'type' in context, "Missing type in context"
        assert 'summary' in context, "Missing summary in context"
        
        # Check extracted concepts
        for concept in sample['expected_concepts']:
            assert any(concept.lower() in c.lower() for c in context['concepts']), \
                f"Expected concept '{concept}' not found in {context['concepts']}"
        
        print(f"  Extracted {len(context['concepts'])} concepts from {sample['expected_type']} code")
    
    print("[OK] Context extraction working correctly")
    return True


def test_query_processing():
    """Test query processing and expansion."""
    print("\nTesting query processing...")
    
    agent = SemanticSearchAgent()
    
    queries = [
        {
            'query': "How to implement OAuth authentication?",
            'expected_keywords': ['oauth', 'authentication', 'implement'],
            'expected_expansions': ['auth', 'login', 'security']
        },
        {
            'query': "Debug database connection timeout",
            'expected_keywords': ['debug', 'database', 'connection', 'timeout'],
            'expected_expansions': ['db', 'error', 'troubleshoot']
        }
    ]
    
    for test in queries:
        processed = agent.process_query(test['query'])
        
        assert 'keywords' in processed, "Missing keywords"
        assert 'expanded_terms' in processed, "Missing expanded terms"
        assert 'query_type' in processed, "Missing query type"
        
        # Check keywords extraction
        for keyword in test['expected_keywords']:
            assert keyword in processed['keywords'], \
                f"Expected keyword '{keyword}' not found"
        
        print(f"  Query: '{test['query'][:30]}...'")
        print(f"    Keywords: {processed['keywords']}")
        print(f"    Type: {processed['query_type']}")
    
    print("[OK] Query processing working correctly")
    return True


def test_semantic_search():
    """Test end-to-end semantic search."""
    print("\nTesting semantic search...")
    
    agent = SemanticSearchAgent()
    
    # Add test memories
    test_memories = [
        {
            'id': 'mem1',
            'content': 'Implemented OAuth2 authentication using passport.js',
            'type': 'implementation',
            'concepts': ['oauth', 'authentication', 'passport']
        },
        {
            'id': 'mem2',
            'content': 'Fixed database connection pool timeout issues',
            'type': 'debugging',
            'concepts': ['database', 'timeout', 'connection', 'pool']
        },
        {
            'id': 'mem3',
            'content': 'Designed microservice architecture with API gateway',
            'type': 'architecture',
            'concepts': ['microservice', 'architecture', 'api', 'gateway']
        }
    ]
    
    # Index memories
    for memory in test_memories:
        agent.index_memory(memory)
    
    # Test searches
    searches = [
        {
            'query': 'authentication system',
            'expected_match': 'mem1',
            'min_score': 0.5
        },
        {
            'query': 'database errors',
            'expected_match': 'mem2',
            'min_score': 0.4
        },
        {
            'query': 'system design',
            'expected_match': 'mem3',
            'min_score': 0.3
        }
    ]
    
    for search in searches:
        results = agent.semantic_search(search['query'], limit=3)
        
        assert len(results) > 0, f"No results for query: {search['query']}"
        
        # Check if expected match is in top results
        top_ids = [r['id'] for r in results[:2]]
        assert search['expected_match'] in top_ids, \
            f"Expected {search['expected_match']} in top results for '{search['query']}'"
        
        # Check minimum score
        assert results[0]['score'] >= search['min_score'], \
            f"Top result score {results[0]['score']} < {search['min_score']}"
        
        print(f"  Query: '{search['query']}' -> Top match: {results[0]['id']} (score: {results[0]['score']:.2f})")
    
    print("[OK] Semantic search working correctly")
    return True


def test_concept_graph():
    """Test concept graph and relationships."""
    print("\nTesting concept graph...")
    
    agent = SemanticSearchAgent()
    
    # Build concept relationships
    concepts = [
        ('authentication', ['security', 'login', 'user']),
        ('database', ['storage', 'query', 'connection']),
        ('api', ['endpoint', 'rest', 'http']),
        ('error', ['exception', 'debug', 'fix'])
    ]
    
    for main_concept, related in concepts:
        for related_concept in related:
            agent.add_concept_relationship(main_concept, related_concept)
    
    # Test concept expansion
    expanded = agent.expand_concept('authentication')
    assert 'security' in expanded, "Related concept 'security' not found"
    assert 'login' in expanded, "Related concept 'login' not found"
    
    # Test concept similarity through graph
    similarity = agent.concept_graph_similarity('authentication', 'security')
    assert similarity > 0.5, f"Expected high similarity between related concepts, got {similarity}"
    
    similarity = agent.concept_graph_similarity('authentication', 'database')
    assert similarity < 0.3, f"Expected low similarity between unrelated concepts, got {similarity}"
    
    print("[OK] Concept graph working correctly")
    return True


def test_code_pattern_matching():
    """Test code pattern matching and extraction."""
    print("\nTesting code pattern matching...")
    
    agent = SemanticSearchAgent()
    
    code_patterns = [
        {
            'pattern': 'function_definition',
            'code': 'def calculate_total(items):\n    return sum(items)',
            'expected_matches': ['calculate_total']
        },
        {
            'pattern': 'class_definition',
            'code': 'class UserManager:\n    def __init__(self):\n        pass',
            'expected_matches': ['UserManager']
        },
        {
            'pattern': 'import_statement',
            'code': 'import pandas as pd\nfrom datetime import datetime',
            'expected_matches': ['pandas', 'datetime']
        },
        {
            'pattern': 'error_handling',
            'code': 'try:\n    risky_operation()\nexcept ValueError as e:\n    handle_error(e)',
            'expected_matches': ['ValueError', 'handle_error']
        }
    ]
    
    for test in code_patterns:
        matches = agent.match_code_pattern(test['code'], test['pattern'])
        
        for expected in test['expected_matches']:
            assert any(expected in str(m) for m in matches), \
                f"Expected match '{expected}' not found for pattern {test['pattern']}"
        
        print(f"  Pattern '{test['pattern']}': Found {len(matches)} matches")
    
    print("[OK] Code pattern matching working correctly")
    return True


def test_memory_relevance_decay():
    """Test temporal relevance decay for memories."""
    print("\nTesting memory relevance decay...")
    
    from datetime import datetime, timedelta
    
    agent = SemanticSearchAgent()
    
    # Create memories with different ages
    now = datetime.now()
    memories = [
        {'id': 'recent', 'timestamp': now - timedelta(hours=1), 'content': 'Recent memory'},
        {'id': 'yesterday', 'timestamp': now - timedelta(days=1), 'content': 'Yesterday memory'},
        {'id': 'old', 'timestamp': now - timedelta(days=7), 'content': 'Week old memory'},
        {'id': 'ancient', 'timestamp': now - timedelta(days=30), 'content': 'Month old memory'}
    ]
    
    for memory in memories:
        agent.index_memory(memory)
    
    # Calculate relevance with temporal decay
    for memory in memories:
        relevance = agent.calculate_temporal_relevance(memory['timestamp'])
        age_hours = (now - memory['timestamp']).total_seconds() / 3600
        
        # More recent should have higher relevance
        if age_hours < 24:
            assert relevance > 0.8, f"Recent memory relevance too low: {relevance}"
        elif age_hours < 168:  # 1 week
            assert 0.3 < relevance < 0.8, f"Week-old memory relevance out of range: {relevance}"
        else:
            assert relevance < 0.3, f"Old memory relevance too high: {relevance}"
        
        print(f"  Memory '{memory['id']}' (age: {age_hours:.1f}h): relevance = {relevance:.2f}")
    
    print("[OK] Temporal relevance decay working correctly")
    return True


def test_multi_modal_search():
    """Test searching across different content types."""
    print("\nTesting multi-modal search...")
    
    agent = SemanticSearchAgent()
    
    # Index different content types
    content_types = [
        {'type': 'code', 'content': 'def login(user, password): return authenticate(user, password)'},
        {'type': 'documentation', 'content': 'The login function handles user authentication'},
        {'type': 'error', 'content': 'LoginError: Invalid credentials for user admin'},
        {'type': 'conversation', 'content': 'User asked about implementing login functionality'}
    ]
    
    for item in content_types:
        agent.index_content(item['content'], content_type=item['type'])
    
    # Search across all types
    results = agent.multi_modal_search('login authentication', types=['code', 'documentation'])
    
    assert len(results) > 0, "No multi-modal results found"
    
    # Check that results include multiple types
    result_types = set(r.get('type') for r in results)
    assert len(result_types) >= 2, f"Expected multiple content types, got {result_types}"
    
    print(f"[OK] Multi-modal search found {len(results)} results across {len(result_types)} types")
    return True


def test_batch_operations():
    """Test batch indexing and searching."""
    print("\nTesting batch operations...")
    
    agent = SemanticSearchAgent()
    
    # Batch index memories
    batch_memories = [
        {'id': f'mem_{i}', 'content': f'Memory content {i}'} 
        for i in range(100)
    ]
    
    start_count = agent.get_index_size()
    agent.batch_index_memories(batch_memories)
    end_count = agent.get_index_size()
    
    assert end_count - start_count == 100, f"Expected 100 new memories, got {end_count - start_count}"
    
    # Batch search
    queries = ['Memory content 1', 'Memory content 50', 'Memory content 99']
    batch_results = agent.batch_search(queries)
    
    assert len(batch_results) == len(queries), "Batch search result count mismatch"
    
    for i, results in enumerate(batch_results):
        assert len(results) > 0, f"No results for query {i}"
    
    print(f"[OK] Batch indexed {len(batch_memories)} memories and searched {len(queries)} queries")
    return True


def test_similarity_metrics():
    """Test different similarity metrics."""
    print("\nTesting similarity metrics...")
    
    agent = SemanticSearchAgent()
    
    # Test different metrics
    text1 = "implement user authentication"
    text2 = "create login system"
    
    metrics = ['cosine', 'jaccard', 'levenshtein', 'semantic']
    
    for metric in metrics:
        similarity = agent.calculate_similarity(text1, text2, metric=metric)
        assert 0 <= similarity <= 1, f"Similarity out of range for {metric}: {similarity}"
        print(f"  {metric}: {similarity:.3f}")
    
    # Test that identical texts have similarity = 1
    for metric in metrics:
        similarity = agent.calculate_similarity(text1, text1, metric=metric)
        assert similarity > 0.99, f"Identical texts should have similarity ~1 for {metric}, got {similarity}"
    
    print("[OK] All similarity metrics working correctly")
    return True


def test_concept_extraction_advanced():
    """Test advanced concept extraction with NLP."""
    print("\nTesting advanced concept extraction...")
    
    agent = SemanticSearchAgent()
    
    texts = [
        {
            'text': "The UserAuthenticationManager class handles OAuth2 login flows",
            'expected_entities': ['UserAuthenticationManager', 'OAuth2'],
            'expected_concepts': ['authentication', 'login', 'class']
        },
        {
            'text': "DatabaseConnectionPool manages PostgreSQL connections with retry logic",
            'expected_entities': ['DatabaseConnectionPool', 'PostgreSQL'],
            'expected_concepts': ['database', 'connection', 'retry']
        }
    ]
    
    for test in texts:
        extraction = agent.extract_concepts_advanced(test['text'])
        
        assert 'entities' in extraction, "Missing entities"
        assert 'concepts' in extraction, "Missing concepts"
        assert 'keywords' in extraction, "Missing keywords"
        
        # Check entity extraction
        for entity in test['expected_entities']:
            assert any(entity.lower() in e.lower() for e in extraction['entities']), \
                f"Expected entity '{entity}' not found"
        
        # Check concept extraction
        for concept in test['expected_concepts']:
            assert any(concept.lower() in c.lower() for c in extraction['concepts']), \
                f"Expected concept '{concept}' not found"
        
        print(f"  Extracted {len(extraction['entities'])} entities, {len(extraction['concepts'])} concepts")
    
    print("[OK] Advanced concept extraction working correctly")
    return True


def test_search_result_ranking():
    """Test search result ranking and scoring."""
    print("\nTesting search result ranking...")
    
    agent = SemanticSearchAgent()
    
    # Add memories with known relevance
    memories = [
        {'id': 'exact', 'content': 'implement user authentication system', 'expected_rank': 1},
        {'id': 'close', 'content': 'create login authentication module', 'expected_rank': 2},
        {'id': 'related', 'content': 'security and user management', 'expected_rank': 3},
        {'id': 'distant', 'content': 'database optimization techniques', 'expected_rank': 4}
    ]
    
    for memory in memories:
        agent.index_memory(memory)
    
    # Search and check ranking
    query = "implement user authentication"
    results = agent.semantic_search(query, limit=4)
    
    # Check that results are properly ranked
    result_ids = [r['id'] for r in results]
    
    # Exact match should be first
    assert result_ids[0] == 'exact', f"Expected 'exact' to be first, got {result_ids[0]}"
    
    # Check scores are descending
    scores = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True), f"Scores not in descending order: {scores}"
    
    print(f"[OK] Results properly ranked: {result_ids}")
    return True


def test_caching_mechanism():
    """Test query result caching."""
    print("\nTesting caching mechanism...")
    
    agent = SemanticSearchAgent()
    
    # Add test data
    for i in range(10):
        agent.index_memory({'id': f'mem_{i}', 'content': f'Content {i}'})
    
    # First search (should cache)
    query = "test query for caching"
    results1 = agent.semantic_search(query, cache=True)
    
    # Second search (should use cache)
    results2 = agent.semantic_search(query, cache=True)
    
    # Results should be identical
    assert len(results1) == len(results2), "Cached results differ in length"
    for r1, r2 in zip(results1, results2):
        assert r1['id'] == r2['id'], "Cached results differ in content"
    
    # Clear cache and search again
    agent.clear_cache()
    results3 = agent.semantic_search(query, cache=False)
    
    # Should work but might have slight score differences
    assert len(results3) == len(results1), "Non-cached search failed"
    
    print("[OK] Caching mechanism working correctly")
    return True


if __name__ == "__main__":
    print("=== SemanticSearchAgent Test Suite ===\n")
    
    tests = [
        ("Initialization", test_semantic_search_agent_initialization),
        ("Codebase Indexing", test_codebase_indexing),
        ("Semantic Similarity", test_semantic_similarity),
        ("Context Extraction", test_context_extraction),
        ("Query Processing", test_query_processing),
        ("Semantic Search", test_semantic_search),
        ("Concept Graph", test_concept_graph),
        ("Code Pattern Matching", test_code_pattern_matching),
        ("Temporal Relevance", test_memory_relevance_decay),
        ("Multi-modal Search", test_multi_modal_search),
        ("Batch Operations", test_batch_operations),
        ("Similarity Metrics", test_similarity_metrics),
        ("Advanced Concept Extraction", test_concept_extraction_advanced),
        ("Result Ranking", test_search_result_ranking),
        ("Caching", test_caching_mechanism)
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
        print("[SUCCESS] All SemanticSearchAgent tests passed!")
    else:
        print("[WARNING] Some tests failed. Check output above.")