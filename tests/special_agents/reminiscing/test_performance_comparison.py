#!/usr/bin/env python3
"""
Performance comparison tests for ReminiscingAgent configurations.

This test suite compares:
1. Basic ReminiscingAgent (hash-based embeddings, no Serena)
2. Enhanced ReminiscingAgent (real embeddings, no Serena)  
3. Enhanced + Serena ReminiscingAgent (real embeddings + Serena)

Metrics collected:
- Response time
- Memory usage
- Result quality/relevance
- Token usage (estimated)
"""

import sys
import os
import time
import json
import tempfile
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
from special_agents.reminiscing.reminiscing_agent_enhanced import EnhancedReminiscingAgent
from special_agents.reminiscing.vector_store import ConversationVectorStore
from special_agents.reminiscing.enhanced_vector_store_v2 import EnhancedVectorStoreV2

# Test queries of different types
TEST_QUERIES = {
    "conversation_recall": [
        "What did we discuss about authentication?",
        "What were the project requirements?",
        "What decisions were made in the last meeting?"
    ],
    "code_search": [
        "Find the Agent class implementation",
        "Where is the error handling for database connections?",
        "Show me the test functions for vector store"
    ],
    "mixed_context": [
        "How did we implement the authentication feature?",
        "What bugs were reported in the vector store?",
        "Explain the architecture decisions for the memory system"
    ],
    "symbol_search": [
        "Find all references to ReminiscingAgent",
        "Where is the store_conversation method used?",
        "List all imports from langgraph"
    ]
}

# Sample data for testing
SAMPLE_CONVERSATIONS = [
    {
        "task": "Implement user authentication",
        "messages": [
            "We need JWT-based authentication",
            "Use bcrypt for password hashing",
            "Store sessions in Redis"
        ],
        "timestamp": "2024-01-15T10:00:00"
    },
    {
        "task": "Fix database connection issues",
        "messages": [
            "Connection pool exhausted error",
            "Increased max connections to 100",
            "Added retry logic with exponential backoff"
        ],
        "timestamp": "2024-01-16T14:30:00"
    },
    {
        "task": "Design vector store architecture",
        "messages": [
            "Use in-memory storage with optional persistence",
            "Implement cosine similarity for search",
            "Add metadata filtering capabilities"
        ],
        "timestamp": "2024-01-17T09:15:00"
    }
]

SAMPLE_CODE_CONTEXTS = [
    {
        "file_path": "agent/agent.py",
        "code": '''class Agent:
    """Base agent class for the Talk framework."""
    
    def __init__(self, roles=None, **kwargs):
        self.roles = roles or []
        self.messages = []
    
    def run(self, input_text: str) -> str:
        """Execute agent logic."""
        raise NotImplementedError''',
        "language": "python",
        "functions": ["__init__", "run"],
        "classes": ["Agent"]
    },
    {
        "file_path": "vector_store.py",
        "code": '''def store_conversation(self, conversation_data: Dict) -> str:
    """Store conversation in vector store."""
    memory_id = self._generate_memory_id(conversation_data)
    embedding = self._generate_embedding(conversation_data)
    self.conversations.append(conversation_data)
    return memory_id''',
        "language": "python",
        "functions": ["store_conversation", "_generate_memory_id", "_generate_embedding"]
    },
    {
        "file_path": "test_vector_store.py",
        "code": '''def test_store_and_retrieve():
    """Test storing and retrieving memories."""
    store = VectorStore()
    memory_id = store.store_conversation({"task": "test"})
    assert memory_id is not None
    results = store.search_memories("test")
    assert len(results) > 0''',
        "language": "python",
        "functions": ["test_store_and_retrieve"]
    }
]


class PerformanceComparison:
    """Compare performance of different ReminiscingAgent configurations."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.temp_dir = None
    
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="reminiscing_test_")
        
        if self.verbose:
            print(f"Test directory: {self.temp_dir}")
        
        # Populate test data
        self._populate_test_data()
        
        # Create agents
        self.agents = self._create_agents()
    
    def cleanup(self):
        """Clean up test environment."""
        import shutil
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _populate_test_data(self):
        """Populate vector stores with test data."""
        # Create a temporary store to populate data
        store = EnhancedVectorStoreV2(
            storage_path=os.path.join(self.temp_dir, "test_memories.json"),
            use_real_embeddings=True
        )
        
        # Add conversations
        for conv in SAMPLE_CONVERSATIONS:
            store.store_conversation(conv)
        
        # Add code contexts
        for code in SAMPLE_CODE_CONTEXTS:
            store.store_code_context(code)
        
        if self.verbose:
            stats = store.get_stats()
            print(f"Populated store: {stats['total_conversations']} conversations, "
                  f"{stats['total_code_contexts']} code contexts")
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create different agent configurations."""
        storage_path = os.path.join(self.temp_dir, "test_memories.json")
        
        agents = {}
        
        # 1. Basic agent (original ReminiscingAgent)
        try:
            agents["basic"] = ReminiscingAgent(
                storage_path=storage_path,
                name="BasicAgent"
            )
            if self.verbose:
                print("✓ Basic agent created")
        except Exception as e:
            print(f"✗ Failed to create basic agent: {e}")
        
        # 2. Enhanced agent without Serena
        try:
            agents["enhanced_no_serena"] = EnhancedReminiscingAgent(
                storage_path=storage_path,
                enable_semantic_search=False,
                use_enhanced_vector_store=True,
                name="EnhancedNoSerena"
            )
            if self.verbose:
                print("✓ Enhanced agent (no Serena) created")
        except Exception as e:
            print(f"✗ Failed to create enhanced agent: {e}")
        
        # 3. Enhanced agent with Serena
        try:
            agents["enhanced_with_serena"] = EnhancedReminiscingAgent(
                storage_path=storage_path,
                enable_semantic_search=True,
                use_enhanced_vector_store=True,
                auto_route_to_serena=True,
                name="EnhancedWithSerena"
            )
            if self.verbose:
                print("✓ Enhanced agent (with Serena) created")
        except Exception as e:
            print(f"✗ Failed to create enhanced+Serena agent: {e}")
        
        return agents
    
    def run_query_test(self, query: str, query_type: str) -> Dict[str, Any]:
        """Run a single query across all agents."""
        results = {
            "query": query,
            "type": query_type,
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            if self.verbose:
                print(f"\n  Testing {agent_name}...")
            
            # Start monitoring
            tracemalloc.start()
            start_time = time.time()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            try:
                # Run query
                response = agent.run(query)
                
                # Stop monitoring
                elapsed = time.time() - start_time
                end_memory = tracemalloc.get_traced_memory()[0]
                memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
                tracemalloc.stop()
                
                # Analyze response
                response_analysis = self._analyze_response(response, query_type)
                
                results["agents"][agent_name] = {
                    "success": True,
                    "response_time": elapsed,
                    "memory_used_mb": memory_used,
                    "response_length": len(response),
                    "quality_metrics": response_analysis,
                    "error": None
                }
                
                if self.verbose:
                    print(f"    Time: {elapsed:.3f}s, Memory: {memory_used:.2f}MB")
                
            except Exception as e:
                tracemalloc.stop()
                results["agents"][agent_name] = {
                    "success": False,
                    "response_time": None,
                    "memory_used_mb": None,
                    "response_length": 0,
                    "quality_metrics": {},
                    "error": str(e)
                }
                
                if self.verbose:
                    print(f"    Error: {e}")
        
        return results
    
    def _analyze_response(self, response: str, query_type: str) -> Dict[str, Any]:
        """Analyze response quality metrics."""
        metrics = {
            "has_conversation_memories": "CONVERSATION MEMORIES" in response or "MEMORY_TRACES" in response,
            "has_code_context": "CODE CONTEXT" in response or "code" in response.lower(),
            "has_serena_analysis": "Serena" in response,
            "confidence_score": 0.0,
            "response_type": "unknown"
        }
        
        # Extract confidence if present
        if "CONFIDENCE:" in response:
            try:
                conf_line = [l for l in response.split('\n') if "CONFIDENCE:" in l][0]
                metrics["confidence_score"] = float(conf_line.split(":")[-1].strip())
            except:
                pass
        
        # Determine response type
        if "Response Type:" in response:
            type_line = [l for l in response.split('\n') if "Response Type:" in l][0]
            metrics["response_type"] = type_line.split(":")[-1].strip()
        
        # Query-type specific checks
        if query_type == "code_search":
            metrics["appropriate_for_query"] = metrics["has_code_context"]
        elif query_type == "conversation_recall":
            metrics["appropriate_for_query"] = metrics["has_conversation_memories"]
        else:
            metrics["appropriate_for_query"] = True
        
        return metrics
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        print("\n" + "=" * 60)
        print("REMINISCING AGENT PERFORMANCE COMPARISON")
        print("=" * 60)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "query_results": [],
            "summary": {}
        }
        
        # Run tests for each query type
        for query_type, queries in TEST_QUERIES.items():
            print(f"\n[{query_type.upper()}]")
            
            for query in queries:
                print(f"\nQuery: {query[:50]}...")
                result = self.run_query_test(query, query_type)
                all_results["query_results"].append(result)
        
        # Generate summary
        all_results["summary"] = self._generate_summary(all_results["query_results"])
        
        return all_results
    
    def _generate_summary(self, query_results: List[Dict]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {}
        
        # Aggregate by agent
        for agent_name in self.agents.keys():
            agent_results = []
            
            for query_result in query_results:
                if agent_name in query_result["agents"]:
                    agent_results.append(query_result["agents"][agent_name])
            
            # Calculate averages
            successful = [r for r in agent_results if r["success"]]
            
            if successful:
                avg_time = sum(r["response_time"] for r in successful) / len(successful)
                avg_memory = sum(r["memory_used_mb"] for r in successful) / len(successful)
                avg_confidence = sum(r["quality_metrics"].get("confidence_score", 0) for r in successful) / len(successful)
                appropriate_count = sum(1 for r in successful if r["quality_metrics"].get("appropriate_for_query", False))
                
                summary[agent_name] = {
                    "success_rate": len(successful) / len(agent_results) * 100,
                    "avg_response_time": avg_time,
                    "avg_memory_mb": avg_memory,
                    "avg_confidence": avg_confidence,
                    "appropriateness_rate": appropriate_count / len(successful) * 100 if successful else 0
                }
            else:
                summary[agent_name] = {
                    "success_rate": 0,
                    "avg_response_time": None,
                    "avg_memory_mb": None,
                    "avg_confidence": 0,
                    "appropriateness_rate": 0
                }
        
        return summary
    
    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        
        # Print comparison table
        print("\n%-25s %10s %10s %10s %10s" % (
            "Agent", "Avg Time", "Memory MB", "Confidence", "Appropriate"
        ))
        print("-" * 75)
        
        for agent_name, metrics in summary.items():
            print("%-25s %10.3f %10.2f %10.2f %9.1f%%" % (
                agent_name,
                metrics["avg_response_time"] or 0,
                metrics["avg_memory_mb"] or 0,
                metrics["avg_confidence"],
                metrics["appropriateness_rate"]
            ))
        
        # Determine winner for each category
        print("\n" + "=" * 60)
        print("CATEGORY WINNERS")
        print("=" * 60)
        
        # Speed
        fastest = min(summary.items(), key=lambda x: x[1]["avg_response_time"] or float('inf'))
        print(f"Fastest: {fastest[0]} ({fastest[1]['avg_response_time']:.3f}s)")
        
        # Memory efficiency
        most_efficient = min(summary.items(), key=lambda x: x[1]["avg_memory_mb"] or float('inf'))
        print(f"Most Memory Efficient: {most_efficient[0]} ({most_efficient[1]['avg_memory_mb']:.2f}MB)")
        
        # Quality
        highest_confidence = max(summary.items(), key=lambda x: x[1]["avg_confidence"])
        print(f"Highest Confidence: {highest_confidence[0]} ({highest_confidence[1]['avg_confidence']:.2f})")
        
        # Appropriateness
        most_appropriate = max(summary.items(), key=lambda x: x[1]["appropriateness_rate"])
        print(f"Most Appropriate: {most_appropriate[0]} ({most_appropriate[1]['appropriateness_rate']:.1f}%)")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
        
        return filepath


def main():
    """Run performance comparison tests."""
    comparison = PerformanceComparison(verbose=True)
    
    try:
        # Setup
        comparison.setup()
        
        # Run tests
        results = comparison.run_all_tests()
        
        # Print summary
        comparison.print_summary(results)
        
        # Save results
        comparison.save_results(results)
        
        return True
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        comparison.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)