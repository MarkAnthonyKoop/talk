#!/usr/bin/env python3
"""
Demo script showing various agent queries on YouTube history.
"""

from youtube_history_agent import YouTubeHistoryAgent
from pathlib import Path
import time

def demo_queries():
    """Run a series of demo queries to showcase the agent's capabilities."""
    
    db_path = "youtube_fast.db"
    if not Path(db_path).exists():
        print("Error: Database not found. Run build_db_fast.py first.")
        return
    
    print("YouTube History Agent Demo")
    print("=" * 70)
    print("This demo shows how the agent can answer various questions about")
    print("your YouTube viewing history using natural language.\n")
    
    agent = YouTubeHistoryAgent(db_path)
    
    # Demo queries
    queries = [
        "What codebase analysis or AST-related videos have I watched?",
        "Show me content about building AI agents, multi-agent systems, or autonomous AI",
        "What Docker, Kubernetes, or DevOps content is in my history?",
        "Have I watched anything about static analysis, code quality, or testing?",
        "What are my top 5 most technical channels based on AI/coding relevance?",
        "Based on my viewing history, what knowledge gaps should I fill for better codebase analysis skills?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print("-" * 70)
        
        try:
            result = agent.analyze(query)
            
            # Truncate if too long for demo
            if len(result) > 1500:
                result = result[:1500] + "\n\n[... truncated for demo ...]"
            
            print(result)
            
            # Small delay between queries
            if i < len(queries):
                time.sleep(2)
                
        except Exception as e:
            print(f"Error: {e}")
    
    agent.close()
    
    print(f"\n{'=' * 70}")
    print("Demo Complete!")
    print("\nThe agent can answer any question about your YouTube history:")
    print("- Content discovery ('What have I watched about X?')")
    print("- Pattern analysis ('What are my viewing habits?')")
    print("- Learning paths ('What should I watch next?')")
    print("- Gap analysis ('What am I missing in my learning?')")
    print("\nRun: python3 youtube_history_agent.py")
    print("for interactive mode where you can ask your own questions!")


if __name__ == "__main__":
    demo_queries()