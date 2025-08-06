#!/usr/bin/env python3
"""
Example usage of SerenaAgent within Talk framework.

This demonstrates how to use the SerenaAgent for semantic code analysis
while following the Talk framework patterns.
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from special_agents.reminiscing.serena_agent import SerenaAgent

def example_usage():
    """Example usage of SerenaAgent following Talk patterns."""
    print("SerenaAgent Usage Example")
    print("=" * 30)
    
    # Create the agent (follows Talk pattern)
    agent = SerenaAgent(name="SemanticAnalyzer")
    
    # Example 1: Find specific symbols
    print("\n1. Symbol Search:")
    result = agent.run("Find the Agent class definition in the codebase")
    print("Result:", result[:200] + "...")
    
    # Example 2: Analyze codebase structure  
    print("\n2. Codebase Overview:")
    result = agent.run("Provide overview of the special_agents module structure")
    print("Result:", result[:200] + "...")
    
    # Example 3: Reference analysis
    print("\n3. Reference Analysis:")
    result = agent.run("Find all references to the run method across the project")
    print("Result:", result[:200] + "...")
    
    print("\n✨ Key Benefits:")
    print("- No dashboard interference (server runs headless)")
    print("- Results stored in structured .talk/serena/ files")
    print("- Clean server lifecycle (auto start/stop)")
    print("- Talk contract compliance (prompt in ==> completion out)")
    print("- Semantic search vs reading entire files")
    print("- LSP-based understanding across 13+ languages")

def show_result_files():
    """Show what result files were created."""
    results_dir = Path.cwd() / ".talk" / "serena"
    
    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))
        print(f"\n📁 Result Files Created ({len(json_files)}):")
        
        for f in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size = f.stat().st_size
            print(f"- {f.name} ({size} bytes)")
    else:
        print("\n📁 No result files created yet")

if __name__ == "__main__":
    example_usage()
    show_result_files()
    
    print(f"\n🎯 This demonstrates the solution to the YouTube video's")
    print(f"   '30% potential' problem:")
    print(f"   - Serena provides semantic search via LSP")
    print(f"   - Talk gets focused context, not entire files")
    print(f"   - LLM performance improves dramatically")
    print(f"   - Token usage is optimized")
    print(f"   - Both frameworks coexist cleanly (pip + UV)")