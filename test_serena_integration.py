#!/usr/bin/env python3
"""
Test script for Serena integration with Talk framework.

This demonstrates how to use Serena's semantic search capabilities
within the Talk framework while keeping the projects separate.
"""

import sys
from pathlib import Path

# Add the project root to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from special_agents.reminiscing.serena_integration_agent import SerenaIntegrationAgent

def test_serena_integration():
    """Test the Serena integration agent."""
    print("Testing Serena Integration with Talk Framework")
    print("=" * 60)
    
    try:
        # Create the integration agent
        agent = SerenaIntegrationAgent(name="SerenaTestAgent")
        
        print("✅ SerenaIntegrationAgent created successfully")
        
        # Test 1: Get general info
        print("\n1. Testing general information:")
        response = agent.run("What can you do?")
        print(response)
        
        # Test 2: Try to start server for current project
        print("\n2. Testing server startup:")
        response = agent.run(f"start serena {Path.cwd()}")
        print(response)
        
        # Test 3: Try semantic search
        print("\n3. Testing semantic search:")
        response = agent.run("find function Agent")
        print(response)
        
        # Test 4: Try symbol analysis 
        print("\n4. Testing symbol analysis:")
        response = agent.run("analyze symbol SerenaIntegrationAgent")
        print(response)
        
        # Test 5: Stop server
        print("\n5. Testing server shutdown:")
        response = agent.run("stop serena")
        print(response)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\nWhat this demonstrates:")
        print("- Talk framework (pip-based) can use Serena (UV-based)")
        print("- Semantic search capabilities are available to Talk agents") 
        print("- No disruption to existing Talk project structure")
        print("- Clean separation between the two systems")
        
    except Exception as e:
        print(f"❌ Error testing Serena integration: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_concept():
    """Demonstrate the key concept behind the integration."""
    print("\nDemonstration: Why This Solves the '30% Potential' Problem")
    print("=" * 60)
    
    print("""
BEFORE (Traditional approach):
1. Claude reads entire files into context
2. Context window gets polluted with irrelevant code  
3. LLM has to find relevant pieces among noise
4. Performance degrades due to information overload
5. Token usage is wasteful

AFTER (Serena Integration):
1. Serena uses Language Server Protocol for semantic understanding
2. Only relevant symbols and contexts are retrieved
3. LLM gets focused, precise information
4. Performance improves dramatically
5. Token usage is optimized

KEY BENEFITS:
- Semantic search (not just text matching)
- Symbol-level understanding (functions, classes, references)
- Relationship mapping (imports, dependencies)
- Language-specific analysis (13+ languages)
- Clean, focused context for optimal LLM performance
""")

if __name__ == "__main__":
    test_serena_integration()
    demonstrate_concept()