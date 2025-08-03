#!/usr/bin/env python3
"""
Test Script for Conversation Intelligence System

Quick test to verify the conversation intelligence and codebase knowledge
system is working correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.conversation_intelligence_tool import ConversationIntelligenceTool

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

async def test_tool_initialization():
    """Test basic tool initialization."""
    print("Testing tool initialization...")
    
    tool = ConversationIntelligenceTool()
    
    # Check initial status
    status = tool.get_tool_status()
    print(f"  Initial status: {status['initialized']}")
    
    # Initialize
    success = await tool.initialize()
    
    if success:
        print("  ✓ Tool initialized successfully")
        
        # Check status after initialization
        status = tool.get_tool_status()
        print(f"  Knowledge agent active: {status['knowledge_agent_active']}")
        print(f"  Conversation agent active: {status['conversation_agent_active']}")
        print(f"  Capabilities: {len(status['capabilities'])}")
        
        return True
    else:
        print("  ✗ Tool initialization failed")
        return False

async def test_codebase_overview():
    """Test codebase overview functionality."""
    print("\nTesting codebase overview...")
    
    tool = ConversationIntelligenceTool()
    
    overview = await tool.get_codebase_overview(detail_level="basic")
    
    if overview.get("status") == "completed":
        arch_data = overview.get('architecture_overview', {})
        summary = arch_data.get('system_summary', {})
        
        file_count = summary.get('total_files', 0)
        component_count = summary.get('total_components', 0)
        
        print(f"  ✓ Analysis completed")
        print(f"  Files analyzed: {file_count}")
        print(f"  Components found: {component_count}")
        
        if file_count > 0:
            print(f"  Architecture: {summary.get('architectural_style', 'Unknown')}")
            technologies = arch_data.get('key_technologies', [])
            if technologies:
                print(f"  Technologies: {', '.join(technologies[:3])}")
            return True
        else:
            print("  ⚠ No files found - check codebase paths")
            return False
    else:
        print(f"  ✗ Overview failed: {overview.get('message', 'Unknown error')}")
        return False

async def test_conversation_analysis():
    """Test conversation analysis functionality."""
    print("\nTesting conversation analysis...")
    
    tool = ConversationIntelligenceTool()
    
    analysis = await tool.analyze_recent_conversations(
        time_window_hours=24,
        analysis_type="basic"
    )
    
    if analysis.get("status") == "completed":
        conv_count = analysis.get('conversations_analyzed', 0)
        results = analysis.get('results', {})
        
        print(f"  ✓ Analysis completed")
        print(f"  Conversations analyzed: {conv_count}")
        
        if conv_count > 0:
            patterns = results.get('patterns', [])
            print(f"  Patterns detected: {len(patterns)}")
            
            topics = results.get('key_topics', [])
            if topics:
                print(f"  Key topics: {', '.join(topics[:3])}")
            
            return True
        else:
            print("  ⚠ No recent conversations found")
            return True  # Not necessarily an error
    else:
        print(f"  ✗ Analysis failed: {analysis.get('message', 'Unknown error')}")
        return False

async def test_contextual_assistance():
    """Test contextual assistance functionality."""
    print("\nTesting contextual assistance...")
    
    tool = ConversationIntelligenceTool()
    
    assistance = await tool.get_contextual_assistance(
        current_task="test the conversation intelligence system",
        keywords=["test", "conversation", "intelligence"],
        assistance_type="code_development"
    )
    
    if assistance.get("status") == "completed":
        recommendations = assistance.get('recommendations', [])
        codebase_context = assistance.get('codebase_context', {})
        
        print(f"  ✓ Assistance provided")
        print(f"  Recommendations: {len(recommendations)}")
        
        related_files = codebase_context.get('related_files', [])
        if related_files:
            print(f"  Related files: {len(related_files)}")
        
        confidence = assistance.get('confidence_score', 0) * 100
        print(f"  Confidence: {confidence:.0f}%")
        
        return True
    else:
        print(f"  ✗ Assistance failed: {assistance.get('message', 'Unknown error')}")
        return False

async def test_knowledge_query():
    """Test knowledge base querying."""
    print("\nTesting knowledge queries...")
    
    tool = ConversationIntelligenceTool()
    
    # Test file search
    query_result = await tool.query_knowledge_base(
        query="agent",
        query_type="file_search"
    )
    
    if query_result.get("status") == "completed":
        results = query_result.get('results', [])
        print(f"  ✓ Query completed")
        print(f"  Results found: {len(results)}")
        
        if results:
            # Show a sample result
            sample = results[0]
            file_path = sample.get('path', 'unknown')
            file_name = Path(file_path).name
            print(f"  Sample result: {file_name}")
        
        return True
    else:
        print(f"  ✗ Query failed: {query_result.get('message', 'Unknown error')}")
        return False

async def run_tests():
    """Run all tests."""
    print("Conversation Intelligence System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Tool Initialization", test_tool_initialization),
        ("Codebase Overview", test_codebase_overview),
        ("Conversation Analysis", test_conversation_analysis),
        ("Contextual Assistance", test_contextual_assistance),
        ("Knowledge Queries", test_knowledge_query)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            results.append((test_name, False))
            log.error(f"Test {test_name} failed", exc_info=True)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    elif passed > 0:
        print("⚠ Some tests passed. The system has basic functionality.")
    else:
        print("❌ All tests failed. Check system configuration.")
    
    return passed == total

async def main():
    """Main test function."""
    try:
        success = await run_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        log.error("Unexpected test error", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)