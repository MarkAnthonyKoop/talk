#!/usr/bin/env python3
"""
Simple verification script for the conversation intelligence system.
"""

import sys
import os
from pathlib import Path

def verify_file_structure():
    """Verify the file structure is correct."""
    print("Verifying file structure...")
    
    base_path = Path("/home/xx/talk_tries/code")
    required_files = [
        "autonomous_agents/conversation_intelligence.py",
        "autonomous_agents/codebase_knowledge_agent.py", 
        "tools/conversation_intelligence_tool.py",
        "demo_conversation_intelligence.py",
        "integrate_conversation_intelligence.py",
        "test_conversation_intelligence.py",
        "README_CONVERSATION_INTELLIGENCE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  ✗ {file_path}")
        return False
    
    print("✓ All required files present")
    return True

def verify_imports():
    """Verify that imports work correctly."""
    print("\nVerifying imports...")
    
    # Add paths for testing
    sys.path.insert(0, "/home/xx/talk_tries/code")
    sys.path.insert(0, "/home/xx/talk_tries/code/tools")
    
    try:
        # Test conversation intelligence import
        from autonomous_agents.conversation_intelligence import ConversationIntelligenceAgent
        print("  ✓ ConversationIntelligenceAgent import")
        
        # Test codebase knowledge import  
        from autonomous_agents.codebase_knowledge_agent import CodebaseKnowledgeAgent
        print("  ✓ CodebaseKnowledgeAgent import")
        
        # Test tool import
        from tools.conversation_intelligence_tool import ConversationIntelligenceTool
        print("  ✓ ConversationIntelligenceTool import")
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def verify_basic_instantiation():
    """Verify basic class instantiation."""
    print("\nVerifying basic instantiation...")
    
    sys.path.insert(0, "/home/xx/talk_tries/code")
    sys.path.insert(0, "/home/xx/talk_tries/code/tools")
    
    try:
        # Test sandbox creation
        from tools.conversation_intelligence_tool import SimpleSandbox
        sandbox = SimpleSandbox()
        print("  ✓ SimpleSandbox created")
        
        # Test conversation agent creation
        from autonomous_agents.conversation_intelligence import ConversationIntelligenceAgent
        conv_agent = ConversationIntelligenceAgent(
            name="test_conv", 
            capabilities=["test"], 
            sandbox=sandbox
        )
        print("  ✓ ConversationIntelligenceAgent created")
        
        # Test codebase agent creation
        from autonomous_agents.codebase_knowledge_agent import CodebaseKnowledgeAgent
        kb_agent = CodebaseKnowledgeAgent(
            name="test_kb",
            capabilities=["test"],
            sandbox=sandbox
        )
        print("  ✓ CodebaseKnowledgeAgent created")
        
        # Test tool creation
        from tools.conversation_intelligence_tool import ConversationIntelligenceTool
        tool = ConversationIntelligenceTool()
        print("  ✓ ConversationIntelligenceTool created")
        
        print("✓ All instantiations successful")
        return True
        
    except Exception as e:
        print(f"  ✗ Instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_codebase_access():
    """Verify codebase access works."""
    print("\nVerifying codebase access...")
    
    # Check if key directories exist
    paths_to_check = [
        "/home/xx/talk_tries",
        "/home/xx/talk_tries/code", 
        "/home/xx/code"
    ]
    
    accessible_paths = []
    for path in paths_to_check:
        if os.path.exists(path) and os.path.isdir(path):
            accessible_paths.append(path)
            print(f"  ✓ {path} accessible")
        else:
            print(f"  ✗ {path} not accessible")
    
    if accessible_paths:
        print(f"✓ {len(accessible_paths)} codebase paths accessible")
        return True
    else:
        print("✗ No codebase paths accessible")
        return False

def main():
    """Main verification function."""
    print("Conversation Intelligence System - Verification")
    print("=" * 50)
    
    tests = [
        ("File Structure", verify_file_structure),
        ("Import System", verify_imports),
        ("Basic Instantiation", verify_basic_instantiation),
        ("Codebase Access", verify_codebase_access)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Verification Results")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} verifications passed")
    
    if passed == total:
        print("\n🎉 System verification successful!")
        print("The conversation intelligence system is properly installed and ready to use.")
        
        print("\nNext steps:")
        print("1. Run the demo: python3 /home/xx/talk_tries/code/demo_conversation_intelligence.py")
        print("2. Test integration: python3 /home/xx/talk_tries/code/integrate_conversation_intelligence.py")
        print("3. Use in conversations: from tools.conversation_intelligence_tool import ConversationIntelligenceTool")
        
    elif passed > 0:
        print("\n⚠ Partial verification successful.")
        print("Some components may not work correctly.")
    else:
        print("\n❌ System verification failed.")
        print("Check installation and file permissions.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)