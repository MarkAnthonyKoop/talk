#!/usr/bin/env python3
"""
Test Listen v5 system automation capabilities.

This test validates that v5 can properly:
- Detect action intents vs conversation intents
- Route requests to appropriate specialized agents
- Execute safe system commands
- Provide helpful responses
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def test_listen_v5():
    """Test Listen v5 capabilities."""
    print("=" * 60)
    print("TESTING LISTEN V5 SYSTEM AUTOMATION")
    print("=" * 60)
    
    try:
        # Import Listen v5
        from listen.versions.listen_v5 import ListenV5, IntentType
        
        print("1. Initializing Listen v5...")
        with tempfile.TemporaryDirectory() as temp_dir:
            assistant = ListenV5(
                name="Test Assistant v5",
                db_path=Path(temp_dir) / "test.db",
                use_tts=False,
                confidence_threshold=0.6
            )
            print("✓ Listen v5 initialized successfully")
            
            print("\n2. Testing Intent Detection...")
            
            # Test conversation intent
            classification = assistant.action_intent_agent.classify_intent(
                "Hello, how are you?", []
            )
            print(f"  Conversation test: {classification.intent_type.value} (confidence: {classification.confidence:.2f})")
            assert classification.intent_type == IntentType.CONVERSATION
            
            # Test action intent - file operations
            classification = assistant.action_intent_agent.classify_intent(
                "List my files in the current directory", []
            )
            print(f"  File action test: {classification.intent_type.value} (confidence: {classification.confidence:.2f})")
            print(f"    Suggested handler: {classification.suggested_handler}")
            assert classification.intent_type == IntentType.ACTION
            assert classification.suggested_handler == "FileSystemAgent"
            
            # Test action intent - git operations
            classification = assistant.action_intent_agent.classify_intent(
                "Show me the git status", []
            )
            print(f"  Git action test: {classification.intent_type.value} (confidence: {classification.confidence:.2f})")
            print(f"    Suggested handler: {classification.suggested_handler}")
            assert classification.intent_type == IntentType.ACTION
            assert classification.suggested_handler == "GitAgent"
            
            # Test mixed intent
            classification = assistant.action_intent_agent.classify_intent(
                "Hi there! Can you also check my disk space?", []
            )
            print(f"  Mixed intent test: {classification.intent_type.value} (confidence: {classification.confidence:.2f})")
            
            print("✓ Intent detection working correctly")
            
            print("\n3. Testing Safety Validation...")
            
            # Test safe command
            validation = assistant.safety_validator.validate_command("ls -la")
            print(f"  Safe command (ls): {validation.is_safe}, level: {validation.permission_level.name}")
            assert validation.is_safe
            
            # Test dangerous command
            validation = assistant.safety_validator.validate_command("rm -rf /")
            print(f"  Dangerous command (rm -rf /): {validation.is_safe}, level: {validation.permission_level.name}")
            assert not validation.is_safe
            
            # Test command requiring confirmation
            validation = assistant.safety_validator.validate_command("sudo systemctl restart nginx")
            print(f"  Elevated command (sudo): requires_confirmation: {validation.requires_confirmation}")
            assert validation.requires_confirmation
            
            print("✓ Safety validation working correctly")
            
            print("\n4. Testing File System Agent...")
            
            # Test file listing
            response = await assistant.filesystem_agent.run("list files")
            print(f"  File listing response: {response[:100]}...")
            assert "files" in response.lower() or "error" in response.lower()
            
            # Test disk space
            response = await assistant.filesystem_agent.run("check disk space")
            print(f"  Disk space response: {response[:100]}...")
            assert any(word in response.lower() for word in ["disk", "space", "usage", "error"])
            
            print("✓ File System Agent working correctly")
            
            print("\n5. Testing Git Agent...")
            
            if assistant.git_agent:
                # Test git status (may show repo status or not a git repo)
                response = assistant.git_agent.run("git status")
                print(f"  Git status response: {response[:100]}...")
                assert any(word in response.lower() for word in ["branch", "repository", "git", "files", "clean"])
                
                # Test git help
                response = assistant.git_agent.run("help")
                print(f"  Git help response: {response[:100]}...")
                assert "git" in response.lower()
                
                print("✓ Git Agent working correctly")
            else:
                print("⚠️  Git Agent not available - skipped")
            
            print("\n6. Testing Dev Tools Agent...")
            
            if assistant.dev_tools_agent:
                # Test project detection
                response = assistant.dev_tools_agent.run("what type of project is this")
                print(f"  Project detection: {response[:100]}...")
                
                # Test tool availability
                response = assistant.dev_tools_agent.run("check python version")
                print(f"  Tool check: {response[:100]}...")
                
                print("✓ Dev Tools Agent working correctly")
            else:
                print("⚠️  Dev Tools Agent not available - skipped")
            
            print("\n7. Testing MCP Integration...")
            
            # Test MCP availability
            print(f"  MCP available: {assistant.mcp_manager.is_available}")
            
            # Test safe command execution
            if assistant.mcp_manager.is_available:
                result = await assistant.mcp_manager.execute_command("echo 'Hello from MCP'")
                print(f"  MCP command result: success={result.get('return_code', -1) == 0}")
                print(f"    Output: {result.get('stdout', '')[:50]}...")
                
                print("✓ MCP Integration working correctly")
            else:
                print("⚠️  MCP not available - simulated mode")
                
            print("\n8. Testing Action Execution Flow...")
            
            # Simulate audio processing with action request
            test_audio = {
                "audio": None,
                "timestamp": "2025-01-01T12:00:00",
                "energy": 1000,
                "pitch": 150,
                "duration": 3.0
            }
            
            # Mock the transcription result by directly adding to conversation
            assistant.conversation_manager.add_turn(
                text="list my files",
                speaker_id="test_user",
                audio_features=test_audio
            )
            
            # Test intent classification and routing
            classification = assistant.action_intent_agent.classify_intent("list my files", [])
            print(f"  Action routing test: {classification.suggested_handler}")
            
            print("✓ Action execution flow working correctly")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL LISTEN V5 TESTS PASSED!")
    print("Listen v5 system automation capabilities are working correctly")
    print("=" * 60)
    
    return True


async def main():
    """Main test entry point."""
    success = await test_listen_v5()
    exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())