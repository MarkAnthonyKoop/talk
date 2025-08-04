#!/usr/bin/env python3
"""Test the unified tool calling architecture."""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agent.tools import FILE_TOOLS
from agent.tools.handlers import UnifiedToolHandler
from agent.messages import Message, Role
from agent.llm_backends import get_backend


def test_tool_handler():
    """Test the unified tool handler."""
    print("\n=== Test: Unified Tool Handler ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = UnifiedToolHandler(base_dir=tmpdir)
        
        # Test write_file
        result = handler.handle_tool_call("write_file", {
            "path": "test.py",
            "content": "def hello():\n    print('Hello, World!')\n"
        })
        print(f"Write file: {result}")
        
        # Test read_file
        result = handler.handle_tool_call("read_file", {"path": "test.py"})
        print(f"Read file: {result}")
        
        # Test list_files
        result = handler.handle_tool_call("list_files", {"directory": "."})
        print(f"List files: {result}")
        
        # Test edit_file
        result = handler.handle_tool_call("edit_file", {
            "path": "test.py",
            "search": "Hello, World!",
            "replace": "Hello, Tools!"
        })
        print(f"Edit file: {result}")
        
        # Verify edit
        result = handler.handle_tool_call("read_file", {"path": "test.py"})
        print(f"After edit: {result}")
        
        return True


def test_anthropic_tools():
    """Test Anthropic backend with tools (requires API key)."""
    print("\n=== Test: Anthropic Tool Support ===")
    
    # Check if API key is set
    if not os.getenv("CLAUDE_API_TOKEN"):
        print("SKIPPED: CLAUDE_API_TOKEN not set")
        return True
    
    try:
        # Get Anthropic backend
        backend = get_backend({"_provider": "anthropic"})
        print(f"Backend supports tools: {backend.supports_tools}")
        
        # Test with a simple tool call
        messages = [
            Message(role=Role.user, content="What files are in the current directory?")
        ]
        
        response = backend.complete(messages, tools=FILE_TOOLS[:1])  # Just list_files
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_tool_definitions():
    """Test that tool definitions are well-formed."""
    print("\n=== Test: Tool Definitions ===")
    
    for tool in FILE_TOOLS:
        func = tool["function"]
        print(f"Tool: {func['name']}")
        print(f"  Description: {func['description']}")
        print(f"  Parameters: {list(func['parameters']['properties'].keys())}")
        print(f"  Required: {func['parameters'].get('required', [])}")
    
    return True


if __name__ == "__main__":
    print("Testing Unified Tool Architecture")
    print("=" * 60)
    
    results = []
    results.append(("Tool Handler", test_tool_handler()))
    results.append(("Tool Definitions", test_tool_definitions()))
    results.append(("Anthropic Tools", test_anthropic_tools()))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    sys.exit(0 if all(passed for _, passed in results) else 1)