#!/usr/bin/env python3
"""
Simple test script for Claude Sonnet API.
Tests both direct API call and through Talk framework.
"""

import os
import sys
import anthropic
from datetime import datetime

def test_direct_api():
    """Test Claude API directly using anthropic library."""
    print("\n" + "="*60)
    print("DIRECT API TEST")
    print("="*60)
    
    api_key = os.getenv("CLAUDE_API_TOKEN")
    print("api key is ",api_key)
    if not api_key:
        print("❌ CLAUDE_API_TOKEN not set")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("✓ Client initialized")
        
        print("\nSending request to Claude 3.5 Sonnet...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": "Say hello and tell me the current date"
            }]
        )
        
        print("✓ Response received!")
        print(f"\nClaude says: {response.content[0].text}")
        print(f"\nUsage: {response.usage.input_tokens} input + {response.usage.output_tokens} output tokens")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_talk_framework():
    """Test Claude through Talk framework settings."""
    print("\n" + "="*60)
    print("TALK FRAMEWORK TEST")
    print("="*60)
    
    try:
        # Add parent directories to path to import Talk modules
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        
        # Set model to trigger provider switch
        os.environ["TALK_FORCE_MODEL"] = "claude-3-5-sonnet-20241022"
        
        from agent.settings import Settings
        from agent.llm_backends import get_backend
        from agent.messages import Message, Role
        
        print("✓ Imports successful")
        
        # Resolve settings
        settings = Settings.resolve()
        print(f"✓ Provider: {settings.llm.provider}")
        print(f"✓ Model: {settings.provider.anthropic.model_name}")
        
        # Get provider settings and create backend
        provider_settings = settings.get_provider_settings()
        provider_dict = provider_settings.model_dump(mode="python")
        provider_dict["_provider"] = settings.llm.provider
        
        backend = get_backend(provider_dict)
        print(f"✓ Backend: {backend.__class__.__name__}")
        
        # Test completion
        messages = [
            Message(role=Role.user, content="Say hello from Talk framework")
        ]
        
        print("\nSending request through Talk backend...")
        response = backend.complete(messages)
        print(f"✓ Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curl_command():
    """Show equivalent curl command for testing."""
    print("\n" + "="*60)
    print("EQUIVALENT CURL COMMAND")
    print("="*60)
    
    api_key = os.getenv("CLAUDE_API_TOKEN", "your-api-key-here")
    
    curl_cmd = f'''curl https://api.anthropic.com/v1/messages \\
  -H "content-type: application/json" \\
  -H "x-api-key: {api_key}" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [
      {{"role": "user", "content": "Say hello"}}
    ]
  }}'
'''
    
    print(curl_cmd)
    print("\nNote: Replace 'your-api-key-here' with actual key if not set in env")

if __name__ == "__main__":
    print(f"Testing Claude Sonnet API - {datetime.now()}")
    
    # Run tests
    direct_ok = test_direct_api()
    framework_ok = test_talk_framework()
    test_curl_command()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Direct API Test: {'✓ PASSED' if direct_ok else '❌ FAILED'}")
    print(f"Talk Framework Test: {'✓ PASSED' if framework_ok else '❌ FAILED'}")
    
    sys.exit(0 if direct_ok and framework_ok else 1)
