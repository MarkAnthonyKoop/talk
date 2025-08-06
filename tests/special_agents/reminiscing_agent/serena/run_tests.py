#!/usr/bin/env python3
"""
Test runner for SerenaAgent tests.

Quick way to verify SerenaAgent functionality without dashboard popups.
"""

import sys
import subprocess
from pathlib import Path

def run_serena_tests():
    """Run all SerenaAgent tests."""
    test_file = Path(__file__).parent / "test_serena_agent.py"
    
    print("🧪 Running SerenaAgent Tests")
    print("=" * 40)
    print("Testing:")
    print("- Dashboard disabled (no popups)")
    print("- Talk contract compliance") 
    print("- Server lifecycle management")
    print("- Result file storage")
    print("- Semantic analysis capabilities")
    print()
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, str(test_file)], 
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent.parent
        )
        
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            print("\nSerenaAgent is ready for:")
            print("- Integration with Talk framework")
            print("- Semantic code analysis via Serena MCP")
            print("- No dashboard interference")
            print("- Production use")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_serena_tests()
    sys.exit(0 if success else 1)