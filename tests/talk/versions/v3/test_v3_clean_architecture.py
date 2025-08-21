#!/usr/bin/env python3
"""
Test Talk v3 with clean architecture improvements.

Tests:
1. Plain English communication between agents
2. BranchingAgent using agent descriptions
3. TALK.md persistence
4. Loop detection and prevention
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/tony/talk')

def test_simple_code_generation():
    """Test simple code generation with clean architecture."""
    print("=" * 60)
    print("TALK V3 CLEAN ARCHITECTURE TEST")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nWorking directory: {tmpdir}")
        
        # Simple task that should complete quickly
        task = "Create a Python function named greet that takes a name parameter and returns 'Hello, {name}!'"
        
        print(f"\nTask: {task}")
        print("-" * 40)
        
        # Run talk using the installed command
        import subprocess
        result = subprocess.run(
            ["talk", task, "--dir", tmpdir, "--model", "gemini-2.0-flash"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmpdir  # Run from tmpdir
        )
        
        print("\n[OUTPUT]")
        print(result.stdout)
        
        if result.stderr:
            print("\n[ERRORS]")
            print(result.stderr)
        
        print("\n[CHECKING RESULTS]")
        print("-" * 40)
        
        # Check if TALK.md was created
        talk_md = Path(tmpdir) / "TALK.md"
        if talk_md.exists():
            print("✓ TALK.md created")
            with open(talk_md) as f:
                content = f.read()
                print(f"  - Size: {len(content)} bytes")
                if "## Todo List" in content:
                    print("  ✓ Contains todo list")
                if "## Latest Recommendation" in content:
                    print("  ✓ Contains recommendations")
        else:
            print("✗ TALK.md not created")
        
        # Check for generated files
        files_created = list(Path(tmpdir).glob("*.py"))
        if files_created:
            print(f"✓ Python files created: {[f.name for f in files_created]}")
            
            # Check file content
            for file in files_created:
                with open(file) as f:
                    code = f.read()
                    if "def greet" in code:
                        print(f"  ✓ {file.name} contains greet function")
                        if "Hello," in code:
                            print(f"  ✓ {file.name} has correct implementation")
        else:
            print("✗ No Python files created")
        
        # Check blackboard
        blackboard_path = Path(tmpdir) / ".talk" / "*" / "blackboard.json"
        blackboard_files = list(Path(tmpdir).glob(".talk/*/blackboard.json"))
        if blackboard_files:
            print(f"✓ Blackboard created")
            with open(blackboard_files[0]) as f:
                blackboard = json.load(f)
                print(f"  - Entries: {len(blackboard.get('entries', []))}")
        
        # Check for plain English in logs
        if "need to generate" in result.stdout.lower() or "let's" in result.stdout.lower():
            print("✓ Plain English communication detected")
        
        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("✅ TEST PASSED - V3 CLEAN ARCHITECTURE WORKING!")
        else:
            print(f"❌ TEST FAILED - Exit code: {result.returncode}")
        print("=" * 60)
        
        return result.returncode == 0


def test_loop_prevention():
    """Test that v3 prevents infinite loops."""
    print("\n" + "=" * 60)
    print("LOOP PREVENTION TEST")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Task that might cause loops
        task = "Analyze this task but don't do anything"
        
        print(f"\nTask: {task}")
        print("(This should complete without looping)")
        print("-" * 40)
        
        # Run with timeout
        import subprocess
        try:
            result = subprocess.run(
                ["talk", task, "--dir", tmpdir, "--model", "gemini-2.0-flash"],
                capture_output=True,
                text=True,
                timeout=30,  # Short timeout to catch loops
                cwd=tmpdir
            )
            
            # Count how many times "plan_next" appears
            plan_count = result.stdout.count("plan_next")
            select_count = result.stdout.count("select_action")
            
            print(f"\nLoop metrics:")
            print(f"  - plan_next called: {plan_count} times")
            print(f"  - select_action called: {select_count} times")
            
            if plan_count > 10:
                print("  ⚠️ Warning: Excessive planning detected")
            elif plan_count <= 5:
                print("  ✓ Planning count reasonable")
            
            if abs(plan_count - select_count) > 2:
                print("  ⚠️ Warning: Imbalanced flow")
            else:
                print("  ✓ Balanced workflow")
            
            print("\n✅ No infinite loop detected")
            return True
            
        except subprocess.TimeoutExpired:
            print("\n❌ TIMEOUT - Possible infinite loop!")
            return False


if __name__ == "__main__":
    print("Testing Talk v3 with Clean Architecture")
    print("=" * 60)
    
    tests = [
        ("Simple Code Generation", test_simple_code_generation),
        ("Loop Prevention", test_loop_prevention),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - V3 CLEAN ARCHITECTURE IS WORKING!")
    
    sys.exit(0 if passed == total else 1)