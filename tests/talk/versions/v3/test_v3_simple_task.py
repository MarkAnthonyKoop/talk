#!/usr/bin/env python3
"""
Simple task test for Talk v3 clean architecture.
"""

import sys
import json
import tempfile
from pathlib import Path
import subprocess

def test_simple_task():
    """Test a very simple task with v3."""
    print("=" * 60)
    print("TALK V3 SIMPLE TASK TEST")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working directory: {tmpdir}")
        
        # Very simple task
        task = "Create a file named hello.py with a print statement that says hello world"
        
        print(f"Task: {task}")
        print("-" * 40)
        
        # Run talk with short timeout
        try:
            result = subprocess.run(
                ["talk", task, "--dir", tmpdir, "--model", "gemini-2.0-flash"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir
            )
            
            print(f"Exit code: {result.returncode}")
            
            # Show last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("\nLast 10 lines of output:")
                for line in lines[-10:]:
                    print(f"  {line}")
            
            # Check for created file
            hello_file = Path(tmpdir) / "hello.py"
            if hello_file.exists():
                print("\n✓ hello.py created")
                with open(hello_file) as f:
                    content = f.read()
                    print(f"Content:\n{content}")
                    if "print" in content and "hello" in content.lower():
                        print("✓ Contains print statement with hello")
                        return True
            else:
                print("\n✗ hello.py not created")
                # Check scratch directory
                scratch_dir = Path(tmpdir) / ".talk_scratch"
                if scratch_dir.exists():
                    scratch_files = list(scratch_dir.glob("*"))
                    print(f"Scratch files: {[f.name for f in scratch_files]}")
                
            # Check TALK.md
            talk_md = Path(tmpdir) / "TALK.md"
            if talk_md.exists():
                print("✓ TALK.md exists")
                with open(talk_md) as f:
                    print(f"TALK.md preview:\n{f.read()[:300]}...")
            
            return hello_file.exists()
            
        except subprocess.TimeoutExpired:
            print("⏱️ Test timed out (30s)")
            return False

if __name__ == "__main__":
    try:
        success = test_simple_task()
        print("\n" + "=" * 60)
        if success:
            print("✅ TEST PASSED - File created successfully")
        else:
            print("❌ TEST FAILED - File not created")
        print("=" * 60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)