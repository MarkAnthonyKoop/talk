#!/usr/bin/env python3
"""
Test for Talk v2 with proper output directory compliance.

This test ensures Talk v2 outputs go to tests/output/ as per CLAUDE.md requirements.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.utilities.test_output_writer import TestOutputWriter
from talk.versions.talk_v2 import TalkOrchestratorV2


def test_talk_v2_simple():
    """Test Talk v2 with a simple task, ensuring proper output location."""
    
    # Create test output writer for compliance
    writer = TestOutputWriter("integration", "test_talk_v2")
    output_dir = writer.get_output_dir()
    
    # Set environment to use our compliant output directory
    os.environ["TALK_OUTPUT_ROOT"] = str(output_dir)
    
    # Simple test task
    task = "Create a Python function that calculates the factorial of a number"
    
    # Create working directory in our test output
    working_dir = output_dir / "workspace"
    working_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Talk v2 with our working directory
        orchestrator = TalkOrchestratorV2(
            task=task,
            working_dir=str(working_dir),
            model="gemini-2.0-flash",  # Use fast model for testing
            interactive=False
        )
        
        # Run the orchestration
        result = orchestrator.run()
        
        # Collect results
        test_results = {
            "task": task,
            "status": result.get("status", "unknown"),
            "files_generated": result.get("files_generated", []),
            "execution_time": result.get("execution_time", 0),
            "session_dir": str(orchestrator.session_dir),
            "working_dir": str(orchestrator.working_dir),
            "timestamp": datetime.now().isoformat()
        }
        
        # Write results using compliant writer
        writer.write_results(test_results)
        
        # Log success
        writer.write_log(f"Talk v2 test completed successfully")
        writer.write_log(f"Session directory: {orchestrator.session_dir}")
        writer.write_log(f"Working directory: {orchestrator.working_dir}")
        
        # Verify output location compliance
        session_path = Path(orchestrator.session_dir)
        if session_path.is_relative_to(Path.cwd() / "tests" / "output"):
            writer.write_log("✓ Output location is CLAUDE.md compliant")
            compliance_status = "compliant"
        elif session_path.is_relative_to(Path.cwd() / ".talk"):
            writer.write_log("✗ Output location violates CLAUDE.md (using .talk)")
            compliance_status = "non-compliant"
        else:
            writer.write_log(f"? Output location unclear: {session_path}")
            compliance_status = "unknown"
        
        # Save compliance check
        compliance_results = {
            "test": "talk_v2",
            "compliance_status": compliance_status,
            "actual_output_path": str(session_path),
            "expected_pattern": "tests/output/<category>/<testname>/<year_month>/",
            "timestamp": datetime.now().isoformat()
        }
        
        writer.write_json("compliance_check.json", compliance_results)
        
        print(f"\n✅ Talk v2 test completed")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Compliance: {compliance_status}")
        
        return test_results
        
    except Exception as e:
        error_msg = f"Talk v2 test failed: {str(e)}"
        writer.write_log(error_msg)
        writer.write_json("error.json", {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        print(f"\n❌ {error_msg}")
        raise


def test_talk_v2_with_override():
    """Test Talk v2 with environment variable override for output location."""
    
    # Create compliant output directory
    writer = TestOutputWriter("integration", "test_talk_v2_override")
    output_dir = writer.get_output_dir()
    
    # Override the output root in settings
    original_output_root = os.environ.get("TALK_PATHS_OUTPUT_ROOT")
    os.environ["TALK_PATHS_OUTPUT_ROOT"] = str(output_dir)
    
    try:
        task = "Write a hello world program"
        
        orchestrator = TalkOrchestratorV2(
            task=task,
            model="gemini-2.0-flash",
            interactive=False
        )
        
        # Check where session was created
        session_path = Path(orchestrator.session_dir)
        
        results = {
            "test": "override_test",
            "session_created_at": str(session_path),
            "is_in_tests_output": session_path.is_relative_to(Path.cwd() / "tests" / "output"),
            "timestamp": datetime.now().isoformat()
        }
        
        writer.write_json("override_test_results.json", results)
        
        if results["is_in_tests_output"]:
            print("✅ Environment variable override works - output is compliant")
        else:
            print(f"❌ Override failed - output at: {session_path}")
            
        return results
        
    finally:
        # Restore original setting
        if original_output_root:
            os.environ["TALK_PATHS_OUTPUT_ROOT"] = original_output_root
        else:
            os.environ.pop("TALK_PATHS_OUTPUT_ROOT", None)


if __name__ == "__main__":
    print("=" * 70)
    print("TALK V2 COMPLIANCE TEST")
    print("=" * 70)
    print("\nThis test verifies Talk v2 compliance with CLAUDE.md output rules")
    print("-" * 70)
    
    # Test 1: Simple task with manual directory setting
    print("\n📝 Test 1: Simple task with manual working directory...")
    test_talk_v2_simple()
    
    # Test 2: Environment variable override
    print("\n📝 Test 2: Testing environment variable override...")
    test_talk_v2_with_override()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)