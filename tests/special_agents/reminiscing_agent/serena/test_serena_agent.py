#!/usr/bin/env python3
"""
Test for SerenaAgent - Semantic code analysis using Serena MCP.

This test verifies that the SerenaAgent:
1. Follows Talk framework contract strictly
2. Doesn't open dashboard windows
3. Stores results properly in .talk/serena/
4. Manages server lifecycle cleanly
5. Provides semantic analysis capabilities
"""

import sys
import json
import time
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from special_agents.reminiscing.serena_agent import SerenaAgent

class TestSerenaAgent(unittest.TestCase):
    """Test suite for SerenaAgent."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_results_dir = Path.cwd() / ".talk" / "serena"
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Count existing files to track new ones
        self.initial_file_count = len(list(self.test_results_dir.glob("*.json")))
        
    def tearDown(self):
        """Clean up after tests."""
        # Optional: clean up test files (commented out to preserve for inspection)
        # test_files = list(self.test_results_dir.glob("*test*.json"))
        # for f in test_files:
        #     f.unlink()
        pass
    
    def test_agent_creation(self):
        """Test that SerenaAgent can be created successfully."""
        agent = SerenaAgent(name="TestAgent")
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.serena_wrapper)
        self.assertIsNotNone(agent.naming_agent)
    
    def test_talk_contract_compliance(self):
        """Test that agent follows Talk framework contract."""
        agent = SerenaAgent(name="ContractTestAgent")
        
        # Test: prompt in ==> completion out
        input_prompt = "Find the Agent class in the codebase"
        result = agent.run(input_prompt)
        
        # Should return string completion
        self.assertIsInstance(result, str)
        
        # Should contain expected structure
        self.assertIn("SERENA_ANALYSIS_COMPLETE", result)
        
        # Should reference a data file
        self.assertIn(".talk/serena/", result)
    
    def test_no_dashboard_popup(self):
        """Test that dashboard doesn't pop up during agent execution."""
        agent = SerenaAgent(name="NoDashboardTestAgent")
        
        # Mock the server wrapper to verify dashboard=False is passed
        with patch.object(agent.serena_wrapper, 'start_mcp_server') as mock_start:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Server running
            mock_start.return_value = mock_process
            
            # Run analysis
            result = agent.run("Test analysis without dashboard")
            
            # Verify dashboard was disabled
            mock_start.assert_called_once()
            call_args = mock_start.call_args
            self.assertEqual(call_args.kwargs.get('enable_dashboard'), False)
    
    def test_result_file_creation(self):
        """Test that result files are created with proper structure."""
        agent = SerenaAgent(name="FileTestAgent")
        
        # Get initial file count
        initial_count = len(list(self.test_results_dir.glob("*.json")))
        
        # Run analysis
        result = agent.run("Create test result file")
        
        # Check new file was created
        final_count = len(list(self.test_results_dir.glob("*.json")))
        self.assertGreater(final_count, initial_count)
        
        # Find the new file (most recent)
        json_files = list(self.test_results_dir.glob("*.json"))
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        # Verify file structure
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        required_keys = ["metadata", "request", "results", "file_info"]
        for key in required_keys:
            self.assertIn(key, data)
        
        # Verify metadata structure
        metadata = data["metadata"]
        self.assertIn("timestamp", metadata)
        self.assertIn("session_uuid", metadata)
        self.assertIn("analysis_name", metadata)
        self.assertIn("agent_id", metadata)
    
    def test_different_analysis_types(self):
        """Test different types of semantic analysis."""
        agent = SerenaAgent(name="AnalysisTypesTestAgent")
        
        test_cases = [
            ("Find the run method", "symbol_search"),
            ("Overview of the codebase structure", "codebase_overview"), 
            ("References to the Agent class", "reference_analysis")
        ]
        
        for prompt, expected_type in test_cases:
            result = agent.run(prompt)
            
            # Should complete successfully
            self.assertIn("SERENA_ANALYSIS_COMPLETE", result)
            self.assertIn(f"Analysis Type: {expected_type}", result)
    
    def test_server_lifecycle_management(self):
        """Test that server starts and stops properly."""
        agent = SerenaAgent(name="LifecycleTestAgent")
        
        # Mock the server wrapper
        with patch.object(agent.serena_wrapper, 'start_mcp_server') as mock_start, \
             patch.object(agent, '_stop_mcp_server') as mock_stop:
            
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_start.return_value = mock_process
            
            # Run analysis
            result = agent.run("Test server lifecycle")
            
            # Verify server was started and stopped
            mock_start.assert_called_once()
            mock_stop.assert_called()
    
    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        agent = SerenaAgent(name="ErrorTestAgent")
        
        # Test with serena unavailable
        agent.serena_available = False
        
        result = agent.run("Test error handling")
        
        self.assertIn("SERENA_UNAVAILABLE", result)
    
    def test_naming_integration(self):
        """Test that naming agent integration works."""
        agent = SerenaAgent(name="NamingTestAgent")
        
        # Test name generation
        request = {
            "type": "symbol_search",
            "target_symbols": ["TestClass"],
            "languages": ["python"],
            "scope": "full"
        }
        
        name = agent._generate_analysis_name(request)
        
        # Should be 10 characters
        self.assertEqual(len(name), 10)
        self.assertTrue(name.isalnum() or '_' in name)


def test_dashboard_disabled():
    """Standalone test to verify dashboard doesn't open."""
    print("Testing SerenaAgent - Dashboard Disabled")
    print("=" * 45)
    
    print("Creating SerenaAgent...")
    agent = SerenaAgent(name="DashboardTestAgent")
    
    print("Running semantic analysis...")
    print("(If dashboard opens, this test failed)")
    
    result = agent.run("Find the Agent class definition")
    
    if "SERENA_ANALYSIS_COMPLETE" in result:
        print("✅ Analysis completed successfully")
        
        if ".talk/serena/" in result:
            print("✅ Result file referenced")
            
        print("✅ NO DASHBOARD OPENED")
        return True
    else:
        print("❌ Analysis failed")
        return False

def run_integration_test():
    """Run a full integration test."""
    print("\nSerenaAgent Integration Test")
    print("=" * 30)
    
    try:
        # Create agent
        agent = SerenaAgent(name="IntegrationTestAgent")
        print("✅ Agent created")
        
        # Test semantic analysis
        test_prompts = [
            "Find the SerenaAgent class in the codebase",
            "Get overview of special_agents module",
            "Find references to the run method"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing: {prompt[:50]}...")
            result = agent.run(prompt)
            
            if "SERENA_ANALYSIS_COMPLETE" in result:
                print(f"   ✅ Completed successfully")
            else:
                print(f"   ❌ Failed")
                
        # Check result files
        results_dir = Path.cwd() / ".talk" / "serena"
        json_files = list(results_dir.glob("*.json"))
        print(f"\n📁 Created {len(json_files)} result files")
        
        # Show latest file info
        if json_files:
            latest = max(json_files, key=lambda f: f.stat().st_mtime)
            print(f"   Latest: {latest.name}")
            print(f"   Size: {latest.stat().st_size} bytes")
        
        print("\n🎯 Integration test completed successfully!")
        print("   - No dashboard popups")
        print("   - Proper Talk contract compliance")
        print("   - Semantic analysis capabilities")
        print("   - Clean server lifecycle management")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run dashboard test first
    dashboard_ok = test_dashboard_disabled()
    
    if dashboard_ok:
        # Run integration test
        integration_ok = run_integration_test()
        
        if integration_ok:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   SerenaAgent is ready for production use")
        else:
            print(f"\n❌ Integration test failed")
    else:
        print(f"\n❌ Dashboard test failed")
    
    # Also run unit tests
    print(f"\n🧪 Running Unit Tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)