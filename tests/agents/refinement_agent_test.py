#!/usr/bin/env python3
"""
Tests for the RefinementAgent.

This test suite verifies that the RefinementAgent correctly manages
the iterative code development cycle.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from special_agents.refinement_agent import RefinementAgent, RefinementStatus, RefinementResult


class TestRefinementAgent(unittest.TestCase):
    """Test cases for RefinementAgent."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.agent = RefinementAgent(base_dir=self.temp_dir, max_iterations=3)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.base_dir, self.temp_dir)
        self.assertEqual(self.agent.max_iterations, 3)
        self.assertIsNotNone(self.agent.code_agent)
        self.assertIsNotNone(self.agent.file_agent)
        self.assertIsNotNone(self.agent.test_agent)
    
    @patch('special_agents.refinement_agent.CodeAgent')
    @patch('special_agents.refinement_agent.FileAgent')
    @patch('special_agents.refinement_agent.TestAgent')
    def test_successful_refinement(self, mock_test, mock_file, mock_code):
        """Test successful refinement in first iteration."""
        # Mock sub-agents
        mock_code.return_value.run.return_value = "def hello(): return 'Hello'"
        mock_file.return_value.run.return_value = "Created hello.py"
        mock_test.return_value.run.return_value = "All tests PASSED"
        
        # Mock evaluation to return success
        self.agent._evaluate_test_results = Mock(return_value={
            "success": True,
            "critical_errors": [],
            "improvements_needed": [],
            "improvement_priority": "",
            "improvement_summary": "All good"
        })
        
        # Run refinement
        result = self.agent.refine_code("Create a hello function")
        
        # Verify result
        self.assertEqual(result.status, RefinementStatus.SUCCESS)
        self.assertEqual(result.iterations, 1)
        self.assertIn("Hello", result.final_output)
        self.assertIn("PASSED", result.test_results)
    
    @patch('special_agents.refinement_agent.CodeAgent')
    @patch('special_agents.refinement_agent.FileAgent')
    @patch('special_agents.refinement_agent.TestAgent')
    def test_refinement_with_retry(self, mock_test, mock_file, mock_code):
        """Test refinement that needs one retry."""
        # First iteration fails
        mock_code.return_value.run.side_effect = [
            "def hello(): return 'Helo'",  # Typo
            "def hello(): return 'Hello'"   # Fixed
        ]
        mock_file.return_value.run.return_value = "File updated"
        mock_test.return_value.run.side_effect = [
            "Test FAILED: Expected 'Hello' got 'Helo'",
            "All tests PASSED"
        ]
        
        # Mock evaluation
        self.agent._evaluate_test_results = Mock(side_effect=[
            {
                "success": False,
                "critical_errors": ["Typo in output"],
                "improvements_needed": ["Fix spelling"],
                "improvement_priority": "Fix typo",
                "improvement_summary": "Output has typo"
            },
            {
                "success": True,
                "critical_errors": [],
                "improvements_needed": [],
                "improvement_priority": "",
                "improvement_summary": "All good"
            }
        ])
        
        # Run refinement
        result = self.agent.refine_code("Create a hello function")
        
        # Verify result
        self.assertEqual(result.status, RefinementStatus.SUCCESS)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.improvements_made), 1)
        self.assertIn("typo", result.improvements_made[0].lower())
    
    @patch('special_agents.refinement_agent.CodeAgent')
    @patch('special_agents.refinement_agent.FileAgent')
    @patch('special_agents.refinement_agent.TestAgent')
    def test_max_iterations_reached(self, mock_test, mock_file, mock_code):
        """Test refinement that reaches max iterations."""
        # All iterations fail
        mock_code.return_value.run.return_value = "def hello(): return 'Wrong'"
        mock_file.return_value.run.return_value = "File updated"
        mock_test.return_value.run.return_value = "Test FAILED"
        
        # Mock evaluation to always fail
        self.agent._evaluate_test_results = Mock(return_value={
            "success": False,
            "critical_errors": ["Wrong output"],
            "improvements_needed": ["Fix output"],
            "improvement_priority": "Fix output",
            "improvement_summary": "Output incorrect"
        })
        
        # Run refinement
        result = self.agent.refine_code("Create a hello function")
        
        # Verify result
        self.assertEqual(result.status, RefinementStatus.MAX_ITERATIONS)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(len(result.improvements_made), 3)
    
    @patch('special_agents.refinement_agent.CodeAgent')
    def test_refinement_with_exception(self, mock_code):
        """Test refinement that encounters an exception."""
        # Mock code agent to raise exception
        mock_code.return_value.run.side_effect = Exception("API error")
        
        # Run refinement
        result = self.agent.refine_code("Create a hello function")
        
        # Verify result
        self.assertEqual(result.status, RefinementStatus.FAILED)
        self.assertEqual(result.iterations, 1)
        self.assertIn("API error", result.final_output)
    
    def test_evaluate_test_results_success(self):
        """Test evaluation of successful test results."""
        # Mock successful response
        self.agent.reply = Mock(return_value=json.dumps({
            "success": True,
            "critical_errors": [],
            "improvements_needed": [],
            "improvement_priority": "",
            "improvement_summary": "All tests passed"
        }))
        
        result = self.agent._evaluate_test_results("All tests PASSED", "task")
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["critical_errors"]), 0)
    
    def test_evaluate_test_results_failure(self):
        """Test evaluation of failed test results."""
        # Mock failure response
        self.agent.reply = Mock(return_value=json.dumps({
            "success": False,
            "critical_errors": ["TypeError", "AssertionError"],
            "improvements_needed": ["Fix type handling", "Fix logic"],
            "improvement_priority": "Fix TypeError first",
            "improvement_summary": "Multiple test failures"
        }))
        
        result = self.agent._evaluate_test_results("Tests FAILED", "task")
        
        self.assertFalse(result["success"])
        self.assertEqual(len(result["critical_errors"]), 2)
        self.assertIn("TypeError", result["critical_errors"])
    
    def test_evaluate_test_results_json_error(self):
        """Test evaluation with JSON parse error."""
        # Mock invalid JSON response
        self.agent.reply = Mock(return_value="Not valid JSON")
        
        result = self.agent._evaluate_test_results("Test output", "task")
        
        # Should return fallback result
        self.assertFalse(result["success"])
        self.assertIn("Failed to parse", result["critical_errors"][0])
    
    def test_prepare_improvement_prompt(self):
        """Test improvement prompt generation."""
        evaluation = {
            "critical_errors": ["TypeError", "ValueError"],
            "improvement_priority": "Fix TypeError",
            "improvements_needed": ["Type checking", "Input validation"]
        }
        
        prompt = self.agent._prepare_improvement_prompt(
            original_task="Create function",
            test_results="Test failed",
            evaluation=evaluation,
            previous_attempts=1
        )
        
        self.assertIn("Iteration 2", prompt)
        self.assertIn("Create function", prompt)
        self.assertIn("TypeError", prompt)
        self.assertIn("Fix TypeError", prompt)
    
    def test_run_method_json_output(self):
        """Test that run method returns valid JSON."""
        # Mock successful refinement
        self.agent.refine_code = Mock(return_value=RefinementResult(
            status=RefinementStatus.SUCCESS,
            iterations=1,
            final_output="Code output",
            test_results="Tests passed",
            improvements_made=["Fixed bug"]
        ))
        
        result = self.agent.run("Create function")
        
        # Verify JSON structure
        data = json.loads(result)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["iterations"], 1)
        self.assertEqual(data["final_output"], "Code output")
        self.assertEqual(len(data["improvements_made"]), 1)


if __name__ == '__main__':
    unittest.main()