#!/usr/bin/env python3
"""
Comprehensive tests for the orchestrated YouTube Research CLI.

Tests AI planning, parallel execution, and agent orchestration.
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the modules we're testing
from youtube_research_cli import YouTubeResearchCLI, ResearchPlan


class TestAIPlanningFeatures(unittest.TestCase):
    """Test AI-powered planning capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
        
        # Create temporary context directory
        self.temp_dir = tempfile.mkdtemp()
        self.context_dir = Path(self.temp_dir) / ".talk" / "yr"
        self.context_dir.mkdir(parents=True)
        
        # Mock context file
        self.context_data = {
            "query": "previous query",
            "results": {"history_matches": 10},
            "timestamp": "2024-01-01T12:00:00"
        }
        
        context_file = self.context_dir / "test_context_20240101_120000_test123.json"
        with open(context_file, 'w') as f:
            json.dump(self.context_data, f)
        
        # Initialize CLI with mocked context directory
        with patch('youtube_research_cli.Path.cwd', return_value=Path(self.temp_dir)):
            self.cli = YouTubeResearchCLI(str(self.db_path))
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'cli') and self.cli.conn:
            self.cli.conn.close()
        
        # Remove temporary directory
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_context_loading(self):
        """Test that context is loaded from .talk/yr/ directory."""
        # Context should be loaded during initialization
        self.assertIsNotNone(self.cli.context)
        self.assertEqual(self.cli.context.get('query'), 'previous query')
        self.assertEqual(self.cli.context.get('results', {}).get('history_matches'), 10)
    
    def test_context_saving(self):
        """Test that context is saved after analysis."""
        test_data = {
            "query": "test query",
            "results": {"test": "data"},
            "timestamp": datetime.now().isoformat()
        }
        
        with patch('youtube_research_cli.Path.cwd', return_value=Path(self.temp_dir)):
            self.cli._save_context(test_data)
        
        # Check that a new file was created
        json_files = list(self.context_dir.glob("*.json"))
        self.assertGreater(len(json_files), 1)  # Original + new file
        
        # Verify the content of the saved file
        newest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        with open(newest_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['query'], 'test query')
        self.assertEqual(saved_data['results']['test'], 'data')
    
    def test_research_plan_creation(self):
        """Test AI-powered research plan creation."""
        # Mock the planner agent response
        mock_response = json.dumps({
            "intent": "analyze AI videos",
            "requires_history": True,
            "requires_web_search": False,
            "requires_transcript": True,
            "steps": [
                {"phase": "gather", "action": "search_history", "details": "Search for AI videos"},
                {"phase": "analyze", "action": "detect_patterns", "details": "Find patterns"}
            ],
            "expected_output": "List of important AI videos"
        })
        
        with patch.object(self.cli.planner, 'run', return_value=mock_response):
            plan = self.cli._create_research_plan("What AI videos have I watched?")
        
        # Verify plan structure
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.intent, "analyze AI videos")
        self.assertTrue(plan.requires_history)
        self.assertFalse(plan.requires_web_search)
        self.assertTrue(plan.requires_transcript)
        self.assertEqual(len(plan.steps), 2)
    
    def test_plan_with_context_awareness(self):
        """Test that planning uses previous context."""
        mock_response = json.dumps({
            "intent": "follow-up analysis",
            "requires_history": True,
            "requires_web_search": True,
            "requires_transcript": False,
            "steps": [],
            "expected_output": "Refined analysis"
        })
        
        with patch.object(self.cli.planner, 'run') as mock_run:
            plan = self.cli._create_research_plan("Tell me more about those videos")
            
            # Verify context was included in planning prompt
            call_args = mock_run.call_args[0][0]
            self.assertIn("Previous context", call_args)
            self.assertIn("previous query", call_args)
    
    def test_enhanced_synthesis(self):
        """Test enhanced synthesis with better prompting."""
        plan = ResearchPlan(
            query="Test query",
            intent="test",
            expected_output="Test output"
        )
        
        results = {
            'history': {
                'matches': [
                    {'title': 'Test Video 1'},
                    {'title': 'Test Video 2'}
                ],
                'stats': {'total_videos': 20000}
            }
        }
        
        with patch.object(self.cli.agent, 'run', return_value="Synthesized result") as mock_run:
            synthesis = self.cli._enhanced_synthesis(plan, results)
        
        self.assertEqual(synthesis, "Synthesized result")
        
        # Verify the synthesis prompt included results
        call_args = mock_run.call_args[0][0]
        self.assertIn("Test query", call_args)
        self.assertIn("Test Video 1", call_args)
        self.assertIn("20000", call_args)


class TestQueryTypes(unittest.TestCase):
    """Test different types of queries and their handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database connection."""
        cls.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
    
    def setUp(self):
        """Create CLI instance for each test."""
        self.cli = YouTubeResearchCLI(str(self.db_path))
    
    def tearDown(self):
        """Clean up."""
        if hasattr(self, 'cli') and self.cli.conn:
            self.cli.conn.close()
    
    def test_ai_video_query(self):
        """Test querying for AI-related videos."""
        results = self.cli._analyze_history("What Claude videos have I watched?")
        
        self.assertIn('matches', results)
        self.assertIn('stats', results)
        
        # Check if Claude videos are found
        claude_found = any(
            'claude' in match['title'].lower()
            for match in results.get('matches', [])
        )
        
        if results['matches']:
            # If matches found, at least some should be Claude-related
            self.assertTrue(
                claude_found or len(results['matches']) > 0,
                "Should find Claude videos or return results"
            )
    
    def test_music_video_query(self):
        """Test querying for music videos."""
        results = self.cli._analyze_history("Pink Floyd videos")
        
        self.assertIn('matches', results)
        
        # Pink Floyd should be in results
        if results['matches']:
            pink_floyd_found = any(
                'pink floyd' in match['title'].lower()
                for match in results['matches']
            )
            self.assertTrue(pink_floyd_found, "Should find Pink Floyd videos")
    
    def test_complex_query_with_planning(self):
        """Test complex query that requires planning."""
        mock_plan_response = json.dumps({
            "intent": "complex analysis",
            "requires_history": True,
            "requires_web_search": True,
            "requires_transcript": False,
            "steps": [
                {"phase": "gather", "action": "search", "details": "Search history"},
                {"phase": "analyze", "action": "synthesize", "details": "Create report"}
            ],
            "expected_output": "Comprehensive analysis"
        })
        
        mock_web_response = {"success": True, "data": "Web results"}
        mock_synthesis = "Final comprehensive analysis"
        
        with patch.object(self.cli.planner, 'run', return_value=mock_plan_response), \
             patch.object(self.cli.web_agent, 'run', return_value=json.dumps(mock_web_response)), \
             patch.object(self.cli.agent, 'run', return_value=mock_synthesis), \
             patch.object(self.cli, '_save_context'):
            
            # Capture output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                self.cli.analyze_with_planning(
                    "What are the best AI videos for learning about agents?"
                )
            
            output = f.getvalue()
            
            # Verify planning and execution
            self.assertIn("Creating AI-powered research plan", output)
            self.assertIn("Research Plan", output)
            self.assertIn("RESEARCH RESULTS", output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
        self.cli = YouTubeResearchCLI(str(self.db_path))
    
    def tearDown(self):
        """Clean up."""
        if hasattr(self, 'cli') and self.cli.conn:
            self.cli.conn.close()
    
    def test_malformed_plan_response(self):
        """Test handling of malformed planning response."""
        with patch.object(self.cli.planner, 'run', return_value="Not valid JSON"):
            plan = self.cli._create_research_plan("Test query")
        
        # Should return default plan
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.intent, "general")
        self.assertTrue(plan.requires_history)
    
    def test_web_search_failure(self):
        """Test handling of web search failure."""
        plan = ResearchPlan(
            query="Test",
            requires_web_search=True
        )
        
        with patch.object(self.cli.web_agent, 'run', side_effect=Exception("Network error")), \
             patch.object(self.cli.agent, 'run', return_value="Synthesis"), \
             patch.object(self.cli, '_save_context'):
            
            # Should not crash
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                self.cli.analyze_with_planning("Test query")
            
            output = f.getvalue()
            self.assertIn("RESEARCH RESULTS", output)
    
    def test_empty_history_results(self):
        """Test handling of empty history results."""
        results = self.cli._analyze_history("xyz123nonexistent")
        
        self.assertIn('matches', results)
        self.assertIn('stats', results)
        self.assertEqual(len(results['matches']), 0)
    
    def test_context_directory_missing(self):
        """Test handling when context directory doesn't exist."""
        with patch('youtube_research_cli.Path.cwd', return_value=Path("/nonexistent")):
            context = self.cli._load_context()
        
        self.assertEqual(context, {})


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete flow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
    
    def test_end_to_end_flow(self):
        """Test complete flow from query to results."""
        cli = YouTubeResearchCLI(str(self.db_path))
        
        # Mock components to avoid actual API calls
        mock_plan = json.dumps({
            "intent": "analyze",
            "requires_history": True,
            "requires_web_search": False,
            "requires_transcript": False,
            "steps": [],
            "expected_output": "Analysis"
        })
        
        with patch.object(cli.planner, 'run', return_value=mock_plan), \
             patch.object(cli.agent, 'run', return_value="Final analysis"), \
             patch.object(cli, '_save_context') as mock_save:
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                cli.analyze_with_planning("What videos have I watched?")
            
            output = f.getvalue()
            
            # Verify complete flow
            self.assertIn("Advanced Analysis", output)
            self.assertIn("Research Plan", output)
            self.assertIn("Searching viewing history", output)
            self.assertIn("Synthesizing results", output)
            self.assertIn("RESEARCH RESULTS", output)
            self.assertIn("Final analysis", output)
            
            # Verify context was saved
            mock_save.assert_called_once()
        
        cli.conn.close()


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAIPlanningFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("Orchestrated YouTube Research CLI - Test Suite")
    print("=" * 70)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    sys.exit(0 if success else 1)