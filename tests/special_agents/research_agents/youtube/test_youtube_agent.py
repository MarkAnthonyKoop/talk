#!/usr/bin/env python3

"""
Comprehensive tests for YoutubeAgent.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent, YoutubeAgentIntegration


class TestYoutubeAgent(unittest.TestCase):
    """Test cases for YoutubeAgent."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.takeout_path = project_root / "special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip"
        cls.test_scratch_dir = Path(".talk/scratch")
        cls.test_scratch_dir.mkdir(parents=True, exist_ok=True)
    
    def setUp(self):
        """Set up each test."""
        # Clean scratch directory before each test
        for file in self.test_scratch_dir.glob("youtube_analysis_*.json"):
            file.unlink()
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        self.assertIsNotNone(agent)
        self.assertEqual(agent.takeout_path, str(self.takeout_path))
        self.assertIsInstance(agent.data_cache, dict)
    
    def test_agent_without_takeout(self):
        """Test agent initialization without takeout path."""
        agent = YoutubeAgent()
        self.assertIsNone(agent.takeout_path)
        result = agent.run("Analyze my YouTube data")
        self.assertIn("not available", result.lower())
    
    def test_load_takeout_data(self):
        """Test loading and parsing takeout data."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        agent._load_takeout_data()
        
        # Check that some data was loaded
        self.assertGreater(len(agent.data_cache), 0)
        
        # Check for expected data types
        possible_keys = ['watch_history', 'search_history', 'subscriptions', 'playlists']
        loaded_keys = [key for key in possible_keys if key in agent.data_cache]
        self.assertGreater(len(loaded_keys), 0, "Should load at least some data from takeout")
    
    def test_analysis_type_determination(self):
        """Test determining analysis type from input."""
        agent = YoutubeAgent()
        
        test_cases = [
            ("Show me my watch history", "watch_history"),
            ("What did I search for?", "search_history"),
            ("List my subscriptions", "subscriptions"),
            ("Show my playlists", "playlists"),
            ("Give me a general overview", "general"),
            ("Random request", "comprehensive")
        ]
        
        for input_text, expected_type in test_cases:
            result = agent._determine_analysis_type(input_text)
            self.assertEqual(result, expected_type, f"Failed for: {input_text}")
    
    def test_general_overview(self):
        """Test general overview analysis."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        result = agent.run("Give me a general overview of my YouTube data")
        
        # Check that result contains expected sections
        self.assertIn("OVERVIEW", result.upper())
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 100)  # Should have substantial content
    
    def test_watch_history_analysis(self):
        """Test watch history analysis."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        result = agent.run("Analyze my watch history")
        
        # Check for expected content
        self.assertIn("HISTORY", result.upper())
        self.assertIsNotNone(result)
    
    def test_search_history_analysis(self):
        """Test search history analysis."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        result = agent.run("What are my most common search terms?")
        
        # Check for expected content
        self.assertIn("SEARCH", result.upper())
        self.assertIsNotNone(result)
    
    def test_subscription_analysis(self):
        """Test subscription analysis."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        result = agent.run("Show me my subscriptions")
        
        # Check for expected content
        self.assertIn("SUBSCRIPTION", result.upper())
        self.assertIsNotNone(result)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        result = agent.run("Perform a comprehensive analysis")
        
        # Should contain multiple analysis sections
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 200)  # Comprehensive should be longer
    
    def test_results_saved_to_scratch(self):
        """Test that results are saved to scratch directory."""
        if not self.takeout_path.exists():
            self.skipTest(f"Takeout file not found at {self.takeout_path}")
        
        agent = YoutubeAgent(takeout_path=str(self.takeout_path))
        
        # Count files before
        files_before = list(self.test_scratch_dir.glob("youtube_analysis_*.json"))
        
        # Run analysis
        agent.run("Analyze my YouTube data")
        
        # Count files after
        files_after = list(self.test_scratch_dir.glob("youtube_analysis_*.json"))
        
        # Should have created a new file
        self.assertGreater(len(files_after), len(files_before))
        
        # Verify file content
        if files_after:
            newest_file = max(files_after, key=lambda f: f.stat().st_mtime)
            with open(newest_file, 'r') as f:
                data = json.load(f)
                self.assertIn('timestamp', data)
                self.assertIn('analysis_result', data)
                self.assertIn('insights', data)
                self.assertIn('data_summary', data)
    
    def test_error_handling(self):
        """Test error handling for invalid data."""
        agent = YoutubeAgent(takeout_path="/nonexistent/path.zip")
        result = agent.run("Analyze my data")
        
        # Should handle gracefully
        self.assertIn("ERROR", result.upper())
        self.assertIsNotNone(result)
    
    @patch('special_agents.research_agents.youtube.youtube_agent.zipfile.ZipFile')
    def test_corrupted_zip_handling(self, mock_zipfile):
        """Test handling of corrupted zip file."""
        mock_zipfile.side_effect = Exception("Corrupted zip")
        
        agent = YoutubeAgent(takeout_path="fake.zip")
        result = agent.run("Analyze my data")
        
        # Should handle error gracefully
        self.assertIn("ERROR", result.upper())


class TestYoutubeAgentIntegration(unittest.TestCase):
    """Test cases for YoutubeAgentIntegration helper."""
    
    def test_should_analyze_youtube_data(self):
        """Test detection of YouTube-related tasks."""
        test_cases = [
            ("Analyze my YouTube watch history", True),
            ("Show me my YouTube subscriptions", True),
            ("What videos have I watched?", True),
            ("Process my takeout data", True),
            ("Fix a bug in the code", False),
            ("Refactor this function", False),
            ("Add comments to the file", False),
        ]
        
        for task, expected in test_cases:
            result = YoutubeAgentIntegration.should_analyze_youtube_data(task)
            self.assertEqual(result, expected, f"Failed for task: {task}")
    
    def test_create_analysis_prompt(self):
        """Test analysis prompt creation."""
        task = "Analyze my viewing habits"
        path = "/path/to/takeout.zip"
        
        prompt = YoutubeAgentIntegration.create_analysis_prompt(task, path)
        
        self.assertIn(task, prompt)
        self.assertIn(path, prompt)
        self.assertIn("Analyze", prompt)


class TestYoutubeAgentWithMockData(unittest.TestCase):
    """Test YoutubeAgent with mock data."""
    
    def setUp(self):
        """Create mock data for testing."""
        self.agent = YoutubeAgent()
        
        # Mock watch history
        self.agent.data_cache['watch_history'] = [
            {'title': 'Video 1', 'channel': 'Channel A', 'timestamp': '2025-01-01, 10:00 AM'},
            {'title': 'Video 2', 'channel': 'Channel A', 'timestamp': '2025-01-01, 11:00 AM'},
            {'title': 'Video 3', 'channel': 'Channel B', 'timestamp': '2025-01-01, 12:00 PM'},
            {'title': 'Video 4', 'channel': 'Channel A', 'timestamp': '2025-01-02, 09:00 AM'},
            {'title': 'Video 5', 'channel': 'Channel C', 'timestamp': '2025-01-02, 02:00 PM'},
        ]
        
        # Mock search history
        self.agent.data_cache['search_history'] = [
            {'query': 'python tutorial', 'timestamp': '2025-01-01, 09:00 AM'},
            {'query': 'machine learning basics', 'timestamp': '2025-01-01, 10:30 AM'},
            {'query': 'python debugging', 'timestamp': '2025-01-02, 11:00 AM'},
        ]
        
        # Mock subscriptions
        self.agent.data_cache['subscriptions'] = [
            {'Channel Title': 'Tech Channel'},
            {'Channel Title': 'Education Hub'},
            {'Channel Title': 'Coding Masters'},
        ]
        
        # Mock playlists
        self.agent.data_cache['playlists'] = {
            'Watch later': [
                {'Video Title': 'Learn Python'},
                {'Video Title': 'Advanced Git'},
            ],
            'Favorites': [
                {'Video Title': 'Best Practices'},
            ]
        }
    
    def test_watch_history_with_mock_data(self):
        """Test watch history analysis with mock data."""
        result = self.agent._analyze_watch_history("Analyze watch history")
        
        self.assertIn("Total videos watched: 5", result)
        self.assertIn("Channel A: 3 videos", result)
        self.assertIn("Channel B: 1 videos", result)
        self.assertIn("Channel C: 1 videos", result)
    
    def test_search_history_with_mock_data(self):
        """Test search history analysis with mock data."""
        result = self.agent._analyze_search_history("Analyze searches")
        
        self.assertIn("Total searches: 3", result)
        self.assertIn("python", result.lower())
    
    def test_subscription_with_mock_data(self):
        """Test subscription analysis with mock data."""
        result = self.agent._analyze_subscriptions("Show subscriptions")
        
        self.assertIn("Total subscriptions: 3", result)
        self.assertIn("Tech Channel", result)
        self.assertIn("Education Hub", result)
    
    def test_playlist_with_mock_data(self):
        """Test playlist analysis with mock data."""
        result = self.agent._analyze_playlists("Show playlists")
        
        self.assertIn("Total playlists: 2", result)
        self.assertIn("Watch later: 2 videos", result)
        self.assertIn("Favorites: 1 videos", result)


def run_specific_test(test_name=None):
    """Run a specific test or all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_name:
        # Run specific test
        suite.addTest(TestYoutubeAgent(test_name))
    else:
        # Run all tests
        suite.addTests(loader.loadTestsFromTestCase(TestYoutubeAgent))
        suite.addTests(loader.loadTestsFromTestCase(TestYoutubeAgentIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestYoutubeAgentWithMockData))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YoutubeAgent')
    parser.add_argument('--test', help='Specific test to run', default=None)
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        unittest.main(verbosity=2, argv=[''])
    else:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)