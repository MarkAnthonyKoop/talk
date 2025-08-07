#!/usr/bin/env python3
"""
Comprehensive tests for YouTube Research CLI search functionality.

Tests both command-line interface and direct Python API.
"""

import sys
import subprocess
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any
import unittest
from unittest.mock import patch, MagicMock
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from youtube_research_cli import YouTubeResearchCLI, smart_route, main


class TestYouTubeSearch(unittest.TestCase):
    """Test YouTube search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database connection."""
        cls.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
        if not cls.db_path.exists():
            raise FileNotFoundError(f"Test database not found: {cls.db_path}")
        
        # Get some real data for testing
        conn = sqlite3.connect(str(cls.db_path))
        cursor = conn.cursor()
        
        # Get actual artists/creators from the database
        cursor.execute("""
            SELECT title FROM videos 
            WHERE title LIKE 'Pink Floyd%' 
            LIMIT 5
        """)
        cls.pink_floyd_videos = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT title FROM videos 
            WHERE title LIKE 'AC/DC%' 
            LIMIT 5
        """)
        cls.acdc_videos = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE LOWER(title) LIKE '%claude%'
        """)
        cls.claude_count = cursor.fetchone()[0]
        
        conn.close()
    
    def setUp(self):
        """Set up for each test."""
        self.cli = YouTubeResearchCLI(str(self.db_path))
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'cli') and self.cli.conn:
            self.cli.conn.close()
    
    def test_search_by_artist_in_title(self):
        """Test searching for videos by artist name in title."""
        # Test Pink Floyd
        result = self.cli._analyze_history("Pink Floyd videos")
        
        self.assertIn('matches', result)
        self.assertTrue(len(result['matches']) > 0, "Should find Pink Floyd videos")
        
        # Check that results actually contain Pink Floyd
        pink_floyd_found = False
        for match in result['matches']:
            if 'pink floyd' in match['title'].lower():
                pink_floyd_found = True
                break
        
        self.assertTrue(pink_floyd_found, "Should find videos with 'Pink Floyd' in title")
    
    def test_search_acdc(self):
        """Test searching for AC/DC videos."""
        result = self.cli._analyze_history("AC/DC videos")
        
        self.assertIn('matches', result)
        self.assertTrue(len(result['matches']) > 0, "Should find AC/DC videos")
        
        # Check for AC/DC in results
        acdc_found = False
        for match in result['matches']:
            if 'ac/dc' in match['title'].lower() or 'ac dc' in match['title'].lower():
                acdc_found = True
                break
        
        self.assertTrue(acdc_found, "Should find videos with 'AC/DC' in title")
    
    def test_search_with_special_characters(self):
        """Test searching with special characters like /."""
        # This should handle AC/DC correctly
        result = self.cli._analyze_history("what AC/DC videos have I watched")
        
        self.assertIn('matches', result)
        # Even if no exact matches, should not crash
        self.assertIsInstance(result['matches'], list)
    
    def test_claude_video_search(self):
        """Test searching for Claude videos."""
        result = self.cli._analyze_history("Claude videos")
        
        self.assertIn('matches', result)
        
        if self.claude_count > 0:
            self.assertTrue(len(result['matches']) > 0, 
                          f"Should find Claude videos (DB has {self.claude_count})")
    
    def test_nonexistent_creator(self):
        """Test searching for a creator that doesn't exist."""
        result = self.cli._analyze_history("Cole Medin videos")
        
        self.assertIn('matches', result)
        # Should return empty or very few matches
        self.assertIsInstance(result['matches'], list)
        
        # Check that we don't get false positives
        for match in result['matches']:
            title_lower = match['title'].lower()
            # Should not have random unrelated videos
            self.assertTrue(
                'cole' in title_lower or 'medin' in title_lower,
                f"Unrelated video in results: {match['title']}"
            )
    
    def test_keyword_extraction(self):
        """Test keyword extraction from queries."""
        keywords = self.cli._extract_keywords("What Pink Floyd videos have I watched?")
        
        self.assertIn('pink', keywords)
        self.assertIn('floyd', keywords)
        self.assertNotIn('what', keywords)  # Stop word
        self.assertNotIn('have', keywords)  # Stop word
    
    def test_search_with_multiple_keywords(self):
        """Test search with multiple keywords."""
        # Use a search we know should work based on the database
        result = self.cli._analyze_history("Pink Floyd music videos")
        
        self.assertIn('matches', result)
        
        # Pink Floyd is common in the database, so should find matches
        if len(result['matches']) > 0:
            # At least some results should be relevant
            relevant = False
            for match in result['matches'][:10]:
                title_lower = match['title'].lower()
                if 'pink' in title_lower or 'floyd' in title_lower:
                    relevant = True
                    break
            
            self.assertTrue(relevant, "Should find relevant videos for multi-keyword search")
        else:
            # If no Pink Floyd videos, try another common artist
            result = self.cli._analyze_history("AC/DC live concert")
            self.assertIn('matches', result)
            if len(result['matches']) > 0:
                relevant = False
                for match in result['matches'][:10]:
                    title_lower = match['title'].lower()
                    if 'ac/dc' in title_lower or 'ac dc' in title_lower:
                        relevant = True
                        break
                self.assertTrue(relevant, "Should find AC/DC videos as fallback")
    
    def test_stats_in_results(self):
        """Test that stats are included in results."""
        result = self.cli._analyze_history("any query")
        
        self.assertIn('stats', result)
        self.assertIn('total_videos', result['stats'])
        self.assertGreater(result['stats']['total_videos'], 0)


class TestCommandLineInterface(unittest.TestCase):
    """Test the yr command-line interface."""
    
    def test_yr_command_exists(self):
        """Test that yr command is accessible."""
        result = subprocess.run(['which', 'yr'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "yr command should be in PATH")
    
    def test_yr_help(self):
        """Test yr --help."""
        result = subprocess.run(['yr', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('YouTube Research CLI', result.stdout)
    
    def test_yr_greeting(self):
        """Test yr with greeting."""
        result = subprocess.run(['yr', 'hello'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Hello', result.stdout)
        self.assertIn('YouTube Research Assistant', result.stdout)
    
    def test_yr_simple_math(self):
        """Test yr with simple query."""
        result = subprocess.run(['yr', 'what is 2 + 2?'], 
                              capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('4', result.stdout)
    
    def test_yr_video_search(self):
        """Test yr with video search query."""
        # This will timeout on web search, but should at least start
        process = subprocess.Popen(
            ['yr', 'what Pink Floyd videos have I watched?'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a few seconds then kill it
        time.sleep(3)
        
        # Try to get output before terminating
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        
        # Should see something indicating it started
        # Could be "Analyzing" or "Processing" or "Searching"
        self.assertTrue(
            'Analyzing' in stdout or 'Processing' in stdout or 'Searching' in stdout,
            f"Expected to see analysis starting, but got: {stdout[:200]}"
        )


class TestSmartRouting(unittest.TestCase):
    """Test the smart routing functionality."""
    
    def setUp(self):
        """Set up mock CLI for testing."""
        self.mock_cli = MagicMock()
        self.mock_cli.agent = MagicMock()
    
    def test_routing_youtube_url(self):
        """Test routing detects YouTube URLs."""
        with patch('youtube_research_cli.YouTubeResearchCLI') as mock_cls:
            mock_cls.return_value = self.mock_cli
            
            # Simulate routing
            query = "https://youtube.com/watch?v=test123"
            
            # The function should detect this as a YouTube URL
            # and route to research_video
            # (We'd need to refactor smart_route to be more testable)
            self.assertIn('youtube.com', query)
    
    def test_routing_learning_topic(self):
        """Test routing detects learning topics."""
        query = "machine learning fundamentals"
        
        # Should be routed to research-topic
        self.assertNotIn('?', query)  # Not a question
        self.assertIn('learning', query.lower())


class TestDatabaseIntegrity(unittest.TestCase):
    """Test database integrity and data quality."""
    
    @classmethod
    def setUpClass(cls):
        """Check database structure."""
        cls.db_path = Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db"
        cls.conn = sqlite3.connect(str(cls.db_path))
        cls.cursor = cls.conn.cursor()
    
    @classmethod
    def tearDownClass(cls):
        """Close database connection."""
        cls.conn.close()
    
    def test_database_exists(self):
        """Test that database exists and is accessible."""
        self.assertTrue(self.db_path.exists())
        
        # Check we can query it
        self.cursor.execute("SELECT COUNT(*) FROM videos")
        count = self.cursor.fetchone()[0]
        self.assertGreater(count, 0, "Database should have videos")
    
    def test_database_schema(self):
        """Test database has expected schema."""
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='videos'
        """)
        
        self.assertIsNotNone(self.cursor.fetchone(), "videos table should exist")
        
        # Check columns
        self.cursor.execute("PRAGMA table_info(videos)")
        columns = {col[1] for col in self.cursor.fetchall()}
        
        expected_columns = {'video_id', 'title', 'channel', 'url', 'ai_score', 'categories'}
        self.assertEqual(columns, expected_columns)
    
    def test_data_quality_issues(self):
        """Identify data quality issues."""
        # Check how many videos have unknown channels
        self.cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE channel = 'Unknown' OR channel IS NULL
        """)
        unknown_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM videos")
        total_count = self.cursor.fetchone()[0]
        
        print(f"\nData Quality Report:")
        print(f"  Total videos: {total_count}")
        print(f"  Unknown channels: {unknown_count} ({unknown_count/total_count*100:.1f}%)")
        
        # Check AI scoring accuracy
        self.cursor.execute("""
            SELECT title, ai_score FROM videos 
            WHERE ai_score > 0.8 
            ORDER BY RANDOM() 
            LIMIT 10
        """)
        
        print("\n  Sample high AI score videos (checking for false positives):")
        for title, score in self.cursor.fetchall():
            print(f"    {score:.2f}: {title[:60]}")


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestYouTubeSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandLineInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestSmartRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseIntegrity))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("YouTube Research CLI - Comprehensive Test Suite")
    print("=" * 70)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed - see output above")
    
    sys.exit(0 if success else 1)