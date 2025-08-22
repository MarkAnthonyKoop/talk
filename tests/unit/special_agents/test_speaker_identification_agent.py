#!/usr/bin/env python3
"""
Unit tests for SpeakerIdentificationAgent.

Tests the speaker identification functionality including:
- Embedding extraction
- Speaker enrollment
- Speaker identification
- Profile management
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

# Imports relative to PYTHONPATH (~/talk)
from special_agents.speaker_identification_agent import (
    SpeakerIdentificationAgent, 
    SpeakerProfile
)
from tests.utilities.test_output_writer import TestOutputWriter


class TestSpeakerProfile(unittest.TestCase):
    """Test the SpeakerProfile class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile = SpeakerProfile("test_speaker", "Test User")
    
    def test_profile_creation(self):
        """Test creating a speaker profile."""
        self.assertEqual(self.profile.speaker_id, "test_speaker")
        self.assertEqual(self.profile.name, "Test User")
        self.assertEqual(len(self.profile.embeddings), 0)
        self.assertEqual(self.profile.total_samples, 0)
    
    def test_add_embedding(self):
        """Test adding embeddings to profile."""
        # Add embeddings
        for i in range(5):
            embedding = np.random.randn(256)
            self.profile.add_embedding(embedding)
        
        self.assertEqual(len(self.profile.embeddings), 5)
        self.assertEqual(self.profile.total_samples, 5)
    
    def test_embedding_limit(self):
        """Test that embeddings are limited to 100."""
        # Add 150 embeddings
        for i in range(150):
            embedding = np.random.randn(256)
            self.profile.add_embedding(embedding)
        
        # Should only keep last 100
        self.assertEqual(len(self.profile.embeddings), 100)
        self.assertEqual(self.profile.total_samples, 150)
    
    def test_centroid_embedding(self):
        """Test centroid embedding calculation."""
        # Add known embeddings
        embeddings = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        for emb in embeddings:
            self.profile.add_embedding(emb)
        
        # Centroid should be average
        centroid = self.profile.get_centroid_embedding()
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(centroid, expected)
    
    def test_similarity_calculation(self):
        """Test similarity calculation."""
        # Add embeddings
        base_embedding = np.array([1, 0, 0, 0])
        self.profile.add_embedding(base_embedding)
        
        # Test identical embedding
        similarity = self.profile.calculate_similarity(base_embedding)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test orthogonal embedding
        orthogonal = np.array([0, 1, 0, 0])
        similarity = self.profile.calculate_similarity(orthogonal)
        self.assertAlmostEqual(similarity, 0.0, places=5)


class TestSpeakerIdentificationAgent(unittest.TestCase):
    """Test the SpeakerIdentificationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_speakers.db"
        
        # Create agent with mock embeddings
        self.agent = SpeakerIdentificationAgent(
            db_path=self.db_path,
            similarity_threshold=0.75,
            use_mock=True
        )
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertTrue(self.agent.use_mock)
        self.assertEqual(self.agent.similarity_threshold, 0.75)
        self.assertEqual(self.agent.embedding_dim, 256)
        self.assertTrue(self.db_path.exists())
    
    def test_extract_embedding(self):
        """Test embedding extraction."""
        # Test with dict audio data
        audio_data = {"speaker_hint": "test_speaker"}
        embedding = self.agent.extract_embedding(audio_data)
        
        self.assertEqual(embedding.shape[0], 256)
        self.assertTrue(np.all(np.isfinite(embedding)))
        
        # Test consistency (same hint = same embedding in mock)
        embedding2 = self.agent.extract_embedding(audio_data)
        np.testing.assert_array_almost_equal(embedding, embedding2)
    
    def test_enroll_speaker(self):
        """Test enrolling a new speaker."""
        # Create mock audio samples
        samples = [
            {"speaker_hint": "alice", "sample": i}
            for i in range(3)
        ]
        
        # Enroll speaker
        speaker_id = self.agent.enroll_speaker("Alice Smith", samples)
        
        # Verify enrollment
        self.assertIn(speaker_id, self.agent.profiles)
        profile = self.agent.profiles[speaker_id]
        self.assertEqual(profile.name, "Alice Smith")
        self.assertEqual(len(profile.embeddings), 3)
    
    def test_identify_known_speaker(self):
        """Test identifying a known speaker."""
        # Enroll a speaker
        samples = [{"speaker_hint": "bob"} for _ in range(3)]
        speaker_id = self.agent.enroll_speaker("Bob Jones", samples)
        
        # Identify with similar audio
        audio_data = {"speaker_hint": "bob"}
        identified_id, confidence, metadata = self.agent.identify_speaker(audio_data)
        
        # Should identify as Bob
        self.assertEqual(identified_id, speaker_id)
        self.assertGreater(confidence, 0.75)
        self.assertEqual(metadata["name"], "Bob Jones")
        self.assertTrue(metadata["threshold_met"])
    
    def test_identify_unknown_speaker(self):
        """Test identifying an unknown speaker."""
        # Identify without any enrolled speakers
        audio_data = {"speaker_hint": "stranger"}
        identified_id, confidence, metadata = self.agent.identify_speaker(audio_data)
        
        # Should create temporary profile
        self.assertTrue(identified_id.startswith("unknown_"))
        self.assertEqual(confidence, 0.0)
        self.assertTrue(metadata["temporary"])
        self.assertTrue(metadata["new_speaker"])
        self.assertFalse(metadata["threshold_met"])
    
    def test_update_speaker_name(self):
        """Test updating speaker name."""
        # Create temporary profile
        audio_data = {"speaker_hint": "temp_user"}
        temp_id, _, _ = self.agent.identify_speaker(audio_data)
        
        # Update name (convert to permanent)
        success = self.agent.update_speaker_name(temp_id, "Carol Davis")
        self.assertTrue(success)
        
        # Verify conversion
        self.assertNotIn(temp_id, self.agent.temp_profiles)
        
        # Find new permanent profile
        carol_profile = None
        for profile in self.agent.profiles.values():
            if profile.name == "Carol Davis":
                carol_profile = profile
                break
        
        self.assertIsNotNone(carol_profile)
    
    def test_merge_speakers(self):
        """Test merging speaker profiles."""
        # Create two speakers
        samples1 = [{"speaker_hint": "dave1"} for _ in range(2)]
        speaker_id1 = self.agent.enroll_speaker("Dave Part1", samples1)
        
        samples2 = [{"speaker_hint": "dave2"} for _ in range(3)]
        speaker_id2 = self.agent.enroll_speaker("Dave Part2", samples2)
        
        # Merge them
        merged_id = self.agent.merge_speakers(speaker_id1, speaker_id2, "Dave Complete")
        
        # Verify merge
        self.assertEqual(merged_id, speaker_id1)
        self.assertNotIn(speaker_id2, self.agent.profiles)
        
        merged_profile = self.agent.profiles[merged_id]
        self.assertEqual(merged_profile.name, "Dave Complete")
        self.assertEqual(len(merged_profile.embeddings), 5)
    
    def test_get_all_speakers(self):
        """Test listing all speakers."""
        # Enroll some speakers
        self.agent.enroll_speaker("Eve", [{"speaker_hint": "eve"}])
        self.agent.enroll_speaker("Frank", [{"speaker_hint": "frank"}])
        
        # Create temp profile
        self.agent.identify_speaker({"speaker_hint": "temp"})
        
        # Get all speakers
        speakers = self.agent.get_all_speakers()
        
        # Should have 2 permanent + 1 temporary
        self.assertEqual(len(speakers), 3)
        
        # Check for temporary flag
        temp_speakers = [s for s in speakers if s.get("temporary")]
        self.assertEqual(len(temp_speakers), 1)
    
    def test_run_identify_command(self):
        """Test the run method with identify command."""
        # Enroll a speaker
        samples = [{"speaker_hint": "grace"}]
        self.agent.enroll_speaker("Grace", samples)
        
        # Run identify command
        command = json.dumps({
            "command": "identify",
            "audio_data": {"speaker_hint": "grace"}
        })
        
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("speaker_id", result)
        self.assertIn("confidence", result)
        self.assertIn("metadata", result)
    
    def test_run_enroll_command(self):
        """Test the run method with enroll command."""
        command = json.dumps({
            "command": "enroll",
            "name": "Henry",
            "samples": [{"speaker_hint": "henry"} for _ in range(3)]
        })
        
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("speaker_id", result)
        self.assertTrue(result["enrolled"])
    
    def test_run_list_command(self):
        """Test the run method with list command."""
        # Enroll a speaker
        self.agent.enroll_speaker("Iris", [{"speaker_hint": "iris"}])
        
        # Run list command
        command = json.dumps({"command": "list"})
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("speakers", result)
        self.assertIn("total", result)
        self.assertEqual(result["total"], 1)
    
    def test_run_stats_command(self):
        """Test the run method with stats command."""
        # Enroll a speaker
        speaker_id = self.agent.enroll_speaker("Jack", [{"speaker_hint": "jack"}])
        
        # Get stats for specific speaker
        command = json.dumps({
            "command": "stats",
            "speaker_id": speaker_id
        })
        
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertEqual(result["speaker_id"], speaker_id)
        self.assertEqual(result["name"], "Jack")
        self.assertIn("total_samples", result)
        
        # Get overall stats
        command = json.dumps({"command": "stats"})
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("total_profiles", result)
        self.assertIn("temp_profiles", result)
    
    def test_persistence(self):
        """Test that profiles persist across agent instances."""
        # Enroll speaker with first agent
        speaker_id = self.agent.enroll_speaker("Kate", [{"speaker_hint": "kate"}])
        
        # Create new agent with same database
        new_agent = SpeakerIdentificationAgent(
            db_path=self.db_path,
            use_mock=True
        )
        
        # Should load the profile
        self.assertIn(speaker_id, new_agent.profiles)
        self.assertEqual(new_agent.profiles[speaker_id].name, "Kate")


class TestIntegration(unittest.TestCase):
    """Integration tests for speaker identification workflow."""
    
    def test_enrollment_to_identification_workflow(self):
        """Test complete workflow from enrollment to identification."""
        # Create agent
        agent = SpeakerIdentificationAgent(use_mock=True)
        
        # Enroll multiple speakers
        alice_id = agent.enroll_speaker("Alice", [
            {"speaker_hint": "alice", "sample": i} for i in range(5)
        ])
        
        bob_id = agent.enroll_speaker("Bob", [
            {"speaker_hint": "bob", "sample": i} for i in range(5)
        ])
        
        # Test identification accuracy
        test_cases = [
            ({"speaker_hint": "alice"}, alice_id),
            ({"speaker_hint": "bob"}, bob_id),
            ({"speaker_hint": "alice"}, alice_id),  # Repeat
        ]
        
        for audio_data, expected_id in test_cases:
            identified_id, confidence, metadata = agent.identify_speaker(audio_data)
            self.assertEqual(identified_id, expected_id)
            self.assertGreater(confidence, 0.7)
    
    def test_temporary_to_permanent_conversion(self):
        """Test converting temporary profiles to permanent."""
        agent = SpeakerIdentificationAgent(use_mock=True)
        
        # Create temporary profiles through identification
        temp_ids = []
        for i in range(3):
            audio = {"speaker_hint": f"unknown_{i}"}
            temp_id, _, _ = agent.identify_speaker(audio)
            temp_ids.append(temp_id)
        
        # Convert to permanent
        for i, temp_id in enumerate(temp_ids):
            success = agent.update_speaker_name(temp_id, f"Person {i+1}")
            self.assertTrue(success)
        
        # Verify all converted
        self.assertEqual(len(agent.temp_profiles), 0)
        self.assertEqual(len(agent.profiles), 3)


def run_tests():
    """Run all tests."""
    # Initialize test output writer
    writer = TestOutputWriter("unit", "test_speaker_identification_agent")
    output_dir = writer.get_output_dir()
    
    # Start test run
    start_time = datetime.now()
    print(f"\nTest output directory: {output_dir}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpeakerProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestSpeakerIdentificationAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Prepare results
    test_results = {
        "test_name": "test_speaker_identification_agent",
        "category": "unit",
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful()
    }
    
    # Add failure details
    if result.failures:
        test_results["failures"] = [
            {
                "test": str(test[0]),
                "traceback": test[1]
            }
            for test in result.failures
        ]
    
    if result.errors:
        test_results["errors"] = [
            {
                "test": str(test[0]),
                "traceback": test[1]
            }
            for test in result.errors
        ]
    
    # Write results
    writer.write_results(test_results)
    
    # Write summary log
    summary = f"""Speaker Identification Agent Test Results
=========================================
Timestamp: {start_time.isoformat()}
Duration: {duration:.2f} seconds
Total Tests: {result.testsRun}
Passed: {test_results['passed']}
Failed: {test_results['failed']}
Errors: {test_results['errors']}
Success: {result.wasSuccessful()}
"""
    writer.write_log(summary)
    
    print(f"\nTest results saved to: {output_dir}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)