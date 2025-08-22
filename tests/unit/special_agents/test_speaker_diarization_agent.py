#!/usr/bin/env python3
"""
Unit tests for SpeakerDiarizationAgent.

Tests the speaker diarization functionality including:
- Audio segmentation
- Speaker change detection
- Voice activity detection
- Segment merging
- Timeline generation
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

# Imports relative to PYTHONPATH (~/talk)
from special_agents.speaker_diarization_agent import (
    SpeakerDiarizationAgent,
    AudioSegment
)
from special_agents.speaker_identification_agent import SpeakerIdentificationAgent
from tests.utilities.test_output_writer import TestOutputWriter


class TestAudioSegment(unittest.TestCase):
    """Test the AudioSegment class."""
    
    def test_segment_creation(self):
        """Test creating an audio segment."""
        segment = AudioSegment(
            start_time=1.0,
            end_time=3.5,
            speaker_id="speaker_1",
            confidence=0.95
        )
        
        self.assertEqual(segment.start_time, 1.0)
        self.assertEqual(segment.end_time, 3.5)
        self.assertEqual(segment.duration, 2.5)
        self.assertEqual(segment.speaker_id, "speaker_1")
        self.assertEqual(segment.confidence, 0.95)
    
    def test_segment_to_dict(self):
        """Test converting segment to dictionary."""
        segment = AudioSegment(
            start_time=0.0,
            end_time=5.0,
            speaker_id="alice",
            confidence=0.85
        )
        segment.transcript = "Hello, this is Alice"
        segment.metadata = {"quality": "good"}
        
        data = segment.to_dict()
        
        self.assertEqual(data["start_time"], 0.0)
        self.assertEqual(data["end_time"], 5.0)
        self.assertEqual(data["duration"], 5.0)
        self.assertEqual(data["speaker_id"], "alice")
        self.assertEqual(data["confidence"], 0.85)
        self.assertEqual(data["transcript"], "Hello, this is Alice")
        self.assertEqual(data["metadata"], {"quality": "good"})


class TestSpeakerDiarizationAgent(unittest.TestCase):
    """Test the SpeakerDiarizationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock speaker agent
        self.speaker_agent = SpeakerIdentificationAgent(use_mock=True)
        
        # Create diarization agent
        self.agent = SpeakerDiarizationAgent(
            speaker_agent=self.speaker_agent,
            vad_threshold=0.5,
            min_segment_duration=0.5,
            max_segment_duration=10.0,
            use_mock=True
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.vad_threshold, 0.5)
        self.assertEqual(self.agent.min_segment_duration, 0.5)
        self.assertEqual(self.agent.max_segment_duration, 10.0)
        self.assertTrue(self.agent.use_mock)
        self.assertEqual(len(self.agent.current_segments), 0)
    
    def test_process_audio_stream(self):
        """Test processing audio stream chunks."""
        # Process first chunk
        audio1 = {"speaker_hint": "alice", "chunk": 1}
        segments = self.agent.process_audio_stream(audio1, timestamp=0.0)
        
        # Should buffer, not create segment yet
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(self.agent.processing_buffer), 1)
        
        # Process more chunks from same speaker
        for i in range(2, 5):
            audio = {"speaker_hint": "alice", "chunk": i}
            segments = self.agent.process_audio_stream(audio, timestamp=i * 0.1)
        
        # Process chunk from different speaker (triggers change)
        audio_bob = {"speaker_hint": "bob", "chunk": 5}
        segments = self.agent.process_audio_stream(audio_bob, timestamp=0.5)
        
        # Should have created a segment
        self.assertGreater(len(segments), 0)
        segment = segments[0]
        self.assertIsNotNone(segment.speaker_id)
    
    def test_speaker_change_detection(self):
        """Test detecting speaker changes."""
        # First audio chunk
        audio1 = {"speaker_hint": "carol"}
        changed = self.agent._detect_speaker_change(audio1)
        self.assertFalse(changed)  # First chunk, no previous
        
        # Same speaker
        audio2 = {"speaker_hint": "carol"}
        changed = self.agent._detect_speaker_change(audio2)
        self.assertFalse(changed)
        
        # Different speaker
        audio3 = {"speaker_hint": "dave"}
        changed = self.agent._detect_speaker_change(audio3)
        self.assertTrue(changed)
    
    def test_finalize_segment(self):
        """Test finalizing audio segments."""
        # Add chunks to buffer
        for i in range(5):
            self.agent.processing_buffer.append({
                "data": {"speaker_hint": "eve", "chunk": i},
                "timestamp": i * 0.2
            })
        
        # Finalize segment
        segments = self.agent._finalize_current_segment(end_time=1.0)
        
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.start_time, 0.0)
        self.assertEqual(segment.end_time, 1.0)
        self.assertIsNotNone(segment.speaker_id)
        self.assertGreater(segment.confidence, 0)
        
        # Buffer should be cleared
        self.assertEqual(len(self.agent.processing_buffer), 0)
    
    def test_min_segment_duration(self):
        """Test minimum segment duration enforcement."""
        # Add very short segment
        self.agent.processing_buffer.append({
            "data": {"speaker_hint": "frank"},
            "timestamp": 0.0
        })
        
        # Try to finalize too early
        segments = self.agent._finalize_current_segment(end_time=0.3)
        
        # Should skip segment (too short)
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(self.agent.processing_buffer), 0)
    
    def test_max_segment_duration(self):
        """Test maximum segment duration enforcement."""
        # Add first chunk
        audio = {"speaker_hint": "grace", "chunk": 0}
        self.agent.process_audio_stream(audio, timestamp=0.0)
        
        # Add chunks up to max duration
        for i in range(1, 100):
            audio = {"speaker_hint": "grace", "chunk": i}
            segments = self.agent.process_audio_stream(audio, timestamp=i * 0.2)
            
            # Check if segment was created due to max duration
            if segments:
                segment = segments[0]
                self.assertLessEqual(segment.duration, self.agent.max_segment_duration)
                break
    
    def test_combine_audio_chunks(self):
        """Test combining audio chunks."""
        chunks = [
            {"speaker_hint": "henry", "part": 1},
            {"speaker_hint": "henry", "part": 2},
            {"speaker_hint": "henry", "part": 3}
        ]
        
        combined = self.agent._combine_audio_chunks(chunks)
        
        # In mock mode, should combine metadata
        self.assertIn("chunk_count", combined)
        self.assertEqual(combined["chunk_count"], 3)
    
    def test_diarize_file(self):
        """Test diarizing an audio file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Diarize file
            segments = self.agent.diarize_file(tmp_path)
            
            # Should create mock segments
            self.assertGreater(len(segments), 0)
            
            # Check segments have required fields
            for segment in segments:
                self.assertIsNotNone(segment.start_time)
                self.assertIsNotNone(segment.end_time)
                self.assertIsNotNone(segment.speaker_id)
                self.assertGreater(segment.confidence, 0)
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_voice_activity_detection(self):
        """Test VAD functionality."""
        audio_data = {"mock": True}
        
        regions = self.agent.apply_voice_activity_detection(audio_data)
        
        # Should return mock regions
        self.assertGreater(len(regions), 0)
        
        for start, end in regions:
            self.assertLess(start, end)
            self.assertGreaterEqual(start, 0)
    
    def test_merge_segments(self):
        """Test merging adjacent segments."""
        # Create segments
        segments = [
            AudioSegment(0.0, 2.0, "iris", 0.9),
            AudioSegment(2.2, 4.0, "iris", 0.85),  # Small gap, same speaker
            AudioSegment(4.1, 6.0, "iris", 0.88),  # Small gap, same speaker
            AudioSegment(7.0, 9.0, "jack", 0.92),  # Large gap, different speaker
            AudioSegment(9.1, 11.0, "jack", 0.90)  # Small gap, same speaker
        ]
        
        # Add transcripts
        for i, seg in enumerate(segments):
            seg.transcript = f"Segment {i+1}"
        
        # Merge segments
        merged = self.agent.merge_segments(segments, max_gap=0.5)
        
        # Should merge segments with small gaps and same speaker
        self.assertLess(len(merged), len(segments))
        
        # Check first merged segment (iris)
        self.assertEqual(merged[0].speaker_id, "iris")
        self.assertEqual(merged[0].start_time, 0.0)
        self.assertEqual(merged[0].end_time, 6.0)
        
        # Check second segment (jack)
        self.assertEqual(merged[1].speaker_id, "jack")
        self.assertEqual(merged[1].start_time, 7.0)
    
    def test_export_timeline(self):
        """Test exporting timeline."""
        # Create segments
        segments = [
            AudioSegment(0.0, 5.0, "kate", 0.95),
            AudioSegment(5.5, 10.0, "liam", 0.88),
            AudioSegment(10.5, 15.0, "kate", 0.92)
        ]
        
        segments[0].transcript = "Hello, I'm Kate"
        segments[1].transcript = "Hi Kate, I'm Liam"
        segments[2].transcript = "Nice to meet you"
        
        # Export timeline
        timeline = self.agent.export_timeline(segments)
        
        # Check timeline contains expected information
        self.assertIn("Speaker Timeline", timeline)
        self.assertIn("kate", timeline)
        self.assertIn("liam", timeline)
        self.assertIn("95.0%", timeline)  # Kate's confidence
        self.assertIn("88.0%", timeline)  # Liam's confidence
    
    def test_run_process_stream(self):
        """Test run method with process_stream command."""
        command = json.dumps({
            "command": "process_stream",
            "audio_data": {"speaker_hint": "mary"},
            "timestamp": 1.5
        })
        
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("segments", result)
        self.assertIn("buffer_size", result)
    
    def test_run_diarize_file(self):
        """Test run method with diarize_file command."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            command = json.dumps({
                "command": "diarize_file",
                "file_path": str(tmp_path)
            })
            
            response = self.agent.run(command)
            result = json.loads(response)
            
            self.assertIn("segments", result)
            self.assertIn("total_duration", result)
            self.assertIn("speaker_count", result)
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_run_timeline(self):
        """Test run method with timeline command."""
        # Add some segments first
        self.agent.current_segments = [
            AudioSegment(0.0, 3.0, "nina", 0.9),
            AudioSegment(3.5, 6.0, "oscar", 0.85)
        ]
        
        command = json.dumps({"command": "timeline"})
        response = self.agent.run(command)
        
        self.assertIn("Speaker Timeline", response)
        self.assertIn("nina", response)
        self.assertIn("oscar", response)
    
    def test_run_merge(self):
        """Test run method with merge command."""
        # Add segments to merge
        self.agent.current_segments = [
            AudioSegment(0.0, 2.0, "paul", 0.9),
            AudioSegment(2.1, 4.0, "paul", 0.88)
        ]
        
        command = json.dumps({
            "command": "merge",
            "max_gap": 0.5
        })
        
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertIn("original_count", result)
        self.assertIn("merged_count", result)
        self.assertIn("segments", result)
        self.assertLess(result["merged_count"], result["original_count"])
    
    def test_run_stats(self):
        """Test run method with stats command."""
        # Set up some state
        self.agent.current_segments = [
            AudioSegment(0.0, 3.0, "quinn", 0.9)
        ]
        self.agent.last_speaker_id = "quinn"
        self.agent.last_change_time = 3.0
        
        command = json.dumps({"command": "stats"})
        response = self.agent.run(command)
        result = json.loads(response)
        
        self.assertEqual(result["total_segments"], 1)
        self.assertEqual(result["last_speaker"], "quinn")
        self.assertEqual(result["last_change_time"], 3.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for diarization workflow."""
    
    def test_stream_to_segments_workflow(self):
        """Test complete workflow from stream to segments."""
        # Create agents
        speaker_agent = SpeakerIdentificationAgent(use_mock=True)
        diarization_agent = SpeakerDiarizationAgent(
            speaker_agent=speaker_agent,
            use_mock=True
        )
        
        # Simulate conversation stream
        conversation = [
            ("alice", 0.0, 2.0),
            ("alice", 2.0, 4.0),
            ("bob", 4.5, 6.5),
            ("bob", 6.5, 8.0),
            ("alice", 8.5, 10.0)
        ]
        
        all_segments = []
        
        for speaker, start, end in conversation:
            # Simulate streaming chunks
            for t in np.arange(start, end, 0.1):
                audio = {"speaker_hint": speaker, "time": t}
                segments = diarization_agent.process_audio_stream(audio, t)
                all_segments.extend(segments)
        
        # Finalize last segment
        final_segments = diarization_agent._finalize_current_segment(10.0)
        all_segments.extend(final_segments)
        
        # Should have created segments
        self.assertGreater(len(all_segments), 0)
        
        # Merge segments
        merged = diarization_agent.merge_segments(all_segments)
        
        # Should have fewer merged segments
        self.assertLessEqual(len(merged), len(all_segments))
    
    def test_file_diarization_workflow(self):
        """Test file-based diarization workflow."""
        agent = SpeakerDiarizationAgent(use_mock=True)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Diarize file
            segments = agent.diarize_file(tmp_path)
            
            # Apply VAD
            vad_regions = agent.apply_voice_activity_detection({"file": str(tmp_path)})
            
            # Export timeline
            timeline = agent.export_timeline(segments)
            
            # Verify results
            self.assertGreater(len(segments), 0)
            self.assertGreater(len(vad_regions), 0)
            self.assertIn("Speaker Timeline", timeline)
        
        finally:
            tmp_path.unlink(missing_ok=True)


def run_tests():
    """Run all tests."""
    # Initialize test output writer
    writer = TestOutputWriter("unit", "test_speaker_diarization_agent")
    output_dir = writer.get_output_dir()
    
    # Start test run
    start_time = datetime.now()
    print(f"\nTest output directory: {output_dir}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAudioSegment))
    suite.addTests(loader.loadTestsFromTestCase(TestSpeakerDiarizationAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Prepare results
    test_results = {
        "test_name": "test_speaker_diarization_agent",
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
    summary = f"""Speaker Diarization Agent Test Results
=======================================
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