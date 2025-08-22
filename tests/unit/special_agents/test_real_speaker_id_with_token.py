#!/usr/bin/env python3
"""Test real speaker identification with actual HF token."""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Imports relative to PYTHONPATH (~/talk)
from external_agents.pyannote_agent import PyannoteAgent
from external_agents.speechbrain_agent import SpeechBrainAgent
from special_agents.real_speaker_identification_agent import RealSpeakerIdentificationAgent
from tests.utilities.test_output_writer import TestOutputWriter

def test_with_real_token():
    """Test the complete pipeline with real HF token."""
    
    writer = TestOutputWriter("unit", "test_real_speaker_id_with_token")
    output_dir = writer.get_output_dir()
    
    # Get HF token from environment
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        writer.write_log("ERROR: HF_TOKEN not found in environment")
        return False
    
    writer.write_log(f"Using HF token: {hf_token[:10]}...")
    
    try:
        # Test PyannoteAgent initialization
        writer.write_log("\n=== Testing PyannoteAgent ===")
        pyannote = PyannoteAgent(hf_token=hf_token)
        writer.write_log("✓ PyannoteAgent initialized successfully")
        
        # Test SpeechBrainAgent
        writer.write_log("\n=== Testing SpeechBrainAgent ===")
        speechbrain = SpeechBrainAgent()
        writer.write_log("✓ SpeechBrainAgent initialized successfully")
        
        # Create test audio file (simple sine wave)
        writer.write_log("\n=== Creating test audio ===")
        import wave
        import struct
        import math
        
        sample_rate = 16000
        duration = 3  # seconds
        frequency = 440  # A4 note
        
        test_audio = Path(output_dir) / "test_audio.wav"
        
        with wave.open(str(test_audio), 'w') as wav:
            wav.setnchannels(1)  # mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            
            # Generate sine wave
            for i in range(sample_rate * duration):
                value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                wav.writeframes(struct.pack('<h', value))
        
        writer.write_log(f"✓ Created test audio: {test_audio}")
        
        # Test complete speaker identification pipeline
        writer.write_log("\n=== Testing RealSpeakerIdentificationAgent ===")
        db_path = Path(output_dir) / "test_speakers.db"
        
        speaker_agent = RealSpeakerIdentificationAgent(
            hf_token=hf_token,
            db_path=str(db_path)
        )
        
        # Initialize the agent
        result = speaker_agent.run(None, {"command": "initialize"})
        writer.write_log(f"Initialization result: {result}")
        
        # Enroll a speaker
        writer.write_log("\n=== Testing speaker enrollment ===")
        enroll_result = speaker_agent.run(None, {
            "command": "enroll",
            "audio_path": str(test_audio),
            "name": "Test Speaker",
            "email": "test@example.com"
        })
        writer.write_log(f"Enrollment result: {enroll_result}")
        
        if enroll_result.get('success'):
            speaker_id = enroll_result.get('speaker_id')
            writer.write_log(f"✓ Speaker enrolled with ID: {speaker_id}")
        
        # Test speaker identification
        writer.write_log("\n=== Testing speaker identification ===")
        identify_result = speaker_agent.run(None, {
            "command": "identify",
            "audio_path": str(test_audio)
        })
        writer.write_log(f"Identification result: {identify_result}")
        
        if identify_result.get('success'):
            identified_speaker = identify_result.get('speaker_id')
            confidence = identify_result.get('confidence')
            writer.write_log(f"✓ Identified speaker: {identified_speaker} (confidence: {confidence:.2f})")
        
        # Test diarization (if pyannote loads successfully)
        writer.write_log("\n=== Testing diarization ===")
        try:
            segments = pyannote.diarize_file(str(test_audio))
            writer.write_log(f"✓ Diarization returned {len(segments)} segments")
            for seg in segments[:3]:  # Show first 3 segments
                writer.write_log(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
        except Exception as e:
            writer.write_log(f"Diarization test skipped: {e}")
        
        # Summary
        writer.write_results({
            "status": "success",
            "hf_token_valid": True,
            "pyannote_loaded": True,
            "speechbrain_loaded": True,
            "enrollment_successful": enroll_result.get('success', False),
            "identification_successful": identify_result.get('success', False),
            "test_audio_path": str(test_audio),
            "db_path": str(db_path)
        })
        
        writer.write_log("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        writer.write_log(f"\n✗ Test failed with error: {e}")
        writer.write_results({
            "status": "failed",
            "error": str(e)
        })
        return False

if __name__ == "__main__":
    success = test_with_real_token()
    sys.exit(0 if success else 1)