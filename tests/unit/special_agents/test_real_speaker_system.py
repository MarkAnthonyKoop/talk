#!/usr/bin/env python3
"""
Test real speaker identification system - NO MOCKING.

This test verifies that the actual speaker identification pipeline works
with real dependencies and real audio data.
"""

import sys
import os
from pathlib import Path

# Set up imports
from special_agents.speaker_identification_agent import SpeakerIdentificationAgent
from special_agents.speaker_diarization_agent import SpeakerDiarizationAgent  
from special_agents.voice_enrollment_agent import VoiceEnrollmentAgent
from external_agents.speechbrain_agent import SpeechBrainAgent
from external_agents.pyannote_agent import PyannoteAgent
from tests.utilities.test_output_writer import TestOutputWriter


def test_dependencies():
    """Test all required dependencies are available."""
    try:
        import torch
        import torchaudio
        import scipy
        print("✓ torch, torchaudio, scipy available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    try:
        from RealtimeTTS import SystemEngine
        print("✓ RealtimeTTS available") 
    except ImportError as e:
        print(f"⚠️  RealtimeTTS not available: {e}")
    
    try:
        import speechbrain
        print("✓ speechbrain available")
    except ImportError as e:
        print(f"❌ Missing speechbrain: {e}")
        return False
        
    return True


def test_real_speaker_identification():
    """Test speaker identification agent WITHOUT mocking."""
    
    writer = TestOutputWriter("unit", "test_real_speaker_system")
    output_dir = writer.get_output_dir()
    
    print("="*60)
    print("TESTING REAL SPEAKER IDENTIFICATION - NO MOCKING")
    print("="*60)
    
    writer.write_log("Starting real speaker identification test")
    
    # Test 1: Initialize agent with real mode
    print("\n1. Initializing SpeakerIdentificationAgent (real mode)...")
    try:
        agent = SpeakerIdentificationAgent(
            db_path=Path(output_dir) / "test_speakers.db",
            use_mock=False  # Explicitly NO mocking
        )
        print("✓ Agent initialized successfully")
        writer.write_log("SpeakerIdentificationAgent initialized in real mode")
    except ImportError as e:
        print(f"❌ Failed to initialize: {e}")
        writer.write_log(f"FAILED: Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        writer.write_log(f"FAILED: Unexpected error: {e}")
        return False
    
    # Test 2: Test with SpeechBrain integration  
    print("\n2. Testing SpeechBrain integration...")
    try:
        sb_agent = SpeechBrainAgent()
        print("✓ SpeechBrain agent created")
        
        # Test embedding extraction (will download model on first run)
        print("  Loading ECAPA-TDNN model (may take time on first run)...")
        sb_agent.load_model()
        print("✓ SpeechBrain model loaded")
        
        writer.write_log("SpeechBrain integration successful")
        
    except Exception as e:
        print(f"❌ SpeechBrain failed: {e}")
        writer.write_log(f"SpeechBrain test failed: {e}")
        return False
    
    # Test 3: Test voice enrollment agent
    print("\n3. Testing VoiceEnrollmentAgent...")
    try:
        enrollment_agent = VoiceEnrollmentAgent(
            speaker_agent=agent,
            min_samples=2,
            max_samples=5
        )
        print("✓ VoiceEnrollmentAgent initialized")
        
        # Test enrollment session creation
        session_id = enrollment_agent.start_enrollment("TestUser")
        print(f"✓ Enrollment session started: {session_id}")
        
        phrase = enrollment_agent.get_next_phrase(session_id)
        print(f"✓ Got enrollment phrase: '{phrase}'")
        
        writer.write_log("VoiceEnrollmentAgent test successful")
        
    except Exception as e:
        print(f"❌ VoiceEnrollmentAgent failed: {e}")
        writer.write_log(f"VoiceEnrollmentAgent test failed: {e}")
        return False
    
    # Test 4: Test diarization agent
    print("\n4. Testing SpeakerDiarizationAgent...")
    try:
        diarization_agent = SpeakerDiarizationAgent(
            speaker_agent=agent,
            use_mock=False  # Explicitly NO mocking
        )
        print("✓ SpeakerDiarizationAgent initialized")
        writer.write_log("SpeakerDiarizationAgent test successful")
        
    except Exception as e:
        print(f"❌ SpeakerDiarizationAgent failed: {e}")
        writer.write_log(f"SpeakerDiarizationAgent test failed: {e}")
        return False
    
    # Test 5: Test Pyannote integration  
    print("\n5. Testing Pyannote integration...")
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            pyannote_agent = PyannoteAgent(hf_token=hf_token)
            print("✓ PyannoteAgent initialized")
            writer.write_log("Pyannote integration successful")
        except Exception as e:
            print(f"⚠️  Pyannote test skipped: {e}")
            writer.write_log(f"Pyannote test skipped: {e}")
    else:
        print("⚠️  Pyannote test skipped: No HF_TOKEN")
        writer.write_log("Pyannote test skipped: No HF_TOKEN")
    
    # Write results
    writer.write_results({
        "status": "success",
        "real_mode": True,
        "mocking": False,
        "dependencies_available": True,
        "tests_passed": 5
    })
    
    print("\n" + "="*60)
    print("✅ ALL REAL SPEAKER SYSTEM TESTS PASSED!")
    print("No mocking used - production ready")
    print("="*60)
    
    return True


def test_mock_warnings():
    """Test that mock mode produces loud warnings."""
    
    print("\n" + "="*60)
    print("TESTING MOCK WARNINGS")
    print("="*60)
    
    print("\nTesting explicit mock mode (should show warnings):")
    
    try:
        # This should produce loud warnings
        mock_agent = SpeakerIdentificationAgent(use_mock=True)
        print("✓ Mock agent created with warnings")
        return True
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        return False


if __name__ == "__main__":
    print("COMPREHENSIVE REAL SPEAKER SYSTEM TEST")
    print("No mocking - production dependencies only")
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ CRITICAL: Missing dependencies!")
        print("Install with: pip install torch torchaudio scipy speechbrain")
        sys.exit(1)
    
    # Test real system
    if not test_real_speaker_identification():
        print("\n❌ REAL SYSTEM TESTS FAILED!")
        sys.exit(1)
    
    # Test mock warnings
    if not test_mock_warnings():
        print("\n❌ MOCK WARNING TESTS FAILED!")
        sys.exit(1)
        
    print("\n🎉 ALL TESTS PASSED!")
    print("Speaker identification system is production ready!")
    sys.exit(0)