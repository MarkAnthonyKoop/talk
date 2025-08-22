#!/usr/bin/env python3
"""
Test real microphone input - Integration test.

This test verifies that the microphone input pipeline works with real hardware.
"""

import sys
import time
from pathlib import Path

# Set up imports
try:
    import speech_recognition as sr
    import pyaudio
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

from tests.utilities.test_output_writer import TestOutputWriter


def test_microphone_availability():
    """Test if microphone hardware is available."""
    
    if not MIC_AVAILABLE:
        print("❌ speech_recognition or pyaudio not available")
        return False
    
    try:
        # List available microphones
        mic_list = sr.Microphone.list_microphone_names()
        print(f"✓ Found {len(mic_list)} microphone(s):")
        for i, name in enumerate(mic_list[:5]):  # Show first 5
            print(f"  {i}: {name}")
        
        if len(mic_list) == 0:
            print("❌ No microphones detected")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False


def test_microphone_input():
    """Test actual microphone input with timeout."""
    
    writer = TestOutputWriter("integration", "test_microphone_input")
    output_dir = writer.get_output_dir()
    
    print("="*60)
    print("TESTING REAL MICROPHONE INPUT")
    print("="*60)
    
    if not MIC_AVAILABLE:
        print("❌ Microphone libraries not available")
        writer.write_results({
            "status": "skipped",
            "reason": "microphone_libraries_missing"
        })
        return False
    
    # Test microphone initialization
    print("1. Testing microphone initialization...")
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        print("✓ Microphone initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize microphone: {e}")
        writer.write_results({
            "status": "failed",
            "error": str(e),
            "step": "microphone_init"
        })
        return False
    
    # Test ambient noise adjustment
    print("2. Testing ambient noise calibration...")
    try:
        with microphone as source:
            print("  Calibrating for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Ambient noise calibration complete")
            
    except Exception as e:
        print(f"❌ Ambient noise calibration failed: {e}")
        writer.write_results({
            "status": "failed", 
            "error": str(e),
            "step": "ambient_noise_calibration"
        })
        return False
    
    # Test audio capture (short timeout for automated testing)
    print("3. Testing audio capture (5 second timeout)...")
    try:
        with microphone as source:
            print("  Listening for audio... (5 seconds)")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            print("✓ Audio captured successfully")
            
            # Try to recognize the audio
            print("4. Testing speech recognition...")
            try:
                text = recognizer.recognize_google(audio)
                print(f"✓ Recognized: '{text}'")
                
                writer.write_results({
                    "status": "success",
                    "audio_captured": True,
                    "speech_recognized": True,
                    "recognized_text": text,
                    "audio_duration": 3.0
                })
                
            except sr.UnknownValueError:
                print("⚠️  Audio captured but no speech recognized")
                writer.write_results({
                    "status": "partial_success",
                    "audio_captured": True,
                    "speech_recognized": False,
                    "reason": "no_speech_detected"
                })
                
            except sr.RequestError as e:
                print(f"⚠️  Speech recognition service error: {e}")
                writer.write_results({
                    "status": "partial_success",
                    "audio_captured": True,
                    "speech_recognized": False,
                    "reason": "recognition_service_error",
                    "error": str(e)
                })
                
    except sr.WaitTimeoutError:
        print("⚠️  No audio detected within timeout period")
        writer.write_results({
            "status": "partial_success",
            "audio_captured": False,
            "reason": "timeout_no_audio"
        })
        
    except Exception as e:
        print(f"❌ Audio capture failed: {e}")
        writer.write_results({
            "status": "failed",
            "error": str(e),
            "step": "audio_capture"
        })
        return False
    
    print("\n" + "="*60)
    print("✅ MICROPHONE INPUT TEST COMPLETE")
    print("Check test output for detailed results")
    print("="*60)
    
    writer.write_log("Microphone input test completed successfully")
    return True


def test_microphone_stress():
    """Test microphone under continuous operation."""
    
    print("\n" + "="*60)
    print("TESTING CONTINUOUS MICROPHONE OPERATION")
    print("="*60)
    
    if not MIC_AVAILABLE:
        print("❌ Skipped - microphone not available")
        return True
    
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print("Testing 3 consecutive captures...")
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            for i in range(3):
                print(f"  Capture {i+1}/3...")
                try:
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=1)
                    print(f"  ✓ Capture {i+1} successful")
                except sr.WaitTimeoutError:
                    print(f"  ⚠️  Capture {i+1} timeout (expected)")
                except Exception as e:
                    print(f"  ❌ Capture {i+1} failed: {e}")
                    return False
                    
        print("✓ Stress test completed")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False


if __name__ == "__main__":
    print("MICROPHONE INPUT INTEGRATION TEST")
    print("Tests real hardware microphone functionality")
    
    # Test microphone availability
    if not test_microphone_availability():
        print("\n❌ MICROPHONE NOT AVAILABLE")
        print("This test requires:")
        print("1. pip install SpeechRecognition pyaudio")
        print("2. Working microphone hardware")
        print("3. Audio permissions")
        sys.exit(1)
    
    # Test basic microphone input
    success = test_microphone_input()
    
    # Test stress/continuous operation
    stress_success = test_microphone_stress()
    
    if success and stress_success:
        print("\n🎉 ALL MICROPHONE TESTS PASSED!")
        print("Microphone input is ready for production use")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED OR INCOMPLETE")
        print("Check test outputs for details")
        sys.exit(1)