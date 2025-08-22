#!/usr/bin/env python3
"""Test TTS with ffmpeg installed."""

try:
    from RealtimeTTS import TextToAudioStream, SystemEngine
    
    print("Testing RealtimeTTS with ffmpeg...")
    
    engine = SystemEngine()
    stream = TextToAudioStream(engine)
    
    test_text = "Hello! This is Listen version 4. I can now speak back to you!"
    print(f"Speaking: '{test_text}'")
    
    stream.feed(test_text)
    stream.play()
    
    print("✅ RealtimeTTS test successful!")
    
except Exception as e:
    print(f"❌ RealtimeTTS test failed: {e}")