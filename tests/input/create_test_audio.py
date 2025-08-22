#!/usr/bin/env python3
"""
Create a test audio file with simulated multiple speakers.
This creates a WAV file with different frequency tones to simulate different speakers.
"""

import wave
import struct
import math
import random

def create_multi_speaker_audio(output_path, duration=30):
    """
    Create a test audio file simulating multiple speakers.
    
    Args:
        output_path: Path to save the WAV file
        duration: Duration in seconds
    """
    sample_rate = 16000
    
    # Define "speakers" with different characteristics
    speakers = [
        {"id": "speaker1", "base_freq": 120, "name": "Alice"},  # Lower voice
        {"id": "speaker2", "base_freq": 200, "name": "Bob"},    # Medium voice  
        {"id": "speaker3", "base_freq": 280, "name": "Carol"},  # Higher voice
    ]
    
    # Create conversation segments
    segments = []
    current_time = 0.0
    
    # Simulate a conversation
    conversation = [
        (0, 3, 0, "Hello everyone, welcome to the meeting."),
        (3.5, 2, 1, "Thanks for having us."),
        (6, 2.5, 2, "Great to be here."),
        (9, 4, 0, "Let's start with the first topic."),
        (13.5, 3, 1, "I have some thoughts on that."),
        (17, 2, 2, "Me too."),
        (19.5, 3, 0, "Excellent, please share."),
        (23, 3.5, 1, "Well, I think we should consider..."),
        (27, 2.5, 2, "That's a good point."),
    ]
    
    # Generate audio data
    audio_data = []
    
    for sample_idx in range(sample_rate * duration):
        time = sample_idx / sample_rate
        value = 0
        
        # Check which speaker is active
        for start, dur, speaker_idx, text in conversation:
            if start <= time < start + dur:
                speaker = speakers[speaker_idx]
                # Generate tone with some variation
                freq = speaker["base_freq"] * (1 + 0.1 * math.sin(2 * math.pi * 5 * time))
                value = int(16383 * math.sin(2 * math.pi * freq * time))
                
                # Add some noise for realism
                value += random.randint(-1000, 1000)
                break
        
        # Ensure value is in valid range
        value = max(-32767, min(32767, value))
        audio_data.append(value)
    
    # Write WAV file
    with wave.open(output_path, 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        
        for value in audio_data:
            wav.writeframes(struct.pack('<h', value))
    
    print(f"Created test audio: {output_path}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print("\nSimulated conversation:")
    for start, dur, speaker_idx, text in conversation:
        speaker = speakers[speaker_idx]
        print(f"  {start:5.1f}s - {start+dur:5.1f}s: [{speaker['name']}] {text}")
    
    return conversation, speakers

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    output_path = Path("multi_speaker_test.wav")
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    create_multi_speaker_audio(str(output_path))