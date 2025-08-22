#!/usr/bin/env python3
"""
Test speaker diarization on a WAV file using real pyannote models.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Imports relative to PYTHONPATH
from external_agents.pyannote_agent import PyannoteAgent
from external_agents.speechbrain_agent import SpeechBrainAgent
from tests.utilities.test_output_writer import TestOutputWriter


def format_time(seconds):
    """Format seconds to MM:SS.S format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def test_diarization(audio_path: str):
    """
    Test speaker diarization on an audio file.
    
    Args:
        audio_path: Path to WAV file
    """
    writer = TestOutputWriter("listen", "test_diarization_wav")
    output_dir = writer.get_output_dir()
    
    # Check HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: Please set HF_TOKEN environment variable")
        return 1
    
    print("=" * 60)
    print("SPEAKER DIARIZATION TEST")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"HF Token: {hf_token[:10]}...")
    print(f"Output dir: {output_dir}")
    print()
    
    # Initialize pyannote agent
    print("Loading pyannote diarization pipeline...")
    pyannote = PyannoteAgent(hf_token=hf_token)
    
    # Load the pipeline (this downloads models on first run)
    print("Loading models (this may take a while on first run)...")
    pyannote.load_pipeline()
    print("✓ Pipeline loaded")
    print()
    
    # Perform diarization
    print("Performing speaker diarization...")
    print("-" * 40)
    
    start_time = datetime.now()
    segments = pyannote.diarize_file(
        audio_path,
        min_speakers=2,  # We know there are at least 2 speakers
        max_speakers=5   # But no more than 5
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Display results
    print(f"\nDiarization completed in {elapsed:.2f} seconds")
    print(f"Found {len(segments)} speech segments")
    print()
    
    # Group by speaker
    speakers = {}
    for seg in segments:
        speaker = seg['speaker']
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(seg)
    
    print(f"Identified {len(speakers)} unique speakers")
    print()
    
    # Show timeline
    print("SPEAKER TIMELINE:")
    print("-" * 40)
    
    for seg in segments[:20]:  # Show first 20 segments
        start = format_time(seg['start'])
        end = format_time(seg['end'])
        duration = seg['end'] - seg['start']
        speaker = seg['speaker']
        
        # Create visual bar
        bar_length = int(duration * 5)  # Scale for display
        bar = "█" * min(bar_length, 30)
        
        print(f"{start} - {end} [{speaker:>10}] {bar} ({duration:.1f}s)")
    
    if len(segments) > 20:
        print(f"... and {len(segments) - 20} more segments")
    
    print()
    
    # Show speaker statistics
    print("SPEAKER STATISTICS:")
    print("-" * 40)
    
    for speaker, segs in speakers.items():
        total_time = sum(s['end'] - s['start'] for s in segs)
        avg_duration = total_time / len(segs)
        
        print(f"{speaker}:")
        print(f"  Total speaking time: {total_time:.1f}s")
        print(f"  Number of segments: {len(segs)}")
        print(f"  Average segment duration: {avg_duration:.2f}s")
        print()
    
    # Save results
    results = {
        "audio_file": audio_path,
        "processing_time": elapsed,
        "total_segments": len(segments),
        "unique_speakers": len(speakers),
        "segments": segments[:50],  # Save first 50 for analysis
        "speaker_stats": {
            speaker: {
                "total_time": sum(s['end'] - s['start'] for s in segs),
                "segment_count": len(segs)
            }
            for speaker, segs in speakers.items()
        }
    }
    
    writer.write_results(results)
    writer.write_log(f"Diarization test completed: {len(segments)} segments, {len(speakers)} speakers")
    
    print("=" * 60)
    print("✓ Diarization test complete!")
    print(f"Results saved to: {output_dir}")
    
    return 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test speaker diarization on WAV files"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="tests/input/multi_speaker_test.wav",
        help="Path to WAV file (default: tests/input/multi_speaker_test.wav)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    return test_diarization(args.audio_file)


if __name__ == "__main__":
    sys.exit(main())