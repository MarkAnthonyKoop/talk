#!/usr/bin/env python3
"""
PyannoteAgent - Wrapper for pyannote.audio speaker diarization.

This agent provides real speaker diarization using pyannote.audio v3.1.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio import Audio
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    log.warning("pyannote.audio not available. Install with: pip install pyannote.audio")


class PyannoteAgent:
    """
    Real speaker diarization using pyannote.audio.
    
    This agent performs:
    - Voice Activity Detection (VAD)
    - Speaker segmentation
    - Speaker clustering
    - Overlap detection
    """
    
    def __init__(self, 
                 hf_token: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize PyannoteAgent.
        
        Args:
            hf_token: Hugging Face API token for model access
            device: Device to run on ('cuda' or 'cpu')
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio not installed")
        
        self.hf_token = hf_token
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.audio = Audio(sample_rate=16000, mono="downmix")
        
        log.info(f"PyannoteAgent initialized on {self.device}")
    
    def load_pipeline(self):
        """Load the diarization pipeline."""
        if self.pipeline is None:
            if not self.hf_token:
                raise ValueError("Hugging Face token required. Get one at https://huggingface.co/settings/tokens")
            
            try:
                print(f"Loading pipeline with token: {self.hf_token[:10]}...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                if self.pipeline:
                    self.pipeline.to(torch.device(self.device))
                    log.info("Loaded pyannote speaker-diarization-3.1 pipeline")
                else:
                    raise Exception("Pipeline is None - likely auth issue")
            except Exception as e:
                log.error(f"Failed to load pipeline: {e}")
                print("\n⚠️  IMPORTANT: You need to accept the model conditions:")
                print("1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("2. Click 'Agree and access repository'")
                print("3. Also visit: https://huggingface.co/pyannote/segmentation-3.0")
                print("4. Click 'Agree and access repository' there too")
                print("\nThen try again.")
                raise
    
    def diarize_file(self, audio_path: str, 
                     min_speakers: Optional[int] = None,
                     max_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            List of segments with speaker labels
        """
        self.load_pipeline()
        
        # Configure speaker counts if provided
        params = {}
        if min_speakers is not None:
            params['min_speakers'] = min_speakers
        if max_speakers is not None:
            params['max_speakers'] = max_speakers
        
        # Run diarization
        diarization = self.pipeline(audio_path, **params)
        
        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'speaker': speaker,
                'confidence': 1.0  # pyannote doesn't provide per-segment confidence
            })
        
        return segments
    
    def extract_speaker_segments(self, audio_path: str,
                                segments: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
        """
        Extract audio for each speaker segment.
        
        Args:
            audio_path: Path to audio file
            segments: List of segment dictionaries
            
        Returns:
            Dictionary mapping speaker IDs to audio arrays
        """
        speaker_audio = {}
        
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_audio:
                speaker_audio[speaker] = []
            
            # Extract audio for this segment
            seg = Segment(segment['start'], segment['end'])
            waveform, sample_rate = self.audio.crop(audio_path, seg)
            
            # Convert to numpy array
            audio_array = waveform.numpy().squeeze()
            speaker_audio[speaker].append(audio_array)
        
        return speaker_audio
    
    def detect_voice_activity(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Detect voice activity regions.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start, end) tuples for speech regions
        """
        # Use the pipeline's VAD component
        self.load_pipeline()
        
        # Run VAD through the pipeline
        # Note: pyannote v3.1 includes VAD in the main pipeline
        diarization = self.pipeline(audio_path)
        
        # Extract speech regions
        speech_regions = []
        for segment in diarization.get_timeline():
            speech_regions.append((segment.start, segment.end))
        
        return speech_regions
    
    def get_speaker_statistics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from diarization results.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_speakers': 0,
            'total_duration': 0,
            'speakers': {}
        }
        
        speaker_times = {}
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']
            
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += duration
            stats['total_duration'] += duration
        
        stats['total_speakers'] = len(speaker_times)
        
        for speaker, time in speaker_times.items():
            stats['speakers'][speaker] = {
                'total_time': time,
                'percentage': (time / stats['total_duration'] * 100) if stats['total_duration'] > 0 else 0
            }
        
        return stats