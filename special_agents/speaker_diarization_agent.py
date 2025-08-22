#!/usr/bin/env python3
"""
SpeakerDiarizationAgent - Segments audio into speaker turns.

This agent handles:
- Audio segmentation by speaker
- Voice activity detection (VAD)
- Speaker change detection
- Turn boundary identification
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from agent.agent import Agent
from special_agents.speaker_identification_agent import SpeakerIdentificationAgent

log = logging.getLogger(__name__)


class AudioSegment:
    """Represents a segment of audio with speaker information."""
    
    def __init__(self,
                 start_time: float,
                 end_time: float,
                 speaker_id: Optional[str] = None,
                 confidence: float = 0.0,
                 audio_data: Optional[Any] = None):
        """
        Initialize an audio segment.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            speaker_id: Identified speaker (can be None)
            confidence: Confidence score for speaker identification
            audio_data: Raw audio data for this segment
        """
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.speaker_id = speaker_id
        self.confidence = confidence
        self.audio_data = audio_data
        self.transcript = ""
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
            "transcript": self.transcript,
            "metadata": self.metadata
        }


class SpeakerDiarizationAgent(Agent):
    """
    Agent that segments audio by speaker.
    
    This agent:
    1. Detects voice activity
    2. Identifies speaker changes
    3. Segments audio into speaker turns
    4. Associates segments with speaker IDs
    """
    
    def __init__(self,
                 speaker_agent: Optional[SpeakerIdentificationAgent] = None,
                 vad_threshold: float = 0.5,
                 min_segment_duration: float = 0.5,
                 max_segment_duration: float = 30.0,
                 use_mock: bool = False,
                 **kwargs):
        """
        Initialize the diarization agent.
        
        Args:
            speaker_agent: Speaker identification agent to use
            vad_threshold: Voice activity detection threshold
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            use_mock: Use mock processing for testing
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are an audio diarization specialist.",
            "You segment audio streams by speaker.",
            "You detect when speakers change.",
            "You identify who is speaking in each segment."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Don't default to mocking for speaker agent
        if speaker_agent is None:
            # Only use mock if explicitly requested
            self.speaker_agent = SpeakerIdentificationAgent(use_mock=use_mock)
        else:
            self.speaker_agent = speaker_agent
            
        self.vad_threshold = vad_threshold
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.use_mock = use_mock
        
        if self.use_mock:
            print("\n" + "!"*60)
            print("⚠️  WARNING: USING MOCK SPEAKER DIARIZATION")
            print("This is for testing only - no real diarization!")
            print("Set use_mock=False for production use")
            print("!"*60 + "\n")
        
        # Diarization state
        self.current_segments: List[AudioSegment] = []
        self.processing_buffer = []
        self.last_speaker_id = None
        self.last_change_time = 0.0
        
        log.info(f"Initialized SpeakerDiarizationAgent (VAD threshold: {vad_threshold})")
    
    def process_audio_stream(self, audio_data: Any, 
                            timestamp: float = None) -> List[AudioSegment]:
        """
        Process a chunk of audio stream.
        
        Args:
            audio_data: Audio data chunk
            timestamp: Current timestamp in seconds
            
        Returns:
            List of completed audio segments
        """
        if timestamp is None:
            timestamp = len(self.processing_buffer) * 0.1  # Assume 100ms chunks
        
        # Add to processing buffer
        self.processing_buffer.append({
            "data": audio_data,
            "timestamp": timestamp
        })
        
        # Check for speaker change
        if self._detect_speaker_change(audio_data):
            # Finalize current segment
            segments = self._finalize_current_segment(timestamp)
            return segments
        
        # Check if segment is too long
        if (self.processing_buffer and 
            timestamp - self.processing_buffer[0]["timestamp"] >= self.max_segment_duration):
            segments = self._finalize_current_segment(timestamp)
            return segments
        
        return []
    
    def _detect_speaker_change(self, audio_data: Any) -> bool:
        """
        Detect if speaker has changed.
        
        Args:
            audio_data: Current audio chunk
            
        Returns:
            True if speaker change detected
        """
        if self.use_mock:
            # Mock detection based on audio hints
            if isinstance(audio_data, dict):
                current_hint = audio_data.get("speaker_hint", "unknown")
                if hasattr(self, "_last_hint"):
                    changed = current_hint != self._last_hint
                    self._last_hint = current_hint
                    return changed
                self._last_hint = current_hint
                return False
            # Random change for testing
            return np.random.random() < 0.1
        
        # Real implementation would:
        # 1. Extract features from audio
        # 2. Compare with recent audio characteristics
        # 3. Use speaker embeddings to detect change
        # 4. Apply smoothing to avoid false positives
        
        return False
    
    def _finalize_current_segment(self, end_time: float) -> List[AudioSegment]:
        """
        Finalize the current audio segment.
        
        Args:
            end_time: End timestamp for the segment
            
        Returns:
            List containing the finalized segment
        """
        if not self.processing_buffer:
            return []
        
        start_time = self.processing_buffer[0]["timestamp"]
        duration = end_time - start_time
        
        # Skip very short segments
        if duration < self.min_segment_duration:
            self.processing_buffer.clear()
            return []
        
        # Combine audio data
        combined_audio = self._combine_audio_chunks(
            [chunk["data"] for chunk in self.processing_buffer]
        )
        
        # Identify speaker
        speaker_id, confidence, metadata = self.speaker_agent.identify_speaker(combined_audio)
        
        # Create segment
        segment = AudioSegment(
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            confidence=confidence,
            audio_data=combined_audio
        )
        segment.metadata = metadata
        
        # Clear buffer for next segment
        self.processing_buffer.clear()
        self.last_speaker_id = speaker_id
        self.last_change_time = end_time
        
        # Add to history
        self.current_segments.append(segment)
        
        log.debug(f"Finalized segment: {speaker_id} ({start_time:.1f}-{end_time:.1f}s)")
        
        return [segment]
    
    def _combine_audio_chunks(self, chunks: List[Any]) -> Any:
        """
        Combine multiple audio chunks into one.
        
        Args:
            chunks: List of audio chunks
            
        Returns:
            Combined audio data
        """
        if self.use_mock:
            # For mock, just return metadata from all chunks
            combined = {}
            for chunk in chunks:
                if isinstance(chunk, dict):
                    combined.update(chunk)
            combined["chunk_count"] = len(chunks)
            return combined
        
        # Real implementation would concatenate audio arrays
        # np.concatenate(chunks) for numpy arrays
        # Or use audio library specific methods
        
        return chunks[0] if chunks else None
    
    def diarize_file(self, audio_file: Path) -> List[AudioSegment]:
        """
        Diarize an entire audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of audio segments with speaker labels
        """
        log.info(f"Diarizing file: {audio_file}")
        
        if self.use_mock:
            # Mock diarization for testing
            segments = self._mock_diarize_file(audio_file)
        else:
            # Real implementation would:
            # 1. Load audio file
            # 2. Apply VAD to find speech regions
            # 3. Segment by speaker change points
            # 4. Cluster segments by speaker
            # 5. Assign speaker IDs
            segments = []
        
        log.info(f"Diarized into {len(segments)} segments")
        return segments
    
    def _mock_diarize_file(self, audio_file: Path) -> List[AudioSegment]:
        """
        Mock file diarization for testing.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Mock segments
        """
        # Create mock segments
        segments = []
        current_time = 0.0
        speakers = ["speaker_1", "speaker_2", "unknown_1"]
        
        for i in range(10):  # Create 10 segments
            duration = np.random.uniform(2.0, 10.0)
            speaker = speakers[i % len(speakers)]
            
            segment = AudioSegment(
                start_time=current_time,
                end_time=current_time + duration,
                speaker_id=speaker,
                confidence=np.random.uniform(0.7, 1.0),
                audio_data={"mock": True, "file": str(audio_file)}
            )
            
            segment.transcript = f"Mock transcript for segment {i+1}"
            segments.append(segment)
            
            current_time += duration
        
        return segments
    
    def apply_voice_activity_detection(self, audio_data: Any) -> List[Tuple[float, float]]:
        """
        Apply VAD to find speech regions.
        
        Args:
            audio_data: Audio data to process
            
        Returns:
            List of (start, end) tuples for speech regions
        """
        if self.use_mock:
            # Mock VAD regions
            return [
                (0.0, 2.5),
                (3.0, 5.5),
                (6.0, 8.0),
                (9.0, 12.0)
            ]
        
        # Real implementation would use:
        # - Energy-based VAD
        # - WebRTC VAD
        # - Neural network VAD
        # - Spectral features
        
        return []
    
    def merge_segments(self, segments: List[AudioSegment],
                      max_gap: float = 0.5) -> List[AudioSegment]:
        """
        Merge adjacent segments from same speaker.
        
        Args:
            segments: List of segments to merge
            max_gap: Maximum gap between segments to merge
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Check if should merge
            gap = next_seg.start_time - current.end_time
            same_speaker = next_seg.speaker_id == current.speaker_id
            
            if same_speaker and gap <= max_gap:
                # Merge segments
                current = AudioSegment(
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    speaker_id=current.speaker_id,
                    confidence=min(current.confidence, next_seg.confidence),
                    audio_data=self._combine_audio_chunks([
                        current.audio_data,
                        next_seg.audio_data
                    ])
                )
                current.transcript = f"{current.transcript} {next_seg.transcript}".strip()
            else:
                # Save current and start new
                merged.append(current)
                current = next_seg
        
        # Add last segment
        merged.append(current)
        
        log.debug(f"Merged {len(segments)} segments into {len(merged)}")
        return merged
    
    def export_timeline(self, segments: List[AudioSegment]) -> str:
        """
        Export segments as timeline.
        
        Args:
            segments: List of segments
            
        Returns:
            Timeline as formatted string
        """
        timeline = ["Speaker Timeline:", "=" * 50]
        
        for seg in segments:
            start = timedelta(seconds=seg.start_time)
            end = timedelta(seconds=seg.end_time)
            speaker = seg.speaker_id or "Unknown"
            conf = seg.confidence * 100
            
            timeline.append(
                f"[{start} - {end}] {speaker} ({conf:.1f}%)"
            )
            
            if seg.transcript:
                timeline.append(f"  > {seg.transcript[:100]}...")
        
        return "\n".join(timeline)
    
    def run(self, prompt: str) -> str:
        """
        Process diarization commands.
        
        Args:
            prompt: Command or audio data
            
        Returns:
            Response as JSON or text
        """
        try:
            # Parse JSON commands
            if prompt.startswith("{"):
                data = json.loads(prompt)
                command = data.get("command", "help")
                
                if command == "process_stream":
                    # Process audio stream chunk
                    audio_data = data.get("audio_data", {})
                    timestamp = data.get("timestamp")
                    
                    segments = self.process_audio_stream(audio_data, timestamp)
                    
                    return json.dumps({
                        "segments": [seg.to_dict() for seg in segments],
                        "buffer_size": len(self.processing_buffer)
                    }, indent=2)
                
                elif command == "diarize_file":
                    # Diarize entire file
                    file_path = Path(data.get("file_path", ""))
                    
                    if not file_path.exists():
                        return json.dumps({"error": "File not found"})
                    
                    segments = self.diarize_file(file_path)
                    
                    return json.dumps({
                        "segments": [seg.to_dict() for seg in segments],
                        "total_duration": segments[-1].end_time if segments else 0,
                        "speaker_count": len(set(s.speaker_id for s in segments))
                    }, indent=2)
                
                elif command == "timeline":
                    # Export timeline
                    timeline = self.export_timeline(self.current_segments)
                    return timeline
                
                elif command == "merge":
                    # Merge segments
                    max_gap = data.get("max_gap", 0.5)
                    merged = self.merge_segments(self.current_segments, max_gap)
                    
                    return json.dumps({
                        "original_count": len(self.current_segments),
                        "merged_count": len(merged),
                        "segments": [seg.to_dict() for seg in merged]
                    }, indent=2)
                
                elif command == "stats":
                    # Get diarization stats
                    return json.dumps({
                        "total_segments": len(self.current_segments),
                        "buffer_size": len(self.processing_buffer),
                        "last_speaker": self.last_speaker_id,
                        "last_change_time": self.last_change_time
                    }, indent=2)
                
                else:
                    return json.dumps({
                        "error": f"Unknown command: {command}",
                        "available_commands": [
                            "process_stream", "diarize_file", "timeline",
                            "merge", "stats"
                        ]
                    })
            
            # Natural language processing
            else:
                return """I'm the Speaker Diarization Agent.
                
I can help you:
- Segment audio by speaker
- Detect speaker changes
- Identify voice activity regions
- Create speaker timelines

Provide audio data or a file path to diarize."""
        
        except Exception as e:
            log.error(f"Error in SpeakerDiarizationAgent: {e}")
            return json.dumps({"error": str(e)})
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Segment audio by speaker and detect speaker changes"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest diarization is needed."""
        return ["diarize", "segment", "speaker change", "who said", "timeline"]