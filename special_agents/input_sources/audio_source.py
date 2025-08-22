"""
AudioSource - Audio input source for Listen v2.

This module provides real-time audio capture and transcription
with speaker diarization support.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
import threading
import queue

try:
    import speech_recognition as sr
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not installed. Install with:")
    print("  pip install SpeechRecognition pyaudio")

from special_agents.input_sources import InputSource

log = logging.getLogger(__name__)


class AudioSource(InputSource):
    """
    Audio input source with real-time transcription.
    
    This source captures audio from the microphone, transcribes it,
    and provides speaker identification hints.
    """
    
    def __init__(self,
                 device_index: Optional[int] = None,
                 energy_threshold: int = 4000,
                 pause_threshold: float = 0.8,
                 phrase_time_limit: Optional[int] = 5,
                 **kwargs):
        """
        Initialize the audio source.
        
        Args:
            device_index: Microphone device index (None for default)
            energy_threshold: Energy threshold for speech detection
            pause_threshold: Seconds of silence before phrase ends
            phrase_time_limit: Maximum seconds for a phrase
            **kwargs: Additional arguments for InputSource
        """
        super().__init__(name="audio", priority=8, **kwargs)
        
        if not AUDIO_AVAILABLE:
            log.warning("Audio libraries not available")
            self.available = False
            return
        
        self.device_index = device_index
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.phrase_time_limit = phrase_time_limit
        
        # Audio components
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.microphone = sr.Microphone(device_index=device_index)
        
        # Transcription queue
        self.audio_queue = queue.Queue()
        self.transcription_queue = asyncio.Queue()
        
        # Threading for audio capture
        self.capture_thread = None
        self.transcribe_thread = None
        self.stop_event = threading.Event()
        
        # Speaker tracking (simplified)
        self.current_speaker = "user"
        self.speaker_features = {}
        
        self.available = True
        
        # Calibrate for ambient noise
        try:
            with self.microphone as source:
                log.info("Calibrating for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                log.info(f"Energy threshold set to: {self.recognizer.energy_threshold}")
        except Exception as e:
            log.error(f"Failed to calibrate microphone: {e}")
            self.available = False
    
    def validate(self) -> bool:
        """Validate that audio capture is available."""
        return AUDIO_AVAILABLE and self.available
    
    async def start(self) -> None:
        """Start audio capture."""
        await super().start()
        
        if not self.available:
            log.warning("Audio source not available")
            return
        
        # Start capture threads
        self.stop_event.clear()
        
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()
        
        self.transcribe_thread = threading.Thread(
            target=self._transcribe_loop,
            daemon=True
        )
        self.transcribe_thread.start()
        
        log.info("Audio capture started")
    
    async def stop(self) -> None:
        """Stop audio capture."""
        await super().stop()
        
        # Stop threads
        self.stop_event.set()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.transcribe_thread:
            self.transcribe_thread.join(timeout=2)
        
        log.info("Audio capture stopped")
    
    def _capture_loop(self):
        """Background thread that captures audio."""
        with self.microphone as source:
            while not self.stop_event.is_set():
                try:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=self.phrase_time_limit
                    )
                    
                    # Queue for transcription
                    self.audio_queue.put(audio)
                    
                except sr.WaitTimeoutError:
                    # No speech detected, continue
                    pass
                except Exception as e:
                    log.error(f"Error capturing audio: {e}")
                    time.sleep(0.1)
    
    def _transcribe_loop(self):
        """Background thread that transcribes audio."""
        while not self.stop_event.is_set():
            try:
                # Get audio from queue with timeout
                try:
                    audio = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Transcribe
                try:
                    text = self.recognizer.recognize_google(audio)
                    
                    # Extract audio features for speaker identification
                    audio_features = self._extract_audio_features(audio)
                    
                    # Create transcription data
                    transcription = {
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.9,  # Google doesn't provide confidence
                        "audio_features": audio_features,
                        "speaker_hint": self._identify_speaker(audio_features)
                    }
                    
                    # Put in async queue
                    asyncio.run_coroutine_threadsafe(
                        self.transcription_queue.put(transcription),
                        asyncio.get_event_loop()
                    )
                    
                    log.debug(f"Transcribed: {text[:50]}...")
                    
                except sr.UnknownValueError:
                    log.debug("Could not understand audio")
                except sr.RequestError as e:
                    log.error(f"Speech recognition error: {e}")
                
            except Exception as e:
                log.error(f"Error in transcription loop: {e}")
                time.sleep(0.1)
    
    def _extract_audio_features(self, audio) -> Dict[str, Any]:
        """
        Extract features from audio for speaker identification.
        
        This is a simplified version. In production, use proper
        voice analysis libraries.
        """
        features = {
            "duration": len(audio.frame_data) / audio.sample_rate if audio.sample_rate else 0,
            "sample_rate": audio.sample_rate,
            "sample_width": audio.sample_width
        }
        
        # Simplified pitch estimation
        # In production, use librosa or similar for proper analysis
        if audio.frame_data:
            # Very rough energy calculation
            import struct
            samples = struct.unpack(
                f"{len(audio.frame_data)//2}h",
                audio.frame_data[:len(audio.frame_data)//2*2]
            )
            avg_amplitude = sum(abs(s) for s in samples) / len(samples) if samples else 0
            
            # Map to rough pitch range
            features["pitch"] = min(400, max(80, avg_amplitude / 50))
            features["energy"] = avg_amplitude
        
        return features
    
    def _identify_speaker(self, audio_features: Dict[str, Any]) -> str:
        """
        Identify speaker based on audio features.
        
        This is a simplified heuristic. In production, use
        proper speaker diarization models.
        """
        # Simple pitch-based identification
        pitch = audio_features.get("pitch", 150)
        
        if pitch > 200:
            return "speaker_high"  # Higher pitched voice
        elif pitch < 120:
            return "speaker_low"   # Lower pitched voice
        else:
            return "speaker_mid"   # Medium pitched voice
    
    async def capture(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Capture and yield transcribed audio data.
        
        Yields:
            Transcription data with audio features
        """
        while self.is_active:
            try:
                # Get transcription with timeout
                transcription = await asyncio.wait_for(
                    self.transcription_queue.get(),
                    timeout=1.0
                )
                
                # Format output
                output = self.format_output(
                    data=transcription["text"],
                    category="conversation",
                    confidence=transcription["confidence"],
                    audio_features=transcription["audio_features"],
                    speaker_hint=transcription["speaker_hint"],
                    raw_transcription=transcription
                )
                
                yield output
                
            except asyncio.TimeoutError:
                # No transcription available, continue
                continue
            except Exception as e:
                log.error(f"Error in audio capture: {e}")
                await asyncio.sleep(0.1)
    
    def set_energy_threshold(self, threshold: int):
        """Adjust energy threshold for speech detection."""
        if self.recognizer:
            self.recognizer.energy_threshold = threshold
            log.info(f"Energy threshold set to: {threshold}")
    
    def list_microphones(self) -> List[str]:
        """List available microphone devices."""
        if not AUDIO_AVAILABLE:
            return []
        
        devices = []
        for i in range(sr.Microphone.list_microphone_names().__len__()):
            devices.append(sr.Microphone.list_microphone_names()[i])
        
        return devices