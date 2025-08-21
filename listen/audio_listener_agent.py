#!/usr/bin/env python3
"""
AudioListenerAgent - Real-time audio capture and transcription agent.

This agent continuously listens to microphone input, transcribes it to text,
and passes relevant content through the blackboard for task processing.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import speech_recognition as sr
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not installed. Install with:")
    print("  pip install SpeechRecognition pyaudio")

from agent.agent import Agent

log = logging.getLogger(__name__)


class AudioListenerAgent(Agent):
    """
    Agent that listens to microphone input and transcribes to text in real-time.
    
    This agent:
    1. Continuously captures audio from the microphone
    2. Transcribes audio to text using speech recognition
    3. Timestamps and stores transcriptions
    4. Filters for task-relevant content
    """
    
    def __init__(self, 
                 task_description: Optional[str] = None,
                 continuous: bool = True,
                 **kwargs):
        """
        Initialize the AudioListenerAgent.
        
        Args:
            task_description: The task to listen for relevant content
            continuous: Whether to continuously listen or single capture
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are an audio listening and transcription agent.",
            "You capture audio from the microphone and convert it to text.",
            "You identify task-relevant content from the transcriptions.",
            "You maintain a timeline of audio events and transcriptions."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.task_description = task_description
        self.continuous = continuous
        self.transcription_queue = queue.Queue()
        self.is_listening = False
        self.listener_thread = None
        
        # Initialize speech recognition
        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                log.info("Calibrating for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                log.info("Calibration complete. Ready to listen.")
        else:
            self.recognizer = None
            self.microphone = None
            log.warning("Audio libraries not available")
        
        # Store transcription history
        self.transcription_history = []
        self.relevance_keywords = self._extract_keywords(task_description)
    
    def _extract_keywords(self, task: Optional[str]) -> List[str]:
        """Extract keywords from task description for relevance filtering."""
        if not task:
            return []
        
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        words = re.findall(r'\b\w+\b', task.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'shall'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        log.info(f"Task keywords: {keywords}")
        return keywords
    
    def start_listening(self):
        """Start the background listening thread."""
        if not AUDIO_AVAILABLE:
            log.error("Cannot start listening - audio libraries not available")
            return
        
        if self.is_listening:
            log.warning("Already listening")
            return
        
        self.is_listening = True
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()
        log.info("Started audio listening thread")
    
    def stop_listening(self):
        """Stop the background listening thread."""
        self.is_listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
        log.info("Stopped audio listening")
    
    def _listen_loop(self):
        """Background thread that continuously listens and transcribes."""
        while self.is_listening:
            try:
                # Listen for audio
                with self.microphone as source:
                    # Use shorter timeout for more responsive listening
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Transcribe in background to avoid blocking
                threading.Thread(target=self._transcribe_audio, args=(audio,), daemon=True).start()
                
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except Exception as e:
                log.error(f"Error in listen loop: {e}")
                time.sleep(1)
    
    def _transcribe_audio(self, audio):
        """Transcribe audio to text and process."""
        try:
            # Try multiple recognition engines for robustness
            text = None
            timestamp = datetime.now()
            
            # Try Google Speech Recognition (free, no API key needed)
            try:
                text = self.recognizer.recognize_google(audio)
                engine = "google"
            except sr.UnknownValueError:
                log.debug("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                log.error(f"Google Speech Recognition error: {e}")
            
            # If Google fails, try Sphinx (offline)
            if not text:
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    engine = "sphinx"
                except sr.UnknownValueError:
                    log.debug("Sphinx could not understand audio")
                except sr.RequestError as e:
                    log.error(f"Sphinx error: {e}")
            
            if text:
                # Process the transcription
                transcription = {
                    "text": text,
                    "timestamp": timestamp.isoformat(),
                    "engine": engine,
                    "relevance_score": self._calculate_relevance(text)
                }
                
                self.transcription_history.append(transcription)
                self.transcription_queue.put(transcription)
                
                log.info(f"Transcribed: {text[:100]}... (relevance: {transcription['relevance_score']:.2f})")
                
                # If highly relevant, trigger immediate processing
                if transcription['relevance_score'] > 0.5:
                    log.info(f"HIGH RELEVANCE detected: {text[:200]}")
        
        except Exception as e:
            log.error(f"Error transcribing audio: {e}")
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score of transcribed text to task."""
        if not self.relevance_keywords:
            return 0.5  # Neutral score if no task specified
        
        text_lower = text.lower()
        matches = sum(1 for keyword in self.relevance_keywords if keyword in text_lower)
        
        # Simple relevance score based on keyword matches
        score = matches / max(len(self.relevance_keywords), 1)
        return min(score, 1.0)
    
    def get_recent_transcriptions(self, seconds: int = 30) -> List[Dict[str, Any]]:
        """Get transcriptions from the last N seconds."""
        cutoff = datetime.now().timestamp() - seconds
        recent = []
        
        for trans in reversed(self.transcription_history):
            trans_time = datetime.fromisoformat(trans['timestamp']).timestamp()
            if trans_time >= cutoff:
                recent.append(trans)
            else:
                break
        
        return list(reversed(recent))
    
    def get_relevant_transcriptions(self, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Get transcriptions above a relevance threshold."""
        return [t for t in self.transcription_history if t['relevance_score'] >= min_score]
    
    def run(self, prompt: str) -> str:
        """
        Process audio listening request.
        
        Args:
            prompt: Instructions for what to listen for
            
        Returns:
            Summary of captured audio or current listening status
        """
        try:
            # Parse any specific instructions
            if "start" in prompt.lower():
                self.start_listening()
                return "Started audio listening. Capturing and transcribing in real-time."
            
            elif "stop" in prompt.lower():
                self.stop_listening()
                summary = self._generate_summary()
                return summary
            
            elif "status" in prompt.lower():
                return self._get_status()
            
            elif "recent" in prompt.lower():
                recent = self.get_recent_transcriptions(30)
                return json.dumps(recent, indent=2)
            
            elif "relevant" in prompt.lower():
                relevant = self.get_relevant_transcriptions()
                return json.dumps(relevant, indent=2)
            
            else:
                # Single capture mode
                if not self.continuous:
                    return self._single_capture()
                
                # Return current status for continuous mode
                return self._get_status()
        
        except Exception as e:
            log.error(f"Error in AudioListenerAgent: {e}")
            return f"Error processing audio: {e}"
    
    def _single_capture(self) -> str:
        """Capture a single audio segment and transcribe."""
        if not AUDIO_AVAILABLE:
            return "Audio libraries not available"
        
        try:
            with self.microphone as source:
                log.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            log.info("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            
            transcription = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "relevance_score": self._calculate_relevance(text)
            }
            
            self.transcription_history.append(transcription)
            
            return f"Captured: {text}\nRelevance: {transcription['relevance_score']:.2f}"
        
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        except Exception as e:
            return f"Error capturing audio: {e}"
    
    def _get_status(self) -> str:
        """Get current listening status."""
        status = {
            "is_listening": self.is_listening,
            "total_transcriptions": len(self.transcription_history),
            "recent_count": len(self.get_recent_transcriptions(30)),
            "relevant_count": len(self.get_relevant_transcriptions()),
            "task": self.task_description,
            "keywords": self.relevance_keywords
        }
        
        if self.transcription_history:
            latest = self.transcription_history[-1]
            status["latest_transcription"] = {
                "text": latest["text"][:100],
                "timestamp": latest["timestamp"],
                "relevance": latest["relevance_score"]
            }
        
        return json.dumps(status, indent=2)
    
    def _generate_summary(self) -> str:
        """Generate a summary of captured audio."""
        if not self.transcription_history:
            return "No audio captured"
        
        total = len(self.transcription_history)
        relevant = self.get_relevant_transcriptions()
        
        summary = f"Captured {total} audio segments\n"
        summary += f"Found {len(relevant)} relevant segments\n\n"
        
        if relevant:
            summary += "Most relevant transcriptions:\n"
            for trans in sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)[:5]:
                summary += f"- [{trans['relevance_score']:.2f}] {trans['text'][:100]}...\n"
        
        return summary
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Listen to audio and transcribe relevant content in real-time"
    
    @property  
    def triggers(self) -> List[str]:
        """Words that suggest audio listening is needed."""
        return ["listen", "audio", "microphone", "speech", "transcribe", "hear"]