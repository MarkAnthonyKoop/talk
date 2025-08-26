#!/usr/bin/env python3
"""
Voice Processing Agent

Handles audio transcription with multi-service fallback.
Part of Listen v7's agentic architecture.
"""

import logging
import time
from typing import Dict, Any, Optional
import numpy as np
from agent.agent import Agent

log = logging.getLogger(__name__)


class VoiceProcessingAgent(Agent):
    """
    Specialized agent for voice-to-text processing.
    
    Manages multiple transcription services with intelligent fallback:
    - Deepgram Nova-3 (premium)
    - AssemblyAI (alternative premium)
    - Google Speech API (free tier)
    - Local Whisper (offline fallback)
    """
    
    def __init__(self, deepgram_key: Optional[str] = None, 
                 service_tier: str = "standard", **kwargs):
        roles = [
            "You transcribe audio to text with high accuracy",
            "You manage multiple speech recognition services",
            "You optimize for speed and accuracy based on requirements",
            "You handle audio preprocessing and format conversion",
            "You provide confidence scores and timing information"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        self.service_tier = service_tier
        self.deepgram_key = deepgram_key
        self.services_available = self._check_available_services()
        
        log.info(f"VoiceProcessingAgent initialized with {len(self.services_available)} services")
    
    def _check_available_services(self) -> Dict[str, bool]:
        """Check which transcription services are available."""
        services = {}
        
        # Check Deepgram
        try:
            import deepgram
            services["deepgram"] = bool(self.deepgram_key)
        except ImportError:
            services["deepgram"] = False
        
        # Check speech_recognition (Google)
        try:
            import speech_recognition as sr
            services["google_speech"] = True
        except ImportError:
            services["google_speech"] = False
        
        # Check for local Whisper
        try:
            import whisper
            services["whisper"] = True
        except ImportError:
            services["whisper"] = False
        
        # Mock service always available for testing
        services["mock"] = True
        
        return services
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of voice processing services."""
        return {
            "status": "healthy" if any(self.services_available.values()) else "unhealthy",
            "services_available": self.services_available,
            "primary_service": self._select_primary_service()
        }
    
    def _select_primary_service(self) -> str:
        """Select the best available service based on tier and availability."""
        if self.service_tier == "premium" and self.services_available.get("deepgram"):
            return "deepgram"
        elif self.services_available.get("google_speech"):
            return "google_speech"
        elif self.services_available.get("whisper"):
            return "whisper"
        else:
            return "mock"
    
    async def transcribe(self, audio_data: bytes, service_tier: str = None) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        This is the main action called by PlanRunner.
        """
        start_time = time.time()
        tier = service_tier or self.service_tier
        
        # Select service based on tier
        service = self._select_primary_service() if tier != "economy" else "google_speech"
        
        # Try primary service
        try:
            if service == "deepgram" and self.services_available.get("deepgram"):
                result = await self._transcribe_with_deepgram(audio_data)
            elif service == "google_speech" and self.services_available.get("google_speech"):
                result = await self._transcribe_with_google(audio_data)
            elif service == "whisper" and self.services_available.get("whisper"):
                result = await self._transcribe_with_whisper(audio_data)
            else:
                result = await self._transcribe_with_mock(audio_data)
                
            # Add processing metadata
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            result["service_tier"] = tier
            
            return result
            
        except Exception as e:
            log.error(f"Primary service {service} failed: {e}")
            # Fallback cascade
            return await self._fallback_transcription(audio_data, start_time)
    
    async def transcribe_fast(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Fast transcription for quick commands.
        
        Optimizes for speed over accuracy.
        """
        # Use fastest available service
        if self.services_available.get("deepgram"):
            # Deepgram with speed optimization
            return await self._transcribe_with_deepgram(audio_data, fast_mode=True)
        else:
            # Mock for speed
            return await self._transcribe_with_mock(audio_data)
    
    async def _transcribe_with_deepgram(self, audio_data: bytes, fast_mode: bool = False) -> Dict[str, Any]:
        """Transcribe using Deepgram Nova-3."""
        # This would use actual Deepgram SDK
        # For now, return mock
        return await self._transcribe_with_mock(audio_data)
    
    async def _transcribe_with_google(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe using Google Speech API."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            audio = sr.AudioData(audio_data, 16000, 2)
            
            transcript = recognizer.recognize_google(audio)
            
            return {
                "transcript": transcript,
                "confidence": 0.85,  # Google doesn't provide confidence in free tier
                "service_used": "google_speech",
                "language": "en-US"
            }
        except Exception as e:
            log.error(f"Google Speech failed: {e}")
            raise
    
    async def _transcribe_with_whisper(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        # This would use OpenAI Whisper
        # For now, return mock
        return await self._transcribe_with_mock(audio_data)
    
    async def _transcribe_with_mock(self, audio_data: bytes) -> Dict[str, Any]:
        """Mock transcription for testing."""
        # Simulate transcription based on audio length
        audio_length = len(audio_data) if audio_data else 0
        
        if audio_length < 10000:
            transcript = "list files"
        elif audio_length < 50000:
            transcript = "list my files"
        else:
            transcript = "list my files in the current directory"
        
        return {
            "transcript": transcript,
            "confidence": 0.75,
            "service_used": "mock",
            "language": "en-US"
        }
    
    async def _fallback_transcription(self, audio_data: bytes, start_time: float) -> Dict[str, Any]:
        """Fallback cascade through available services."""
        services_to_try = ["google_speech", "whisper", "mock"]
        
        for service in services_to_try:
            if self.services_available.get(service):
                try:
                    if service == "google_speech":
                        result = await self._transcribe_with_google(audio_data)
                    elif service == "whisper":
                        result = await self._transcribe_with_whisper(audio_data)
                    else:
                        result = await self._transcribe_with_mock(audio_data)
                    
                    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
                    result["fallback_used"] = True
                    return result
                    
                except Exception as e:
                    log.warning(f"Fallback service {service} failed: {e}")
                    continue
        
        # Ultimate fallback
        return {
            "transcript": "Could not transcribe audio",
            "confidence": 0.0,
            "service_used": "none",
            "error": "All transcription services failed",
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Agent interface for LLM-based processing.
        
        Can be used for analyzing transcription quality or other tasks.
        """
        if "transcribe" in prompt:
            # Extract audio data from context
            audio_data = context.get("audio_data") if context else None
            if audio_data:
                result = await self.transcribe(audio_data)
                return f"Transcribed: {result['transcript']} (confidence: {result['confidence']})"
        
        return f"VoiceProcessingAgent: {prompt}"
    
    async def cleanup(self):
        """Clean up voice processing resources."""
        log.info("VoiceProcessingAgent cleanup complete")