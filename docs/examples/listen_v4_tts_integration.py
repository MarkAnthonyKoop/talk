#!/usr/bin/env python3
"""
Listen v4 TTS Integration Examples

This file contains practical examples for integrating TTS and context detection
into Listen v4, based on the research findings.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Mock imports - replace with actual implementations
# from realtimetts import RealtimeTTS, TextToAudioStream
# import openwakeword
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ResponseUrgency(Enum):
    """Response urgency levels affecting TTS selection."""
    IMMEDIATE = "immediate"  # < 100ms, use fastest local
    NORMAL = "normal"       # < 500ms, balanced approach  
    HIGH_QUALITY = "high_quality"  # Best quality, higher latency OK
    BACKGROUND = "background"  # Non-critical, can queue


class TTSEngine(Enum):
    """Available TTS engines."""
    REALTIME_TTS = "realtimetts"
    DEEPGRAM_AURA = "deepgram_aura"
    MELO_TTS = "melo_tts"
    AZURE_TTS = "azure_tts"
    OPENAI_TTS = "openai_tts"
    SYSTEM_TTS = "system_tts"  # macOS say command, etc.


@dataclass
class VoiceProfile:
    """Voice profile settings for a speaker."""
    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0
    emotion: str = "neutral"
    accent: Optional[str] = None
    
    
@dataclass
class ContextSignals:
    """Signals indicating conversation context relevance."""
    wake_word_detected: bool = False
    wake_confidence: float = 0.0
    direct_address: bool = False
    address_confidence: float = 0.0
    intent_detected: str = "none"
    intent_confidence: float = 0.0
    conversation_flow_score: float = 0.0
    silence_duration: float = 0.0
    voice_energy: float = 0.0
    speaker_change: bool = False


class MockTTSEngine:
    """Mock TTS engine for demonstration."""
    
    def __init__(self, engine_type: TTSEngine, latency_ms: int):
        self.engine_type = engine_type
        self.latency_ms = latency_ms
        self.available = True
    
    async def synthesize_async(self, text: str, voice_profile: VoiceProfile = None) -> bytes:
        """Mock synthesis with simulated latency."""
        await asyncio.sleep(self.latency_ms / 1000.0)  # Simulate latency
        log.info(f"{self.engine_type.value} synthesized: '{text[:50]}...' "
                f"(latency: {self.latency_ms}ms)")
        return b"mock_audio_data"  # Would be actual audio bytes


class TTSManager:
    """
    Manages multiple TTS engines with fallback and quality selection.
    """
    
    def __init__(self):
        # Initialize TTS engines in order of preference for different scenarios
        self.engines = {
            TTSEngine.REALTIME_TTS: MockTTSEngine(TTSEngine.REALTIME_TTS, 100),
            TTSEngine.DEEPGRAM_AURA: MockTTSEngine(TTSEngine.DEEPGRAM_AURA, 200),
            TTSEngine.MELO_TTS: MockTTSEngine(TTSEngine.MELO_TTS, 300),
            TTSEngine.AZURE_TTS: MockTTSEngine(TTSEngine.AZURE_TTS, 400),
            TTSEngine.SYSTEM_TTS: MockTTSEngine(TTSEngine.SYSTEM_TTS, 150),
        }
        
        # Engine preference by urgency
        self.engine_preferences = {
            ResponseUrgency.IMMEDIATE: [
                TTSEngine.REALTIME_TTS,
                TTSEngine.SYSTEM_TTS,
                TTSEngine.MELO_TTS
            ],
            ResponseUrgency.NORMAL: [
                TTSEngine.REALTIME_TTS,
                TTSEngine.DEEPGRAM_AURA,
                TTSEngine.MELO_TTS
            ],
            ResponseUrgency.HIGH_QUALITY: [
                TTSEngine.DEEPGRAM_AURA,
                TTSEngine.AZURE_TTS,
                TTSEngine.OPENAI_TTS
            ],
            ResponseUrgency.BACKGROUND: [
                TTSEngine.MELO_TTS,
                TTSEngine.AZURE_TTS
            ]
        }
        
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.fallback_voice = VoiceProfile("default_voice")
    
    async def synthesize_response(self, 
                                text: str, 
                                urgency: ResponseUrgency = ResponseUrgency.NORMAL,
                                speaker_id: Optional[str] = None) -> Tuple[bytes, TTSEngine]:
        """
        Synthesize text to speech with appropriate engine selection.
        
        Args:
            text: Text to synthesize
            urgency: Response urgency level
            speaker_id: Speaker ID for voice profile lookup
            
        Returns:
            Tuple of (audio_data, engine_used)
        """
        # Get voice profile
        voice_profile = self.voice_profiles.get(speaker_id, self.fallback_voice)
        
        # Try engines in preference order
        preferred_engines = self.engine_preferences[urgency]
        
        for engine_type in preferred_engines:
            engine = self.engines.get(engine_type)
            if engine and engine.available:
                try:
                    audio_data = await engine.synthesize_async(text, voice_profile)
                    log.info(f"Successfully used {engine_type.value} for synthesis")
                    return audio_data, engine_type
                except Exception as e:
                    log.warning(f"Engine {engine_type.value} failed: {e}")
                    continue
        
        # All preferred engines failed, try any available engine
        for engine_type, engine in self.engines.items():
            if engine.available:
                try:
                    audio_data = await engine.synthesize_async(text, voice_profile)
                    log.warning(f"Fallback to {engine_type.value}")
                    return audio_data, engine_type
                except Exception:
                    continue
        
        raise RuntimeError("All TTS engines failed")
    
    def set_voice_profile(self, speaker_id: str, voice_profile: VoiceProfile):
        """Set voice profile for a speaker."""
        self.voice_profiles[speaker_id] = voice_profile
        log.info(f"Set voice profile for {speaker_id}: {voice_profile.voice_id}")
    
    def get_engine_status(self) -> Dict[str, bool]:
        """Get status of all engines."""
        return {engine.value: engine.available for engine, engine in self.engines.items()}


class WakeWordDetector:
    """Wake word detection using openWakeWord (mocked)."""
    
    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or ["hey assistant", "listen", "computer"]
        self.threshold = 0.5
        
    def detect(self, audio_features: Dict[str, Any]) -> Tuple[bool, float, Optional[str]]:
        """
        Detect wake words in audio.
        
        Args:
            audio_features: Audio feature dictionary
            
        Returns:
            Tuple of (detected, confidence, wake_word)
        """
        # Mock implementation - in reality would use openWakeWord
        # Simulate wake word detection based on mock audio features
        mock_wake_score = audio_features.get("wake_score", 0.0)
        
        if mock_wake_score > self.threshold:
            detected_word = self.wake_words[0]  # Mock detection
            return True, mock_wake_score, detected_word
        
        return False, mock_wake_score, None


class IntentDetector:
    """Intent detection for conversation context."""
    
    def __init__(self):
        # Mock intent categories
        self.intent_categories = {
            "question": ["what", "how", "why", "when", "where", "?"],
            "request": ["please", "can you", "would you", "help me"],
            "command": ["do", "make", "create", "delete", "show"],
            "greeting": ["hello", "hi", "hey", "good morning"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"],
            "confirmation": ["yes", "okay", "sure", "alright"],
            "negation": ["no", "nope", "don't", "stop"]
        }
    
    def detect_intent(self, text: str, context: List[str] = None) -> Tuple[str, float]:
        """
        Detect intent from text.
        
        Args:
            text: Input text
            context: Previous conversation context
            
        Returns:
            Tuple of (intent, confidence)
        """
        text_lower = text.lower()
        best_intent = "unknown"
        best_confidence = 0.0
        
        for intent, keywords in self.intent_categories.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = min(matches / len(keywords), 1.0)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent
        
        # Boost confidence for direct questions
        if "?" in text:
            if best_intent == "question":
                best_confidence = min(best_confidence + 0.3, 1.0)
        
        return best_intent, best_confidence


class AddressingDetector:
    """Detect if speech is addressing the assistant."""
    
    def __init__(self):
        self.addressing_patterns = {
            "direct_names": ["assistant", "computer", "hey", "listen"],
            "second_person": ["you", "your", "yourself"],
            "imperatives": ["please", "help", "tell me", "show me"],
            "questions_to_assistant": ["what do you", "can you", "would you", "do you know"]
        }
    
    def is_addressing_assistant(self, 
                              text: str, 
                              speaker_id: str,
                              conversation_history: List[Dict]) -> Tuple[bool, float]:
        """
        Determine if the text is addressing the assistant.
        
        Args:
            text: Input text
            speaker_id: ID of the speaker
            conversation_history: Recent conversation turns
            
        Returns:
            Tuple of (is_addressing, confidence)
        """
        text_lower = text.lower()
        addressing_score = 0.0
        
        # Direct name mention
        for name in self.addressing_patterns["direct_names"]:
            if name in text_lower:
                addressing_score += 0.4
                break
        
        # Second person pronouns
        for pronoun in self.addressing_patterns["second_person"]:
            if pronoun in text_lower:
                addressing_score += 0.2
                break
        
        # Imperative mood
        for imperative in self.addressing_patterns["imperatives"]:
            if imperative in text_lower:
                addressing_score += 0.3
                break
        
        # Questions directed at assistant
        for question in self.addressing_patterns["questions_to_assistant"]:
            if question in text_lower:
                addressing_score += 0.4
                break
        
        # Context boost: if assistant spoke recently
        if conversation_history:
            last_turn = conversation_history[-1]
            if (last_turn.get("speaker_id") == "assistant" and 
                time.time() - last_turn.get("timestamp", 0) < 10):
                addressing_score += 0.2
        
        is_addressing = addressing_score > 0.5
        return is_addressing, min(addressing_score, 1.0)


class ContextAnalyzer:
    """
    Comprehensive context analysis for determining when to respond.
    """
    
    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.intent_detector = IntentDetector()
        self.addressing_detector = AddressingDetector()
        
        # Response thresholds
        self.response_threshold = 0.6
        self.confidence_weights = {
            "wake_word": 0.4,
            "direct_address": 0.3,
            "intent": 0.2,
            "conversation_flow": 0.1
        }
    
    def analyze_context(self, 
                       text: str,
                       audio_features: Dict[str, Any],
                       speaker_id: str,
                       conversation_history: List[Dict]) -> ContextSignals:
        """
        Comprehensive context analysis.
        
        Args:
            text: Transcribed speech
            audio_features: Audio analysis data
            speaker_id: Speaker identifier
            conversation_history: Recent conversation turns
            
        Returns:
            ContextSignals object with analysis results
        """
        signals = ContextSignals()
        
        # Wake word detection
        signals.wake_word_detected, signals.wake_confidence, _ = \
            self.wake_detector.detect(audio_features)
        
        # Addressing detection
        signals.direct_address, signals.address_confidence = \
            self.addressing_detector.is_addressing_assistant(
                text, speaker_id, conversation_history
            )
        
        # Intent detection
        signals.intent_detected, signals.intent_confidence = \
            self.intent_detector.detect_intent(text, 
                [turn.get("text", "") for turn in conversation_history[-5:]])
        
        # Conversation flow analysis
        signals.conversation_flow_score = self._analyze_conversation_flow(
            conversation_history, speaker_id
        )
        
        # Audio features
        signals.voice_energy = audio_features.get("energy", 0)
        signals.silence_duration = audio_features.get("silence_duration", 0)
        
        # Speaker change detection
        if conversation_history:
            last_speaker = conversation_history[-1].get("speaker_id")
            signals.speaker_change = (speaker_id != last_speaker)
        
        return signals
    
    def should_respond(self, signals: ContextSignals) -> Tuple[bool, float, str]:
        """
        Determine if assistant should respond based on context signals.
        
        Args:
            signals: ContextSignals from analysis
            
        Returns:
            Tuple of (should_respond, confidence, reason)
        """
        # High confidence triggers
        if signals.wake_word_detected and signals.wake_confidence > 0.7:
            return True, signals.wake_confidence, "wake_word"
        
        if signals.direct_address and signals.address_confidence > 0.8:
            return True, signals.address_confidence, "direct_address"
        
        # Medium confidence triggers
        total_confidence = (
            signals.wake_confidence * self.confidence_weights["wake_word"] +
            signals.address_confidence * self.confidence_weights["direct_address"] +
            signals.intent_confidence * self.confidence_weights["intent"] +
            signals.conversation_flow_score * self.confidence_weights["conversation_flow"]
        )
        
        if total_confidence > self.response_threshold:
            return True, total_confidence, "combined_signals"
        
        # Special cases
        if (signals.intent_detected in ["question", "request"] and 
            signals.intent_confidence > 0.7):
            return True, signals.intent_confidence, "high_intent"
        
        # Silence handling (don't respond during silence)
        if signals.silence_duration > 3.0:
            return False, 0.1, "silence_detected"
        
        return False, total_confidence, "below_threshold"
    
    def _analyze_conversation_flow(self, 
                                 conversation_history: List[Dict],
                                 current_speaker: str) -> float:
        """Analyze conversation flow to determine response appropriateness."""
        if not conversation_history:
            return 0.0
        
        last_turn = conversation_history[-1]
        last_speaker = last_turn.get("speaker_id")
        last_timestamp = last_turn.get("timestamp", 0)
        
        flow_score = 0.0
        
        # If assistant spoke last and someone responds
        if last_speaker == "assistant":
            flow_score += 0.6
        
        # Time since last turn
        time_since_last = time.time() - last_timestamp
        if time_since_last < 5.0:  # Recent conversation
            flow_score += 0.3
        elif time_since_last > 30.0:  # Long pause
            flow_score -= 0.2
        
        # Speaker patterns
        recent_speakers = [turn.get("speaker_id") for turn in conversation_history[-5:]]
        unique_speakers = set(recent_speakers)
        
        # Group conversation (be more selective)
        if len(unique_speakers) > 2:
            flow_score *= 0.7
        
        return max(0.0, min(flow_score, 1.0))


class Listen4ResponseManager:
    """
    Main response manager integrating TTS and context analysis.
    """
    
    def __init__(self):
        self.tts_manager = TTSManager()
        self.context_analyzer = ContextAnalyzer()
        self.conversation_history: List[Dict] = []
        
        # Response templates for different scenarios
        self.response_templates = {
            "acknowledgment": [
                "I understand",
                "Got it",
                "I'm listening",
                "Yes, I hear you"
            ],
            "clarification": [
                "Could you clarify that for me?",
                "I'm not sure I understand",
                "What exactly do you mean?",
                "Can you provide more details?"
            ],
            "help": [
                "I'm here to help",
                "What can I do for you?",
                "How can I assist you?",
                "I'm ready to help"
            ],
            "information": [
                "Here's what I found",
                "According to my information",
                "Let me tell you about that",
                "I can help with that"
            ]
        }
    
    async def process_speech(self, 
                           text: str,
                           audio_features: Dict[str, Any],
                           speaker_id: str) -> Optional[Tuple[bytes, str]]:
        """
        Process incoming speech and generate response if appropriate.
        
        Args:
            text: Transcribed speech
            audio_features: Audio analysis data
            speaker_id: Speaker identifier
            
        Returns:
            Optional tuple of (audio_response, text_response)
        """
        # Add to conversation history
        turn = {
            "text": text,
            "speaker_id": speaker_id,
            "timestamp": time.time()
        }
        self.conversation_history.append(turn)
        
        # Analyze context
        signals = self.context_analyzer.analyze_context(
            text, audio_features, speaker_id, self.conversation_history
        )
        
        # Determine if should respond
        should_respond, confidence, reason = self.context_analyzer.should_respond(signals)
        
        log.info(f"Response decision: {should_respond} (confidence: {confidence:.2f}, reason: {reason})")
        
        if not should_respond:
            return None
        
        # Determine response urgency
        urgency = self._determine_urgency(signals, reason)
        
        # Generate response text
        response_text = self._generate_response_text(signals, text)
        
        # Synthesize response
        try:
            audio_data, engine_used = await self.tts_manager.synthesize_response(
                response_text, urgency, speaker_id
            )
            
            # Add assistant response to history
            assistant_turn = {
                "text": response_text,
                "speaker_id": "assistant", 
                "timestamp": time.time()
            }
            self.conversation_history.append(assistant_turn)
            
            log.info(f"Generated response using {engine_used.value}: '{response_text}'")
            return audio_data, response_text
            
        except Exception as e:
            log.error(f"Failed to generate response: {e}")
            return None
    
    def _determine_urgency(self, signals: ContextSignals, reason: str) -> ResponseUrgency:
        """Determine response urgency based on context."""
        if signals.wake_word_detected or reason == "wake_word":
            return ResponseUrgency.IMMEDIATE
        
        if signals.intent_detected == "request" and signals.intent_confidence > 0.8:
            return ResponseUrgency.NORMAL
        
        if signals.direct_address:
            return ResponseUrgency.NORMAL
        
        return ResponseUrgency.BACKGROUND
    
    def _generate_response_text(self, signals: ContextSignals, original_text: str) -> str:
        """Generate appropriate response text based on context."""
        import random
        
        # Choose response type based on intent
        if signals.intent_detected == "question":
            if "what" in original_text.lower():
                return random.choice(self.response_templates["information"])
            else:
                return random.choice(self.response_templates["clarification"])
        
        elif signals.intent_detected == "request":
            return random.choice(self.response_templates["help"])
        
        elif signals.intent_detected == "greeting":
            return "Hello! How can I help you today?"
        
        elif signals.wake_word_detected:
            return random.choice(self.response_templates["acknowledgment"])
        
        else:
            # Default acknowledgment
            return random.choice(self.response_templates["acknowledgment"])
    
    def set_speaker_voice(self, speaker_id: str, voice_profile: VoiceProfile):
        """Set voice profile for a speaker."""
        self.tts_manager.set_voice_profile(speaker_id, voice_profile)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_turns = len(self.conversation_history)
        assistant_turns = len([t for t in self.conversation_history if t["speaker_id"] == "assistant"])
        
        return {
            "total_turns": total_turns,
            "assistant_turns": assistant_turns,
            "response_rate": assistant_turns / max(total_turns - assistant_turns, 1),
            "recent_speakers": list(set(t["speaker_id"] for t in self.conversation_history[-10:])),
            "engine_status": self.tts_manager.get_engine_status()
        }


# Example usage and testing
async def demo_listen_v4_response():
    """Demonstrate the Listen v4 response system."""
    
    print("🎙️ Listen v4 TTS and Context Detection Demo")
    print("=" * 50)
    
    # Initialize response manager
    manager = Listen4ResponseManager()
    
    # Set up voice profiles for different speakers
    manager.set_speaker_voice("user1", VoiceProfile("jenny_neural", speed=1.1))
    manager.set_speaker_voice("user2", VoiceProfile("david_neural", speed=0.9))
    
    # Simulate conversation scenarios
    scenarios = [
        {
            "text": "Hey assistant, what's the weather like?",
            "audio_features": {"energy": 800, "wake_score": 0.8, "silence_duration": 0.0},
            "speaker_id": "user1",
            "description": "Direct wake word + question"
        },
        {
            "text": "I think it's going to rain tomorrow",
            "audio_features": {"energy": 600, "wake_score": 0.1, "silence_duration": 0.0},
            "speaker_id": "user2", 
            "description": "General conversation (should not respond)"
        },
        {
            "text": "Can you help me with something?",
            "audio_features": {"energy": 700, "wake_score": 0.3, "silence_duration": 0.0},
            "speaker_id": "user1",
            "description": "Direct request to assistant"
        },
        {
            "text": "Yes, please show me the calendar",
            "audio_features": {"energy": 750, "wake_score": 0.2, "silence_duration": 1.0},
            "speaker_id": "user1",
            "description": "Follow-up command"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['description']} ---")
        print(f"Speaker: {scenario['speaker_id']}")
        print(f"Input: '{scenario['text']}'")
        
        # Process the speech
        result = await manager.process_speech(
            scenario["text"],
            scenario["audio_features"], 
            scenario["speaker_id"]
        )
        
        if result:
            audio_data, response_text = result
            print(f"✅ Response: '{response_text}'")
            print(f"   Audio generated: {len(audio_data)} bytes")
        else:
            print("🔇 No response (passive listening)")
        
        # Small delay between scenarios
        await asyncio.sleep(0.5)
    
    # Show final statistics
    print(f"\n--- Final Statistics ---")
    stats = manager.get_conversation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_listen_v4_response())