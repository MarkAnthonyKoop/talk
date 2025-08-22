#!/usr/bin/env python3
"""
Listen v4 - Intelligent Conversational Assistant with Voice Reply.

Building on v3's speaker identification, v4 adds:
- Context relevance detection
- Intelligent reply decision making
- Voice synthesis (TTS) for natural responses
- Multi-modal conversation management
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
import time

# Audio handling
try:
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Install with: pip install SpeechRecognition pyaudio")

# TTS handling
try:
    from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Note: RealtimeTTS not available. Install with: pip install realtimetts")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import special agents
from special_agents.speaker_identification_agent import SpeakerIdentificationAgent
from special_agents.voice_enrollment_agent import VoiceEnrollmentAgent
from special_agents.speaker_diarization_agent import SpeakerDiarizationAgent
from special_agents.conversation_manager import ConversationManager
from special_agents.information_organizer import InformationOrganizer
from special_agents.interjection_agent import InterjectionAgent
from special_agents.multi_source_orchestrator import MultiSourceOrchestrator
from special_agents.execution_planning_agent import ExecutionPlanningAgent
from special_agents.active_listening_agent import ActiveListeningAgent

# Import LLM agent for generating responses
from agent.agent import Agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ContextRelevanceAgent(Agent):
    """
    Agent that determines when to respond in conversation.
    """
    
    def __init__(self, **kwargs):
        """Initialize context relevance agent."""
        roles = [
            "You analyze conversation context to determine if a response is needed.",
            "You detect when users are addressing the assistant directly.",
            "You identify questions, requests, and commands.",
            "You understand conversation flow and timing."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Wake phrases that indicate direct addressing
        self.wake_phrases = [
            "hey listen", "ok listen", "listen up",
            "assistant", "hey assistant", "computer",
            "can you", "could you", "would you",
            "please", "help me", "tell me", "show me",
            "what is", "what's", "how do", "how to",
            "why is", "when will", "where is"
        ]
        
        # Context for recent interactions
        self.last_response_time = 0
        self.response_cooldown = 3.0  # Seconds between responses
        
    def should_respond(self, 
                       text: str,
                       speaker_id: str,
                       conversation_context: List[Dict],
                       confidence_threshold: float = 0.6) -> Tuple[bool, float, str]:
        """
        Determine if assistant should respond.
        
        Args:
            text: Current utterance
            speaker_id: Who is speaking
            conversation_context: Recent conversation history
            confidence_threshold: Minimum confidence to respond
            
        Returns:
            (should_respond, confidence, reason)
        """
        # Check cooldown
        if time.time() - self.last_response_time < self.response_cooldown:
            return False, 0.0, "cooldown"
        
        text_lower = text.lower()
        
        # Check for wake phrases (high confidence)
        for phrase in self.wake_phrases:
            if phrase in text_lower:
                return True, 0.9, f"wake_phrase:{phrase}"
        
        # Check if it's a question
        if text.strip().endswith("?"):
            return True, 0.8, "direct_question"
        
        # Check for commands/requests
        command_words = ["please", "could you", "can you", "would you", "will you"]
        for word in command_words:
            if word in text_lower:
                return True, 0.85, "polite_request"
        
        # Check if previous turn was from assistant
        if conversation_context and len(conversation_context) > 1:
            last_turn = conversation_context[-2]
            if last_turn.get("speaker_id") == "assistant":
                # User might be responding to us
                if len(text.split()) < 10:  # Short response
                    return True, 0.7, "follow_up"
        
        # Analyze with LLM for complex cases
        analysis_prompt = f"""
        Analyze if the assistant should respond to this utterance.
        
        Current text: "{text}"
        Speaker: {speaker_id}
        
        Consider:
        1. Is this directed at the assistant?
        2. Is this a question or request?
        3. Does this need information or help?
        
        Respond with JSON: {{"should_respond": true/false, "confidence": 0.0-1.0, "reason": "..."}}
        """
        
        try:
            result = self.run(analysis_prompt, {"max_tokens": 100})
            if result and "should_respond" in result:
                return result["should_respond"], result.get("confidence", 0.5), result.get("reason", "llm_analysis")
        except:
            pass
        
        # Default: don't respond unless confident
        return False, 0.3, "no_clear_trigger"


class ResponseGenerator(Agent):
    """
    Agent that generates appropriate responses.
    """
    
    def __init__(self, **kwargs):
        """Initialize response generator."""
        roles = [
            "You are a helpful conversational assistant.",
            "You provide concise, relevant responses.",
            "You maintain context from previous conversation.",
            "You speak naturally and conversationally."
        ]
        super().__init__(roles=roles, **kwargs)
        
    def generate_response(self,
                         text: str,
                         speaker_info: Dict,
                         conversation_context: List[Dict],
                         information_context: Dict) -> str:
        """
        Generate an appropriate response.
        
        Args:
            text: Current utterance to respond to
            speaker_info: Information about the speaker
            conversation_context: Recent conversation history
            information_context: Available information/knowledge
            
        Returns:
            Response text
        """
        # Build context
        context_str = ""
        if conversation_context:
            context_str = "Recent conversation:\n"
            for turn in conversation_context[-5:]:  # Last 5 turns
                speaker = turn.get("speaker_id", "unknown")
                context_str += f"{speaker}: {turn.get('text', '')}\n"
        
        # Build prompt
        prompt = f"""
        Generate a helpful response to the user.
        
        {context_str}
        
        Current speaker ({speaker_info.get('name', 'User')}): {text}
        
        Guidelines:
        - Be concise (1-2 sentences unless more detail is needed)
        - Be conversational and natural
        - Use the speaker's name if known
        - Provide helpful information
        - Ask clarifying questions if needed
        
        Response:
        """
        
        try:
            response = self.run(prompt, {"max_tokens": 150})
            if response and isinstance(response, str):
                return response.strip()
        except Exception as e:
            log.error(f"Error generating response: {e}")
        
        # Fallback responses
        if "?" in text:
            return "I'm not sure about that. Could you provide more context?"
        else:
            return "I understand. How can I help you with that?"


class ListenV4:
    """
    Listen v4 - Intelligent conversational assistant with voice reply.
    
    Features:
    - All v3 capabilities (speaker ID, diarization, enrollment)
    - Context relevance detection
    - Intelligent reply decision making
    - Voice synthesis for natural responses
    - Multi-modal conversation management
    """
    
    def __init__(self,
                 name: str = "Listen v4",
                 db_path: Optional[Path] = None,
                 confidence_threshold: float = 0.7,
                 use_tts: bool = True,
                 tts_voice: Optional[str] = None):
        """
        Initialize Listen v4.
        
        Args:
            name: Assistant name
            db_path: Path to speaker database
            confidence_threshold: Confidence threshold for actions
            use_tts: Whether to use TTS for responses
            tts_voice: Specific TTS voice to use
        """
        self.name = name
        self.db_path = db_path or Path.home() / ".listen" / "speakers.db"
        self.confidence_threshold = confidence_threshold
        self.use_tts = use_tts and TTS_AVAILABLE
        
        # Initialize speaker agents (from v3)
        self.speaker_id_agent = SpeakerIdentificationAgent(
            db_path=self.db_path,
            similarity_threshold=0.75,
            use_mock=not AUDIO_AVAILABLE
        )
        
        self.enrollment_agent = VoiceEnrollmentAgent(
            speaker_agent=self.speaker_id_agent,
            min_samples=3,
            max_samples=10
        )
        
        self.diarization_agent = SpeakerDiarizationAgent(
            speaker_agent=self.speaker_id_agent,
            vad_threshold=0.5,
            min_segment_duration=0.5,
            max_segment_duration=30.0,
            use_mock=not AUDIO_AVAILABLE
        )
        
        # Initialize other components (from v3)
        self.orchestrator = MultiSourceOrchestrator()
        self.conversation_manager = ConversationManager()
        self.information_organizer = InformationOrganizer()
        self.interjection_agent = InterjectionAgent(
            confidence_threshold=confidence_threshold
        )
        
        # Planning agents
        self.planning_agent = ExecutionPlanningAgent()
        self.listening_agent = ActiveListeningAgent()
        
        # NEW v4 agents
        self.context_agent = ContextRelevanceAgent()
        self.response_generator = ResponseGenerator()
        
        # Initialize TTS
        if self.use_tts:
            try:
                # Try system TTS first (fastest)
                self.tts_engine = SystemEngine()
                self.tts_stream = TextToAudioStream(self.tts_engine)
                log.info("Initialized system TTS engine")
            except:
                try:
                    # Fallback to Coqui
                    self.tts_engine = CoquiEngine()
                    self.tts_stream = TextToAudioStream(self.tts_engine)
                    log.info("Initialized Coqui TTS engine")
                except:
                    self.use_tts = False
                    log.warning("TTS initialization failed, voice replies disabled")
        
        # Audio components
        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        else:
            self.recognizer = None
            self.microphone = None
        
        # Processing state
        self.is_listening = False
        self.active_task = None
        self.audio_queue = queue.Queue()
        self.segment_buffer = []
        self.response_queue = queue.Queue()
        
        # Speaker state
        self.current_speaker = None
        self.enrollment_sessions = {}
        
        # Response state
        self.last_response_time = 0
        self.min_response_interval = 2.0  # Minimum seconds between responses
        
        log.info(f"Initialized {name} with voice reply capabilities")
    
    async def start(self, task: Optional[str] = None):
        """
        Start the Listen v4 assistant.
        
        Args:
            task: Optional task to focus on
        """
        self.active_task = task
        
        print(f"\n🎙️  {self.name} Starting...")
        print("=" * 50)
        
        # Load speaker profiles
        speakers = self.speaker_id_agent.get_all_speakers()
        print(f"Loaded {len(speakers)} speaker profiles")
        
        if self.use_tts:
            print("✓ Voice synthesis enabled")
        else:
            print("✗ Voice synthesis disabled")
        
        if task:
            print(f"Active Task: {task}")
            # Create execution plan
            plan = self.planning_agent.create_plan(task)
            if plan:
                print(f"Created plan with {len(plan.get('steps', []))} steps")
        
        print("\n💡 Tips:")
        print("- Say 'Hey Listen' to get my attention")
        print("- Ask questions naturally")
        print("- Say 'enroll my voice' to register your voice")
        print("\n🎤 Listening... (Press Ctrl+C to stop)\n")
        
        # Start processing
        await self._run_processing_loop()
    
    async def _run_processing_loop(self):
        """
        Main processing loop.
        """
        self.is_listening = True
        
        # Start audio listener in thread
        if AUDIO_AVAILABLE:
            listener_thread = threading.Thread(
                target=self._audio_listener_thread,
                daemon=True
            )
            listener_thread.start()
        
        # Start TTS thread if available
        if self.use_tts:
            tts_thread = threading.Thread(
                target=self._tts_thread,
                daemon=True
            )
            tts_thread.start()
        
        try:
            while self.is_listening:
                # Process audio from queue
                await self._process_audio_queue()
                
                # Process segments
                await self._process_segments()
                
                # Check for interjections
                await self._check_interjections()
                
                # Small delay
                await asyncio.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nStopping Listen v4...")
        finally:
            self.is_listening = False
    
    def _audio_listener_thread(self):
        """
        Thread to continuously listen for audio.
        """
        if not AUDIO_AVAILABLE:
            return
        
        with self.microphone as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_listening:
                try:
                    # Listen with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=5
                    )
                    
                    # Add to queue with timestamp
                    self.audio_queue.put({
                        "audio": audio,
                        "timestamp": datetime.now()
                    })
                
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    log.error(f"Audio error: {e}")
    
    def _tts_thread(self):
        """
        Thread to handle TTS output.
        """
        while self.is_listening:
            try:
                # Check for responses to speak
                response = self.response_queue.get(timeout=1)
                
                if response and self.tts_stream:
                    print(f"\n🔊 [{self.name}]: {response}\n")
                    
                    # Play the response
                    self.tts_stream.feed(response)
                    self.tts_stream.play_async()
                    
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"TTS error: {e}")
    
    async def _process_audio_queue(self):
        """
        Process audio from the queue.
        """
        while not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()
                await self._process_audio(audio_data)
            except queue.Empty:
                break
    
    async def _process_audio(self, audio_data: Dict[str, Any]):
        """
        Process a single audio chunk.
        
        Args:
            audio_data: Audio data with timestamp
        """
        timestamp = audio_data["timestamp"]
        
        if AUDIO_AVAILABLE:
            # Transcribe audio
            try:
                audio = audio_data["audio"]
                text = self.recognizer.recognize_google(audio)
                
                # Mock audio features for testing
                audio_features = {
                    "energy": 1000,
                    "pitch": 150,
                    "duration": 3.0
                }
            except sr.UnknownValueError:
                return  # Could not understand
            except sr.RequestError as e:
                log.error(f"Recognition error: {e}")
                return
        else:
            # Mock data for testing
            text = "This is a test message"
            audio_features = {"mock": True}
        
        # Identify speaker
        speaker_id, confidence, metadata = self.speaker_id_agent.identify_speaker(
            audio_features
        )
        
        # Check if enrollment is active
        if speaker_id in self.enrollment_sessions:
            await self._handle_enrollment(speaker_id, audio_features)
            return
        
        # Add to conversation
        turn = self.conversation_manager.add_turn(
            text=text,
            speaker_id=speaker_id,
            audio_features=audio_features
        )
        
        # Display what was heard
        speaker_name = metadata.get("name", speaker_id)
        conf_str = f"({confidence*100:.0f}%)" if confidence > 0 else "(new)"
        print(f"[{speaker_name} {conf_str}]: {text}")
        
        # NEW v4: Check if we should respond
        should_respond, resp_confidence, reason = self.context_agent.should_respond(
            text=text,
            speaker_id=speaker_id,
            conversation_context=self.conversation_manager.get_context(num_turns=5),
            confidence_threshold=self.confidence_threshold
        )
        
        if should_respond and resp_confidence >= self.confidence_threshold:
            # Generate and queue response
            await self._generate_response(text, speaker_id, metadata, reason)
        
        # Process through diarization
        segments = self.diarization_agent.process_audio_stream(
            audio_features,
            timestamp.timestamp()
        )
        
        self.segment_buffer.extend(segments)
        
        # Organize information
        category, cat_conf = self.information_organizer.categorize(
            text,
            source=speaker_id
        )
        
        if category != "general":
            print(f"  📁 Categorized as: {category}")
        
        # Check for enrollment request
        if self._is_enrollment_request(text):
            await self._start_enrollment(speaker_id, text)
    
    async def _generate_response(self, text: str, speaker_id: str, 
                                metadata: Dict, reason: str):
        """
        Generate and queue a response.
        
        Args:
            text: User's text
            speaker_id: Speaker identifier
            metadata: Speaker metadata
            reason: Why we're responding
        """
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_response_time < self.min_response_interval:
            return
        
        # Get conversation context
        context = self.conversation_manager.get_context(num_turns=5)
        
        # Get relevant information
        info_items = self.information_organizer.retrieve(query=text, limit=3)
        info_context = {
            "items": [{"content": item.content, "category": item.category} 
                     for item in info_items]
        }
        
        # Generate response
        response = self.response_generator.generate_response(
            text=text,
            speaker_info={"id": speaker_id, "name": metadata.get("name", "User")},
            conversation_context=context,
            information_context=info_context
        )
        
        if response:
            # Add to conversation as assistant turn
            self.conversation_manager.add_turn(
                text=response,
                speaker_id="assistant",
                audio_features={"generated": True, "reason": reason}
            )
            
            # Queue for TTS
            if self.use_tts:
                self.response_queue.put(response)
            else:
                print(f"\n💬 [{self.name}]: {response}\n")
            
            self.last_response_time = current_time
    
    def _is_enrollment_request(self, text: str) -> bool:
        """
        Check if text is requesting voice enrollment.
        
        Args:
            text: Transcribed text
            
        Returns:
            True if enrollment requested
        """
        enrollment_phrases = [
            "enroll my voice",
            "register my voice",
            "add my voice",
            "remember my voice",
            "this is"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in enrollment_phrases)
    
    async def _start_enrollment(self, speaker_id: str, text: str):
        """
        Start voice enrollment process.
        
        Args:
            speaker_id: Current speaker ID
            text: Text that triggered enrollment
        """
        # Extract name if provided
        name = "Unknown"
        if "this is" in text.lower():
            parts = text.lower().split("this is")
            if len(parts) > 1:
                name = parts[1].strip().title()
        elif "my name is" in text.lower():
            parts = text.lower().split("my name is")
            if len(parts) > 1:
                name = parts[1].strip().title()
        
        # Start enrollment
        session_id = self.enrollment_agent.start_enrollment(name)
        self.enrollment_sessions[speaker_id] = session_id
        
        # Get first phrase
        phrase = self.enrollment_agent.get_next_phrase(session_id)
        
        enrollment_msg = f"Starting voice enrollment for {name}. Please say: '{phrase}'"
        
        print(f"\n🎯 {enrollment_msg}\n")
        
        # Speak the instruction
        if self.use_tts:
            self.response_queue.put(enrollment_msg)
    
    async def _handle_enrollment(self, speaker_id: str, audio_data: Any):
        """
        Handle enrollment sample.
        
        Args:
            speaker_id: Speaker ID
            audio_data: Audio sample
        """
        session_id = self.enrollment_sessions.get(speaker_id)
        if not session_id:
            return
        
        # Add sample
        result = self.enrollment_agent.add_voice_sample(session_id, audio_data)
        
        if result.get("accepted"):
            msg = f"Sample {result['samples_collected']} accepted"
            print(f"✅ {msg}")
            
            if result["status"] == "ready_to_complete":
                # Complete enrollment
                completion = self.enrollment_agent.complete_enrollment(session_id)
                if completion.get("success"):
                    complete_msg = f"Enrollment complete! I'll remember you as {completion.get('name', 'User')}"
                    print(f"\n🎉 {complete_msg}")
                    
                    if self.use_tts:
                        self.response_queue.put(complete_msg)
                    
                    del self.enrollment_sessions[speaker_id]
                else:
                    print(f"❌ Enrollment failed: {completion.get('error')}")
            else:
                # Get next phrase
                next_phrase = result.get("next_phrase")
                if next_phrase:
                    instruction = f"Please say: '{next_phrase}'"
                    print(instruction)
                    
                    if self.use_tts:
                        self.response_queue.put(instruction)
        else:
            error_msg = f"Sample rejected: {result.get('reason')}. {result.get('suggestion', 'Please try again')}"
            print(f"❌ {error_msg}")
            
            if self.use_tts:
                self.response_queue.put(error_msg)
    
    async def _process_segments(self):
        """
        Process diarized segments.
        """
        if not self.segment_buffer:
            return
        
        # Process each segment
        for segment in self.segment_buffer:
            # Update current speaker
            if segment.speaker_id != self.current_speaker:
                self.current_speaker = segment.speaker_id
                
                # Get speaker info
                speaker_info = None
                for speaker in self.speaker_id_agent.get_all_speakers():
                    if speaker["speaker_id"] == segment.speaker_id:
                        speaker_info = speaker
                        break
                
                if speaker_info and not speaker_info.get("temporary"):
                    print(f"\n👤 Speaker changed to: {speaker_info['name']}")
        
        # Clear processed segments
        self.segment_buffer.clear()
    
    async def _check_interjections(self):
        """
        Check if we should interject.
        """
        # Get recent context
        context = self.conversation_manager.get_context(num_turns=3)
        if not context:
            return
        
        # Get last turn
        last_turn = context[-1]
        
        # Don't interject on our own responses
        if last_turn.get("speaker_id") == "assistant":
            return
        
        # Search for relevant information
        relevant_info = []
        items = self.information_organizer.retrieve(
            query=last_turn["text"],
            limit=5
        )
        
        for item in items:
            relevant_info.append({
                "content": item.content,
                "relevance_score": 0.8,
                "category": item.category,
                "source": item.source
            })
        
        # Check if should interject
        should_interject, confidence, int_type = self.interjection_agent.should_interject(
            conversation_turn=last_turn,
            available_info=relevant_info
        )
        
        if should_interject and confidence > self.confidence_threshold:
            await self._make_interjection(int_type, relevant_info)
    
    async def _make_interjection(self, int_type: str, info: List[Dict[str, Any]]):
        """
        Make an interjection.
        
        Args:
            int_type: Type of interjection
            info: Relevant information
        """
        # Format interjection based on type
        if int_type == "clarification":
            content = "I have some information that might help clarify..."
        elif int_type == "correction":
            content = "Actually, I should mention..."
        elif int_type == "reminder":
            content = "Just a reminder..."
        else:
            content = "I thought you should know..."
        
        # Add relevant info
        if info:
            content += f" {info[0]['content']}"
        
        # Add to conversation
        self.conversation_manager.add_turn(
            text=content,
            speaker_id="assistant",
            audio_features={"interjection": True}
        )
        
        # Speak the interjection
        if self.use_tts:
            self.response_queue.put(content)
        else:
            print(f"\n💡 [{self.name}]: {content}\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Statistics dictionary
        """
        conv_analysis = self.conversation_manager.analyze_conversation()
        info_summary = self.information_organizer.get_summary()
        
        # Get speaker stats
        all_speakers = self.speaker_id_agent.get_all_speakers()
        permanent_speakers = [s for s in all_speakers if not s.get("temporary")]
        temp_speakers = [s for s in all_speakers if s.get("temporary")]
        
        return {
            "conversation": conv_analysis,
            "information": info_summary,
            "speakers": {
                "total": len(all_speakers),
                "permanent": len(permanent_speakers),
                "temporary": len(temp_speakers),
                "enrolled_names": [s["name"] for s in permanent_speakers]
            },
            "active_task": self.active_task,
            "enrollments_active": len(self.enrollment_sessions),
            "tts_enabled": self.use_tts,
            "responses_generated": self.context_agent.last_response_time > 0
        }


def main():
    """
    Main entry point for Listen v4.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Listen v4 - Intelligent conversational assistant with voice reply"
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Optional task to focus on"
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to speaker database"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for responses (0-1)"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech output"
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="TTS voice to use"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a WAV file instead of microphone"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        print("Demo mode for Listen v4")
        print("Features:")
        print("- Speaker identification & enrollment")
        print("- Context-aware responses")
        print("- Voice synthesis replies")
        print("- Intelligent conversation management")
    else:
        # Create and run Listen v4
        assistant = ListenV4(
            name="Listen v4",
            db_path=args.db,
            confidence_threshold=args.confidence,
            use_tts=not args.no_tts,
            tts_voice=args.voice
        )
        
        # Run async
        asyncio.run(assistant.start(task=args.task))


if __name__ == "__main__":
    main()