#!/usr/bin/env python3
"""
Listen v3 - Advanced Personal Assistant with Speaker Identification.

Integrates speaker identification, enrollment, and diarization agents
for comprehensive voice-based interaction management.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue

# Audio handling
try:
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Install with: pip install SpeechRecognition pyaudio")

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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ListenV3:
    """
    Listen v3 - Advanced voice-aware personal assistant.
    
    Features:
    - Real-time speaker identification and diarization
    - Voice enrollment for new speakers
    - Multi-source input processing
    - Intelligent context management
    - Task-aware active listening
    """
    
    def __init__(self,
                 name: str = "Listen v3",
                 db_path: Optional[Path] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize Listen v3.
        
        Args:
            name: Assistant name
            db_path: Path to speaker database
            confidence_threshold: Confidence threshold for actions
        """
        self.name = name
        self.db_path = db_path or Path.home() / ".listen" / "speakers.db"
        self.confidence_threshold = confidence_threshold
        
        # Initialize speaker agents
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
        
        # Initialize other components
        self.orchestrator = MultiSourceOrchestrator()
        self.conversation_manager = ConversationManager()
        self.information_organizer = InformationOrganizer()
        self.interjection_agent = InterjectionAgent(
            confidence_threshold=confidence_threshold
        )
        
        # Planning agents
        self.planning_agent = ExecutionPlanningAgent()
        self.listening_agent = ActiveListeningAgent()
        
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
        
        # Speaker state
        self.current_speaker = None
        self.enrollment_sessions = {}
        
        log.info(f"Initialized {name} with speaker identification")
    
    async def start(self, task: Optional[str] = None):
        """
        Start the Listen v3 assistant.
        
        Args:
            task: Optional task to focus on
        """
        self.active_task = task
        
        print(f"\n🎙️  {self.name} Starting...")
        print("=" * 50)
        
        # Load speaker profiles
        speakers = self.speaker_id_agent.get_all_speakers()
        print(f"Loaded {len(speakers)} speaker profiles")
        
        if task:
            print(f"Active Task: {task}")
            # Create execution plan
            plan = self.planning_agent.create_plan(task)
            if plan:
                print(f"Created plan with {len(plan.get('steps', []))} steps")
        
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
            print("\n\nStopping Listen v3...")
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
            print("\n🎤 Listening... (Press Ctrl+C to stop)\n")
            
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
        
        print(f"\n🎯 Starting voice enrollment for {name}")
        print(f"Please say: \"{phrase}\"\n")
    
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
            print(f"✅ Sample {result['samples_collected']} accepted")
            
            if result["status"] == "ready_to_complete":
                # Complete enrollment
                completion = self.enrollment_agent.complete_enrollment(session_id)
                if completion.get("success"):
                    print(f"\n🎉 Enrollment complete! Speaker ID: {completion['speaker_id']}")
                    del self.enrollment_sessions[speaker_id]
                else:
                    print(f"❌ Enrollment failed: {completion.get('error')}")
            else:
                # Get next phrase
                next_phrase = result.get("next_phrase")
                if next_phrase:
                    print(f"Please say: \"{next_phrase}\"")
        else:
            print(f"❌ Sample rejected: {result.get('reason')}")
            print(result.get("suggestion", "Please try again"))
    
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
        
        print(f"\n💡 [{self.name}]: {content}\n")
        
        # Add to conversation
        self.conversation_manager.add_turn(
            text=content,
            speaker_id="assistant",
            audio_features={"interjection": True}
        )
    
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
            "enrollments_active": len(self.enrollment_sessions)
        }


def main():
    """
    Main entry point for Listen v3.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Listen v3 - Voice-aware personal assistant"
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
        help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        from tests.listen.demo_listen_v3 import demo_listen_v3
        demo_listen_v3()
    else:
        # Create and run Listen v3
        assistant = ListenV3(
            name="Listen v3",
            db_path=args.db,
            confidence_threshold=args.confidence
        )
        
        # Run async
        asyncio.run(assistant.start(task=args.task))


if __name__ == "__main__":
    main()