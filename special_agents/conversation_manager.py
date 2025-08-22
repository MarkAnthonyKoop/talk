"""
ConversationManager - Tracks conversations with speaker identification.

This module manages conversation flow, speaker diarization, and maintains
context about who said what in the conversation.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path

log = logging.getLogger(__name__)


class Speaker:
    """Represents a speaker in the conversation."""
    
    def __init__(self, speaker_id: str, name: Optional[str] = None):
        """Initialize a speaker."""
        self.id = speaker_id
        self.name = name or f"Speaker_{speaker_id}"
        self.voice_embeddings = []  # For future voice recognition
        self.utterances = []
        self.first_heard = datetime.now()
        self.last_heard = None
        self.metadata = {}
    
    def add_utterance(self, text: str, timestamp: datetime, confidence: float = 1.0):
        """Add an utterance from this speaker."""
        self.utterances.append({
            "text": text,
            "timestamp": timestamp.isoformat(),
            "confidence": confidence
        })
        self.last_heard = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert speaker to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "utterance_count": len(self.utterances),
            "first_heard": self.first_heard.isoformat(),
            "last_heard": self.last_heard.isoformat() if self.last_heard else None,
            "metadata": self.metadata
        }


class ConversationTurn:
    """Represents a single turn in the conversation."""
    
    def __init__(self, 
                 speaker_id: str,
                 text: str,
                 timestamp: datetime,
                 confidence: float = 1.0,
                 metadata: Optional[Dict] = None):
        """Initialize a conversation turn."""
        self.speaker_id = speaker_id
        self.text = text
        self.timestamp = timestamp
        self.confidence = confidence
        self.metadata = metadata or {}
        self.categories = []
        self.entities = []  # Named entities mentioned
        self.intent = None  # Detected intent
        self.sentiment = None  # Sentiment score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "categories": self.categories,
            "entities": self.entities,
            "intent": self.intent,
            "sentiment": self.sentiment,
            "metadata": self.metadata
        }


class ConversationManager:
    """
    Manages conversation tracking with speaker identification.
    
    This class tracks who said what, maintains conversation history,
    and provides analysis of conversation patterns.
    """
    
    def __init__(self, 
                 conversation_id: Optional[str] = None,
                 enable_diarization: bool = True,
                 save_path: Optional[Path] = None):
        """
        Initialize the conversation manager.
        
        Args:
            conversation_id: Unique ID for this conversation
            enable_diarization: Whether to attempt speaker diarization
            save_path: Path to save conversation logs
        """
        self.conversation_id = conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_diarization = enable_diarization
        self.save_path = save_path or Path(".talk_scratch/conversations")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Conversation state
        self.speakers: Dict[str, Speaker] = {}
        self.turns: List[ConversationTurn] = []
        self.current_speaker = None
        self.start_time = datetime.now()
        
        # Analysis caches
        self.topic_history = deque(maxlen=50)
        self.context_window = deque(maxlen=10)
        self.speaker_patterns = defaultdict(lambda: {
            "avg_utterance_length": 0,
            "speaking_time": 0,
            "interruptions": 0,
            "questions_asked": 0
        })
        
        # Diarization state
        self.voice_buffer = []
        self.speaker_embeddings = {}
        
        log.info(f"Initialized conversation: {self.conversation_id}")
    
    def add_speaker(self, speaker_id: str, name: Optional[str] = None) -> Speaker:
        """
        Add a new speaker to the conversation.
        
        Args:
            speaker_id: Unique identifier for the speaker
            name: Optional human-readable name
            
        Returns:
            The Speaker object
        """
        if speaker_id not in self.speakers:
            speaker = Speaker(speaker_id, name)
            self.speakers[speaker_id] = speaker
            log.info(f"Added speaker: {speaker.name}")
        return self.speakers[speaker_id]
    
    def identify_speaker(self, 
                        audio_features: Optional[Dict] = None,
                        text: Optional[str] = None) -> str:
        """
        Identify the current speaker.
        
        This is a simplified version. In a real implementation,
        this would use voice embeddings or other features.
        
        Args:
            audio_features: Audio characteristics
            text: Text content (for pattern matching)
            
        Returns:
            Speaker ID
        """
        if not self.enable_diarization:
            return "default"
        
        # Simplified heuristic-based identification
        # In production, use pyannote.audio or similar
        
        # For now, use simple alternation or patterns
        if audio_features and "pitch" in audio_features:
            # Use pitch to distinguish (simplified)
            pitch = audio_features["pitch"]
            if pitch > 200:  # Higher pitch
                speaker_id = "speaker_1"
            else:
                speaker_id = "speaker_2"
        elif self.current_speaker and len(self.speakers) > 1:
            # Simple alternation for demo
            current_idx = list(self.speakers.keys()).index(self.current_speaker)
            next_idx = (current_idx + 1) % len(self.speakers)
            speaker_id = list(self.speakers.keys())[next_idx]
        else:
            # Default speaker
            speaker_id = "user"
        
        # Ensure speaker exists
        if speaker_id not in self.speakers:
            self.add_speaker(speaker_id)
        
        return speaker_id
    
    def add_turn(self,
                 text: str,
                 speaker_id: Optional[str] = None,
                 audio_features: Optional[Dict] = None,
                 confidence: float = 1.0,
                 metadata: Optional[Dict] = None) -> ConversationTurn:
        """
        Add a turn to the conversation.
        
        Args:
            text: The spoken text
            speaker_id: Optional speaker ID (will identify if not provided)
            audio_features: Audio characteristics for speaker identification
            confidence: Confidence in transcription
            metadata: Additional metadata
            
        Returns:
            The created ConversationTurn
        """
        # Identify speaker if not provided
        if not speaker_id:
            speaker_id = self.identify_speaker(audio_features, text)
        
        # Create turn
        turn = ConversationTurn(
            speaker_id=speaker_id,
            text=text,
            timestamp=datetime.now(),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Add to conversation
        self.turns.append(turn)
        self.context_window.append(turn)
        
        # Update speaker (auto-create if unknown)
        if speaker_id not in self.speakers:
            self.add_speaker(speaker_id, f"Unknown Speaker {len(self.speakers) + 1}")
        speaker = self.speakers[speaker_id]
        speaker.add_utterance(text, turn.timestamp, confidence)
        self.current_speaker = speaker_id
        
        # Update patterns
        self._update_speaker_patterns(speaker_id, text)
        
        # Auto-save periodically
        if len(self.turns) % 10 == 0:
            self.save_conversation()
        
        log.debug(f"[{speaker.name}]: {text[:50]}...")
        
        return turn
    
    def _update_speaker_patterns(self, speaker_id: str, text: str):
        """Update speaking patterns for analysis."""
        patterns = self.speaker_patterns[speaker_id]
        
        # Update average length
        current_avg = patterns["avg_utterance_length"]
        utterance_count = len(self.speakers[speaker_id].utterances)
        patterns["avg_utterance_length"] = (
            (current_avg * (utterance_count - 1) + len(text)) / utterance_count
        )
        
        # Check for questions
        if "?" in text:
            patterns["questions_asked"] += 1
        
        # Check for interruptions (simplified)
        if len(self.turns) > 1:
            last_turn = self.turns[-2]
            if last_turn.speaker_id != speaker_id:
                time_diff = (datetime.now() - last_turn.timestamp).total_seconds()
                if time_diff < 1.0:  # Quick response might be interruption
                    patterns["interruptions"] += 1
    
    def get_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.
        
        Args:
            num_turns: Number of recent turns to return
            
        Returns:
            List of recent turns as dictionaries
        """
        recent_turns = list(self.context_window)[-num_turns:]
        context = []
        
        for turn in recent_turns:
            speaker = self.speakers[turn.speaker_id]
            context.append({
                "speaker": speaker.name,
                "text": turn.text,
                "timestamp": turn.timestamp.isoformat()
            })
        
        return context
    
    def analyze_conversation(self) -> Dict[str, Any]:
        """
        Analyze the conversation for patterns and insights.
        
        Returns:
            Analysis dictionary with various metrics
        """
        if not self.turns:
            return {"status": "no_conversation"}
        
        analysis = {
            "conversation_id": self.conversation_id,
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "total_turns": len(self.turns),
            "speaker_count": len(self.speakers),
            "speakers": {}
        }
        
        # Analyze per speaker
        for speaker_id, speaker in self.speakers.items():
            patterns = self.speaker_patterns[speaker_id]
            analysis["speakers"][speaker.name] = {
                "utterance_count": len(speaker.utterances),
                "avg_utterance_length": patterns["avg_utterance_length"],
                "questions_asked": patterns["questions_asked"],
                "interruptions": patterns["interruptions"],
                "participation_rate": len(speaker.utterances) / len(self.turns)
            }
        
        # Identify dominant speaker
        if self.speakers:
            dominant = max(
                self.speakers.values(),
                key=lambda s: len(s.utterances)
            )
            analysis["dominant_speaker"] = dominant.name
        
        # Topic extraction (simplified)
        all_text = " ".join(turn.text for turn in self.turns[-20:])
        analysis["recent_topics"] = self._extract_topics(all_text)
        
        return analysis
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified version)."""
        # In production, use NLP libraries like spaCy or NLTK
        # For now, extract capitalized words as potential topics
        import re
        
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in {"The", "This", "That", "These", "Those", "A", "An"}:
                word_freq[word] += 1
        
        # Return top 5 topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic for topic, _ in topics]
    
    def save_conversation(self):
        """Save conversation to disk."""
        try:
            filepath = self.save_path / f"{self.conversation_id}.jsonl"
            
            with open(filepath, "w") as f:
                # Write metadata
                f.write(json.dumps({
                    "type": "metadata",
                    "conversation_id": self.conversation_id,
                    "start_time": self.start_time.isoformat(),
                    "speakers": {
                        sid: speaker.to_dict() 
                        for sid, speaker in self.speakers.items()
                    }
                }) + "\n")
                
                # Write turns
                for turn in self.turns:
                    f.write(json.dumps({
                        "type": "turn",
                        **turn.to_dict()
                    }) + "\n")
            
            log.debug(f"Saved conversation to {filepath}")
        
        except Exception as e:
            log.error(f"Failed to save conversation: {e}")
    
    def load_conversation(self, conversation_id: str):
        """Load a previous conversation from disk."""
        filepath = self.save_path / f"{conversation_id}.jsonl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                
                if data["type"] == "metadata":
                    self.conversation_id = data["conversation_id"]
                    self.start_time = datetime.fromisoformat(data["start_time"])
                    
                    # Restore speakers
                    for sid, speaker_data in data["speakers"].items():
                        speaker = self.add_speaker(sid, speaker_data["name"])
                        speaker.first_heard = datetime.fromisoformat(speaker_data["first_heard"])
                        if speaker_data["last_heard"]:
                            speaker.last_heard = datetime.fromisoformat(speaker_data["last_heard"])
                
                elif data["type"] == "turn":
                    turn = ConversationTurn(
                        speaker_id=data["speaker_id"],
                        text=data["text"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        confidence=data["confidence"],
                        metadata=data.get("metadata", {})
                    )
                    turn.categories = data.get("categories", [])
                    turn.entities = data.get("entities", [])
                    turn.intent = data.get("intent")
                    turn.sentiment = data.get("sentiment")
                    
                    self.turns.append(turn)
                    self.context_window.append(turn)
        
        log.info(f"Loaded conversation {conversation_id} with {len(self.turns)} turns")
    
    def export_transcript(self) -> str:
        """
        Export conversation as readable transcript.
        
        Returns:
            Formatted transcript string
        """
        transcript = f"Conversation: {self.conversation_id}\n"
        transcript += f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        transcript += f"Participants: {', '.join(s.name for s in self.speakers.values())}\n"
        transcript += "=" * 50 + "\n\n"
        
        for turn in self.turns:
            speaker = self.speakers[turn.speaker_id]
            timestamp = turn.timestamp.strftime("%H:%M:%S")
            transcript += f"[{timestamp}] {speaker.name}: {turn.text}\n"
        
        return transcript