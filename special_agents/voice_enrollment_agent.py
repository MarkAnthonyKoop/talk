#!/usr/bin/env python3
"""
VoiceEnrollmentAgent - Manages voice enrollment workflow for speakers.

This agent handles the process of enrolling new speakers, collecting
voice samples, and managing the enrollment process.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from agent.agent import Agent
from special_agents.speaker_identification_agent import SpeakerIdentificationAgent

log = logging.getLogger(__name__)


class VoiceEnrollmentAgent(Agent):
    """
    Agent that manages voice enrollment for speaker identification.
    
    This agent:
    1. Guides users through voice enrollment
    2. Collects multiple voice samples
    3. Validates sample quality
    4. Coordinates with SpeakerIdentificationAgent
    """
    
    def __init__(self,
                 speaker_agent: Optional[SpeakerIdentificationAgent] = None,
                 min_samples: int = 3,
                 max_samples: int = 10,
                 **kwargs):
        """
        Initialize the voice enrollment agent.
        
        Args:
            speaker_agent: SpeakerIdentificationAgent instance
            min_samples: Minimum voice samples required
            max_samples: Maximum voice samples to collect
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are a voice enrollment assistant.",
            "You guide users through the process of enrolling their voice.",
            "You ensure high-quality voice samples are collected.",
            "You provide clear instructions and feedback."
        ]
        super().__init__(roles=roles, **kwargs)
        
        if speaker_agent is None:
            log.warning("No SpeakerIdentificationAgent provided to VoiceEnrollmentAgent")
            self.speaker_agent = SpeakerIdentificationAgent(use_mock=False)
        else:
            self.speaker_agent = speaker_agent
        self.min_samples = min_samples
        self.max_samples = max_samples
        
        # Enrollment state
        self.active_enrollments: Dict[str, Dict[str, Any]] = {}
        self.completed_enrollments: List[Dict[str, Any]] = []
        
        # Sample phrases for enrollment
        self.enrollment_phrases = [
            "The quick brown fox jumps over the lazy dog",
            "Hello, my name is {name} and this is my voice",
            "I am enrolling my voice for speaker identification",
            "Testing one two three, testing voice enrollment",
            "The weather today is perfect for voice recording",
            "Please recognize my voice in future conversations",
            "This is sample number {number} of my voice enrollment",
            "I speak clearly and naturally for best results",
            "Voice identification helps personalize my experience",
            "Thank you for enrolling my voice profile"
        ]
        
        log.info(f"Initialized VoiceEnrollmentAgent (min: {min_samples}, max: {max_samples} samples)")
    
    def start_enrollment(self, name: str, email: Optional[str] = None) -> str:
        """
        Start a new voice enrollment session.
        
        Args:
            name: Name of the person enrolling
            email: Optional email for the profile
            
        Returns:
            Enrollment session ID
        """
        session_id = f"enroll_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.lower().replace(' ', '_')}"
        
        self.active_enrollments[session_id] = {
            "name": name,
            "email": email,
            "samples": [],
            "started_at": datetime.now(),
            "status": "active",
            "current_phrase_index": 0,
            "quality_scores": []
        }
        
        log.info(f"Started enrollment session: {session_id} for {name}")
        return session_id
    
    def get_next_phrase(self, session_id: str) -> Optional[str]:
        """
        Get the next phrase for enrollment.
        
        Args:
            session_id: Enrollment session ID
            
        Returns:
            The next phrase to speak, or None if complete
        """
        if session_id not in self.active_enrollments:
            return None
        
        enrollment = self.active_enrollments[session_id]
        
        # Check if we have enough samples
        if len(enrollment["samples"]) >= self.min_samples:
            return None
        
        # Get next phrase
        index = enrollment["current_phrase_index"]
        if index < len(self.enrollment_phrases):
            phrase = self.enrollment_phrases[index]
            
            # Substitute placeholders
            phrase = phrase.format(
                name=enrollment["name"],
                number=index + 1
            )
            
            return phrase
        
        # Repeat phrases if needed
        return self.enrollment_phrases[index % len(self.enrollment_phrases)]
    
    def add_voice_sample(self, session_id: str, audio_data: Any) -> Dict[str, Any]:
        """
        Add a voice sample to the enrollment session.
        
        Args:
            session_id: Enrollment session ID
            audio_data: Audio sample data
            
        Returns:
            Dictionary with quality assessment and status
        """
        if session_id not in self.active_enrollments:
            return {"error": "Invalid session ID"}
        
        enrollment = self.active_enrollments[session_id]
        
        # Assess sample quality
        quality = self._assess_sample_quality(audio_data)
        
        if quality["acceptable"]:
            # Add sample
            enrollment["samples"].append({
                "audio": audio_data,
                "timestamp": datetime.now().isoformat(),
                "quality": quality,
                "phrase_index": enrollment["current_phrase_index"]
            })
            
            enrollment["quality_scores"].append(quality["score"])
            enrollment["current_phrase_index"] += 1
            
            # Check if enrollment is complete
            samples_count = len(enrollment["samples"])
            if samples_count >= self.min_samples:
                status = "ready_to_complete"
            else:
                status = "need_more_samples"
            
            return {
                "accepted": True,
                "quality": quality,
                "samples_collected": samples_count,
                "samples_needed": max(0, self.min_samples - samples_count),
                "status": status,
                "next_phrase": self.get_next_phrase(session_id)
            }
        else:
            # Reject sample
            return {
                "accepted": False,
                "quality": quality,
                "reason": quality.get("reason", "Quality too low"),
                "suggestion": "Please speak clearly and ensure low background noise",
                "retry_same_phrase": True
            }
    
    def _assess_sample_quality(self, audio_data: Any) -> Dict[str, Any]:
        """
        Assess the quality of a voice sample.
        
        Args:
            audio_data: Audio sample to assess
            
        Returns:
            Quality assessment dictionary
        """
        # Mock quality assessment for now
        # Real implementation would check:
        # - Signal-to-noise ratio
        # - Duration
        # - Clipping
        # - Energy levels
        
        if isinstance(audio_data, dict):
            # Check metadata hints
            duration = audio_data.get("duration", 3.0)
            energy = audio_data.get("energy", 1000)
            
            score = min(1.0, duration / 3.0) * min(1.0, energy / 1000)
            
            return {
                "acceptable": score > 0.5,
                "score": score,
                "duration": duration,
                "energy": energy,
                "snr": audio_data.get("snr", 20),  # Signal-to-noise ratio
                "clipping": False
            }
        
        # Default mock assessment
        return {
            "acceptable": True,
            "score": 0.8,
            "duration": 3.0,
            "energy": 1500,
            "snr": 25,
            "clipping": False
        }
    
    def complete_enrollment(self, session_id: str) -> Dict[str, Any]:
        """
        Complete an enrollment session.
        
        Args:
            session_id: Enrollment session ID
            
        Returns:
            Enrollment result with speaker ID
        """
        if session_id not in self.active_enrollments:
            return {"error": "Invalid session ID"}
        
        enrollment = self.active_enrollments[session_id]
        
        if len(enrollment["samples"]) < self.min_samples:
            return {
                "error": "Insufficient samples",
                "samples_collected": len(enrollment["samples"]),
                "samples_needed": self.min_samples
            }
        
        # Extract audio samples
        audio_samples = [s["audio"] for s in enrollment["samples"]]
        
        # Enroll with speaker identification agent
        speaker_id = self.speaker_agent.enroll_speaker(
            name=enrollment["name"],
            audio_samples=audio_samples
        )
        
        # Mark enrollment as complete
        enrollment["status"] = "completed"
        enrollment["completed_at"] = datetime.now()
        enrollment["speaker_id"] = speaker_id
        enrollment["average_quality"] = np.mean(enrollment["quality_scores"])
        
        # Move to completed
        self.completed_enrollments.append(enrollment)
        del self.active_enrollments[session_id]
        
        log.info(f"Completed enrollment: {session_id} -> {speaker_id}")
        
        return {
            "success": True,
            "speaker_id": speaker_id,
            "name": enrollment["name"],
            "samples_used": len(audio_samples),
            "average_quality": enrollment["average_quality"],
            "duration": (enrollment["completed_at"] - enrollment["started_at"]).total_seconds()
        }
    
    def cancel_enrollment(self, session_id: str) -> bool:
        """
        Cancel an enrollment session.
        
        Args:
            session_id: Enrollment session ID
            
        Returns:
            True if cancelled successfully
        """
        if session_id in self.active_enrollments:
            enrollment = self.active_enrollments[session_id]
            enrollment["status"] = "cancelled"
            enrollment["cancelled_at"] = datetime.now()
            
            # Move to completed with cancelled status
            self.completed_enrollments.append(enrollment)
            del self.active_enrollments[session_id]
            
            log.info(f"Cancelled enrollment: {session_id}")
            return True
        
        return False
    
    def get_enrollment_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the status of an enrollment session.
        
        Args:
            session_id: Enrollment session ID
            
        Returns:
            Status dictionary
        """
        if session_id in self.active_enrollments:
            enrollment = self.active_enrollments[session_id]
            return {
                "session_id": session_id,
                "name": enrollment["name"],
                "status": enrollment["status"],
                "samples_collected": len(enrollment["samples"]),
                "samples_needed": max(0, self.min_samples - len(enrollment["samples"])),
                "started_at": enrollment["started_at"].isoformat(),
                "average_quality": np.mean(enrollment["quality_scores"]) if enrollment["quality_scores"] else 0
            }
        
        # Check completed enrollments
        for enrollment in self.completed_enrollments:
            if enrollment.get("session_id") == session_id:
                return {
                    "session_id": session_id,
                    "status": enrollment["status"],
                    "speaker_id": enrollment.get("speaker_id"),
                    "completed_at": enrollment.get("completed_at", "").isoformat() if enrollment.get("completed_at") else None
                }
        
        return {"error": "Session not found"}
    
    def list_active_enrollments(self) -> List[Dict[str, Any]]:
        """List all active enrollment sessions."""
        active = []
        for session_id, enrollment in self.active_enrollments.items():
            active.append({
                "session_id": session_id,
                "name": enrollment["name"],
                "samples_collected": len(enrollment["samples"]),
                "started_at": enrollment["started_at"].isoformat()
            })
        return active
    
    def run(self, prompt: str) -> str:
        """
        Process enrollment commands.
        
        Args:
            prompt: Command or instruction
            
        Returns:
            Response as JSON or text
        """
        try:
            # Parse JSON commands
            if prompt.startswith("{"):
                data = json.loads(prompt)
                command = data.get("command", "help")
                
                if command == "start":
                    # Start new enrollment
                    name = data.get("name", "Unknown")
                    email = data.get("email")
                    session_id = self.start_enrollment(name, email)
                    
                    return json.dumps({
                        "session_id": session_id,
                        "first_phrase": self.get_next_phrase(session_id),
                        "instructions": "Please read the phrase clearly"
                    }, indent=2)
                
                elif command == "add_sample":
                    # Add voice sample
                    session_id = data.get("session_id")
                    audio_data = data.get("audio_data", {})
                    
                    result = self.add_voice_sample(session_id, audio_data)
                    return json.dumps(result, indent=2)
                
                elif command == "complete":
                    # Complete enrollment
                    session_id = data.get("session_id")
                    result = self.complete_enrollment(session_id)
                    return json.dumps(result, indent=2)
                
                elif command == "status":
                    # Get status
                    session_id = data.get("session_id")
                    status = self.get_enrollment_status(session_id)
                    return json.dumps(status, indent=2)
                
                elif command == "list":
                    # List active enrollments
                    active = self.list_active_enrollments()
                    return json.dumps({
                        "active_enrollments": active,
                        "count": len(active)
                    }, indent=2)
                
                elif command == "cancel":
                    # Cancel enrollment
                    session_id = data.get("session_id")
                    success = self.cancel_enrollment(session_id)
                    return json.dumps({"cancelled": success}, indent=2)
                
                else:
                    return json.dumps({
                        "error": f"Unknown command: {command}",
                        "available_commands": [
                            "start", "add_sample", "complete", 
                            "status", "list", "cancel"
                        ]
                    })
            
            # Natural language processing
            else:
                prompt_lower = prompt.lower()
                
                if "enroll" in prompt_lower or "sign up" in prompt_lower:
                    return """To enroll your voice, I'll need to collect a few samples.
                    
Please provide your name and I'll guide you through the process.
You'll need to read 3-5 short phrases clearly.

Ready to start? Say 'Start enrollment for [your name]'."""
                
                elif "start enrollment for" in prompt_lower:
                    # Extract name
                    name = prompt.replace("start enrollment for", "").strip()
                    session_id = self.start_enrollment(name)
                    first_phrase = self.get_next_phrase(session_id)
                    
                    return f"""Enrollment started for {name}!
                    
Session ID: {session_id}

Please read the following phrase clearly:
"{first_phrase}"

When ready, provide the audio sample."""
                
                else:
                    return """I'm the Voice Enrollment Assistant.
                    
I can help you:
- Enroll your voice for speaker identification
- Guide you through the sample collection process
- Ensure high-quality voice samples

Say 'Start enrollment for [your name]' to begin."""
        
        except Exception as e:
            log.error(f"Error in VoiceEnrollmentAgent: {e}")
            return json.dumps({"error": str(e)})
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Manage voice enrollment for speaker identification"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest voice enrollment is needed."""
        return ["enroll", "enrollment", "register", "voice", "signup", "add speaker"]