#!/usr/bin/env python3
"""
RealSpeakerIdentificationAgent - Real speaker identification without mocking.

This agent uses SpeechBrain for embeddings and SQLite for persistence.
NO MOCKING - This is production-ready code.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from agent.agent import Agent
from external_agents.speechbrain_agent import SpeechBrainAgent
from external_agents.speaker_database import SpeakerDatabase

log = logging.getLogger(__name__)


class RealSpeakerIdentificationAgent(Agent):
    """
    Real speaker identification agent using SpeechBrain embeddings.
    
    This agent:
    - Extracts real voice embeddings using SpeechBrain
    - Stores profiles in SQLite database
    - Performs speaker identification using cosine similarity
    - NO MOCKING - uses real audio processing
    """
    
    def __init__(self,
                 db_path: Optional[Path] = None,
                 similarity_threshold: float = 0.75,
                 hf_token: Optional[str] = None,
                 **kwargs):
        """
        Initialize the real speaker identification agent.
        
        Args:
            db_path: Path to speaker database
            similarity_threshold: Minimum similarity for positive ID
            hf_token: Hugging Face token for models
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are a real speaker identification system.",
            "You use SpeechBrain to extract voice embeddings.",
            "You identify speakers using cosine similarity.",
            "You store profiles in a SQLite database."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Initialize database
        self.db_path = db_path or Path.home() / ".talk" / "speakers.db"
        self.database = SpeakerDatabase(self.db_path)
        
        # Initialize SpeechBrain agent
        self.embedding_agent = SpeechBrainAgent()
        
        self.similarity_threshold = similarity_threshold
        self.hf_token = hf_token
        
        # Load existing speakers
        self.speakers_cache = {}
        self._load_speakers_cache()
        
        log.info(f"RealSpeakerIdentificationAgent initialized with database at {self.db_path}")
    
    def _load_speakers_cache(self):
        """Load speaker embeddings into memory for faster matching."""
        speakers = self.database.get_all_speakers()
        
        for speaker in speakers:
            speaker_id = speaker['speaker_id']
            embeddings = self.database.get_speaker_embeddings(speaker_id)
            
            if embeddings:
                # Calculate centroid embedding
                centroid = np.mean(embeddings, axis=0)
                self.speakers_cache[speaker_id] = {
                    'name': speaker['name'],
                    'centroid': centroid,
                    'embeddings': embeddings,
                    'is_temporary': speaker['is_temporary']
                }
        
        log.info(f"Loaded {len(self.speakers_cache)} speakers into cache")
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract real embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_agent.extract_embedding(audio_path)
            return embedding
        except Exception as e:
            log.error(f"Failed to extract embedding: {e}")
            raise
    
    def identify_speaker(self, audio_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Identify speaker from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (speaker_id, confidence, metadata)
        """
        # Extract embedding
        embedding = self.extract_embedding(audio_path)
        
        # Find best match
        best_speaker_id = None
        best_similarity = 0.0
        best_metadata = {}
        
        for speaker_id, cache_data in self.speakers_cache.items():
            # Calculate similarity with centroid
            similarity = self.embedding_agent.calculate_similarity(
                embedding, 
                cache_data['centroid']
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker_id = speaker_id
                best_metadata = {
                    'name': cache_data['name'],
                    'is_temporary': cache_data['is_temporary'],
                    'embedding_count': len(cache_data['embeddings'])
                }
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            # Known speaker
            return best_speaker_id, best_similarity, best_metadata
        else:
            # Unknown speaker - create temporary profile
            temp_id = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add to database as temporary
            self.database.add_speaker(
                speaker_id=temp_id,
                name=f"Unknown Speaker {len(self.speakers_cache) + 1}",
                is_temporary=True
            )
            
            # Add embedding
            self.database.add_embedding(temp_id, embedding, audio_path)
            
            # Update cache
            self.speakers_cache[temp_id] = {
                'name': f"Unknown Speaker {len(self.speakers_cache) + 1}",
                'centroid': embedding,
                'embeddings': [embedding],
                'is_temporary': True
            }
            
            return temp_id, 0.0, {
                'name': f"Unknown Speaker {len(self.speakers_cache) + 1}",
                'is_temporary': True,
                'new_speaker': True
            }
    
    def enroll_speaker(self, name: str, audio_files: List[str],
                       email: Optional[str] = None) -> str:
        """
        Enroll a new speaker with audio samples.
        
        Args:
            name: Speaker's name
            audio_files: List of audio file paths
            email: Speaker's email (optional)
            
        Returns:
            Speaker ID
        """
        # Generate speaker ID
        speaker_id = f"speaker_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add speaker to database
        self.database.add_speaker(speaker_id, name, email, is_temporary=False)
        
        # Extract and store embeddings
        embeddings = []
        for audio_file in audio_files:
            try:
                embedding = self.extract_embedding(audio_file)
                self.database.add_embedding(speaker_id, embedding, audio_file)
                embeddings.append(embedding)
            except Exception as e:
                log.error(f"Failed to process {audio_file}: {e}")
        
        if embeddings:
            # Update cache
            centroid = np.mean(embeddings, axis=0)
            self.speakers_cache[speaker_id] = {
                'name': name,
                'centroid': centroid,
                'embeddings': embeddings,
                'is_temporary': False
            }
            
            log.info(f"Enrolled {name} with {len(embeddings)} samples")
        
        return speaker_id
    
    def verify_speakers(self, audio_path1: str, audio_path2: str) -> Tuple[bool, float]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio_path1: First audio file
            audio_path2: Second audio file
            
        Returns:
            Tuple of (same_speaker, similarity_score)
        """
        return self.embedding_agent.verify_speakers(
            audio_path1, 
            audio_path2,
            threshold=self.similarity_threshold
        )
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """
        Update a speaker's name.
        
        Args:
            speaker_id: Speaker ID
            new_name: New name
            
        Returns:
            True if successful
        """
        success = self.database.update_speaker_name(speaker_id, new_name)
        
        if success and speaker_id in self.speakers_cache:
            self.speakers_cache[speaker_id]['name'] = new_name
            
            # Convert temporary to permanent if it was temporary
            if self.speakers_cache[speaker_id]['is_temporary']:
                self.database.convert_temporary_to_permanent(speaker_id)
                self.speakers_cache[speaker_id]['is_temporary'] = False
        
        return success
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str,
                       new_name: Optional[str] = None) -> str:
        """
        Merge two speaker profiles.
        
        Args:
            speaker_id1: First speaker ID (kept)
            speaker_id2: Second speaker ID (merged)
            new_name: Optional new name
            
        Returns:
            Resulting speaker ID
        """
        # Merge in database
        success = self.database.merge_speakers(speaker_id1, speaker_id2)
        
        if success:
            # Update cache
            if speaker_id1 in self.speakers_cache and speaker_id2 in self.speakers_cache:
                # Combine embeddings
                embeddings1 = self.speakers_cache[speaker_id1]['embeddings']
                embeddings2 = self.speakers_cache[speaker_id2]['embeddings']
                combined = embeddings1 + embeddings2
                
                # Update cache for kept speaker
                self.speakers_cache[speaker_id1]['embeddings'] = combined
                self.speakers_cache[speaker_id1]['centroid'] = np.mean(combined, axis=0)
                
                if new_name:
                    self.speakers_cache[speaker_id1]['name'] = new_name
                    self.database.update_speaker_name(speaker_id1, new_name)
                
                # Remove merged speaker from cache
                del self.speakers_cache[speaker_id2]
        
        return speaker_id1
    
    def get_all_speakers(self) -> List[Dict[str, Any]]:
        """Get all speakers in database."""
        return self.database.get_all_speakers()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self.database.get_statistics()
        stats['cached_speakers'] = len(self.speakers_cache)
        return stats
    
    def run(self, prompt: str) -> str:
        """
        Process speaker identification commands.
        
        This agent doesn't use LLMs - it processes commands directly.
        
        Args:
            prompt: Command as JSON string
            
        Returns:
            Response as JSON string
        """
        try:
            if prompt.startswith("{"):
                data = json.loads(prompt)
                command = data.get("command", "help")
                
                if command == "identify":
                    audio_path = data.get("audio_path")
                    if not audio_path:
                        return json.dumps({"error": "audio_path required"})
                    
                    speaker_id, confidence, metadata = self.identify_speaker(audio_path)
                    return json.dumps({
                        "speaker_id": speaker_id,
                        "confidence": confidence,
                        "metadata": metadata
                    })
                
                elif command == "enroll":
                    name = data.get("name")
                    audio_files = data.get("audio_files", [])
                    email = data.get("email")
                    
                    if not name or not audio_files:
                        return json.dumps({"error": "name and audio_files required"})
                    
                    speaker_id = self.enroll_speaker(name, audio_files, email)
                    return json.dumps({
                        "speaker_id": speaker_id,
                        "enrolled": True
                    })
                
                elif command == "verify":
                    audio1 = data.get("audio_path1")
                    audio2 = data.get("audio_path2")
                    
                    if not audio1 or not audio2:
                        return json.dumps({"error": "audio_path1 and audio_path2 required"})
                    
                    same_speaker, similarity = self.verify_speakers(audio1, audio2)
                    return json.dumps({
                        "same_speaker": same_speaker,
                        "similarity": similarity
                    })
                
                elif command == "list":
                    speakers = self.get_all_speakers()
                    return json.dumps({
                        "speakers": speakers,
                        "total": len(speakers)
                    })
                
                elif command == "stats":
                    stats = self.get_statistics()
                    return json.dumps(stats)
                
                else:
                    return json.dumps({
                        "error": f"Unknown command: {command}",
                        "available_commands": [
                            "identify", "enroll", "verify", "list", "stats"
                        ]
                    })
            else:
                return json.dumps({
                    "error": "This agent requires JSON commands",
                    "example": {
                        "command": "identify",
                        "audio_path": "/path/to/audio.wav"
                    }
                })
        
        except Exception as e:
            log.error(f"Error in RealSpeakerIdentificationAgent: {e}")
            return json.dumps({"error": str(e)})
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Real speaker identification using SpeechBrain embeddings"
    
    @property  
    def triggers(self) -> List[str]:
        """Words that suggest speaker identification is needed."""
        return ["identify", "speaker", "voice", "who is", "recognize"]