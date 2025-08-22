#!/usr/bin/env python3
"""
SpeakerDatabase - Real SQLite database for speaker profiles.

This module provides persistent storage for speaker embeddings and profiles
without any mocking.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pickle

log = logging.getLogger(__name__)


class SpeakerDatabase:
    """
    Real SQLite database for speaker profiles.
    
    Stores:
    - Speaker profiles with names and metadata
    - Voice embeddings for each speaker
    - Enrollment history
    - Session information
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize speaker database.
        
        Args:
            db_path: Path to SQLite database file (can be string or Path)
        """
        self.conn = None  # Initialize conn attribute first
        
        # Handle both string and Path inputs
        if isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path or Path.home() / ".talk" / "speakers.db"
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        log.info(f"SpeakerDatabase initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Create speakers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                speaker_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_temporary BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Create embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                audio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                quality_score REAL DEFAULT 1.0,
                metadata TEXT,
                FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
            )
        """)
        
        # Create enrollment sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollment_sessions (
                session_id TEXT PRIMARY KEY,
                speaker_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'active',
                sample_count INTEGER DEFAULT 0,
                FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_speaker_id ON embeddings(speaker_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporary ON speakers(is_temporary)")
        
        self.conn.commit()
    
    def add_speaker(self, speaker_id: str, name: str, 
                   email: Optional[str] = None,
                   is_temporary: bool = False,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Add a new speaker to the database.
        
        Args:
            speaker_id: Unique speaker identifier
            name: Speaker's name
            email: Speaker's email (optional)
            is_temporary: Whether this is a temporary profile
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO speakers (speaker_id, name, email, is_temporary, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                speaker_id, 
                name, 
                email, 
                1 if is_temporary else 0,
                json.dumps(metadata) if metadata else None
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            log.warning(f"Speaker {speaker_id} already exists")
            return False
        except Exception as e:
            log.error(f"Error adding speaker: {e}")
            return False
    
    def add_embedding(self, speaker_id: str, embedding: np.ndarray,
                     audio_path: Optional[str] = None,
                     quality_score: float = 1.0,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Add an embedding for a speaker.
        
        Args:
            speaker_id: Speaker identifier
            embedding: Voice embedding as numpy array
            audio_path: Path to source audio file
            quality_score: Quality score of the embedding
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Serialize embedding
            embedding_bytes = pickle.dumps(embedding)
            
            cursor.execute("""
                INSERT INTO embeddings (speaker_id, embedding, audio_path, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                speaker_id,
                embedding_bytes,
                audio_path,
                quality_score,
                json.dumps(metadata) if metadata else None
            ))
            
            # Update speaker's updated_at timestamp
            cursor.execute("""
                UPDATE speakers SET updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ?
            """, (speaker_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            log.error(f"Error adding embedding: {e}")
            return False
    
    def get_speaker_embeddings(self, speaker_id: str) -> List[np.ndarray]:
        """
        Get all embeddings for a speaker.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            List of embedding arrays
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT embedding FROM embeddings
            WHERE speaker_id = ?
            ORDER BY created_at DESC
        """, (speaker_id,))
        
        embeddings = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[0])
            embeddings.append(embedding)
        
        return embeddings
    
    def get_all_speakers(self, include_temporary: bool = True) -> List[Dict[str, Any]]:
        """
        Get all speakers in the database.
        
        Args:
            include_temporary: Whether to include temporary profiles
            
        Returns:
            List of speaker dictionaries
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT s.speaker_id, s.name, s.email, s.created_at, s.is_temporary,
                   COUNT(e.embedding_id) as embedding_count
            FROM speakers s
            LEFT JOIN embeddings e ON s.speaker_id = e.speaker_id
        """
        
        if not include_temporary:
            query += " WHERE s.is_temporary = 0"
        
        query += " GROUP BY s.speaker_id ORDER BY s.created_at DESC"
        
        cursor.execute(query)
        
        speakers = []
        for row in cursor.fetchall():
            speakers.append({
                'speaker_id': row[0],
                'name': row[1],
                'email': row[2],
                'created_at': row[3],
                'is_temporary': bool(row[4]),
                'embedding_count': row[5]
            })
        
        return speakers
    
    def update_speaker_name(self, speaker_id: str, new_name: str) -> bool:
        """
        Update a speaker's name.
        
        Args:
            speaker_id: Speaker identifier
            new_name: New name
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE speakers 
                SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ?
            """, (new_name, speaker_id))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            log.error(f"Error updating speaker name: {e}")
            return False
    
    def convert_temporary_to_permanent(self, speaker_id: str) -> bool:
        """
        Convert a temporary profile to permanent.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE speakers 
                SET is_temporary = 0, updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ? AND is_temporary = 1
            """, (speaker_id,))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            log.error(f"Error converting profile: {e}")
            return False
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete a speaker and all their embeddings.
        
        Args:
            speaker_id: Speaker identifier
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Delete embeddings first (foreign key constraint)
            cursor.execute("DELETE FROM embeddings WHERE speaker_id = ?", (speaker_id,))
            
            # Delete enrollment sessions
            cursor.execute("DELETE FROM enrollment_sessions WHERE speaker_id = ?", (speaker_id,))
            
            # Delete speaker
            cursor.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            log.error(f"Error deleting speaker: {e}")
            return False
    
    def merge_speakers(self, keep_id: str, merge_id: str) -> bool:
        """
        Merge two speaker profiles.
        
        Args:
            keep_id: Speaker ID to keep
            merge_id: Speaker ID to merge and delete
            
        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            
            # Move embeddings from merge_id to keep_id
            cursor.execute("""
                UPDATE embeddings 
                SET speaker_id = ?
                WHERE speaker_id = ?
            """, (keep_id, merge_id))
            
            # Move enrollment sessions
            cursor.execute("""
                UPDATE enrollment_sessions
                SET speaker_id = ?
                WHERE speaker_id = ?
            """, (keep_id, merge_id))
            
            # Delete the merged speaker
            cursor.execute("DELETE FROM speakers WHERE speaker_id = ?", (merge_id,))
            
            # Update timestamp on kept speaker
            cursor.execute("""
                UPDATE speakers 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ?
            """, (keep_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            log.error(f"Error merging speakers: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()
        
        # Count speakers
        cursor.execute("SELECT COUNT(*) FROM speakers WHERE is_temporary = 0")
        permanent_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM speakers WHERE is_temporary = 1")
        temporary_count = cursor.fetchone()[0]
        
        # Count embeddings
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        
        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM enrollment_sessions")
        session_count = cursor.fetchone()[0]
        
        return {
            'permanent_speakers': permanent_count,
            'temporary_speakers': temporary_count,
            'total_speakers': permanent_count + temporary_count,
            'total_embeddings': embedding_count,
            'total_sessions': session_count,
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Ensure connection is closed on deletion."""
        self.close()