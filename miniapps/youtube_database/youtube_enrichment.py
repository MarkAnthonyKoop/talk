#!/usr/bin/env python3
"""
YouTube Data Enrichment System

Enriches the YouTube database with:
- Video transcripts (via youtube-transcript-api)
- Additional metadata (duration, views, likes)
- AI-generated summaries and insights
- Research annotations
"""

import sys
import sqlite3
import time
import json
import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent


class TranscriptFetcher:
    """Fetches video transcripts using various methods."""
    
    def __init__(self):
        self.youtube_transcript_available = self._check_youtube_transcript_api()
    
    def _check_youtube_transcript_api(self) -> bool:
        """Check if youtube-transcript-api is available."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            return True
        except ImportError:
            log.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
            return False
    
    def fetch_transcript(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch transcript for a video.
        
        Returns dict with:
        - transcript: Full text transcript
        - source: Source of transcript (youtube_api, generated, etc)
        - language: Language code
        - timestamps: List of timestamped segments
        """
        if not video_id or video_id == 'Unknown':
            return None
        
        # Try youtube-transcript-api first
        if self.youtube_transcript_available:
            result = self._fetch_via_youtube_api(video_id)
            if result:
                return result
        
        # Fallback methods could be added here
        # - Whisper API for audio transcription
        # - Web scraping
        # - Other APIs
        
        return None
    
    def _fetch_via_youtube_api(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Fetch transcript using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine segments into full text
            full_text = ' '.join([segment['text'] for segment in transcript_list])
            
            return {
                'transcript': full_text,
                'source': 'youtube_api',
                'language': 'en',
                'timestamps': transcript_list[:100]  # Keep first 100 segments for reference
            }
            
        except Exception as e:
            log.debug(f"Could not fetch transcript for {video_id}: {e}")
            return None


class YouTubeEnricher:
    """Enriches YouTube database with additional data."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.transcript_fetcher = TranscriptFetcher()
        self.ai_agent = None
        self.progress_queue = queue.Queue()
        
    def initialize_ai(self):
        """Initialize AI agent for summaries and insights."""
        try:
            self.ai_agent = Agent(
                roles=["You are an expert at analyzing video transcripts and generating insights."],
                overrides={"llm": {"provider": "anthropic"}}
            )
            return True
        except Exception as e:
            log.error(f"Failed to initialize AI agent: {e}")
            return False
    
    def enrich_database(self, 
                        fetch_transcripts: bool = True,
                        generate_summaries: bool = True,
                        limit: Optional[int] = None):
        """
        Enrich database with additional data.
        
        Args:
            fetch_transcripts: Whether to fetch video transcripts
            generate_summaries: Whether to generate AI summaries
            limit: Limit number of videos to process (for testing)
        """
        log.info("Starting database enrichment...")
        
        # Get videos to enrich
        videos = self._get_videos_to_enrich(limit)
        log.info(f"Found {len(videos)} videos to enrich")
        
        if not videos:
            log.info("No videos to enrich")
            return
        
        # Initialize AI if needed
        if generate_summaries:
            if not self.initialize_ai():
                log.warning("AI not available, skipping summaries")
                generate_summaries = False
        
        # Process videos
        enriched_count = 0
        transcript_count = 0
        summary_count = 0
        
        for i, (video_id, title, channel) in enumerate(videos):
            log.info(f"Processing {i+1}/{len(videos)}: {title[:50]}...")
            
            # Fetch transcript
            if fetch_transcripts:
                transcript_data = self.transcript_fetcher.fetch_transcript(video_id)
                if transcript_data:
                    self._save_transcript(video_id, transcript_data)
                    transcript_count += 1
                    
                    # Generate summary if we have transcript and AI
                    if generate_summaries and self.ai_agent:
                        summary = self._generate_summary(
                            title, 
                            channel, 
                            transcript_data['transcript']
                        )
                        if summary:
                            self._save_research_data(video_id, 'summary', summary)
                            summary_count += 1
            
            enriched_count += 1
            
            # Progress update
            if i % 10 == 0:
                log.info(f"  Progress: {enriched_count} enriched, {transcript_count} transcripts, {summary_count} summaries")
            
            # Rate limiting
            time.sleep(0.5)  # Be respectful to APIs
        
        log.info(f"Enrichment complete! Enriched {enriched_count} videos")
        log.info(f"  Transcripts fetched: {transcript_count}")
        log.info(f"  Summaries generated: {summary_count}")
    
    def _get_videos_to_enrich(self, limit: Optional[int] = None) -> List[tuple]:
        """Get videos that need enrichment."""
        cursor = self.conn.cursor()
        
        # Get videos without transcripts, prioritizing AI-relevant ones
        query = """
            SELECT video_id, title, channel
            FROM videos
            WHERE transcript IS NULL
            AND video_id != 'Unknown'
            AND title NOT LIKE '%#Shorts%'
            ORDER BY ai_relevance_score DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        return cursor.fetchall()
    
    def _save_transcript(self, video_id: str, transcript_data: Dict[str, Any]):
        """Save transcript to database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE videos 
            SET transcript = ?, 
                transcript_source = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
        """, (
            transcript_data['transcript'],
            transcript_data['source'],
            video_id
        ))
        
        # Save additional research data
        if transcript_data.get('timestamps'):
            self._save_research_data(
                video_id, 
                'transcript_timestamps',
                json.dumps(transcript_data['timestamps'])
            )
        
        self.conn.commit()
    
    def _generate_summary(self, title: str, channel: str, transcript: str) -> Optional[str]:
        """Generate AI summary of video."""
        if not self.ai_agent or not transcript:
            return None
        
        # Truncate transcript if too long
        max_length = 4000
        if len(transcript) > max_length:
            transcript = transcript[:max_length] + "..."
        
        prompt = f"""
        Analyze this YouTube video and provide a concise summary:
        
        Title: {title}
        Channel: {channel}
        
        Transcript excerpt:
        {transcript}
        
        Provide:
        1. A 2-3 sentence summary of the main topic
        2. Key technical concepts mentioned (if any)
        3. Relevance to AI/coding (rate 1-10)
        
        Format as JSON with keys: summary, concepts, ai_relevance_score
        """
        
        try:
            response = self.ai_agent.run(prompt)
            return response
        except Exception as e:
            log.error(f"Failed to generate summary: {e}")
            return None
    
    def _save_research_data(self, video_id: str, data_type: str, data_value: str):
        """Save research data to database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO research_data (video_id, data_type, data_value, source)
            VALUES (?, ?, ?, ?)
        """, (video_id, data_type, data_value, 'ai_enrichment'))
        
        self.conn.commit()
    
    def enrich_in_background(self, **kwargs):
        """Run enrichment in background thread."""
        thread = threading.Thread(
            target=self._background_worker,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
        
        # Monitor progress
        while thread.is_alive():
            while not self.progress_queue.empty():
                msg = self.progress_queue.get()
                print(f"► {msg}")
            time.sleep(1)
        
        # Get final messages
        while not self.progress_queue.empty():
            msg = self.progress_queue.get()
            print(f"► {msg}")
    
    def _background_worker(self, **kwargs):
        """Background worker for enrichment."""
        try:
            self.progress_queue.put("Starting enrichment...")
            self.enrich_database(**kwargs)
            self.progress_queue.put("Enrichment complete!")
        except Exception as e:
            self.progress_queue.put(f"Error: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class YouTubeMetadataFetcher:
    """Fetches additional metadata for videos (placeholder for YouTube API)."""
    
    @staticmethod
    def fetch_metadata(video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch video metadata.
        
        In production, would use YouTube Data API to get:
        - Duration
        - View count
        - Like count
        - Upload date
        - Description
        - Tags
        """
        # Placeholder - would use actual YouTube API
        return None


def main():
    """Run enrichment on existing database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich YouTube database")
    parser.add_argument(
        "--db",
        default="youtube_history.db",
        help="Database path"
    )
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="Fetch transcripts"
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="Generate AI summaries"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of videos to process"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background with progress updates"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run youtube_database_builder.py first to create the database")
        return 1
    
    print("YouTube Database Enrichment")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Fetch transcripts: {args.transcripts}")
    print(f"Generate summaries: {args.summaries}")
    if args.limit:
        print(f"Limit: {args.limit} videos")
    print()
    
    # Create enricher
    enricher = YouTubeEnricher(str(db_path))
    
    try:
        if args.background:
            enricher.enrich_in_background(
                fetch_transcripts=args.transcripts,
                generate_summaries=args.summaries,
                limit=args.limit
            )
        else:
            enricher.enrich_database(
                fetch_transcripts=args.transcripts,
                generate_summaries=args.summaries,
                limit=args.limit
            )
    finally:
        enricher.close()
    
    print("\n✓ Enrichment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())