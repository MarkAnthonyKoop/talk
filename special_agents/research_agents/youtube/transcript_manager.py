#!/usr/bin/env python3
"""
Transcript Manager - Caching and parallel fetching for YouTube transcripts

Features:
- Local caching to avoid re-fetching
- Parallel fetching for multiple videos
- Automatic retry with fallback languages
- Compression for storage efficiency
"""

import json
import sqlite3
import hashlib
import gzip
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
except ImportError:
    print("Please install: pip install youtube-transcript-api")
    YouTubeTranscriptApi = None
    NoTranscriptFound = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptCache:
    """SQLite-based transcript cache with compression."""
    
    def __init__(self, cache_path: Path = None):
        """Initialize cache database."""
        if cache_path is None:
            cache_path = Path.home() / ".cache" / "youtube_transcripts" / "cache.db"
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.conn = sqlite3.connect(str(cache_path))
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                video_id TEXT PRIMARY KEY,
                transcript_compressed BLOB,
                language TEXT,
                fetched_at TIMESTAMP,
                error TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fetched_at 
            ON transcripts(fetched_at)
        """)
        
        self.conn.commit()
    
    def get(self, video_id: str, max_age_days: int = 30) -> Optional[str]:
        """Get cached transcript if fresh enough."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT transcript_compressed, fetched_at, error
            FROM transcripts
            WHERE video_id = ?
        """, (video_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        compressed_transcript, fetched_at, error = row
        
        # Check if error was cached
        if error:
            logger.info(f"Cached error for {video_id}: {error}")
            return None
        
        # Check age
        fetched_time = datetime.fromisoformat(fetched_at)
        if datetime.now() - fetched_time > timedelta(days=max_age_days):
            logger.info(f"Cache expired for {video_id}")
            return None
        
        # Decompress and return
        try:
            transcript = gzip.decompress(compressed_transcript).decode('utf-8')
            logger.info(f"Cache hit for {video_id}")
            return transcript
        except Exception as e:
            logger.error(f"Failed to decompress transcript for {video_id}: {e}")
            return None
    
    def set(self, video_id: str, transcript: str = None, error: str = None, 
            language: str = None, metadata: Dict = None):
        """Cache transcript or error."""
        cursor = self.conn.cursor()
        
        compressed = None
        if transcript:
            compressed = gzip.compress(transcript.encode('utf-8'))
        
        cursor.execute("""
            INSERT OR REPLACE INTO transcripts 
            (video_id, transcript_compressed, language, fetched_at, error, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            compressed,
            language,
            datetime.now().isoformat(),
            error,
            json.dumps(metadata) if metadata else None
        ))
        
        self.conn.commit()
        logger.info(f"Cached {'transcript' if transcript else 'error'} for {video_id}")
    
    def clear_old(self, days: int = 90):
        """Clear entries older than specified days."""
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            DELETE FROM transcripts
            WHERE fetched_at < ?
        """, (cutoff.isoformat(),))
        
        deleted = cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleared {deleted} old cache entries")
        
        return deleted
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transcripts WHERE error IS NOT NULL")
        errors = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT SUM(LENGTH(transcript_compressed)) 
            FROM transcripts 
            WHERE transcript_compressed IS NOT NULL
        """)
        size_bytes = cursor.fetchone()[0] or 0
        
        return {
            'total_entries': total,
            'successful': total - errors,
            'errors': errors,
            'cache_size_mb': size_bytes / 1024 / 1024
        }


class TranscriptManager:
    """Manages transcript fetching with caching and parallel processing."""
    
    def __init__(self, cache_path: Path = None, max_workers: int = 4):
        """Initialize transcript manager."""
        self.cache = TranscriptCache(cache_path)
        self.max_workers = max_workers
        self.api = YouTubeTranscriptApi() if YouTubeTranscriptApi else None
    
    def fetch_single(self, video_id: str, use_cache: bool = True, 
                    languages: List[str] = None) -> Optional[str]:
        """Fetch a single transcript with caching."""
        if not self.api:
            logger.error("youtube-transcript-api not installed")
            return None
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(video_id)
            if cached:
                return cached
        
        # Try to fetch
        try:
            if languages:
                transcript_list = self.api.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(languages).fetch()
            else:
                transcript = self.api.get_transcript(video_id)
            
            # Combine text
            full_text = ' '.join([entry['text'] for entry in transcript])
            
            # Cache it
            self.cache.set(
                video_id, 
                full_text,
                language=languages[0] if languages else 'auto',
                metadata={'segments': len(transcript)}
            )
            
            return full_text
            
        except NoTranscriptFound:
            error = "No transcript available"
            logger.warning(f"{error} for {video_id}")
            self.cache.set(video_id, error=error)
            return None
            
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to fetch transcript for {video_id}: {error}")
            self.cache.set(video_id, error=error)
            return None
    
    def fetch_multiple(self, video_ids: List[str], use_cache: bool = True,
                      progress_callback=None) -> Dict[str, str]:
        """Fetch multiple transcripts in parallel."""
        results = {}
        to_fetch = []
        
        # Check cache first
        if use_cache:
            for video_id in video_ids:
                cached = self.cache.get(video_id)
                if cached:
                    results[video_id] = cached
                else:
                    to_fetch.append(video_id)
        else:
            to_fetch = video_ids
        
        if not to_fetch:
            return results
        
        # Fetch in parallel
        logger.info(f"Fetching {len(to_fetch)} transcripts in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_single, vid, use_cache=False): vid
                for vid in to_fetch
            }
            
            completed = 0
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    transcript = future.result(timeout=30)
                    if transcript:
                        results[video_id] = transcript
                except Exception as e:
                    logger.error(f"Failed to fetch {video_id}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(to_fetch))
        
        return results
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """Analyze transcript for key metrics."""
        words = transcript.split()
        sentences = transcript.split('. ')
        
        # Extract potential topics (simple keyword extraction)
        from collections import Counter
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 
                     'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'this', 'that', 'these', 'those',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                     'which', 'who', 'when', 'where', 'why', 'how', 'all',
                     'each', 'every', 'both', 'few', 'more', 'most', 'other',
                     'some', 'such', 'only', 'own', 'same', 'so', 'than',
                     'too', 'very', 'just', 'about', 'into', 'through'}
        
        # Extract meaningful words
        meaningful_words = [
            w.lower() for w in words 
            if len(w) > 3 and w.lower() not in stop_words and w.isalpha()
        ]
        
        word_freq = Counter(meaningful_words)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'unique_words': len(set(meaningful_words)),
            'top_keywords': word_freq.most_common(10),
            'estimated_duration_minutes': len(words) / 150  # Assuming 150 wpm speaking rate
        }
    
    def batch_analyze(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Fetch and analyze multiple transcripts."""
        transcripts = self.fetch_multiple(video_ids)
        
        analyses = {}
        for video_id, transcript in transcripts.items():
            analyses[video_id] = {
                'has_transcript': True,
                'length': len(transcript),
                'analysis': self.analyze_transcript(transcript)
            }
        
        # Mark videos without transcripts
        for video_id in video_ids:
            if video_id not in analyses:
                analyses[video_id] = {
                    'has_transcript': False,
                    'length': 0,
                    'analysis': None
                }
        
        return analyses


def main():
    """CLI for transcript management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Transcript Manager")
    parser.add_argument("command", choices=['fetch', 'analyze', 'stats', 'clear'])
    parser.add_argument("--video-ids", nargs='+', help="Video IDs to process")
    parser.add_argument("--cache-dir", help="Cache directory path")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache")
    
    args = parser.parse_args()
    
    manager = TranscriptManager(
        cache_path=Path(args.cache_dir) if args.cache_dir else None
    )
    
    if args.command == 'fetch':
        if not args.video_ids:
            print("Please provide --video-ids")
            return
        
        def progress(done, total):
            print(f"Progress: {done}/{total} ({done/total*100:.1f}%)", end='\r')
        
        results = manager.fetch_multiple(
            args.video_ids, 
            use_cache=not args.no_cache,
            progress_callback=progress
        )
        
        print(f"\nFetched {len(results)} transcripts")
        for vid, transcript in results.items():
            print(f"  {vid}: {len(transcript)} chars")
    
    elif args.command == 'analyze':
        if not args.video_ids:
            print("Please provide --video-ids")
            return
        
        analyses = manager.batch_analyze(args.video_ids)
        
        for vid, data in analyses.items():
            print(f"\n{vid}:")
            if data['has_transcript']:
                analysis = data['analysis']
                print(f"  Words: {analysis['word_count']}")
                print(f"  Est. duration: {analysis['estimated_duration_minutes']:.1f} min")
                print(f"  Top keywords: {', '.join([w for w, _ in analysis['top_keywords'][:5]])}")
            else:
                print("  No transcript available")
    
    elif args.command == 'stats':
        stats = manager.cache.stats()
        print("Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    
    elif args.command == 'clear':
        deleted = manager.cache.clear_old(days=30)
        print(f"Cleared {deleted} old entries")


if __name__ == "__main__":
    main()