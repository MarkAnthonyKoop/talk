#!/usr/bin/env python3
"""
YouTube Database Builder

Creates a comprehensive SQLite database from YouTube takeout data with:
- Complete watch history
- Video metadata
- Transcripts (fetched via YouTube API or scraped)
- Channel information
- Search history
- AI/coding categorization
- Research annotations
"""

import sys
import sqlite3
import json
import zipfile
import csv
import io
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'miniapps' / 'youtube_ai_analyzer'))

from youtube_ai_analyzer import AIContentCategorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


@dataclass
class VideoRecord:
    """Represents a video in the database."""
    video_id: str
    title: str
    channel: str
    channel_id: Optional[str]
    watch_time: Optional[datetime]
    url: str
    duration: Optional[int]  # in seconds
    description: Optional[str]
    transcript: Optional[str]
    categories: List[str]
    ai_relevance_score: float
    research_notes: Optional[str]
    tags: List[str]


class YouTubeDatabaseBuilder:
    """Builds and manages YouTube history database."""
    
    def __init__(self, takeout_path: str, db_path: str = "youtube_history.db"):
        self.takeout_path = Path(takeout_path)
        self.db_path = Path(db_path)
        self.conn = None
        self.categorizer = AIContentCategorizer()
        
    def build_database(self):
        """Main entry point to build the database."""
        log.info(f"Building YouTube database from {self.takeout_path}")
        
        # Create database
        self._create_database()
        
        # Extract and process takeout data
        takeout_data = self._extract_takeout_data()
        
        # Process watch history
        if takeout_data.get('watch_history'):
            self._process_watch_history(takeout_data['watch_history'])
        
        # Process search history
        if takeout_data.get('search_history'):
            self._process_search_history(takeout_data['search_history'])
        
        # Process subscriptions
        if takeout_data.get('subscriptions'):
            self._process_subscriptions(takeout_data['subscriptions'])
        
        # Process playlists
        if takeout_data.get('playlists'):
            self._process_playlists(takeout_data['playlists'])
        
        # Add categories and scores
        self._add_categorization()
        
        # Create indexes for performance
        self._create_indexes()
        
        # Generate statistics
        stats = self._generate_statistics()
        
        log.info("Database build complete!")
        return stats
    
    def _create_database(self):
        """Create database schema."""
        log.info(f"Creating database: {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Videos table - main table for watch history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                channel TEXT,
                channel_id TEXT,
                url TEXT,
                watch_time TIMESTAMP,
                duration INTEGER,
                description TEXT,
                transcript TEXT,
                transcript_source TEXT,
                ai_relevance_score REAL DEFAULT 0,
                research_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Watch history table - tracks multiple watches of same video
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                watch_time TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        """)
        
        # Categories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """)
        
        # Video categories junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_categories (
                video_id TEXT,
                category_id INTEGER,
                confidence REAL DEFAULT 1.0,
                PRIMARY KEY (video_id, category_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )
        """)
        
        # Channels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                channel_name TEXT NOT NULL,
                subscriber_count INTEGER,
                video_count INTEGER,
                ai_relevance_score REAL DEFAULT 0,
                is_subscribed BOOLEAN DEFAULT 0,
                last_watched TIMESTAMP,
                research_notes TEXT
            )
        """)
        
        # Search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                search_time TIMESTAMP,
                ai_related BOOLEAN DEFAULT 0
            )
        """)
        
        # Playlists table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                playlist_id TEXT PRIMARY KEY,
                playlist_name TEXT NOT NULL,
                video_count INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        # Playlist videos junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlist_videos (
                playlist_id TEXT,
                video_id TEXT,
                position INTEGER,
                added_at TIMESTAMP,
                PRIMARY KEY (playlist_id, video_id),
                FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        """)
        
        # Tags table for custom tagging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Video tags junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_tags (
                video_id TEXT,
                tag_id INTEGER,
                PRIMARY KEY (video_id, tag_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
        """)
        
        # Research data table for additional analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                data_type TEXT,
                data_value TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        """)
        
        self.conn.commit()
        
        # Insert categories
        self._insert_categories()
    
    def _insert_categories(self):
        """Insert AI/coding categories."""
        cursor = self.conn.cursor()
        
        for cat_name, cat_info in self.categorizer.CATEGORIES.items():
            cursor.execute("""
                INSERT OR IGNORE INTO categories (name, description)
                VALUES (?, ?)
            """, (cat_name, cat_info['description']))
        
        self.conn.commit()
    
    def _extract_takeout_data(self):
        """Extract all data from takeout zip."""
        log.info("Extracting takeout data...")
        data = {
            'watch_history': [],
            'search_history': [],
            'subscriptions': [],
            'playlists': {}
        }
        
        with zipfile.ZipFile(self.takeout_path, 'r') as zf:
            files = zf.namelist()
            
            # Extract watch history
            watch_file = next((f for f in files if 'watch-history.html' in f), None)
            if watch_file:
                log.info("Extracting watch history...")
                with zf.open(watch_file) as f:
                    content = f.read().decode('utf-8')
                    data['watch_history'] = self._parse_watch_history_html(content)
                    log.info(f"  Found {len(data['watch_history'])} videos")
            
            # Extract search history
            search_file = next((f for f in files if 'search-history.html' in f), None)
            if search_file:
                log.info("Extracting search history...")
                with zf.open(search_file) as f:
                    content = f.read().decode('utf-8')
                    data['search_history'] = self._parse_search_history_html(content)
                    log.info(f"  Found {len(data['search_history'])} searches")
            
            # Extract subscriptions
            subs_file = next((f for f in files if 'subscriptions.csv' in f), None)
            if subs_file:
                log.info("Extracting subscriptions...")
                with zf.open(subs_file) as f:
                    content = f.read().decode('utf-8')
                    reader = csv.DictReader(io.StringIO(content))
                    data['subscriptions'] = list(reader)
                    log.info(f"  Found {len(data['subscriptions'])} subscriptions")
            
            # Extract playlists
            playlist_files = [f for f in files if 'playlists/' in f and f.endswith('.csv')]
            if playlist_files:
                log.info(f"Extracting {len(playlist_files)} playlists...")
                for pf in playlist_files:
                    playlist_name = Path(pf).stem.replace('-videos', '')
                    with zf.open(pf) as f:
                        content = f.read().decode('utf-8')
                        reader = csv.DictReader(io.StringIO(content))
                        data['playlists'][playlist_name] = list(reader)
        
        return data
    
    def _parse_watch_history_html(self, html_content: str) -> List[Dict]:
        """Parse watch history HTML with BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        videos = []
        
        # Find all video entries
        for entry in soup.find_all('div', class_='mdl-grid'):
            try:
                # Extract video info
                links = entry.find_all('a')
                if not links:
                    continue
                
                # First link is usually the video
                video_link = links[0]
                title = video_link.text.strip()
                url = video_link.get('href', '')
                
                # Extract video ID from URL
                video_id = None
                if 'watch?v=' in url:
                    video_id = url.split('watch?v=')[1].split('&')[0]
                
                # Second link is usually the channel
                channel = "Unknown"
                channel_id = None
                if len(links) > 1:
                    channel_link = links[1]
                    channel = channel_link.text.strip()
                    channel_url = channel_link.get('href', '')
                    if '/channel/' in channel_url:
                        channel_id = channel_url.split('/channel/')[1].split('?')[0]
                
                # Extract timestamp
                timestamp_text = None
                br_tags = entry.find_all('br')
                if br_tags:
                    # Timestamp is usually after the first <br>
                    for br in br_tags:
                        next_text = br.next_sibling
                        if next_text and isinstance(next_text, str):
                            # Parse timestamp like "Aug 5, 2024, 11:58:32 PM PDT"
                            timestamp_text = next_text.strip()
                            break
                
                videos.append({
                    'video_id': video_id or hashlib.md5(title.encode()).hexdigest(),
                    'title': title,
                    'channel': channel,
                    'channel_id': channel_id,
                    'url': url,
                    'timestamp': timestamp_text
                })
                
            except Exception as e:
                log.warning(f"Error parsing video entry: {e}")
                continue
        
        return videos
    
    def _parse_search_history_html(self, html_content: str) -> List[Dict]:
        """Parse search history HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        searches = []
        
        for entry in soup.find_all('div', class_='mdl-grid'):
            try:
                search_link = entry.find('a')
                if search_link:
                    query = search_link.text.strip()
                    
                    # Extract timestamp
                    timestamp_text = None
                    br_tags = entry.find_all('br')
                    if br_tags:
                        for br in br_tags:
                            next_text = br.next_sibling
                            if next_text and isinstance(next_text, str):
                                timestamp_text = next_text.strip()
                                break
                    
                    searches.append({
                        'query': query,
                        'timestamp': timestamp_text
                    })
            except Exception as e:
                log.warning(f"Error parsing search entry: {e}")
                continue
        
        return searches
    
    def _process_watch_history(self, watch_history: List[Dict]):
        """Process and insert watch history into database."""
        log.info(f"Processing {len(watch_history)} watch history entries...")
        
        cursor = self.conn.cursor()
        channels_data = {}
        
        for video in watch_history:
            # Insert or update video
            cursor.execute("""
                INSERT OR IGNORE INTO videos (video_id, title, channel, channel_id, url, watch_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                video['video_id'],
                video['title'],
                video['channel'],
                video['channel_id'],
                video['url'],
                video['timestamp']
            ))
            
            # Add to watch history
            cursor.execute("""
                INSERT INTO watch_history (video_id, watch_time)
                VALUES (?, ?)
            """, (video['video_id'], video['timestamp']))
            
            # Track channel data
            if video['channel_id']:
                if video['channel_id'] not in channels_data:
                    channels_data[video['channel_id']] = {
                        'name': video['channel'],
                        'video_count': 0,
                        'last_watched': video['timestamp']
                    }
                channels_data[video['channel_id']]['video_count'] += 1
        
        # Insert channels
        for channel_id, channel_info in channels_data.items():
            cursor.execute("""
                INSERT OR REPLACE INTO channels (channel_id, channel_name, video_count, last_watched)
                VALUES (?, ?, ?, ?)
            """, (channel_id, channel_info['name'], channel_info['video_count'], channel_info['last_watched']))
        
        self.conn.commit()
        log.info(f"  Processed {len(channels_data)} unique channels")
    
    def _process_search_history(self, search_history: List[Dict]):
        """Process and insert search history."""
        log.info(f"Processing {len(search_history)} search entries...")
        
        cursor = self.conn.cursor()
        
        for search in search_history:
            # Check if AI-related
            categories = self.categorizer.categorize(search['query'])
            ai_related = 1 if categories and categories[0] != 'uncategorized' else 0
            
            cursor.execute("""
                INSERT INTO search_history (query, search_time, ai_related)
                VALUES (?, ?, ?)
            """, (search['query'], search['timestamp'], ai_related))
        
        self.conn.commit()
    
    def _process_subscriptions(self, subscriptions: List[Dict]):
        """Process and update subscription status."""
        log.info(f"Processing {len(subscriptions)} subscriptions...")
        
        cursor = self.conn.cursor()
        
        for sub in subscriptions:
            channel_id = sub.get('Channel Id', '')
            channel_name = sub.get('Channel Title', '')
            
            if channel_id:
                # Update existing channel or insert new
                cursor.execute("""
                    INSERT INTO channels (channel_id, channel_name, is_subscribed)
                    VALUES (?, ?, 1)
                    ON CONFLICT(channel_id) DO UPDATE SET
                        is_subscribed = 1,
                        channel_name = COALESCE(channel_name, excluded.channel_name)
                """, (channel_id, channel_name))
        
        self.conn.commit()
    
    def _process_playlists(self, playlists: Dict[str, List[Dict]]):
        """Process playlists."""
        log.info(f"Processing {len(playlists)} playlists...")
        
        cursor = self.conn.cursor()
        
        for playlist_name, videos in playlists.items():
            # Generate playlist ID
            playlist_id = hashlib.md5(playlist_name.encode()).hexdigest()
            
            # Insert playlist
            cursor.execute("""
                INSERT OR IGNORE INTO playlists (playlist_id, playlist_name, video_count)
                VALUES (?, ?, ?)
            """, (playlist_id, playlist_name, len(videos)))
            
            # Insert playlist videos
            for position, video in enumerate(videos):
                video_title = video.get('Video Title', '')
                video_id = video.get('Video Id', '')
                
                if not video_id:
                    # Generate ID from title
                    video_id = hashlib.md5(video_title.encode()).hexdigest()
                
                # Insert video if not exists
                cursor.execute("""
                    INSERT OR IGNORE INTO videos (video_id, title, url)
                    VALUES (?, ?, ?)
                """, (video_id, video_title, f"https://www.youtube.com/watch?v={video_id}"))
                
                # Add to playlist
                cursor.execute("""
                    INSERT OR IGNORE INTO playlist_videos (playlist_id, video_id, position)
                    VALUES (?, ?, ?)
                """, (playlist_id, video_id, position))
        
        self.conn.commit()
    
    def _add_categorization(self):
        """Add AI/coding categorization to videos."""
        log.info("Adding AI/coding categorization...")
        
        cursor = self.conn.cursor()
        
        # Get all videos
        cursor.execute("SELECT video_id, title, channel FROM videos")
        videos = cursor.fetchall()
        
        for video_id, title, channel in videos:
            # Categorize
            categories = self.categorizer.categorize(title, channel or '')
            score = self.categorizer.score_relevance(title, channel or '')
            
            # Update score
            cursor.execute("""
                UPDATE videos SET ai_relevance_score = ? WHERE video_id = ?
            """, (score, video_id))
            
            # Add categories
            for cat_name in categories:
                # Get category ID
                cursor.execute("SELECT id FROM categories WHERE name = ?", (cat_name,))
                result = cursor.fetchone()
                if result:
                    category_id = result[0]
                    cursor.execute("""
                        INSERT OR IGNORE INTO video_categories (video_id, category_id, confidence)
                        VALUES (?, ?, ?)
                    """, (video_id, category_id, score))
        
        # Update channel scores
        cursor.execute("""
            UPDATE channels 
            SET ai_relevance_score = (
                SELECT AVG(v.ai_relevance_score)
                FROM videos v
                WHERE v.channel_id = channels.channel_id
            )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        log.info("Creating indexes...")
        
        cursor = self.conn.cursor()
        
        # Video indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_watch_time ON videos(watch_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_ai_score ON videos(ai_relevance_score)")
        
        # Watch history index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_watch_history_time ON watch_history(watch_time)")
        
        # Search history index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_ai ON search_history(ai_related)")
        
        # Category indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_categories_video ON video_categories(video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_categories_category ON video_categories(category_id)")
        
        self.conn.commit()
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Video stats
        cursor.execute("SELECT COUNT(*) FROM videos")
        stats['total_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_relevance_score > 0.5")
        stats['ai_videos'] = cursor.fetchone()[0]
        
        # Channel stats
        cursor.execute("SELECT COUNT(*) FROM channels")
        stats['total_channels'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM channels WHERE is_subscribed = 1")
        stats['subscribed_channels'] = cursor.fetchone()[0]
        
        # Search stats
        cursor.execute("SELECT COUNT(*) FROM search_history")
        stats['total_searches'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM search_history WHERE ai_related = 1")
        stats['ai_searches'] = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute("""
            SELECT c.name, COUNT(vc.video_id) as count
            FROM categories c
            LEFT JOIN video_categories vc ON c.id = vc.category_id
            GROUP BY c.id
            ORDER BY count DESC
        """)
        stats['category_distribution'] = cursor.fetchall()
        
        # Top channels
        cursor.execute("""
            SELECT channel, COUNT(*) as watch_count, AVG(ai_relevance_score) as avg_score
            FROM videos
            WHERE channel IS NOT NULL
            GROUP BY channel
            ORDER BY watch_count DESC
            LIMIT 10
        """)
        stats['top_channels'] = cursor.fetchall()
        
        return stats
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class YouTubeTranscriptFetcher:
    """Fetches transcripts for videos (placeholder for actual implementation)."""
    
    @staticmethod
    def fetch_transcript(video_id: str) -> Optional[str]:
        """
        Fetch transcript for a video.
        
        In production, this would use:
        - youtube-transcript-api library
        - YouTube Data API
        - Web scraping as fallback
        """
        # Placeholder - in real implementation would fetch actual transcripts
        return None


def main():
    """Build the YouTube database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build YouTube history database")
    parser.add_argument(
        "--takeout",
        default="special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip",
        help="Path to YouTube takeout zip"
    )
    parser.add_argument(
        "--db",
        default="youtube_history.db",
        help="Output database path"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    takeout_path = Path(args.takeout)
    if not takeout_path.is_absolute():
        takeout_path = Path.cwd() / takeout_path
    
    if not takeout_path.exists():
        print(f"Error: Takeout file not found at {takeout_path}")
        return 1
    
    print("YouTube Database Builder")
    print("=" * 60)
    print(f"Takeout: {takeout_path.name}")
    print(f"Database: {args.db}")
    print()
    
    # Build database
    builder = YouTubeDatabaseBuilder(str(takeout_path), args.db)
    
    try:
        stats = builder.build_database()
        
        print("\n" + "=" * 60)
        print("Database Build Complete!")
        print("-" * 60)
        print(f"Total videos: {stats['total_videos']}")
        print(f"AI/coding videos: {stats['ai_videos']}")
        print(f"Total channels: {stats['total_channels']}")
        print(f"Subscribed channels: {stats['subscribed_channels']}")
        print(f"Total searches: {stats['total_searches']}")
        print(f"AI-related searches: {stats['ai_searches']}")
        
        print("\nCategory Distribution:")
        for category, count in stats['category_distribution'][:5]:
            print(f"  {category}: {count} videos")
        
        print("\nTop Channels:")
        for channel, count, score in stats['top_channels'][:5]:
            print(f"  {channel}: {count} videos (AI score: {score:.2f})")
        
        print(f"\nDatabase saved to: {args.db}")
        
    finally:
        builder.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())