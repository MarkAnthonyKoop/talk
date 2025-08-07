#!/usr/bin/env python3
"""
Fast YouTube database builder - optimized for large takeout files.
Uses regex instead of BeautifulSoup for faster parsing.
"""

import sys
import sqlite3
import json
import zipfile
import csv
import io
import re
import hashlib
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add path for categorizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'miniapps' / 'youtube_ai_analyzer'))
from youtube_ai_analyzer import AIContentCategorizer


def fast_parse_watch_history(html_content: str):
    """Fast parsing using regex instead of BeautifulSoup."""
    videos = []
    
    # Split by video entries - look for watch links
    video_pattern = re.compile(
        r'<a href="(https://www\.youtube\.com/watch\?v=([^"]+))">([^<]+)</a>'
    )
    
    # Find all videos
    matches = video_pattern.findall(html_content)
    
    for url, video_id, title in matches:
        # Clean video_id (remove extra params)
        video_id = video_id.split('&')[0]
        
        videos.append({
            'video_id': video_id,
            'title': title.strip(),
            'url': url,
            'channel': 'Unknown',  # Will be updated if found
            'timestamp': None
        })
    
    log.info(f"  Fast parsed {len(videos)} videos")
    
    # Try to extract channels (best effort)
    channel_pattern = re.compile(
        r'<a href="https://www\.youtube\.com/channel/([^"]+)">([^<]+)</a>'
    )
    
    channel_matches = channel_pattern.findall(html_content)
    log.info(f"  Found {len(channel_matches)} channel references")
    
    return videos


def build_database_fast(takeout_path: str, db_path: str):
    """Build database quickly."""
    log.info(f"Fast database builder starting...")
    log.info(f"Takeout: {takeout_path}")
    log.info(f"Database: {db_path}")
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create minimal schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            url TEXT,
            ai_score REAL DEFAULT 0,
            categories TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_name TEXT PRIMARY KEY,
            video_count INTEGER,
            ai_score REAL DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            channel_name TEXT PRIMARY KEY,
            channel_id TEXT
        )
    """)
    
    categorizer = AIContentCategorizer()
    
    # Process takeout
    with zipfile.ZipFile(takeout_path, 'r') as zf:
        files = zf.namelist()
        
        # Process watch history
        watch_file = next((f for f in files if 'watch-history.html' in f), None)
        if watch_file:
            log.info("Processing watch history...")
            with zf.open(watch_file) as f:
                content = f.read().decode('utf-8')
                
                # Use fast parsing
                videos = fast_parse_watch_history(content)
                
                # Insert videos
                channel_counts = {}
                for video in videos:
                    # Categorize
                    categories = categorizer.categorize(video['title'], video['channel'])
                    score = categorizer.score_relevance(video['title'], video['channel'])
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO videos (video_id, title, channel, url, ai_score, categories)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        video['video_id'],
                        video['title'][:500],  # Limit title length
                        video['channel'],
                        video['url'],
                        score,
                        json.dumps(categories)
                    ))
                    
                    # Track channels
                    if video['channel'] != 'Unknown':
                        channel_counts[video['channel']] = channel_counts.get(video['channel'], 0) + 1
                
                # Insert channels
                for channel, count in channel_counts.items():
                    score = categorizer.score_relevance("", channel)
                    cursor.execute("""
                        INSERT OR REPLACE INTO channels (channel_name, video_count, ai_score)
                        VALUES (?, ?, ?)
                    """, (channel, count, score))
                
                conn.commit()
                log.info(f"  Inserted {len(videos)} videos")
                log.info(f"  Tracked {len(channel_counts)} channels")
        
        # Process subscriptions
        subs_file = next((f for f in files if 'subscriptions.csv' in f), None)
        if subs_file:
            log.info("Processing subscriptions...")
            with zf.open(subs_file) as f:
                content = f.read().decode('utf-8')
                reader = csv.DictReader(io.StringIO(content))
                
                count = 0
                for row in reader:
                    cursor.execute("""
                        INSERT OR IGNORE INTO subscriptions (channel_name, channel_id)
                        VALUES (?, ?)
                    """, (row.get('Channel Title', ''), row.get('Channel Id', '')))
                    count += 1
                
                conn.commit()
                log.info(f"  Inserted {count} subscriptions")
    
    # Create indexes
    log.info("Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_score ON videos(ai_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories ON videos(categories)")
    conn.commit()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
    ai_videos = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM subscriptions")
    total_subs = cursor.fetchone()[0]
    
    conn.close()
    
    log.info("=" * 60)
    log.info("Database created successfully!")
    log.info(f"Total videos: {total_videos}")
    log.info(f"AI/coding videos: {ai_videos}")
    log.info(f"Subscriptions: {total_subs}")
    log.info(f"Database saved to: {db_path}")
    
    return {
        'total_videos': total_videos,
        'ai_videos': ai_videos,
        'subscriptions': total_subs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast YouTube database builder")
    parser.add_argument("--takeout", required=True, help="Path to takeout zip")
    parser.add_argument("--db", default="youtube_fast.db", help="Output database")
    
    args = parser.parse_args()
    
    takeout_path = Path(args.takeout)
    if not takeout_path.exists():
        print(f"Error: {takeout_path} not found")
        sys.exit(1)
    
    build_database_fast(str(takeout_path), args.db)