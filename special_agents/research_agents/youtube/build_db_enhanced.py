#!/usr/bin/env python3
"""
Enhanced YouTube Database Builder

Improvements:
- Extracts channel names properly
- Captures watch timestamps
- Fixes AI score calculation
- Adds video duration and completion data
"""

import re
import json
import sqlite3
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import html
from collections import defaultdict


class EnhancedYouTubeParser:
    """Enhanced parser for YouTube takeout data with better extraction."""
    
    def __init__(self):
        self.videos = []
        self.stats = defaultdict(int)
        
    def parse_html_file(self, html_path: Path) -> List[Dict]:
        """Parse HTML with enhanced extraction."""
        print(f"📄 Parsing {html_path} ({html_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Enhanced regex patterns
        video_pattern = re.compile(
            r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
            r'.*?href="([^"]+)"[^>]*>([^<]+)</a>'  # URL and title
            r'.*?<br>([^<]+?)<br>'  # Channel info
            r'.*?(\w+ \d+, \d+ at \d+:\d+:\d+ [AP]M \w+)?'  # Timestamp (optional)
            r'.*?</div>',
            re.DOTALL
        )
        
        # Alternative pattern for different format
        alt_pattern = re.compile(
            r'Watched.*?href="([^"]+)"[^>]*>([^<]+)</a>'  # URL and title
            r'.*?(?:<br>)?([^<\n]+)?(?:<br>)?'  # Channel (optional)
            r'.*?(\w+ \d+, \d+,? \d+:\d+:\d+ [AP]M)?',  # Timestamp (optional)
            re.DOTALL
        )
        
        matches = video_pattern.findall(content)
        if not matches:
            matches = alt_pattern.findall(content)
        
        print(f"Found {len(matches)} videos")
        
        videos = []
        for match in matches:
            url = match[0]
            title = html.unescape(match[1].strip())
            
            # Extract channel name more carefully
            channel = "Unknown"
            if len(match) > 2 and match[2]:
                channel_text = html.unescape(match[2].strip())
                # Clean up channel name
                channel_text = channel_text.replace('\n', ' ').strip()
                if channel_text and not channel_text.startswith('<'):
                    channel = channel_text
            
            # Extract timestamp
            watch_time = None
            if len(match) > 3 and match[3]:
                try:
                    # Parse timestamp like "Dec 25, 2023 at 3:45:00 PM EST"
                    time_str = match[3].strip()
                    # Remove timezone
                    time_str = re.sub(r' [A-Z]{3}$', '', time_str)
                    watch_time = datetime.strptime(time_str, "%b %d, %Y at %I:%M:%S %p")
                except:
                    try:
                        # Try alternate format
                        watch_time = datetime.strptime(time_str, "%b %d, %Y, %I:%M:%S %p")
                    except:
                        pass
            
            # Extract video ID
            video_id = self._extract_video_id(url)
            
            # Calculate proper AI score
            ai_score = self._calculate_ai_score(title, channel)
            
            # Detect categories
            categories = self._detect_categories(title, channel)
            
            video = {
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'url': url,
                'watch_time': watch_time.isoformat() if watch_time else None,
                'ai_score': ai_score,
                'categories': json.dumps(categories)
            }
            
            videos.append(video)
            
            # Update stats
            self.stats['total'] += 1
            if channel != "Unknown":
                self.stats['with_channel'] += 1
            if watch_time:
                self.stats['with_timestamp'] += 1
            if ai_score > 0.5:
                self.stats['ai_related'] += 1
        
        return videos
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from URL."""
        patterns = [
            r'watch\?v=([a-zA-Z0-9_-]{11})',
            r'youtu\.be/([a-zA-Z0-9_-]{11})',
            r'&v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return url.split('/')[-1][:11]
    
    def _calculate_ai_score(self, title: str, channel: str) -> float:
        """Calculate accurate AI relevance score."""
        title_lower = title.lower()
        channel_lower = channel.lower()
        
        # Strong AI indicators
        strong_indicators = [
            'claude', 'anthropic', 'gpt', 'chatgpt', 'openai',
            'langchain', 'llm', 'large language model',
            'machine learning', 'deep learning', 'neural network',
            'artificial intelligence', 'ai agent', 'generative ai',
            'prompt engineering', 'transformer', 'bert', 'gemini',
            'copilot', 'codex', 'stable diffusion', 'midjourney'
        ]
        
        # Medium indicators
        medium_indicators = [
            'ai', 'ml', 'agent', 'automation', 'bot',
            'algorithm', 'model', 'training', 'dataset'
        ]
        
        # Weak indicators  
        weak_indicators = [
            'code', 'programming', 'python', 'javascript',
            'data', 'analysis', 'science'
        ]
        
        # Negative indicators (likely not AI)
        negative_indicators = [
            'music', 'song', 'album', 'live', 'concert',
            'game', 'gaming', 'minecraft', 'fortnite',
            'vlog', 'daily', 'routine', 'makeup', 'cooking'
        ]
        
        score = 0.0
        
        # Check negative indicators first
        for indicator in negative_indicators:
            if indicator in title_lower:
                return 0.0
        
        # Check strong indicators
        for indicator in strong_indicators:
            if indicator in title_lower or indicator in channel_lower:
                score = max(score, 0.9)
        
        # Check medium indicators
        if score < 0.9:
            for indicator in medium_indicators:
                # Ensure "ai" is a separate word, not part of another word
                if indicator == 'ai':
                    if re.search(r'\bai\b', title_lower) or re.search(r'\bai\b', channel_lower):
                        score = max(score, 0.6)
                elif indicator in title_lower or indicator in channel_lower:
                    score = max(score, 0.5)
        
        # Check weak indicators
        if score < 0.5:
            for indicator in weak_indicators:
                if indicator in title_lower or indicator in channel_lower:
                    score = max(score, 0.3)
        
        return score
    
    def _detect_categories(self, title: str, channel: str) -> List[str]:
        """Detect video categories."""
        categories = []
        title_lower = title.lower()
        channel_lower = channel.lower()
        
        # Category mappings
        category_rules = {
            'AI/ML': ['ai', 'machine learning', 'deep learning', 'neural', 'llm', 'gpt', 'claude'],
            'Programming': ['python', 'javascript', 'code', 'programming', 'developer', 'software'],
            'Music': ['music', 'song', 'album', 'concert', 'live', 'official video'],
            'Education': ['tutorial', 'course', 'learn', 'lecture', 'class', 'lesson'],
            'Gaming': ['game', 'gaming', 'gameplay', 'walkthrough', 'minecraft', 'fortnite'],
            'Tech': ['tech', 'technology', 'gadget', 'review', 'unboxing'],
            'Science': ['science', 'physics', 'chemistry', 'biology', 'space', 'nasa'],
            'News': ['news', 'breaking', 'update', 'report', 'today', 'latest']
        }
        
        for category, keywords in category_rules.items():
            for keyword in keywords:
                if keyword in title_lower or keyword in channel_lower:
                    categories.append(category)
                    break
        
        return list(set(categories))[:3]  # Max 3 categories


def build_enhanced_database(takeout_path: Path, output_path: Path = Path("youtube_enhanced.db")):
    """Build enhanced database from takeout data."""
    print("🚀 Building enhanced YouTube database...")
    
    # Create parser
    parser = EnhancedYouTubeParser()
    
    # Find and parse HTML file
    if takeout_path.suffix == '.zip':
        print("📦 Extracting from zip file...")
        with zipfile.ZipFile(takeout_path, 'r') as z:
            for file in z.namelist():
                if 'watch-history.html' in file.lower():
                    print(f"Found: {file}")
                    z.extract(file, '/tmp/youtube_extract')
                    html_path = Path('/tmp/youtube_extract') / file
                    videos = parser.parse_html_file(html_path)
                    break
    else:
        # Assume it's a directory
        html_files = list(takeout_path.glob("**/watch-history.html"))
        if html_files:
            videos = parser.parse_html_file(html_files[0])
        else:
            print("❌ No watch-history.html found")
            return
    
    # Create enhanced database
    print(f"\n💾 Creating database with {len(videos)} videos...")
    
    conn = sqlite3.connect(str(output_path))
    cursor = conn.cursor()
    
    # Create enhanced schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            channel TEXT,
            url TEXT,
            watch_time TEXT,
            ai_score REAL,
            categories TEXT,
            duration INTEGER,
            watch_count INTEGER DEFAULT 1,
            last_watched TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_score ON videos(ai_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel ON videos(channel)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_watch_time ON videos(watch_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON videos(title)")
    
    # Insert videos (handle duplicates by updating watch count)
    for video in videos:
        cursor.execute("""
            INSERT INTO videos (video_id, title, channel, url, watch_time, ai_score, categories, last_watched)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                watch_count = watch_count + 1,
                last_watched = excluded.watch_time
        """, (
            video['video_id'],
            video['title'],
            video['channel'],
            video['url'],
            video['watch_time'],
            video['ai_score'],
            video['categories'],
            video['watch_time']
        ))
    
    # Create statistics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    # Store statistics
    stats = [
        ('total_videos', str(parser.stats['total'])),
        ('videos_with_channel', str(parser.stats['with_channel'])),
        ('videos_with_timestamp', str(parser.stats['with_timestamp'])),
        ('ai_related_videos', str(parser.stats['ai_related'])),
        ('last_updated', datetime.now().isoformat())
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO statistics (key, value) VALUES (?, ?)", stats)
    
    conn.commit()
    
    # Print summary
    print("\n📊 Database Summary:")
    print(f"  Total videos: {parser.stats['total']}")
    print(f"  With channel names: {parser.stats['with_channel']} ({parser.stats['with_channel']/parser.stats['total']*100:.1f}%)")
    print(f"  With timestamps: {parser.stats['with_timestamp']} ({parser.stats['with_timestamp']/parser.stats['total']*100:.1f}%)")
    print(f"  AI-related: {parser.stats['ai_related']} ({parser.stats['ai_related']/parser.stats['total']*100:.1f}%)")
    
    # Show sample videos with channels
    cursor.execute("""
        SELECT title, channel, ai_score
        FROM videos
        WHERE channel != 'Unknown'
        ORDER BY ai_score DESC
        LIMIT 5
    """)
    
    print("\n🎯 Sample videos with channel names:")
    for title, channel, score in cursor.fetchall():
        print(f"  [{score:.2f}] {title[:50]} - {channel}")
    
    conn.close()
    print(f"\n✅ Enhanced database saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build enhanced YouTube database from takeout")
    parser.add_argument("takeout", help="Path to takeout zip or directory")
    parser.add_argument("--output", default="youtube_enhanced.db", help="Output database path")
    
    args = parser.parse_args()
    
    takeout_path = Path(args.takeout)
    output_path = Path(args.output)
    
    if not takeout_path.exists():
        print(f"❌ Takeout path not found: {takeout_path}")
        exit(1)
    
    build_enhanced_database(takeout_path, output_path)