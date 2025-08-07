#!/usr/bin/env python3
"""
YouTube CLI - Complete command-line interface for YouTube history analysis

Features:
- Query your viewing history with natural language
- Fetch and save transcripts for videos
- AI-powered analysis and insights
- Export data in various formats
- Recent videos analysis
"""

import sys
import sqlite3
import json
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent


class YouTubeCLI:
    """Complete CLI for YouTube history analysis."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the CLI."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"❌ Database not found: {db_path}")
            print("Run: python3 build_db_fast.py --takeout <path_to_takeout.zip>")
            sys.exit(1)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.agent = None
        self.transcript_cache = {}
    
    def init_agent(self):
        """Initialize AI agent on demand."""
        if not self.agent:
            print("🤖 Initializing AI agent...")
            self.agent = Agent(
                roles=[
                    "You are an expert YouTube viewing history analyst.",
                    "You can identify patterns, extract insights, and provide recommendations.",
                    "You understand technical content, especially AI, machine learning, and programming.",
                    "Be concise but comprehensive in your analysis."
                ],
                overrides={"llm": {"provider": "anthropic"}}
            )
    
    def query(self, prompt: str, limit: int = 20) -> None:
        """Query the database with natural language."""
        self.init_agent()
        
        # First, extract what to search for
        cursor = self.conn.cursor()
        data = {}
        
        # Extract keywords
        keywords = self._extract_keywords(prompt)
        
        # Search for matching videos
        if keywords:
            all_matches = []
            for keyword in keywords:
                cursor.execute("""
                    SELECT title, channel, url, ai_score, categories
                    FROM videos
                    WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
                    ORDER BY ai_score DESC
                    LIMIT ?
                """, (f'%{keyword}%', f'%{keyword}%', limit))
                
                matches = [dict(row) for row in cursor.fetchall()]
                all_matches.extend(matches)
            
            # Deduplicate
            seen = set()
            unique_matches = []
            for match in all_matches:
                if match['url'] not in seen:
                    seen.add(match['url'])
                    unique_matches.append(match)
            
            data['matches'] = unique_matches[:limit]
        
        # Get AI analysis
        analysis_prompt = f"""
        Analyze this YouTube viewing history query: "{prompt}"
        
        Found videos:
        {json.dumps(data.get('matches', []), indent=2)}
        
        Provide a clear, insightful response that:
        1. Directly answers the question
        2. Lists specific videos with titles
        3. Identifies patterns if relevant
        4. Offers insights or recommendations
        
        Format with clear sections and bullet points.
        """
        
        response = self.agent.run(analysis_prompt)
        print("\n" + response)
        
        # Also show raw matches
        if data.get('matches'):
            print("\n📺 Specific Videos Found:")
            print("-" * 60)
            for i, video in enumerate(data['matches'][:10], 1):
                print(f"\n{i}. {video['title']}")
                if video['channel'] != 'Unknown':
                    print(f"   Channel: {video['channel']}")
                print(f"   URL: {video['url']}")
                if video.get('categories') and video['categories'] != '[]':
                    cats = json.loads(video['categories'])
                    if cats:
                        print(f"   Categories: {', '.join(cats)}")
                print(f"   AI Score: {video['ai_score']:.2f}")
    
    def recent(self, days: int = 30, ai_only: bool = False) -> None:
        """Show recent videos from the last N days."""
        print(f"\n📅 Recent Videos (last {days} days)")
        print("=" * 60)
        
        # Note: Our simple database doesn't have proper timestamps
        # This would need watch_time parsing in a real implementation
        cursor = self.conn.cursor()
        
        if ai_only:
            query = """
                SELECT title, channel, url, ai_score, categories
                FROM videos
                WHERE ai_score > 0.5
                ORDER BY RANDOM()
                LIMIT 20
            """
            print("(Showing random AI/coding videos as timestamps not available)")
        else:
            query = """
                SELECT title, channel, url, ai_score, categories
                FROM videos
                ORDER BY RANDOM()
                LIMIT 20
            """
            print("(Showing random videos as timestamps not available)")
        
        cursor.execute(query)
        videos = [dict(row) for row in cursor.fetchall()]
        
        for i, video in enumerate(videos, 1):
            print(f"\n{i}. {video['title'][:80]}")
            if video['channel'] != 'Unknown':
                print(f"   Channel: {video['channel']}")
            if video['ai_score'] > 0.5:
                print(f"   🤖 AI/Coding Content (score: {video['ai_score']:.2f})")
    
    def transcript(self, video_url: str) -> None:
        """Fetch and display transcript for a video."""
        # Extract video ID
        video_id = None
        if 'watch?v=' in video_url:
            video_id = video_url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
        else:
            video_id = video_url  # Assume it's already an ID
        
        print(f"\n📝 Fetching transcript for video ID: {video_id}")
        
        # Check cache first
        if video_id in self.transcript_cache:
            print("(From cache)")
            print(self.transcript_cache[video_id])
            return
        
        # Try to fetch with youtube-transcript-api
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Format transcript
            full_transcript = []
            for segment in transcript_list:
                time = segment['start']
                text = segment['text']
                minutes = int(time // 60)
                seconds = int(time % 60)
                full_transcript.append(f"[{minutes:02d}:{seconds:02d}] {text}")
            
            transcript_text = "\n".join(full_transcript)
            
            # Cache it
            self.transcript_cache[video_id] = transcript_text
            
            # Save to file
            output_file = f"transcript_{video_id}.txt"
            with open(output_file, 'w') as f:
                f.write(transcript_text)
            
            print(f"✅ Transcript saved to: {output_file}")
            
            # Show preview
            lines = transcript_text.split('\n')
            print("\n--- Transcript Preview (first 20 lines) ---")
            for line in lines[:20]:
                print(line)
            if len(lines) > 20:
                print(f"\n... and {len(lines) - 20} more lines")
            
        except ImportError:
            print("❌ youtube-transcript-api not installed")
            print("Install with: pip install youtube-transcript-api")
        except Exception as e:
            print(f"❌ Could not fetch transcript: {e}")
    
    def ai_videos(self, months: int = 3) -> None:
        """Show AI-related videos from recent months."""
        print(f"\n🤖 AI/Coding Videos (Recent {months} months simulation)")
        print("=" * 60)
        
        cursor = self.conn.cursor()
        
        # Get high-scoring AI videos
        cursor.execute("""
            SELECT title, channel, url, ai_score, categories
            FROM videos
            WHERE ai_score > 0.7
            ORDER BY ai_score DESC
            LIMIT 50
        """)
        
        videos = [dict(row) for row in cursor.fetchall()]
        
        # Categorize by type
        categories_map = {
            'llm_tutorials': [],
            'ai_coding': [],
            'ai_agents': [],
            'machine_learning': [],
            'codebase_analysis': []
        }
        
        for video in videos:
            if video.get('categories'):
                try:
                    cats = json.loads(video['categories'])
                    for cat in cats:
                        if cat in categories_map:
                            categories_map[cat].append(video)
                except:
                    pass
        
        # Show by category
        for category, vids in categories_map.items():
            if vids:
                print(f"\n📂 {category.replace('_', ' ').title()}")
                print("-" * 40)
                for video in vids[:5]:
                    print(f"• {video['title'][:70]}")
                    if video['channel'] != 'Unknown':
                        print(f"  Channel: {video['channel']}")
                    print(f"  Score: {video['ai_score']:.2f}")
        
        # AI analysis of trends
        if self.agent is None:
            self.init_agent()
        
        analysis_prompt = f"""
        Based on these AI/coding videos from the viewing history:
        - Total AI videos: {len(videos)}
        - LLM tutorials: {len(categories_map.get('llm_tutorials', []))}
        - AI coding tools: {len(categories_map.get('ai_coding', []))}
        - AI agents: {len(categories_map.get('ai_agents', []))}
        - Machine learning: {len(categories_map.get('machine_learning', []))}
        
        Provide 3-5 interesting insights about the AI learning journey and trends.
        What topics are well-covered? What's missing? What should be explored next?
        """
        
        print("\n💡 AI Learning Insights:")
        print("-" * 40)
        insights = self.agent.run(analysis_prompt)
        print(insights)
    
    def stats(self) -> None:
        """Show database statistics."""
        cursor = self.conn.cursor()
        
        print("\n📊 YouTube History Statistics")
        print("=" * 60)
        
        # Total videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        total = cursor.fetchone()[0]
        print(f"Total videos: {total:,}")
        
        # AI videos
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
        ai_count = cursor.fetchone()[0]
        print(f"AI/Coding videos: {ai_count:,} ({ai_count/total*100:.1f}%)")
        
        # Top channels
        cursor.execute("""
            SELECT channel, COUNT(*) as count
            FROM videos
            WHERE channel != 'Unknown'
            GROUP BY channel
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("\n🏆 Top 10 Channels:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} videos")
        
        # Category distribution
        cursor.execute("""
            SELECT categories, COUNT(*) as count
            FROM videos
            WHERE categories IS NOT NULL AND categories != '[]'
            GROUP BY categories
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("\n📚 Top Categories:")
        for row in cursor.fetchall():
            cats = json.loads(row[0]) if row[0] else []
            if cats:
                print(f"  {', '.join(cats)}: {row[1]} videos")
    
    def export(self, format: str = 'json', ai_only: bool = False) -> None:
        """Export videos to file."""
        cursor = self.conn.cursor()
        
        if ai_only:
            query = """
                SELECT title, channel, url, ai_score, categories
                FROM videos
                WHERE ai_score > 0.5
                ORDER BY ai_score DESC
            """
            filename = f"youtube_ai_export.{format}"
        else:
            query = """
                SELECT title, channel, url, ai_score, categories
                FROM videos
                ORDER BY ai_score DESC
            """
            filename = f"youtube_export.{format}"
        
        cursor.execute(query)
        videos = [dict(row) for row in cursor.fetchall()]
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(videos, f, indent=2)
        elif format == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                if videos:
                    writer = csv.DictWriter(f, fieldnames=videos[0].keys())
                    writer.writeheader()
                    writer.writerows(videos)
        elif format == 'txt':
            with open(filename, 'w') as f:
                for video in videos:
                    f.write(f"{video['title']}\n")
                    f.write(f"  Channel: {video['channel']}\n")
                    f.write(f"  URL: {video['url']}\n")
                    f.write(f"  AI Score: {video['ai_score']}\n\n")
        
        print(f"✅ Exported {len(videos)} videos to: {filename}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'could', 'need', 'needn\'t',
                     'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                     'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                     'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                     'their', 'theirs', 'themselves', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'watching', 'watched', 'video', 'videos', 'youtube'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube CLI - Analyze your YouTube viewing history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  youtube_cli query "What LangChain videos have I watched?"
  youtube_cli recent --days 7 --ai-only
  youtube_cli transcript https://youtube.com/watch?v=VIDEO_ID
  youtube_cli ai-videos --months 6
  youtube_cli stats
  youtube_cli export --format json --ai-only
        """
    )
    
    parser.add_argument('--db', default='youtube_fast.db', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query with natural language')
    query_parser.add_argument('prompt', nargs='+', help='Your question')
    query_parser.add_argument('--limit', type=int, default=20, help='Max results')
    
    # Recent command
    recent_parser = subparsers.add_parser('recent', help='Show recent videos')
    recent_parser.add_argument('--days', type=int, default=30, help='Days to look back')
    recent_parser.add_argument('--ai-only', action='store_true', help='Only AI/coding videos')
    
    # Transcript command
    transcript_parser = subparsers.add_parser('transcript', help='Fetch video transcript')
    transcript_parser.add_argument('url', help='Video URL or ID')
    
    # AI videos command
    ai_parser = subparsers.add_parser('ai-videos', help='Show AI-related videos')
    ai_parser.add_argument('--months', type=int, default=3, help='Months to analyze')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export videos')
    export_parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json')
    export_parser.add_argument('--ai-only', action='store_true', help='Only AI/coding videos')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = YouTubeCLI(args.db)
    
    # Execute command
    if args.command == 'query':
        prompt = ' '.join(args.prompt)
        cli.query(prompt, args.limit)
    elif args.command == 'recent':
        cli.recent(args.days, args.ai_only)
    elif args.command == 'transcript':
        cli.transcript(args.url)
    elif args.command == 'ai-videos':
        cli.ai_videos(args.months)
    elif args.command == 'stats':
        cli.stats()
    elif args.command == 'export':
        cli.export(args.format, args.ai_only)
    else:
        parser.print_help()
        print("\n💡 Try: youtube_cli query 'What Python tutorials have I watched?'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())