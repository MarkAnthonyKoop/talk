#!/usr/bin/env python3
"""
YouTube History Agent

An intelligent agent that can search through your YouTube history database
and answer questions about your viewing patterns, preferences, and learning journey.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent


class YouTubeHistoryAgent:
    """Agent that can intelligently query and analyze YouTube viewing history."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the YouTube history agent."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Initialize AI agent
        self.agent = Agent(
            roles=[
                "You are an expert at analyzing YouTube viewing history.",
                "You can identify patterns, preferences, and learning paths.",
                "You provide insightful analysis about viewing habits and content consumption.",
                "You can recommend content based on past viewing patterns.",
                "You understand technical content and can identify knowledge gaps."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Load database statistics for context
        self.stats = self._load_statistics()
    
    def _load_statistics(self) -> Dict[str, Any]:
        """Load basic statistics about the database."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        stats['total_videos'] = cursor.fetchone()[0]
        
        # AI videos
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
        stats['ai_videos'] = cursor.fetchone()[0]
        
        # Categories distribution
        cursor.execute("""
            SELECT categories, COUNT(*) as count
            FROM videos
            WHERE categories IS NOT NULL AND categories != '[]'
            GROUP BY categories
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_categories'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def analyze(self, prompt: str) -> str:
        """
        Analyze YouTube history based on a natural language prompt.
        
        Args:
            prompt: Natural language question about YouTube history
            
        Returns:
            Detailed analysis and information
        """
        # First, understand what the user is asking for
        analysis_prompt = f"""
        Analyze this question about YouTube viewing history:
        "{prompt}"
        
        Determine what type of information is being requested:
        1. Specific videos or channels
        2. Patterns or trends
        3. Learning progress
        4. Recommendations
        5. Statistics
        
        List the key search terms and what database queries would help answer this.
        Be specific about what to look for.
        """
        
        query_plan = self.agent.run(analysis_prompt)
        
        # Extract relevant data from database
        data = self._extract_relevant_data(prompt, query_plan)
        
        # Generate comprehensive analysis
        final_analysis = self._generate_analysis(prompt, data)
        
        return final_analysis
    
    def _extract_relevant_data(self, prompt: str, query_plan: str) -> Dict[str, Any]:
        """Extract relevant data based on the prompt."""
        data = {}
        prompt_lower = prompt.lower()
        cursor = self.conn.cursor()
        
        # Extract key terms from prompt
        keywords = self._extract_keywords(prompt)
        
        # Search for specific topics/channels/videos
        if keywords:
            data['keyword_matches'] = []
            for keyword in keywords:
                cursor.execute("""
                    SELECT title, channel, url, ai_score, categories
                    FROM videos
                    WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
                    ORDER BY ai_score DESC
                    LIMIT 10
                """, (f'%{keyword}%', f'%{keyword}%'))
                
                matches = [dict(row) for row in cursor.fetchall()]
                if matches:
                    data['keyword_matches'].append({
                        'keyword': keyword,
                        'videos': matches
                    })
        
        # Get viewing patterns
        if any(term in prompt_lower for term in ['pattern', 'trend', 'habit', 'when', 'time']):
            # Get video count by categories
            cursor.execute("""
                SELECT categories, COUNT(*) as count
                FROM videos
                WHERE categories IS NOT NULL AND categories != '[]'
                GROUP BY categories
                ORDER BY count DESC
                LIMIT 20
            """)
            data['category_distribution'] = [dict(row) for row in cursor.fetchall()]
            
            # Get channel frequency
            cursor.execute("""
                SELECT channel, COUNT(*) as count
                FROM videos
                WHERE channel != 'Unknown'
                GROUP BY channel
                ORDER BY count DESC
                LIMIT 20
            """)
            data['top_channels'] = [dict(row) for row in cursor.fetchall()]
        
        # Get AI/coding specific content
        if any(term in prompt_lower for term in ['ai', 'code', 'coding', 'programming', 'machine learning', 'llm']):
            cursor.execute("""
                SELECT title, channel, url, ai_score, categories
                FROM videos
                WHERE ai_score > 0.7
                ORDER BY ai_score DESC
                LIMIT 30
            """)
            data['ai_content'] = [dict(row) for row in cursor.fetchall()]
        
        # Get learning progression
        if any(term in prompt_lower for term in ['learn', 'progress', 'journey', 'evolution', 'improve']):
            # Get category evolution (simplified - would need timestamps for real progression)
            cursor.execute("""
                SELECT categories, COUNT(*) as count, AVG(ai_score) as avg_score
                FROM videos
                WHERE categories LIKE '%tutorials%' OR categories LIKE '%learning%'
                GROUP BY categories
                ORDER BY count DESC
                LIMIT 10
            """)
            data['learning_categories'] = [dict(row) for row in cursor.fetchall()]
        
        # Get recommendations
        if any(term in prompt_lower for term in ['recommend', 'suggest', 'should watch', 'next']):
            # Find under-watched high-value categories
            cursor.execute("""
                SELECT DISTINCT categories
                FROM videos
                WHERE ai_score > 0.8
                GROUP BY categories
                HAVING COUNT(*) < 5
                LIMIT 10
            """)
            data['recommended_categories'] = [row[0] for row in cursor.fetchall()]
        
        # Add general statistics
        data['stats'] = self.stats
        
        return data
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract meaningful keywords from prompt."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
                     'did', 'do', 'does', 'have', 'has', 'had', 'is', 'are', 'was', 'were',
                     'been', 'being', 'my', 'your', 'our', 'their', 'i', 'you', 'we', 'they',
                     'watch', 'watched', 'video', 'videos', 'youtube'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', prompt.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add technical terms if present
        tech_terms = re.findall(r'\b(?:python|javascript|react|ai|ml|llm|gpt|claude|docker|kubernetes|aws|api|git)\b', 
                                prompt.lower())
        keywords.extend(tech_terms)
        
        return list(set(keywords))[:5]  # Limit to 5 keywords
    
    def _generate_analysis(self, prompt: str, data: Dict[str, Any]) -> str:
        """Generate comprehensive analysis using AI."""
        # Prepare data summary for AI
        data_summary = self._prepare_data_summary(data)
        
        analysis_prompt = f"""
        Based on the YouTube viewing history data, answer this question:
        "{prompt}"
        
        Available data from the database:
        {data_summary}
        
        Provide a comprehensive, insightful answer that:
        1. Directly addresses the user's question
        2. Includes specific examples from their viewing history
        3. Identifies patterns or trends if relevant
        4. Offers recommendations or insights
        5. Uses the actual video titles and channels when possible
        
        Be specific and reference actual videos from their history.
        Format the response in a clear, readable way.
        """
        
        analysis = self.agent.run(analysis_prompt)
        
        # Add raw data if useful
        if data.get('keyword_matches'):
            analysis += "\n\n### Specific Videos Found:\n"
            for match in data['keyword_matches']:
                analysis += f"\n**Related to '{match['keyword']}':**\n"
                for video in match['videos'][:3]:
                    analysis += f"- {video['title']}\n"
                    if video['channel'] != 'Unknown':
                        analysis += f"  Channel: {video['channel']}\n"
                    if video.get('categories') and video['categories'] != '[]':
                        analysis += f"  Categories: {video['categories']}\n"
        
        return analysis
    
    def _prepare_data_summary(self, data: Dict[str, Any]) -> str:
        """Prepare a summary of the data for the AI."""
        summary = []
        
        # Add statistics
        if 'stats' in data:
            summary.append(f"Total videos watched: {data['stats']['total_videos']}")
            summary.append(f"AI/coding videos: {data['stats']['ai_videos']}")
        
        # Add keyword matches
        if 'keyword_matches' in data:
            for match in data['keyword_matches']:
                if match['videos']:
                    summary.append(f"\nVideos related to '{match['keyword']}':")
                    for video in match['videos'][:5]:
                        summary.append(f"  - {video['title']} (Score: {video['ai_score']:.2f})")
        
        # Add top channels
        if 'top_channels' in data:
            summary.append("\nMost watched channels:")
            for channel in data['top_channels'][:10]:
                summary.append(f"  - {channel['channel']}: {channel['count']} videos")
        
        # Add AI content
        if 'ai_content' in data:
            summary.append(f"\nFound {len(data['ai_content'])} high-relevance AI/coding videos")
            for video in data['ai_content'][:5]:
                summary.append(f"  - {video['title']}")
        
        # Add category distribution
        if 'category_distribution' in data:
            summary.append("\nContent categories:")
            for cat in data['category_distribution'][:5]:
                if cat['categories'] != '[]':
                    summary.append(f"  - {cat['categories']}: {cat['count']} videos")
        
        return "\n".join(summary)
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("YouTube History Intelligence Agent")
        print("=" * 60)
        print(f"Database: {self.stats['total_videos']:,} videos")
        print(f"AI/Coding content: {self.stats['ai_videos']:,} videos")
        print("\nAsk me anything about your YouTube viewing history!")
        print("Examples:")
        print("  - What machine learning content have I watched?")
        print("  - Show me my Python programming videos")
        print("  - What are my viewing patterns?")
        print("  - What should I watch next to improve my AI knowledge?")
        print("  - Have I watched anything about LangChain or agents?")
        print("\nType 'quit' to exit\n")
        
        while True:
            prompt = input("\n🎥 Your question: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print("\n🔍 Analyzing your viewing history...")
            
            try:
                analysis = self.analyze(prompt)
                print("\n" + "=" * 60)
                print(analysis)
                print("=" * 60)
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    """Run the YouTube History Agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube History Intelligence Agent")
    parser.add_argument("--db", default="youtube_fast.db", help="Database path")
    parser.add_argument("--prompt", help="Single prompt to analyze")
    
    args = parser.parse_args()
    
    # Check database exists
    if not Path(args.db).exists():
        print(f"Error: Database {args.db} not found")
        print("Run build_db_fast.py first to create the database")
        return 1
    
    # Create agent
    agent = YouTubeHistoryAgent(args.db)
    
    try:
        if args.prompt:
            # Single prompt mode
            print("Analyzing your YouTube history...")
            result = agent.analyze(args.prompt)
            print("\n" + "=" * 60)
            print(result)
            print("=" * 60)
        else:
            # Interactive mode
            agent.interactive_mode()
    finally:
        agent.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())