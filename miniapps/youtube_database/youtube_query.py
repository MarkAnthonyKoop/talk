#!/usr/bin/env python3
"""
YouTube Database Query Interface

Provides an interactive interface to query and analyze the YouTube history database.
Includes predefined queries and natural language query support via AI.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent


class YouTubeQueryInterface:
    """Query interface for YouTube database."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.ai_agent = None
    
    def initialize_ai(self):
        """Initialize AI for natural language queries."""
        try:
            self.ai_agent = Agent(
                roles=["You are a SQL expert who helps query YouTube viewing history databases."],
                overrides={"llm": {"provider": "anthropic"}}
            )
            return True
        except Exception as e:
            print(f"AI not available: {e}")
            return False
    
    # === Predefined Queries ===
    
    def get_top_channels(self, limit: int = 10) -> List[Dict]:
        """Get most watched channels."""
        query = """
            SELECT 
                channel,
                COUNT(*) as video_count,
                AVG(ai_relevance_score) as avg_ai_score,
                MAX(watch_time) as last_watched
            FROM videos
            WHERE channel IS NOT NULL
            GROUP BY channel
            ORDER BY video_count DESC
            LIMIT ?
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_ai_coding_videos(self, min_score: float = 0.7) -> List[Dict]:
        """Get high-relevance AI/coding videos."""
        query = """
            SELECT 
                v.title,
                v.channel,
                v.ai_relevance_score,
                v.watch_time,
                GROUP_CONCAT(c.name) as categories
            FROM videos v
            LEFT JOIN video_categories vc ON v.video_id = vc.video_id
            LEFT JOIN categories c ON vc.category_id = c.id
            WHERE v.ai_relevance_score > ?
            GROUP BY v.video_id
            ORDER BY v.ai_relevance_score DESC
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (min_score,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_videos_by_category(self, category: str) -> List[Dict]:
        """Get videos in a specific category."""
        query = """
            SELECT 
                v.title,
                v.channel,
                v.watch_time,
                vc.confidence
            FROM videos v
            JOIN video_categories vc ON v.video_id = vc.video_id
            JOIN categories c ON vc.category_id = c.id
            WHERE c.name = ?
            ORDER BY vc.confidence DESC
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (category,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_watch_patterns(self) -> Dict[str, Any]:
        """Analyze watching patterns."""
        cursor = self.conn.cursor()
        
        # Videos per day of week (if timestamp parsing works)
        # Videos per month
        # Most productive watching periods
        # Binge patterns (multiple videos from same channel in sequence)
        
        patterns = {}
        
        # Total watch time
        cursor.execute("SELECT COUNT(*) FROM watch_history")
        patterns['total_watches'] = cursor.fetchone()[0]
        
        # Unique videos
        cursor.execute("SELECT COUNT(DISTINCT video_id) FROM watch_history")
        patterns['unique_videos'] = cursor.fetchone()[0]
        
        # Average rewatches
        patterns['avg_rewatches'] = patterns['total_watches'] / max(patterns['unique_videos'], 1)
        
        # Most rewatched
        cursor.execute("""
            SELECT v.title, COUNT(*) as watch_count
            FROM watch_history wh
            JOIN videos v ON wh.video_id = v.video_id
            GROUP BY wh.video_id
            HAVING watch_count > 1
            ORDER BY watch_count DESC
            LIMIT 5
        """)
        patterns['most_rewatched'] = [dict(row) for row in cursor.fetchall()]
        
        return patterns
    
    def get_learning_progression(self) -> List[Dict]:
        """Track learning progression over time."""
        query = """
            SELECT 
                DATE(watch_time) as date,
                AVG(ai_relevance_score) as avg_ai_score,
                COUNT(*) as videos_watched
            FROM videos
            WHERE watch_time IS NOT NULL
            GROUP BY DATE(watch_time)
            ORDER BY date
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def search_videos(self, search_term: str) -> List[Dict]:
        """Search videos by title or channel."""
        query = """
            SELECT 
                title,
                channel,
                watch_time,
                ai_relevance_score
            FROM videos
            WHERE title LIKE ? OR channel LIKE ?
            ORDER BY ai_relevance_score DESC
            LIMIT 20
        """
        
        search_pattern = f"%{search_term}%"
        cursor = self.conn.cursor()
        cursor.execute(query, (search_pattern, search_pattern))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_transcribed_videos(self) -> List[Dict]:
        """Get videos with transcripts."""
        query = """
            SELECT 
                title,
                channel,
                transcript_source,
                LENGTH(transcript) as transcript_length
            FROM videos
            WHERE transcript IS NOT NULL
            ORDER BY watch_time DESC
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_channel_evolution(self, channel_name: str) -> List[Dict]:
        """Track watching pattern for a specific channel."""
        query = """
            SELECT 
                title,
                watch_time,
                ai_relevance_score
            FROM videos
            WHERE channel = ?
            ORDER BY watch_time
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (channel_name,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recommendations(self) -> Dict[str, List]:
        """Generate viewing recommendations."""
        cursor = self.conn.cursor()
        
        recommendations = {}
        
        # Under-explored high-value categories
        cursor.execute("""
            SELECT 
                c.name,
                c.description,
                COUNT(vc.video_id) as video_count
            FROM categories c
            LEFT JOIN video_categories vc ON c.id = vc.category_id
            WHERE c.name IN ('codebase_analysis', 'ai_agents', 'ai_coding')
            GROUP BY c.id
            HAVING video_count < 10
            ORDER BY video_count
        """)
        recommendations['explore_categories'] = [dict(row) for row in cursor.fetchall()]
        
        # High-value channels to follow
        cursor.execute("""
            SELECT 
                channel,
                COUNT(*) as video_count,
                AVG(ai_relevance_score) as avg_score
            FROM videos
            WHERE ai_relevance_score > 0.7
            GROUP BY channel
            HAVING video_count >= 3
            ORDER BY avg_score DESC
            LIMIT 5
        """)
        recommendations['follow_channels'] = [dict(row) for row in cursor.fetchall()]
        
        return recommendations
    
    def natural_language_query(self, question: str) -> Any:
        """Convert natural language to SQL query using AI."""
        if not self.ai_agent:
            return "AI not available. Please use predefined queries."
        
        # Get schema information
        schema = self._get_schema_info()
        
        prompt = f"""
        Convert this question to a SQL query for a YouTube history database:
        
        Question: {question}
        
        Database schema:
        {schema}
        
        Return ONLY the SQL query, no explanation.
        The query should be valid SQLite SQL.
        """
        
        try:
            sql_query = self.ai_agent.run(prompt).strip()
            
            # Clean up the query
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Execute the query
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            
            # Return results
            results = cursor.fetchall()
            if results:
                return [dict(row) for row in results]
            else:
                return "No results found"
                
        except Exception as e:
            return f"Query error: {e}"
    
    def _get_schema_info(self) -> str:
        """Get database schema for AI context."""
        cursor = self.conn.cursor()
        
        # Get table info
        cursor.execute("""
            SELECT name, sql 
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        schema_info = []
        for table_name, create_sql in cursor.fetchall():
            # Simplify schema for AI
            schema_info.append(f"Table: {table_name}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                schema_info.append(f"  - {col[1]} ({col[2]})")
        
        return "\n".join(schema_info)
    
    def export_results(self, results: List[Dict], format: str = 'json', output_path: Optional[str] = None):
        """Export query results."""
        if not results:
            print("No results to export")
            return
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"youtube_query_{timestamp}.{format}"
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        else:
            print(f"Unsupported format: {format}")
            return
        
        print(f"Results exported to: {output_path}")
    
    def interactive_mode(self):
        """Run interactive query mode."""
        print("YouTube Database Query Interface")
        print("=" * 60)
        print("Commands:")
        print("  1. Top channels")
        print("  2. AI/coding videos")
        print("  3. Search videos")
        print("  4. Watch patterns")
        print("  5. Recommendations")
        print("  6. Natural language query (requires AI)")
        print("  q. Quit")
        print()
        
        while True:
            choice = input("\nEnter command (1-6 or q): ").strip()
            
            if choice == 'q':
                break
            elif choice == '1':
                results = self.get_top_channels()
                self._display_results(results)
            elif choice == '2':
                results = self.get_ai_coding_videos()
                self._display_results(results)
            elif choice == '3':
                term = input("Search term: ")
                results = self.search_videos(term)
                self._display_results(results)
            elif choice == '4':
                patterns = self.get_watch_patterns()
                print(json.dumps(patterns, indent=2, default=str))
            elif choice == '5':
                recs = self.get_recommendations()
                print(json.dumps(recs, indent=2, default=str))
            elif choice == '6':
                if not self.ai_agent:
                    self.initialize_ai()
                question = input("Your question: ")
                result = self.natural_language_query(question)
                if isinstance(result, list):
                    self._display_results(result)
                else:
                    print(result)
            else:
                print("Invalid choice")
    
    def _display_results(self, results: List[Dict], limit: int = 10):
        """Display query results."""
        if not results:
            print("No results found")
            return
        
        print(f"\nShowing {min(len(results), limit)} of {len(results)} results:")
        print("-" * 60)
        
        for i, row in enumerate(results[:limit]):
            print(f"\n{i+1}. ", end="")
            for key, value in row.items():
                if value is not None:
                    print(f"{key}: {str(value)[:100]}", end=" | ")
            print()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    """Run the query interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query YouTube database")
    parser.add_argument(
        "--db",
        default="youtube_history.db",
        help="Database path"
    )
    parser.add_argument(
        "--query",
        help="Run a specific query"
    )
    parser.add_argument(
        "--export",
        help="Export results to file"
    )
    parser.add_argument(
        "--format",
        choices=['json', 'csv'],
        default='json',
        help="Export format"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Run youtube_database_builder.py first to create the database")
        return 1
    
    # Create query interface
    query_interface = YouTubeQueryInterface(str(db_path))
    
    try:
        if args.query:
            # Natural language query
            query_interface.initialize_ai()
            result = query_interface.natural_language_query(args.query)
            
            if isinstance(result, list):
                query_interface._display_results(result)
                if args.export:
                    query_interface.export_results(result, args.format, args.export)
            else:
                print(result)
        else:
            # Interactive mode
            query_interface.interactive_mode()
    
    finally:
        query_interface.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())