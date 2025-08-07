#!/usr/bin/env python3
"""
YouTubeAgent - Specialized agent for YouTube research integrated with Talk framework

This agent can be called by other agents in the Talk framework to:
- Search YouTube viewing history
- Fetch and analyze transcripts  
- Generate learning paths
- Provide viewing analytics
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import base Agent class
from agent.agent import Agent

# Import YouTube-specific modules
import sys
youtube_path = Path(__file__).parent / "research_agents" / "youtube"
sys.path.insert(0, str(youtube_path))

try:
    from transcript_manager import TranscriptManager
    from learning_path_generator import LearningPathGenerator
except ImportError:
    TranscriptManager = None
    LearningPathGenerator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeAgent(Agent):
    """
    Specialized agent for YouTube research and analysis.
    
    This agent integrates with the Talk framework and can be called
    by other agents to perform YouTube-related tasks.
    """
    
    def __init__(self, db_path: str = None, **kwargs):
        """Initialize YouTube agent with database and specialized capabilities."""
        
        # Set up YouTube-specific system prompts
        youtube_roles = [
            "You are a YouTube research specialist with access to viewing history.",
            "You can search videos, analyze transcripts, and generate learning paths.",
            "You provide data-driven insights from YouTube viewing patterns.",
            "You help other agents find relevant video content and tutorials.",
            "You can identify knowledge gaps and suggest learning resources."
        ]
        
        # Combine with any existing roles
        roles = kwargs.pop("roles", [])
        roles = youtube_roles + roles
        
        # Initialize base agent
        super().__init__(roles=roles, **kwargs)
        
        # Find database
        self.db_path = self._find_database(db_path)
        if self.db_path:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        else:
            self.conn = None
            logger.warning("No YouTube database found")
        
        # Initialize specialized components
        self.transcript_manager = TranscriptManager() if TranscriptManager else None
        self.learning_generator = LearningPathGenerator(str(self.db_path)) if self.db_path and LearningPathGenerator else None
        
        # Cache for performance
        self.cache = {}
    
    def _find_database(self, db_path: str = None) -> Optional[Path]:
        """Find YouTube database in standard locations."""
        if db_path:
            path = Path(db_path)
            if path.exists():
                return path
        
        # Standard locations
        locations = [
            Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db",
            Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_enhanced.db",
            Path.home() / ".youtube_research" / "youtube.db",
            Path("youtube_fast.db"),
            Path("youtube_enhanced.db")
        ]
        
        for loc in locations:
            if loc.exists():
                logger.info(f"Found YouTube database: {loc}")
                return loc
        
        return None
    
    def run(self, input_text: str) -> str:
        """
        Process YouTube-related requests from other agents.
        
        Input format can be:
        - Plain text query: "Find videos about machine learning"
        - JSON command: {"command": "search", "query": "...", "options": {...}}
        """
        
        # Try to parse as JSON command first
        try:
            command = json.loads(input_text)
            return self._execute_command(command)
        except json.JSONDecodeError:
            # Treat as plain text query
            return self._process_query(input_text)
    
    def _execute_command(self, command: Dict) -> str:
        """Execute structured command."""
        cmd_type = command.get("command", "search")
        
        if cmd_type == "search":
            return self._search_videos(
                command.get("query", ""),
                command.get("options", {})
            )
        
        elif cmd_type == "transcript":
            return self._get_transcript(
                command.get("video_id"),
                command.get("analyze", False)
            )
        
        elif cmd_type == "learning_path":
            return self._generate_learning_path(
                command.get("goal", ""),
                command.get("current_knowledge", [])
            )
        
        elif cmd_type == "analytics":
            return self._get_analytics(
                command.get("metric", "overview"),
                command.get("options", {})
            )
        
        elif cmd_type == "find_tutorials":
            return self._find_tutorials(
                command.get("topic", ""),
                command.get("level", "all")
            )
        
        else:
            return f"Unknown command: {cmd_type}"
    
    def _process_query(self, query: str) -> str:
        """Process natural language query."""
        query_lower = query.lower()
        
        # Determine intent
        if any(word in query_lower for word in ["find", "search", "videos", "watched"]):
            return self._search_videos(query, {})
        
        elif any(word in query_lower for word in ["transcript", "captions", "said"]):
            # Extract video ID if present
            import re
            video_id_match = re.search(r'([a-zA-Z0-9_-]{11})', query)
            if video_id_match:
                return self._get_transcript(video_id_match.group(1), analyze=True)
            else:
                return "Please provide a video ID for transcript fetching"
        
        elif any(word in query_lower for word in ["learning", "path", "curriculum", "plan"]):
            # Extract goal from query
            goal = query.replace("learning path", "").replace("for", "").strip()
            return self._generate_learning_path(goal, [])
        
        elif any(word in query_lower for word in ["stats", "analytics", "patterns", "trends"]):
            return self._get_analytics("overview", {})
        
        elif any(word in query_lower for word in ["tutorial", "how to", "guide"]):
            return self._find_tutorials(query, "all")
        
        else:
            # Default to search
            return self._search_videos(query, {"limit": 10})
    
    def _search_videos(self, query: str, options: Dict) -> str:
        """Search YouTube viewing history."""
        if not self.conn:
            return "ERROR: No database connection"
        
        cursor = self.conn.cursor()
        limit = options.get("limit", 20)
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Build search query
        conditions = []
        params = []
        
        for keyword in keywords[:5]:
            conditions.append("LOWER(title) LIKE ?")
            params.append(f"%{keyword}%")
        
        where_clause = " OR ".join(conditions) if conditions else "1=1"
        
        # Execute search
        cursor.execute(f"""
            SELECT video_id, title, channel, url, ai_score, watch_time
            FROM videos
            WHERE {where_clause}
            ORDER BY ai_score DESC
            LIMIT ?
        """, params + [limit])
        
        results = cursor.fetchall()
        
        if not results:
            return json.dumps({
                "status": "success",
                "message": "No videos found",
                "count": 0,
                "videos": []
            })
        
        videos = []
        for row in results:
            videos.append({
                "video_id": row["video_id"],
                "title": row["title"],
                "channel": row["channel"],
                "url": row["url"],
                "ai_score": row["ai_score"],
                "watch_time": row["watch_time"]
            })
        
        response = {
            "status": "success",
            "message": f"Found {len(videos)} videos",
            "count": len(videos),
            "query": query,
            "keywords": keywords,
            "videos": videos
        }
        
        return json.dumps(response, indent=2)
    
    def _get_transcript(self, video_id: str, analyze: bool = False) -> str:
        """Get and optionally analyze transcript."""
        if not self.transcript_manager:
            return json.dumps({
                "status": "error",
                "message": "Transcript manager not available"
            })
        
        # Fetch transcript
        transcript = self.transcript_manager.fetch_single(video_id)
        
        if not transcript:
            return json.dumps({
                "status": "error",
                "message": "Could not fetch transcript",
                "video_id": video_id
            })
        
        response = {
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(transcript)
        }
        
        if analyze:
            analysis = self.transcript_manager.analyze_transcript(transcript)
            response["analysis"] = analysis
            response["transcript_excerpt"] = transcript[:500]
        else:
            response["transcript"] = transcript
        
        return json.dumps(response, indent=2)
    
    def _generate_learning_path(self, goal: str, current_knowledge: List[str]) -> str:
        """Generate learning path."""
        if not self.learning_generator:
            return json.dumps({
                "status": "error",
                "message": "Learning path generator not available"
            })
        
        try:
            path = self.learning_generator.generate_path(goal, current_knowledge)
            
            response = {
                "status": "success",
                "goal": path.goal,
                "current_level": path.current_level,
                "target_level": path.target_level,
                "duration_hours": path.total_duration_hours,
                "phases": {
                    phase: [{"title": n.title, "level": n.level} for n in nodes[:3]]
                    for phase, nodes in path.phases.items()
                },
                "knowledge_gaps": path.knowledge_gaps[:5],
                "recommendations": path.recommended_resources[:3]
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    def _get_analytics(self, metric: str, options: Dict) -> str:
        """Get viewing analytics."""
        if not self.conn:
            return json.dumps({"status": "error", "message": "No database"})
        
        cursor = self.conn.cursor()
        
        if metric == "overview":
            # Get overview statistics
            cursor.execute("SELECT COUNT(*) as total FROM videos")
            total = cursor.fetchone()["total"]
            
            cursor.execute("SELECT COUNT(*) as ai_count FROM videos WHERE ai_score > 0.5")
            ai_count = cursor.fetchone()["ai_count"]
            
            cursor.execute("SELECT COUNT(DISTINCT channel) as channels FROM videos WHERE channel != 'Unknown'")
            channels = cursor.fetchone()["channels"]
            
            cursor.execute("""
                SELECT title, channel, ai_score
                FROM videos
                ORDER BY ai_score DESC
                LIMIT 5
            """)
            top_ai = [dict(row) for row in cursor.fetchall()]
            
            response = {
                "status": "success",
                "metric": "overview",
                "stats": {
                    "total_videos": total,
                    "ai_videos": ai_count,
                    "unique_channels": channels,
                    "ai_percentage": (ai_count / total * 100) if total > 0 else 0
                },
                "top_ai_videos": top_ai
            }
            
        elif metric == "trends":
            # Analyze viewing trends
            cursor.execute("""
                SELECT 
                    DATE(watch_time) as date,
                    COUNT(*) as videos_watched
                FROM videos
                WHERE watch_time IS NOT NULL
                GROUP BY DATE(watch_time)
                ORDER BY date DESC
                LIMIT 30
            """)
            
            daily_stats = [dict(row) for row in cursor.fetchall()]
            
            response = {
                "status": "success",
                "metric": "trends",
                "daily_viewing": daily_stats
            }
        
        else:
            response = {
                "status": "error",
                "message": f"Unknown metric: {metric}"
            }
        
        return json.dumps(response, indent=2)
    
    def _find_tutorials(self, topic: str, level: str) -> str:
        """Find tutorial videos."""
        if not self.conn:
            return json.dumps({"status": "error", "message": "No database"})
        
        cursor = self.conn.cursor()
        
        # Build query based on level
        query = """
            SELECT video_id, title, channel, url, ai_score
            FROM videos
            WHERE LOWER(title) LIKE ?
        """
        
        params = [f"%{topic.lower()}%"]
        
        # Add level filter
        if level == "beginner":
            query += " AND (LOWER(title) LIKE '%beginner%' OR LOWER(title) LIKE '%intro%' OR LOWER(title) LIKE '%basic%')"
        elif level == "advanced":
            query += " AND (LOWER(title) LIKE '%advanced%' OR LOWER(title) LIKE '%expert%' OR LOWER(title) LIKE '%deep%')"
        
        query += " ORDER BY ai_score DESC LIMIT 10"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        tutorials = [dict(row) for row in results]
        
        response = {
            "status": "success",
            "topic": topic,
            "level": level,
            "count": len(tutorials),
            "tutorials": tutorials
        }
        
        return json.dumps(response, indent=2)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        import re
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
                     'what', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'watched', 'videos', 'youtube', 'find', 'search', 'show'}
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    # Methods for Talk framework integration
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Return agent capabilities for registration."""
        return {
            "name": "YouTubeAgent",
            "description": "Specialized agent for YouTube research and analysis",
            "capabilities": [
                "search_videos",
                "fetch_transcripts",
                "generate_learning_paths",
                "analyze_viewing_patterns",
                "find_tutorials"
            ],
            "commands": [
                {"name": "search", "description": "Search viewing history"},
                {"name": "transcript", "description": "Fetch video transcript"},
                {"name": "learning_path", "description": "Generate learning path"},
                {"name": "analytics", "description": "Get viewing analytics"},
                {"name": "find_tutorials", "description": "Find tutorial videos"}
            ]
        }
    
    def collaborate(self, agent_name: str, request: Dict) -> str:
        """Handle collaboration requests from other agents."""
        logger.info(f"Collaboration request from {agent_name}: {request.get('type', 'unknown')}")
        
        # Route based on request type
        if request.get("type") == "find_resource":
            # Another agent is looking for learning resources
            topic = request.get("topic", "")
            return self._find_tutorials(topic, "all")
        
        elif request.get("type") == "check_knowledge":
            # Check if user has watched videos on a topic
            topic = request.get("topic", "")
            results = json.loads(self._search_videos(topic, {"limit": 5}))
            
            has_knowledge = results.get("count", 0) > 0
            return json.dumps({
                "has_knowledge": has_knowledge,
                "video_count": results.get("count", 0),
                "top_video": results.get("videos", [{}])[0].get("title", "None")
            })
        
        else:
            # Default to search
            query = request.get("query", "")
            return self._search_videos(query, request.get("options", {}))


# For backward compatibility
class YouTubeHistoryAgent(YouTubeAgent):
    """Alias for YouTubeAgent focused on history analysis."""
    pass


def register_youtube_agent():
    """Register YouTubeAgent with Talk framework."""
    # This would be called during Talk framework initialization
    from agent.registry import AgentRegistry
    
    youtube_agent = YouTubeAgent()
    AgentRegistry.register("youtube", youtube_agent)
    
    logger.info("YouTubeAgent registered with Talk framework")


if __name__ == "__main__":
    # Test the agent
    agent = YouTubeAgent()
    
    # Test search
    print("\n=== Testing Search ===")
    result = agent.run("Find Claude videos")
    print(result)
    
    # Test structured command
    print("\n=== Testing Command ===")
    command = json.dumps({
        "command": "analytics",
        "metric": "overview"
    })
    result = agent.run(command)
    print(result)
    
    # Test collaboration
    print("\n=== Testing Collaboration ===")
    request = {
        "type": "find_resource",
        "topic": "machine learning"
    }
    result = agent.collaborate("CodebaseAgent", request)
    print(result)