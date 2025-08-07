#!/usr/bin/env python3
"""
YouTube Research CLI - Enhanced YouTube CLI with transcript fetching and web research

Features:
- All original YouTube CLI features
- Transcript fetching from YouTube
- Web research for topics found in videos
- Combined analysis of viewing history + external research
"""

import sys
import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Try to import from installed location first, then fall back to local
try:
    from agent.agent import Agent
    from special_agents.research_agents.web_search_agent import WebSearchAgent
except ImportError:
    # Add project root to path for development
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from agent.agent import Agent
    from special_agents.research_agents.web_search_agent import WebSearchAgent


class YouTubeResearchCLI:
    """Enhanced YouTube CLI with research capabilities."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the research CLI."""
        # Find database path
        db_locations = [
            Path(db_path),
            Path(__file__).parent.parent.parent.parent / "miniapps" / "youtube_database" / db_path,
            Path.home() / "code" / "miniapps" / "youtube_database" / db_path
        ]
        
        self.db_path = None
        for loc in db_locations:
            if loc.exists():
                self.db_path = loc
                break
        
        if not self.db_path:
            print(f"❌ Database not found in any location")
            print("Searched:", db_locations)
            sys.exit(1)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Initialize agents
        self.agent = Agent(
            roles=[
                "You are an expert YouTube content analyst and researcher.",
                "You analyze viewing history and conduct external research.",
                "You synthesize information from multiple sources.",
                "You provide comprehensive insights and recommendations."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        self.web_agent = WebSearchAgent(max_results=5)
        self.transcript_cache = {}
    
    def research_video(self, video_url: str, deep: bool = False) -> None:
        """
        Research a video by fetching transcript and searching related topics.
        
        Args:
            video_url: YouTube video URL or ID
            deep: If True, do deeper research on topics found
        """
        print(f"\n🔬 Researching video: {video_url}")
        print("=" * 60)
        
        # Extract video ID
        video_id = self._extract_video_id(video_url)
        
        # Step 1: Get video info from database
        video_info = self._get_video_info(video_id)
        if video_info:
            print(f"📺 Title: {video_info['title']}")
            print(f"📺 Channel: {video_info['channel']}")
        
        # Step 2: Fetch transcript
        print("\n📝 Fetching transcript...")
        transcript = self._fetch_transcript(video_id)
        
        if not transcript:
            print("❌ Could not fetch transcript")
            return
        
        # Step 3: Extract key topics from transcript
        print("\n🎯 Extracting key topics...")
        topics = self._extract_topics(transcript)
        print(f"Found topics: {', '.join(topics)}")
        
        # Step 4: Research topics on the web
        print("\n🌐 Researching topics online...")
        research_results = {}
        for topic in topics[:3]:  # Limit to top 3 topics
            print(f"  Searching: {topic}")
            results = self.web_agent.run(json.dumps({
                "query": topic,
                "context": f"Technical information about {topic} related to: {video_info.get('title', 'YouTube video')}"
            }))
            research_results[topic] = results
            time.sleep(1)  # Rate limiting
        
        # Step 5: Synthesize findings
        print("\n🧠 Synthesizing research findings...")
        synthesis = self._synthesize_research(video_info, transcript, topics, research_results, deep)
        
        print("\n" + "=" * 60)
        print("📊 RESEARCH SUMMARY")
        print("=" * 60)
        print(synthesis)
        
        # Save research results
        output_file = f"research_{video_id}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "video_id": video_id,
                "video_info": video_info,
                "topics": topics,
                "research": research_results,
                "synthesis": synthesis
            }, f, indent=2)
        print(f"\n✅ Research saved to: {output_file}")
    
    def research_topic(self, topic: str, use_history: bool = True) -> None:
        """
        Research a topic using both viewing history and web search.
        
        Args:
            topic: Topic to research
            use_history: If True, include relevant videos from history
        """
        print(f"\n🔬 Researching topic: {topic}")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Find relevant videos in history
        if use_history:
            print("\n📺 Searching viewing history...")
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT title, channel, url, ai_score
                FROM videos
                WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
                ORDER BY ai_score DESC
                LIMIT 10
            """, (f'%{topic.lower()}%', f'%{topic.lower()}%'))
            
            videos = [dict(row) for row in cursor.fetchall()]
            results['related_videos'] = videos
            
            if videos:
                print(f"Found {len(videos)} related videos in history:")
                for v in videos[:5]:
                    print(f"  • {v['title'][:60]}")
        
        # Step 2: Web research
        print("\n🌐 Searching the web...")
        web_results = self.web_agent.run(json.dumps({
            "query": f"{topic} tutorial documentation best practices",
            "context": "Looking for comprehensive technical information"
        }))
        results['web_research'] = web_results
        
        # Step 3: Get learning path recommendations
        print("\n📚 Generating learning path...")
        learning_path = self._generate_learning_path(topic, results)
        
        print("\n" + "=" * 60)
        print("📊 TOPIC RESEARCH SUMMARY")
        print("=" * 60)
        print(learning_path)
        
        # Save results
        output_file = f"topic_research_{topic.replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "topic": topic,
                "results": results,
                "learning_path": learning_path
            }, f, indent=2)
        print(f"\n✅ Research saved to: {output_file}")
    
    def analyze_and_research(self, prompt: str) -> None:
        """
        Analyze viewing history and conduct web research based on prompt.
        """
        print(f"\n🔍 Analyzing and researching: {prompt}")
        print("=" * 60)
        
        # Step 1: Analyze viewing history
        print("\n📺 Analyzing viewing history...")
        history_data = self._analyze_history(prompt)
        
        # Step 2: Extract topics to research
        topics = self._extract_research_topics(prompt, history_data)
        print(f"\n🎯 Topics to research: {', '.join(topics)}")
        
        # Step 3: Web research for each topic
        print("\n🌐 Conducting web research...")
        research_results = {}
        for topic in topics[:3]:
            print(f"  Researching: {topic}")
            results = self.web_agent.run(json.dumps({
                "query": topic,
                "context": prompt
            }))
            research_results[topic] = results
            time.sleep(1)
        
        # Step 4: Combined analysis
        print("\n🧠 Generating comprehensive analysis...")
        analysis = self._comprehensive_analysis(prompt, history_data, research_results)
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        print(analysis)
    
    def _extract_video_id(self, video_url: str) -> str:
        """Extract video ID from URL."""
        if 'watch?v=' in video_url:
            return video_url.split('watch?v=')[1].split('&')[0]
        elif 'youtu.be/' in video_url:
            return video_url.split('youtu.be/')[1].split('?')[0]
        return video_url  # Assume it's already an ID
    
    def _get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get video info from database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT title, channel, url, ai_score, categories
            FROM videos
            WHERE url LIKE ?
            LIMIT 1
        """, (f'%{video_id}%',))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def _fetch_transcript(self, video_id: str) -> Optional[str]:
        """Fetch transcript from YouTube."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Use the fetch method
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)
            
            # Combine all text - segments have .text attribute
            full_text = ' '.join([segment.text for segment in transcript_list])
            
            # Cache it
            self.transcript_cache[video_id] = full_text
            
            # Save to file
            output_file = f"transcript_{video_id}.txt"
            with open(output_file, 'w') as f:
                f.write(full_text)
            
            print(f"✅ Transcript saved ({len(full_text)} chars)")
            return full_text
            
        except ImportError:
            print("❌ youtube-transcript-api not installed")
            print("Install with: pip install youtube-transcript-api")
            return None
        except Exception as e:
            print(f"❌ Could not fetch transcript: {e}")
            return None
    
    def _extract_topics(self, transcript: str) -> List[str]:
        """Extract key topics from transcript using AI."""
        prompt = f"""
        Extract 3-5 key technical topics from this transcript.
        Return only the topic names, one per line.
        Focus on technologies, concepts, and tools mentioned.
        
        Transcript (first 2000 chars):
        {transcript[:2000]}
        """
        
        response = self.agent.run(prompt)
        topics = [line.strip() for line in response.split('\n') if line.strip()]
        return topics[:5]
    
    def _extract_research_topics(self, prompt: str, history_data: Dict) -> List[str]:
        """Extract topics to research from prompt and history."""
        analysis_prompt = f"""
        Based on this user query: "{prompt}"
        And their viewing history data: {json.dumps(history_data)[:1000]}
        
        Extract 3-5 specific topics that should be researched on the web.
        Return only topic names, one per line.
        Focus on technical concepts, tools, and frameworks.
        """
        
        response = self.agent.run(analysis_prompt)
        topics = [line.strip() for line in response.split('\n') if line.strip()]
        return topics[:5]
    
    def _synthesize_research(self, video_info: Dict, transcript: str, topics: List[str], 
                            research: Dict, deep: bool) -> str:
        """Synthesize all research findings."""
        synthesis_prompt = f"""
        Synthesize research findings for this video:
        
        Video: {video_info.get('title', 'Unknown')}
        Channel: {video_info.get('channel', 'Unknown')}
        
        Key Topics: {', '.join(topics)}
        
        Transcript Summary (first 500 chars): {transcript[:500]}
        
        Web Research Findings:
        {json.dumps(research, indent=2)[:2000]}
        
        Provide:
        1. Key concepts explained
        2. Important resources and links discovered
        3. Related topics to explore
        4. Practical applications
        5. Learning recommendations
        
        {"Include advanced insights and connections between topics." if deep else ""}
        """
        
        return self.agent.run(synthesis_prompt)
    
    def _analyze_history(self, prompt: str) -> Dict:
        """Analyze viewing history based on prompt."""
        cursor = self.conn.cursor()
        keywords = prompt.lower().split()[:5]
        
        data = {'matches': []}
        for keyword in keywords:
            cursor.execute("""
                SELECT title, channel, url, ai_score
                FROM videos
                WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
                ORDER BY ai_score DESC
                LIMIT 5
            """, (f'%{keyword}%', f'%{keyword}%'))
            
            matches = [dict(row) for row in cursor.fetchall()]
            if matches:
                data['matches'].extend(matches)
        
        return data
    
    def _generate_learning_path(self, topic: str, results: Dict) -> str:
        """Generate a learning path for a topic."""
        prompt = f"""
        Create a comprehensive learning path for: {topic}
        
        Based on:
        - Videos watched: {json.dumps(results.get('related_videos', [])[:5])}
        - Web research: {results.get('web_research', '')[:1000]}
        
        Provide:
        1. Current knowledge assessment (based on watched videos)
        2. Knowledge gaps identified
        3. Recommended learning sequence
        4. Specific resources to study
        5. Hands-on projects to build
        6. Advanced topics to explore later
        """
        
        return self.agent.run(prompt)
    
    def _comprehensive_analysis(self, prompt: str, history_data: Dict, research: Dict) -> str:
        """Generate comprehensive analysis combining history and research."""
        analysis_prompt = f"""
        Provide comprehensive analysis for: "{prompt}"
        
        Viewing History Insights:
        {json.dumps(history_data, indent=2)[:1500]}
        
        Web Research Findings:
        {json.dumps(research, indent=2)[:1500]}
        
        Deliver:
        1. Direct answer to the query
        2. Relevant videos from history
        3. Key findings from web research
        4. Synthesis and connections
        5. Actionable recommendations
        6. Further exploration suggestions
        """
        
        return self.agent.run(analysis_prompt)


def smart_route(cli: YouTubeResearchCLI, query: str) -> None:
    """
    Intelligently route a query to the appropriate command using AI.
    """
    # Use AI to determine the best action
    routing_prompt = f"""
    Analyze this user query and determine the best action:
    "{query}"
    
    Available actions:
    1. research-video: If the query contains a YouTube URL or video ID
    2. research-topic: If the query is about researching a specific topic/technology
    3. analyze: If the query is asking a question about viewing history or needs recommendations
    
    Return ONLY one of these exact words: research-video, research-topic, analyze
    
    Examples:
    - "https://youtube.com/watch?v=abc123" -> research-video
    - "LangChain RAG systems" -> research-topic  
    - "What should I learn next?" -> analyze
    - "What videos about Python have I watched?" -> analyze
    - "Machine learning fundamentals" -> research-topic
    """
    
    # Determine action
    action = cli.agent.run(routing_prompt).strip().lower()
    
    # Clean up action (remove any extra text)
    if "research-video" in action:
        action = "research-video"
    elif "research-topic" in action:
        action = "research-topic"
    elif "analyze" in action:
        action = "analyze"
    else:
        # Default fallback based on patterns
        if "youtube.com" in query or "youtu.be" in query or "watch?v=" in query:
            action = "research-video"
        elif "?" in query or any(word in query.lower() for word in ["what", "how", "why", "should", "have i", "did i"]):
            action = "analyze"
        else:
            action = "research-topic"
    
    print(f"🤖 Auto-routing to: {action}\n")
    
    # Execute the appropriate action
    if action == "research-video":
        # Extract URL if present
        import re
        url_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^\s]+)'
        match = re.search(url_pattern, query)
        if match:
            cli.research_video(match.group(1))
        else:
            # Try to extract video ID
            video_id_pattern = r'([a-zA-Z0-9_-]{11})'
            match = re.search(video_id_pattern, query)
            if match:
                cli.research_video(match.group(1))
            else:
                print("❌ Could not extract video URL or ID from query")
                print("💡 Falling back to topic research...")
                cli.research_topic(query)
    elif action == "research-topic":
        cli.research_topic(query)
    elif action == "analyze":
        cli.analyze_and_research(query)


def main():
    """Run the YouTube Research CLI."""
    import sys
    
    # Check if we're in smart mode (no explicit command)
    smart_mode = False
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg not in ['research-video', 'research-topic', 'analyze', '--help', '-h', '--db']:
            smart_mode = True
    
    if smart_mode:
        # Smart mode - process the entire input as a query
        db_path = 'youtube_fast.db'
        
        # Check for --db flag
        if '--db' in sys.argv:
            db_idx = sys.argv.index('--db')
            if db_idx + 1 < len(sys.argv):
                db_path = sys.argv[db_idx + 1]
                # Remove --db and its value from argv
                sys.argv.pop(db_idx)  # Remove --db
                sys.argv.pop(db_idx)  # Remove the value
        
        # Join all remaining args as the query (skip script name)
        query = ' '.join(sys.argv[1:])
        
        if query:
            cli = YouTubeResearchCLI(db_path)
            smart_route(cli, query)
            return 0
    
    # Traditional argument parsing
    parser = argparse.ArgumentParser(
        description="YouTube Research CLI - Enhanced analysis with web research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart mode - let AI decide what to do
  yr "What LangChain videos have I watched?"
  yr "https://youtube.com/watch?v=VIDEO_ID"  
  yr "prompt engineering techniques"
  
  # Explicit commands
  yr research-video https://youtube.com/watch?v=VIDEO_ID
  yr research-topic "LangChain RAG systems"
  yr analyze "What should I learn about AI agents?"
        """
    )
    
    parser.add_argument('--db', default='youtube_fast.db', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Research video command
    video_parser = subparsers.add_parser('research-video', help='Research a specific video')
    video_parser.add_argument('url', help='Video URL or ID')
    video_parser.add_argument('--deep', action='store_true', help='Deep research mode')
    
    # Research topic command
    topic_parser = subparsers.add_parser('research-topic', help='Research a topic')
    topic_parser.add_argument('topic', nargs='+', help='Topic to research')
    topic_parser.add_argument('--no-history', action='store_true', help='Skip viewing history')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze and research')
    analyze_parser.add_argument('prompt', nargs='+', help='Analysis prompt')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = YouTubeResearchCLI(args.db)
    
    # Execute command
    if args.command == 'research-video':
        cli.research_video(args.url, args.deep)
    elif args.command == 'research-topic':
        topic = ' '.join(args.topic)
        cli.research_topic(topic, not args.no_history)
    elif args.command == 'analyze':
        prompt = ' '.join(args.prompt)
        cli.analyze_and_research(prompt)
    else:
        parser.print_help()
        print("\n💡 Try: yr 'What Python tutorials have I watched?'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())