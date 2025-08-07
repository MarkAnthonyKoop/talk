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
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import uuid

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


@dataclass
class ResearchPlan:
    """Represents a research plan with structured steps."""
    query: str
    intent: str = "general"
    requires_history: bool = True
    requires_web_search: bool = False
    requires_transcript: bool = False
    steps: List[Dict] = field(default_factory=list)
    expected_output: str = "Analysis results"


class YouTubeResearchCLI:
    """Enhanced YouTube CLI with research capabilities and AI planning."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the research CLI with context awareness."""
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
        
        # Load context from .talk/yr/ directory
        self.context = self._load_context()
        self.conversation_id = str(uuid.uuid4())[:8]
        
        # Initialize agents with enhanced prompts
        system_prompts = [
            "You are an expert YouTube content analyst and researcher.",
            "You analyze viewing history and conduct external research.",
            "You synthesize information from multiple sources.",
            "You provide comprehensive, structured insights and recommendations.",
            "Think step by step and create detailed plans when needed.",
            "Be specific and actionable in your responses."
        ]
        
        # Add context to system prompts if available
        if self.context:
            system_prompts.append(f"Previous context: {json.dumps(self.context)[:500]}")
        
        self.agent = Agent(
            roles=system_prompts,
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Planning agent for creating research plans
        self.planner = Agent(
            roles=[
                "You are an expert at analyzing queries and creating research plans.",
                "You break down complex questions into actionable steps.",
                "You understand YouTube content analysis and web research.",
                "You identify the most efficient way to gather information."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        self.web_agent = WebSearchAgent(max_results=5)
        self.transcript_cache = {}
    
    def _load_context(self) -> Dict[str, Any]:
        """Load context from .talk/yr/ directory."""
        context = {}
        context_dir = Path.cwd() / ".talk" / "yr"
        
        if context_dir.exists():
            # Find most recent context file
            json_files = sorted(
                glob.glob(str(context_dir / "*_*.json")),
                key=os.path.getmtime,
                reverse=True
            )
            
            if json_files:
                most_recent = json_files[0]
                try:
                    with open(most_recent, 'r') as f:
                        context = json.load(f)
                    print(f"📂 Loaded context from: {Path(most_recent).name}")
                except Exception as e:
                    print(f"⚠️ Could not load context: {e}")
        
        return context
    
    def _save_context(self, data: Dict[str, Any]) -> None:
        """Save context to .talk/yr/ directory."""
        context_dir = Path.cwd() / ".talk" / "yr"
        context_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = context_dir / f"conversation_{timestamp}_{self.conversation_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Context saved to: {filename.name}")
    
    def _create_research_plan(self, query: str) -> ResearchPlan:
        """Create an AI-powered research plan."""
        print("\n🤖 Creating AI-powered research plan...")
        
        # Include context in planning
        context_info = ""
        if self.context:
            context_info = f"\nPrevious context:\n{json.dumps(self.context, indent=2)[:1000]}\n"
        
        planning_prompt = f"""
        Analyze this query and create a research plan: "{query}"
        {context_info}
        
        Available capabilities:
        1. Search YouTube viewing history database (20,000+ videos)
        2. Fetch YouTube video transcripts
        3. Search the web for current information
        4. Analyze patterns in viewing history
        
        Determine:
        1. What is the primary intent?
        2. What data sources are needed?
        3. What analysis should be performed?
        
        Return a JSON structure:
        {{
            "intent": "primary intent",
            "requires_history": true/false,
            "requires_web_search": true/false,
            "requires_transcript": true/false,
            "steps": [
                {{"phase": "gather", "action": "...", "details": "..."}},
                {{"phase": "analyze", "action": "...", "details": "..."}}
            ],
            "expected_output": "description of output"
        }}
        """
        
        try:
            response = self.planner.run(planning_prompt)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                plan = ResearchPlan(
                    query=query,
                    intent=plan_data.get("intent", "general"),
                    requires_history=plan_data.get("requires_history", True),
                    requires_web_search=plan_data.get("requires_web_search", False),
                    requires_transcript=plan_data.get("requires_transcript", False),
                    steps=plan_data.get("steps", []),
                    expected_output=plan_data.get("expected_output", "Analysis results")
                )
                
                # Display plan
                print(f"\n📋 Research Plan:")
                print(f"  Intent: {plan.intent}")
                print(f"  Data sources: history={plan.requires_history}, web={plan.requires_web_search}, transcript={plan.requires_transcript}")
                if plan.steps:
                    print("  Steps:")
                    for step in plan.steps:
                        print(f"    - [{step['phase']}] {step['action']}: {step['details']}")
                
                return plan
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"⚠️ Using default plan: {e}")
            # Return default plan
            return ResearchPlan(
                query=query,
                intent="general",
                requires_history=True,
                requires_web_search=False,
                requires_transcript=False
            )
    
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
    
    def general_query(self, query: str) -> None:
        """
        Handle general queries without YouTube-specific processing.
        Just web search and AI response.
        """
        print(f"\n💬 Processing: {query}")
        print("=" * 60)
        
        # Check if it needs web search
        search_keywords = ['search', 'find', 'look up', 'news', 'latest', 'current', 'today']
        needs_search = any(keyword in query.lower() for keyword in search_keywords)
        
        if needs_search:
            print("\n🌐 Searching the web...")
            search_results = self.web_agent.run(json.dumps({
                "query": query,
                "context": "General web search"
            }))
            
            # Generate response with search results
            response_prompt = f"""
            User query: "{query}"
            
            Web search results:
            {search_results[:2000]}
            
            Provide a helpful, direct response to their query based on the search results.
            Be concise and informative.
            """
            
            # Create fresh agent context for the response
            fresh_agent = Agent(
                roles=["You are a helpful AI assistant."],
                overrides={"llm": {"provider": "anthropic"}}
            )
            response = fresh_agent.run(response_prompt)
            print("\n" + response)
        else:
            # Create fresh agent context for clean response
            fresh_agent = Agent(
                roles=["You are a helpful AI assistant."],
                overrides={"llm": {"provider": "anthropic"}}
            )
            response = fresh_agent.run(query)
            print("\n" + response)
    
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
    
    def analyze_with_planning(self, prompt: str) -> None:
        """
        Analyze with AI-powered planning for better results.
        """
        print(f"\n🔍 Advanced Analysis: {prompt}")
        print("=" * 60)
        
        # Create research plan
        plan = self._create_research_plan(prompt)
        
        results = {}
        
        # Execute plan based on requirements
        if plan.requires_history:
            print("\n📺 Searching viewing history...")
            results['history'] = self._analyze_history(prompt)
        
        if plan.requires_web_search:
            print("\n🌐 Searching the web...")
            try:
                web_results = self.web_agent.run(json.dumps({
                    "query": prompt,
                    "context": "Research for YouTube content analysis"
                }))
                results['web'] = {"success": True, "data": web_results}
            except Exception as e:
                results['web'] = {"success": False, "error": str(e)}
        
        if plan.requires_transcript:
            # Extract video ID if present
            import re
            video_id_match = re.search(r'([a-zA-Z0-9_-]{11})', prompt)
            if video_id_match:
                video_id = video_id_match.group(1)
                print(f"\n📝 Fetching transcript for {video_id}...")
                transcript = self._fetch_transcript(video_id)
                if transcript:
                    results['transcript'] = transcript
        
        # Enhanced synthesis
        print("\n🧠 Synthesizing results...")
        synthesis = self._enhanced_synthesis(plan, results)
        
        print("\n" + "=" * 60)
        print("📊 RESEARCH RESULTS")
        print("=" * 60)
        print(synthesis)
        
        # Save context for future use
        self._save_context({
            "query": prompt,
            "plan": {
                "intent": plan.intent,
                "requires_history": plan.requires_history,
                "requires_web_search": plan.requires_web_search,
                "requires_transcript": plan.requires_transcript,
                "expected_output": plan.expected_output
            },
            "results_summary": {
                "history_matches": len(results.get('history', {}).get('matches', [])),
                "web_search_performed": plan.requires_web_search,
                "transcript_fetched": plan.requires_transcript
            },
            "timestamp": datetime.now().isoformat()
        })
    
    def _enhanced_synthesis(self, plan: ResearchPlan, results: Dict) -> str:
        """Enhanced synthesis with better prompting."""
        synthesis_prompt = f"""
        Query: "{plan.query}"
        Intent: {plan.intent}
        Expected Output: {plan.expected_output}
        
        {f"Previous Context: {json.dumps(self.context, indent=2)[:500]}" if self.context else ""}
        
        Research Results:
        """
        
        if 'history' in results:
            history = results['history']
            matches = history.get('matches', [])
            synthesis_prompt += f"""
        
        Viewing History Analysis:
        - Total videos in database: {history.get('stats', {}).get('total_videos', 0)}
        - Relevant matches found: {len(matches)}
        """
            if matches:
                synthesis_prompt += "\n        Top matching videos:\n"
                for i, match in enumerate(matches[:15], 1):
                    synthesis_prompt += f"        {i}. {match['title']}\n"
        
        if 'web' in results and results['web'].get('success'):
            synthesis_prompt += f"""
        
        Web Research:
        {results['web'].get('data', '')[:1500]}
        """
        
        if 'transcript' in results:
            synthesis_prompt += f"""
        
        Video Transcript (excerpt):
        {results.get('transcript', '')[:1000]}
        """
        
        synthesis_prompt += """
        
        Based on the research, provide a comprehensive response that:
        1. Directly answers the user's question with specific examples
        2. Uses the actual video titles and data from the results
        3. Provides actionable recommendations
        4. Is well-structured with clear sections
        5. Focuses on practical value for the user
        
        Format with markdown for clarity. Be specific, not generic.
        """
        
        return self.agent.run(synthesis_prompt)
    
    def analyze_and_research(self, prompt: str) -> None:
        """
        Analyze viewing history and optionally conduct web research based on prompt.
        Enhanced version with planning.
        """
        # Use the new planning-based approach
        self.analyze_with_planning(prompt)
        return
        
        # Step 1: Analyze viewing history
        print("\n📺 Searching your YouTube history database...")
        history_data = self._analyze_history(prompt)
        
        # Check if this is purely a viewing history query
        history_keywords = ['watched', 'viewed', 'my history', 'have i', 'did i', 'show me', 'tell me what']
        is_history_query = any(keyword in prompt.lower() for keyword in history_keywords)
        
        # Only do web research if it's NOT a pure history query
        research_results = {}
        if not is_history_query:
            # Step 2: Extract topics to research
            topics = self._extract_research_topics(prompt, history_data)
            if topics:
                print(f"\n🎯 Topics to research: {', '.join(topics)}")
                
                # Step 3: Web research for each topic
                print("\n🌐 Conducting web research...")
                for topic in topics[:3]:
                    print(f"  Researching: {topic}")
                    results = self.web_agent.run(json.dumps({
                        "query": topic,
                        "context": prompt
                    }))
                    research_results[topic] = results
                    time.sleep(1)
        
        # Step 4: Combined analysis
        print("\n🧠 Generating analysis...")
        analysis = self._comprehensive_analysis(prompt, history_data, research_results)
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS RESULTS")
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
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'could', 'need', 'i', 'me', 
                     'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                     'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 
                     'them', 'their', 'theirs', 'themselves', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'watched', 'videos', 'youtube', 'give', 'list',
                     'complete', 'show', 'tell'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    def _analyze_history(self, prompt: str) -> Dict:
        """Analyze viewing history based on prompt."""
        cursor = self.conn.cursor()
        data = {'matches': [], 'stats': {}}
        
        # Check if asking about AI videos specifically
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'claude', 'gpt', 
                       'llm', 'neural', 'deep learning', 'langchain', 'agent', 'coding']
        
        # Don't treat general words as AI queries
        prompt_lower = prompt.lower()
        is_ai_query = any(keyword in prompt_lower for keyword in ai_keywords) and \
                     not any(skip in prompt_lower for skip in ['wait', 'said', 'afraid', 'mail'])
        
        if is_ai_query:
            # Search for AI videos by actual keywords in titles
            # The ai_score field has false positives, so we search directly
            ai_search_terms = ['claude', 'gpt', 'chatgpt', 'langchain', 'llm', 
                              'machine learning', 'neural', 'ai agent', 'artificial intelligence',
                              'deep learning', 'openai', 'anthropic', 'gemini', 'copilot']
            
            all_ai_videos = []
            for term in ai_search_terms:
                cursor.execute("""
                    SELECT title, channel, url, ai_score, categories
                    FROM videos
                    WHERE LOWER(title) LIKE ?
                    ORDER BY title
                    LIMIT 50
                """, (f'%{term}%',))
                
                videos = [dict(row) for row in cursor.fetchall()]
                for video in videos:
                    # Avoid duplicates
                    if not any(v['url'] == video['url'] for v in all_ai_videos):
                        all_ai_videos.append(video)
            
            # Also search for general "AI" but filter out false positives
            cursor.execute("""
                SELECT title, channel, url, ai_score, categories
                FROM videos
                WHERE LOWER(title) LIKE '%ai %' OR LOWER(title) LIKE '% ai%'
                ORDER BY title
                LIMIT 200
            """)
            
            general_ai = [dict(row) for row in cursor.fetchall()]
            
            # Filter out likely false positives (music, non-tech content)
            exclude_keywords = ['said', 'wait', 'afraid', 'rain', 'paid', 'daily', 'hair', 'fair', 'mail']
            for video in general_ai:
                title_lower = video['title'].lower()
                # Check if it's likely a real AI video
                if not any(exclude in title_lower for exclude in exclude_keywords):
                    if not any(v['url'] == video['url'] for v in all_ai_videos):
                        all_ai_videos.append(video)
            
            data['matches'] = all_ai_videos[:200]  # Limit to 200 videos
            data['stats']['total_ai_videos'] = len(all_ai_videos)
            
            # Also get some specific keyword matches
            for keyword in ['claude', 'gpt', 'langchain', 'ai', 'agent', 'llm']:
                if keyword in prompt.lower():
                    cursor.execute("""
                        SELECT title, channel, url, ai_score, categories
                        FROM videos
                        WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
                        ORDER BY ai_score DESC
                        LIMIT 20
                    """, (f'%{keyword}%', f'%{keyword}%'))
                    
                    specific_matches = [dict(row) for row in cursor.fetchall()]
                    # Add to matches if not already there
                    for match in specific_matches:
                        if not any(m['url'] == match['url'] for m in data['matches']):
                            data['matches'].append(match)
        else:
            # General keyword search - look for specific patterns
            
            # First, try to find exact artist/creator matches
            # Handle special cases like "AC/DC" 
            special_searches = []
            if "ac/dc" in prompt_lower or "ac dc" in prompt_lower:
                special_searches.append("AC/DC")
            
            # Extract keywords but keep important phrases together
            keywords = self._extract_keywords(prompt)
            
            # Also check for multi-word phrases in the original prompt
            # e.g., "Pink Floyd", "Led Zeppelin"
            words = prompt.split()
            for i in range(len(words) - 1):
                two_word = f"{words[i]} {words[i+1]}"
                if len(two_word) > 4 and two_word[0].isupper():  # Likely a name
                    special_searches.append(two_word)
            
            # Search for special terms first
            for term in special_searches:
                cursor.execute("""
                    SELECT title, channel, url, ai_score, categories
                    FROM videos
                    WHERE title LIKE ?
                    ORDER BY title
                    LIMIT 20
                """, (f'%{term}%',))
                
                matches = [dict(row) for row in cursor.fetchall()]
                for match in matches:
                    if not any(m['url'] == match['url'] for m in data['matches']):
                        data['matches'].append(match)
            
            # Then search for individual keywords
            for keyword in keywords[:5]:
                # Skip very short keywords that might cause false positives
                if len(keyword) < 3:
                    continue
                    
                cursor.execute("""
                    SELECT title, channel, url, ai_score, categories
                    FROM videos
                    WHERE LOWER(title) LIKE ?
                    ORDER BY title
                    LIMIT 10
                """, (f'%{keyword}%',))
                
                matches = [dict(row) for row in cursor.fetchall()]
                for match in matches:
                    if not any(m['url'] == match['url'] for m in data['matches']):
                        data['matches'].append(match)
        
        # Get general stats
        cursor.execute("SELECT COUNT(*) FROM videos")
        data['stats']['total_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
        data['stats']['ai_videos'] = cursor.fetchone()[0]
        
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
        
        # If we have matches, format them nicely
        total_matches = len(history_data.get('matches', []))
        ai_video_count = history_data.get('stats', {}).get('ai_videos', 0)
        
        if total_matches > 0:
            # Format top videos
            top_videos = []
            for video in history_data['matches'][:20]:
                top_videos.append(f"- {video['title']} (Score: {video.get('ai_score', 0):.2f})")
            
            video_list = "\n".join(top_videos)
            
            analysis_prompt = f"""
            User asked: "{prompt}"
            
            Database Statistics:
            - Total videos in history: {history_data.get('stats', {}).get('total_videos', 0)}
            - AI/coding videos: {ai_video_count}
            - Matches found: {total_matches}
            
            Top Matching Videos:
            {video_list}
            
            {"Web Research:" + json.dumps(research, indent=2)[:500] if research else ""}
            
            Provide a clear, helpful response that:
            1. Directly answers their question about their viewing history
            2. Lists specific videos they've watched (use the titles provided)
            3. Identifies patterns in their viewing
            4. {"Includes web research findings" if research else "Provides insights based on their history"}
            5. Offers personalized recommendations based on what they've watched
            
            Be specific and reference actual video titles from their history.
            """
        else:
            analysis_prompt = f"""
            User asked: "{prompt}"
            
            No matching videos found in their history for this query.
            Total videos in database: {history_data.get('stats', {}).get('total_videos', 0)}
            AI videos in database: {ai_video_count}
            
            {"Web Research:" + json.dumps(research, indent=2)[:500] if research else ""}
            
            Provide a helpful response explaining:
            1. That no matching videos were found for their specific query
            2. Suggest they might want to explore this topic
            3. {"Include web research findings" if research else "Offer alternative search suggestions"}
            4. Recommend related content they might find interesting
            """
        
        return self.agent.run(analysis_prompt)


def smart_route(cli: YouTubeResearchCLI, query: str, use_planning: bool = True) -> None:
    """
    Intelligently route a query to the appropriate command using AI.
    Enhanced with planning capabilities.
    """
    # First, quick pattern matching for obvious cases
    if "youtube.com/watch" in query or "youtu.be/" in query:
        print("🎥 Detected YouTube URL - researching video...\n")
        import re
        url_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^\s]+)'
        match = re.search(url_pattern, query)
        if match:
            cli.research_video(match.group(1))
            return
    
    # Use AI to understand intent and determine action
    routing_prompt = f"""
    Analyze this user input: "{query}"
    
    Determine the user's intent and what they need:
    
    1. If it's a greeting or casual message (hello, hi, hey, thanks, etc.) -> respond with: "greeting"
    2. If asking about their YouTube viewing history -> respond with: "analyze" 
    3. If asking what to learn or for recommendations based on their history -> respond with: "analyze"
    4. If it's a learning topic they want to study (programming, AI, etc.) -> respond with: "research-topic"
    5. If contains a YouTube URL or video ID -> respond with: "research-video"
    6. For any other question, request, or query -> respond with: "general"
    
    Return ONLY ONE WORD from: greeting, analyze, research-topic, research-video, general
    
    Examples:
    - "hello" -> greeting
    - "What Python videos have I watched?" -> analyze
    - "machine learning fundamentals" -> research-topic
    - "LangChain tutorials" -> research-topic
    - "What should I learn next?" -> analyze
    - "https://youtube.com/watch?v=abc" -> research-video
    - "search the web for today's news" -> general
    - "what's the weather?" -> general
    - "explain quantum computing" -> general
    - "write a python script" -> general
    """
    
    # Determine action
    action = cli.agent.run(routing_prompt).strip().lower()
    
    # Clean up action response
    if "greeting" in action:
        print("👋 Hello! I'm your YouTube Research Assistant.\n")
        print("I can help you:")
        print("  📺 Analyze your YouTube viewing history")
        print("  🔍 Research topics with web search")
        print("  📝 Fetch and analyze video transcripts")
        print("  💡 Provide learning recommendations\n")
        print("Try asking:")
        print('  yr "What AI videos have I watched?"')
        print('  yr "prompt engineering"')
        print('  yr "https://youtube.com/watch?v=VIDEO_ID"')
        return
    
    elif "general" in action:
        # Handle general queries without YouTube-specific processing
        cli.general_query(query)
        return
    
    elif "unclear" in action:
        print(f"🤔 I'm not sure what you want to do with: '{query}'\n")
        print("Try being more specific:")
        print("  - Ask a question: 'What videos about X have I watched?'")
        print("  - Name a topic: 'machine learning'")
        print("  - Provide a URL: 'https://youtube.com/watch?v=...'")
        return
    
    elif "research-video" in action:
        print("🎥 Researching video...\n")
        # Extract video ID
        import re
        video_id_pattern = r'([a-zA-Z0-9_-]{11})'
        match = re.search(video_id_pattern, query)
        if match:
            cli.research_video(match.group(1))
        else:
            print("❌ Could not extract video ID from query")
    
    elif "research-topic" in action:
        print(f"🔍 Researching topic: {query}\n")
        cli.research_topic(query)
    
    elif "analyze" in action:
        print(f"📊 Analyzing your YouTube history...\n")
        if use_planning:
            cli.analyze_with_planning(query)
        else:
            cli.analyze_and_research(query)
    
    else:
        # Fallback to pattern matching
        if "?" in query or any(word in query.lower() for word in ["what", "how", "why", "should", "have i", "did i", "watched", "viewed"]):
            print(f"📊 Analyzing based on your question...\n")
            if use_planning:
                cli.analyze_with_planning(query)
            else:
                cli.analyze_and_research(query)
        elif any(word in query.lower() for word in ["learn", "tutorial", "course", "study"]):
            print(f"🔍 Researching learning topic: {query}\n")
            cli.research_topic(query)
        else:
            # Default to general query handler
            cli.general_query(query)


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