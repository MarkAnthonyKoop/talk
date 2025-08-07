#!/usr/bin/env python3
"""
YouTube Research CLI - Enhanced with context loading and AI planning

Features:
- Loads context from .talk/yr/ directory
- AI-powered planning with List[Step]
- Improved response quality
- Better structured reasoning
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
    from plan_runner.step import Step
    from plan_runner.plan_runner import PlanRunner
except ImportError:
    # Add project root to path for development
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from agent.agent import Agent
    from special_agents.research_agents.web_search_agent import WebSearchAgent
    from plan_runner.step import Step
    from plan_runner.plan_runner import PlanRunner


@dataclass
class ResearchPlan:
    """Represents a research plan with steps."""
    query: str
    steps: List[Step] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    

class YouTubeResearchCLIEnhanced:
    """Enhanced YouTube CLI with context loading and AI planning."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the enhanced research CLI."""
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
        
        # Initialize agents with context awareness
        system_prompts = [
            "You are an expert YouTube content analyst and researcher.",
            "You analyze viewing history and conduct external research.",
            "You synthesize information from multiple sources.",
            "You provide comprehensive insights and recommendations.",
            "Think step by step and create detailed plans.",
            "Be specific and actionable in your responses."
        ]
        
        # Add context to system prompts if available
        if self.context:
            system_prompts.append(f"Previous context available: {json.dumps(self.context, indent=2)[:500]}")
        
        self.agent = Agent(
            roles=system_prompts,
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Planning agent for creating execution plans
        self.planner = Agent(
            roles=[
                "You are an expert at breaking down complex research tasks into actionable steps.",
                "You create detailed, logical plans that can be executed sequentially or in parallel.",
                "You understand YouTube content analysis, web research, and data synthesis.",
                "You always think about the most efficient way to gather and present information."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        self.web_agent = WebSearchAgent(max_results=5)
        self.transcript_cache = {}
        
        # Save conversation for future context
        self.conversation_id = str(uuid.uuid4())[:8]
        self.conversation_log = []
    
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
        Create a detailed research plan for this query: "{query}"
        {context_info}
        
        Available capabilities:
        1. Search YouTube viewing history database (20,000+ videos)
        2. Fetch YouTube video transcripts
        3. Search the web for current information
        4. Analyze patterns in viewing history
        5. Generate learning paths and recommendations
        
        Based on the query, identify:
        1. What specific information is being requested?
        2. What data sources should be queried (history, transcripts, web)?
        3. What analysis or synthesis is needed?
        4. What format would be most helpful for the response?
        
        Create a step-by-step plan with these phases:
        - Understanding: What exactly does the user want?
        - Gathering: What data do we need to collect?
        - Analysis: How should we process and analyze the data?
        - Synthesis: How should we present the findings?
        
        Return a JSON structure with:
        {{
            "intent": "primary intent of the query",
            "requires_history": true/false,
            "requires_web_search": true/false,
            "requires_transcript": true/false,
            "steps": [
                {{"phase": "understanding", "action": "...", "details": "..."}},
                {{"phase": "gathering", "action": "...", "details": "..."}},
                {{"phase": "analysis", "action": "...", "details": "..."}},
                {{"phase": "synthesis", "action": "...", "details": "..."}}
            ],
            "expected_output": "description of what the final output should contain"
        }}
        """
        
        try:
            response = self.planner.run(planning_prompt)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                # Fallback to basic plan
                plan_data = {
                    "intent": "general research",
                    "requires_history": True,
                    "requires_web_search": False,
                    "requires_transcript": False,
                    "steps": [
                        {"phase": "gathering", "action": "search_history", "details": "Search viewing history"},
                        {"phase": "synthesis", "action": "summarize", "details": "Summarize findings"}
                    ],
                    "expected_output": "Summary of findings"
                }
            
            # Convert to Step objects
            steps = []
            for step_data in plan_data.get("steps", []):
                step = Step(
                    label=f"{step_data['phase']}_{step_data['action']}",
                    agent_key=step_data['action']
                )
                steps.append(step)
            
            plan = ResearchPlan(
                query=query,
                steps=steps,
                context=plan_data
            )
            
            # Display plan
            print("\n📋 Research Plan:")
            print(f"  Intent: {plan_data.get('intent', 'unknown')}")
            print(f"  Requires history: {plan_data.get('requires_history', False)}")
            print(f"  Requires web search: {plan_data.get('requires_web_search', False)}")
            print(f"  Requires transcript: {plan_data.get('requires_transcript', False)}")
            print("\n  Steps:")
            for step in plan_data.get("steps", []):
                print(f"    - [{step['phase']}] {step['action']}: {step['details']}")
            print(f"\n  Expected output: {plan_data.get('expected_output', 'Analysis results')}")
            
            return plan
            
        except Exception as e:
            print(f"⚠️ Plan creation failed: {e}")
            # Return default plan
            return ResearchPlan(
                query=query,
                steps=[
                    Step(label="analyze", agent_key="analyze"),
                    Step(label="synthesize", agent_key="synthesize")
                ],
                context={"intent": "general", "error": str(e)}
            )
    
    def _execute_research_plan(self, plan: ResearchPlan) -> str:
        """Execute a research plan and return results."""
        print("\n🚀 Executing research plan...")
        
        results = {}
        
        # Execute based on plan context
        if plan.context.get("requires_history", False):
            print("\n📺 Searching viewing history...")
            history_data = self._analyze_history(plan.query)
            results['history'] = history_data
        
        if plan.context.get("requires_web_search", False):
            print("\n🌐 Searching the web...")
            web_results = self._perform_web_research(plan.query)
            results['web'] = web_results
        
        if plan.context.get("requires_transcript", False):
            print("\n📝 Fetching transcripts...")
            # Extract video ID if present in query
            video_id = self._extract_video_id_from_query(plan.query)
            if video_id:
                transcript = self._fetch_transcript(video_id)
                results['transcript'] = transcript
        
        # Synthesize results
        synthesis = self._synthesize_results(plan, results)
        
        # Save to context for future conversations
        self._save_context({
            "query": plan.query,
            "plan": plan.context,
            "results": results,
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat()
        })
        
        return synthesis
    
    def _perform_web_research(self, query: str) -> Dict:
        """Perform web research for a query."""
        try:
            results = self.web_agent.run(json.dumps({
                "query": query,
                "context": "Research for YouTube content analysis"
            }))
            return {"success": True, "data": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_video_id_from_query(self, query: str) -> Optional[str]:
        """Extract video ID from query if present."""
        import re
        patterns = [
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'youtu\.be/([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})'  # Just the ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None
    
    def _synthesize_results(self, plan: ResearchPlan, results: Dict) -> str:
        """Synthesize all results into a comprehensive response."""
        print("\n🧠 Synthesizing results...")
        
        # Build synthesis prompt with all context
        synthesis_prompt = f"""
        Query: "{plan.query}"
        
        Research Plan Intent: {plan.context.get('intent', 'general research')}
        Expected Output: {plan.context.get('expected_output', 'comprehensive analysis')}
        
        {f"Previous Context: {json.dumps(self.context, indent=2)[:500]}" if self.context else ""}
        
        Research Results:
        """
        
        if 'history' in results:
            history = results['history']
            synthesis_prompt += f"""
        
        Viewing History Analysis:
        - Total videos in database: {history.get('stats', {}).get('total_videos', 0)}
        - Matches found: {len(history.get('matches', []))}
        - Top matches: {json.dumps([m['title'] for m in history.get('matches', [])[:10]], indent=2)}
        """
        
        if 'web' in results and results['web'].get('success'):
            synthesis_prompt += f"""
        
        Web Research:
        {results['web'].get('data', 'No data')[:1500]}
        """
        
        if 'transcript' in results:
            synthesis_prompt += f"""
        
        Video Transcript (excerpt):
        {results.get('transcript', '')[:1000]}
        """
        
        synthesis_prompt += """
        
        Based on all the above information, provide a comprehensive, well-structured response that:
        1. Directly answers the user's query
        2. Provides specific examples and evidence from the data
        3. Offers insights and patterns discovered
        4. Suggests next steps or recommendations
        5. Is organized with clear sections and bullet points
        
        Be thorough but concise. Use the actual data provided above, not generic statements.
        Format with markdown for clarity.
        """
        
        response = self.agent.run(synthesis_prompt)
        return response
    
    def analyze_with_planning(self, query: str) -> None:
        """Main entry point for AI-planned research."""
        print(f"\n🔍 Advanced Analysis: {query}")
        print("=" * 60)
        
        # Create research plan
        plan = self._create_research_plan(query)
        
        # Execute plan
        results = self._execute_research_plan(plan)
        
        print("\n" + "=" * 60)
        print("📊 RESEARCH RESULTS")
        print("=" * 60)
        print(results)
    
    def _analyze_history(self, prompt: str) -> Dict:
        """Analyze viewing history based on prompt."""
        cursor = self.conn.cursor()
        data = {'matches': [], 'stats': {}}
        
        # Check if asking about AI videos specifically
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'claude', 'gpt', 
                       'llm', 'neural', 'deep learning', 'langchain', 'agent', 'coding']
        
        prompt_lower = prompt.lower()
        is_ai_query = any(keyword in prompt_lower for keyword in ai_keywords) and \
                     not any(skip in prompt_lower for skip in ['wait', 'said', 'afraid', 'mail'])
        
        if is_ai_query:
            # Search for AI videos by actual keywords in titles
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
                    if not any(v['url'] == video['url'] for v in all_ai_videos):
                        all_ai_videos.append(video)
            
            data['matches'] = all_ai_videos[:200]
            data['stats']['total_ai_videos'] = len(all_ai_videos)
        else:
            # General keyword search
            keywords = self._extract_keywords(prompt)
            
            for keyword in keywords[:5]:
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
        
        return data
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'could', 'need', 'i', 'me',
                     'watched', 'videos', 'youtube', 'give', 'list', 'show', 'tell'}
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    def _fetch_transcript(self, video_id: str) -> Optional[str]:
        """Fetch transcript from YouTube."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)
            full_text = ' '.join([segment.text for segment in transcript_list])
            
            self.transcript_cache[video_id] = full_text
            return full_text
            
        except Exception as e:
            print(f"❌ Could not fetch transcript: {e}")
            return None


def smart_route_enhanced(cli: YouTubeResearchCLIEnhanced, query: str) -> None:
    """Enhanced smart routing with AI planning."""
    # Quick pattern matching for obvious cases
    if "youtube.com/watch" in query or "youtu.be/" in query:
        print("🎥 Detected YouTube URL - researching video...\n")
        cli.analyze_with_planning(query)
        return
    
    # For everything else, use AI planning
    cli.analyze_with_planning(query)


def main():
    """Run the enhanced YouTube Research CLI."""
    import sys
    
    parser = argparse.ArgumentParser(
        description="YouTube Research CLI - Enhanced with AI Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('query', nargs='*', help='Your query')
    parser.add_argument('--db', default='youtube_fast.db', help='Database path')
    
    args = parser.parse_args()
    
    if args.query:
        query = ' '.join(args.query)
        cli = YouTubeResearchCLIEnhanced(args.db)
        smart_route_enhanced(cli, query)
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())