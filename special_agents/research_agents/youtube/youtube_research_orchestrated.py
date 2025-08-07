#!/usr/bin/env python3
"""
YouTube Research Orchestrated - Full Talk Framework Integration

This version uses the complete Talk framework with:
- Step objects for structured execution
- Parallel processing capabilities
- Agent orchestration
- Error handling with fallbacks
"""

import sys
import sqlite3
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner
from special_agents.research_agents.web_search_agent import WebSearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class YouTubeStep(Step):
    """Extended Step for YouTube research operations."""
    data: Dict[str, Any] = field(default_factory=dict)
    result_key: str = ""
    dependencies: List[str] = field(default_factory=list)
    

class YouTubeAgentRegistry:
    """Registry of specialized agents for YouTube research."""
    
    def __init__(self, db_path: str, context: Dict[str, Any] = None):
        """Initialize agent registry with database and context."""
        self.db_path = db_path
        self.context = context or {}
        self.agents = {}
        self.results = {}
        
        # Initialize database connection
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create all specialized agents
        self._create_agents()
    
    def _create_agents(self):
        """Create all specialized agents for different tasks."""
        
        # Planning Agent - Creates research plans
        self.agents['planner'] = Agent(
            roles=[
                "You are an expert research planner for YouTube content analysis.",
                "You break down complex queries into actionable, parallel steps.",
                "You identify dependencies between tasks.",
                "You optimize for efficiency and comprehensiveness.",
                f"Context: {json.dumps(self.context)[:500]}" if self.context else ""
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # History Analyzer - Searches viewing history
        self.agents['history_analyzer'] = Agent(
            roles=[
                "You analyze YouTube viewing history data.",
                "You identify patterns and trends in viewing behavior.",
                "You extract meaningful insights from video titles and metadata.",
                "You provide specific, data-driven analysis."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Transcript Analyzer - Processes video transcripts
        self.agents['transcript_analyzer'] = Agent(
            roles=[
                "You analyze YouTube video transcripts.",
                "You extract key topics, concepts, and insights.",
                "You identify technical content and learning points.",
                "You summarize complex information clearly."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Web Researcher - Performs web research
        self.agents['web_researcher'] = WebSearchAgent(max_results=5)
        
        # Synthesizer - Combines all results
        self.agents['synthesizer'] = Agent(
            roles=[
                "You synthesize research from multiple sources.",
                "You create comprehensive, well-structured reports.",
                "You provide actionable recommendations.",
                "You focus on practical value and clarity.",
                "You use markdown formatting for readability."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Pattern Detector - Finds patterns in data
        self.agents['pattern_detector'] = Agent(
            roles=[
                "You identify patterns and trends in data.",
                "You find connections between seemingly unrelated items.",
                "You detect anomalies and interesting outliers.",
                "You provide statistical insights."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
    
    def execute_step(self, step: YouTubeStep) -> Dict[str, Any]:
        """Execute a single step with the appropriate agent."""
        agent_key = step.agent_key
        
        if agent_key not in self.agents:
            logger.warning(f"Unknown agent key: {agent_key}")
            return {"error": f"Unknown agent: {agent_key}"}
        
        agent = self.agents[agent_key]
        
        try:
            # Prepare input based on step type
            if agent_key == 'history_analyzer':
                result = self._analyze_history(step.data.get('query', ''))
            elif agent_key == 'transcript_analyzer':
                result = self._analyze_transcript(step.data.get('video_id', ''))
            elif agent_key == 'web_researcher':
                result = self._research_web(step.data.get('query', ''))
            elif agent_key == 'pattern_detector':
                result = self._detect_patterns(step.data)
            elif agent_key == 'synthesizer':
                result = self._synthesize(step.data)
            else:
                # Generic agent execution
                prompt = step.data.get('prompt', json.dumps(step.data))
                result = agent.run(prompt)
            
            # Store result if key specified
            if step.result_key:
                self.results[step.result_key] = result
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_history(self, query: str) -> Dict[str, Any]:
        """Analyze viewing history for relevant videos."""
        cursor = self.conn.cursor()
        
        # Extract keywords for search
        keywords = self._extract_keywords(query)
        
        all_matches = []
        for keyword in keywords[:10]:
            cursor.execute("""
                SELECT title, channel, url, ai_score, categories
                FROM videos
                WHERE LOWER(title) LIKE ?
                ORDER BY title
                LIMIT 20
            """, (f'%{keyword}%',))
            
            matches = [dict(row) for row in cursor.fetchall()]
            for match in matches:
                if not any(m['url'] == match['url'] for m in all_matches):
                    all_matches.append(match)
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        return {
            "query": query,
            "matches": all_matches[:100],
            "total_matches": len(all_matches),
            "total_videos": total_videos,
            "keywords_used": keywords
        }
    
    def _analyze_transcript(self, video_id: str) -> Dict[str, Any]:
        """Fetch and analyze video transcript."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)
            full_text = ' '.join([segment.text for segment in transcript_list])
            
            # Analyze with agent
            analysis_prompt = f"""
            Analyze this video transcript and extract:
            1. Main topics covered
            2. Key technical concepts
            3. Actionable insights
            4. Learning points
            
            Transcript (first 3000 chars):
            {full_text[:3000]}
            """
            
            analysis = self.agents['transcript_analyzer'].run(analysis_prompt)
            
            return {
                "video_id": video_id,
                "transcript_length": len(full_text),
                "analysis": analysis,
                "excerpt": full_text[:1000]
            }
            
        except Exception as e:
            return {
                "video_id": video_id,
                "error": str(e),
                "transcript_available": False
            }
    
    def _research_web(self, query: str) -> Dict[str, Any]:
        """Perform web research."""
        try:
            results = self.agents['web_researcher'].run(json.dumps({
                "query": query,
                "context": "YouTube content research"
            }))
            return {
                "query": query,
                "results": results,
                "success": True
            }
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "success": False
            }
    
    def _detect_patterns(self, data: Dict) -> Dict[str, Any]:
        """Detect patterns in the provided data."""
        prompt = f"""
        Analyze this data and identify:
        1. Viewing patterns and trends
        2. Topic clusters
        3. Temporal patterns
        4. Interesting anomalies
        
        Data:
        {json.dumps(data)[:3000]}
        """
        
        patterns = self.agents['pattern_detector'].run(prompt)
        return {"patterns": patterns, "data_points": len(data.get('matches', []))}
    
    def _synthesize(self, data: Dict) -> str:
        """Synthesize all research results."""
        prompt = f"""
        Synthesize this research data into a comprehensive report:
        
        Query: {data.get('query', 'Unknown')}
        
        History Analysis:
        {json.dumps(data.get('history', {}))[:2000]}
        
        Pattern Analysis:
        {json.dumps(data.get('patterns', {}))[:1000]}
        
        Web Research:
        {json.dumps(data.get('web', {}))[:1000]}
        
        Transcript Analysis:
        {json.dumps(data.get('transcript', {}))[:1000]}
        
        Create a well-structured report with:
        1. Executive summary
        2. Key findings
        3. Detailed analysis
        4. Actionable recommendations
        5. Next steps
        
        Use markdown formatting for clarity.
        """
        
        return self.agents['synthesizer'].run(prompt)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        import re
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'how', 'what', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'can', 'need', 'watched', 'videos'}
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:15]


class YouTubeResearchOrchestrator:
    """Orchestrates complex YouTube research with parallel execution."""
    
    def __init__(self, db_path: str = "youtube_fast.db"):
        """Initialize the orchestrator."""
        # Find database
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
            raise FileNotFoundError(f"Database not found in any location: {db_locations}")
        
        # Load context
        self.context = self._load_context()
        self.conversation_id = str(uuid.uuid4())[:8]
        
        # Initialize agent registry
        self.registry = YouTubeAgentRegistry(self.db_path, self.context)
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_context(self) -> Dict[str, Any]:
        """Load context from .talk/yr/ directory."""
        context = {}
        context_dir = Path.cwd() / ".talk" / "yr"
        
        if context_dir.exists():
            json_files = sorted(
                glob.glob(str(context_dir / "*_*.json")),
                key=os.path.getmtime,
                reverse=True
            )
            
            if json_files and len(json_files) > 0:
                try:
                    with open(json_files[0], 'r') as f:
                        context = json.load(f)
                    logger.info(f"Loaded context from: {Path(json_files[0]).name}")
                except Exception as e:
                    logger.warning(f"Could not load context: {e}")
        
        return context
    
    def _save_context(self, data: Dict[str, Any]):
        """Save context for future conversations."""
        context_dir = Path.cwd() / ".talk" / "yr"
        context_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = context_dir / f"orchestrated_{timestamp}_{self.conversation_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Context saved to: {filename.name}")
    
    def create_research_plan(self, query: str) -> List[Step]:
        """Create an AI-powered research plan with Step objects."""
        print("\n🤖 Creating orchestrated research plan...")
        
        planning_prompt = f"""
        Create a research plan for: "{query}"
        
        Previous context: {json.dumps(self.context)[:500] if self.context else "None"}
        
        Analyze the query and determine:
        1. What specific information is needed?
        2. Which data sources should be queried?
        3. What can be done in parallel?
        4. What depends on other results?
        
        Return a JSON structure with:
        {{
            "intent": "primary intent",
            "parallel_tasks": [
                {{"task": "search_history", "query": "...", "agent": "history_analyzer"}},
                {{"task": "web_research", "query": "...", "agent": "web_researcher"}}
            ],
            "sequential_tasks": [
                {{"task": "detect_patterns", "depends_on": ["search_history"], "agent": "pattern_detector"}},
                {{"task": "synthesize", "depends_on": ["all"], "agent": "synthesizer"}}
            ],
            "expected_output": "description"
        }}
        """
        
        try:
            response = self.registry.agents['planner'].run(planning_prompt)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON in response")
            
            # Create Step objects
            steps = []
            
            # Create parallel steps
            parallel_steps = []
            for task in plan_data.get('parallel_tasks', []):
                step = YouTubeStep(
                    label=task['task'],
                    agent_key=task.get('agent', 'history_analyzer'),
                    data={'query': task.get('query', query)},
                    result_key=task['task']
                )
                parallel_steps.append(step)
            
            # Add parallel steps as a single step with parallel_steps
            if parallel_steps:
                parallel_step = Step(
                    label="parallel_research",
                    parallel_steps=parallel_steps
                )
                steps.append(parallel_step)
            
            # Add sequential steps
            for task in plan_data.get('sequential_tasks', []):
                step = YouTubeStep(
                    label=task['task'],
                    agent_key=task.get('agent', 'synthesizer'),
                    dependencies=task.get('depends_on', []),
                    result_key=task['task']
                )
                steps.append(step)
            
            # Display plan
            print(f"\n📋 Orchestrated Plan:")
            print(f"  Intent: {plan_data.get('intent', 'research')}")
            print(f"  Parallel tasks: {len(parallel_steps)}")
            for step in parallel_steps:
                print(f"    - {step.label} ({step.agent_key})")
            print(f"  Sequential tasks: {len(plan_data.get('sequential_tasks', []))}")
            for task in plan_data.get('sequential_tasks', []):
                print(f"    - {task['task']} (depends on: {task.get('depends_on', [])})")
            
            return steps
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Fallback plan
            return [
                Step(
                    label="analyze",
                    agent_key="history_analyzer"
                ),
                Step(
                    label="synthesize",
                    agent_key="synthesizer"
                )
            ]
    
    def execute_plan(self, query: str, steps: List[Step]) -> str:
        """Execute the research plan with parallel processing."""
        print("\n🚀 Executing orchestrated research plan...")
        
        results = {"query": query}
        
        for step in steps:
            if step.parallel_steps:
                # Execute parallel steps
                print(f"\n⚡ Executing {len(step.parallel_steps)} tasks in parallel...")
                
                futures = {}
                for parallel_step in step.parallel_steps:
                    if isinstance(parallel_step, YouTubeStep):
                        future = self.executor.submit(
                            self.registry.execute_step,
                            parallel_step
                        )
                        futures[future] = parallel_step.label
                
                # Collect results
                for future in as_completed(futures):
                    label = futures[future]
                    try:
                        result = future.result(timeout=30)
                        results[label] = result
                        print(f"  ✓ Completed: {label}")
                    except Exception as e:
                        logger.error(f"Parallel task {label} failed: {e}")
                        results[label] = {"error": str(e)}
            
            elif isinstance(step, YouTubeStep):
                # Execute sequential step
                print(f"\n📝 Executing: {step.label}")
                
                # Prepare data with dependencies
                if step.dependencies:
                    step.data.update({
                        dep: results.get(dep, {})
                        for dep in step.dependencies
                    })
                
                # Add all results if synthesizing
                if step.agent_key == 'synthesizer':
                    step.data.update(results)
                
                result = self.registry.execute_step(step)
                results[step.label] = result
                print(f"  ✓ Completed: {step.label}")
        
        # Final synthesis
        print("\n🧠 Creating final synthesis...")
        
        synthesis_data = {
            "query": query,
            "history": results.get('search_history', {}).get('result', {}),
            "patterns": results.get('detect_patterns', {}).get('result', {}),
            "web": results.get('web_research', {}).get('result', {}),
            "transcript": results.get('fetch_transcript', {}).get('result', {})
        }
        
        final_synthesis = self.registry._synthesize(synthesis_data)
        
        # Save context
        self._save_context({
            "query": query,
            "results": results,
            "synthesis": final_synthesis,
            "timestamp": datetime.now().isoformat()
        })
        
        return final_synthesis
    
    def research(self, query: str) -> None:
        """Main entry point for orchestrated research."""
        print(f"\n🔬 Orchestrated Research: {query}")
        print("=" * 60)
        
        try:
            # Create plan
            steps = self.create_research_plan(query)
            
            # Execute plan
            synthesis = self.execute_plan(query, steps)
            
            print("\n" + "=" * 60)
            print("📊 ORCHESTRATED RESEARCH RESULTS")
            print("=" * 60)
            print(synthesis)
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            print(f"\n❌ Research failed: {e}")
        
        finally:
            # Clean up
            self.executor.shutdown(wait=False)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YouTube Research Orchestrator - Full Talk Framework Integration"
    )
    
    parser.add_argument('query', nargs='*', help='Your research query')
    parser.add_argument('--db', default='youtube_fast.db', help='Database path')
    
    args = parser.parse_args()
    
    if args.query:
        query = ' '.join(args.query)
        orchestrator = YouTubeResearchOrchestrator(args.db)
        orchestrator.research(query)
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())