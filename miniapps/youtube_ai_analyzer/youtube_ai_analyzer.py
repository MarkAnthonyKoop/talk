#!/usr/bin/env python3
"""
YouTube AI Content Analyzer

A mini application that uses the Talk framework to analyze YouTube history
and identify AI/coding content that could enable better codebase analysis and design.

This app orchestrates multiple agents:
- YoutubeAgent: Extracts and analyzes YouTube takeout data
- WebSearchAgent: Researches identified channels and videos
- PlanningAgent: Creates execution plans
- CodeAgent: Generates analysis code and reports
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Talk framework components
from agent.agent import Agent
from plan_runner.blackboard import Blackboard
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import specialized agents
from special_agents.research_agents.youtube.youtube_agent import YoutubeAgent
from special_agents.research_agents.web_search_agent import WebSearchAgent
from special_agents.planning_agent import PlanningAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class AIContentCategorizer:
    """Categorizes YouTube content for AI/coding relevance."""
    
    # Categories for AI/coding content
    CATEGORIES = {
        "llm_tutorials": {
            "keywords": ["llm", "gpt", "claude", "gemini", "language model", "prompt engineering"],
            "description": "LLM usage and prompt engineering tutorials"
        },
        "ai_coding": {
            "keywords": ["ai coding", "copilot", "cursor", "codewhisperer", "ai development"],
            "description": "AI-assisted coding tools and techniques"
        },
        "codebase_analysis": {
            "keywords": ["code analysis", "static analysis", "ast", "parsing", "code review"],
            "description": "Codebase analysis tools and techniques"
        },
        "software_architecture": {
            "keywords": ["software architecture", "system design", "design patterns", "clean code"],
            "description": "Software design and architecture principles"
        },
        "ai_agents": {
            "keywords": ["ai agents", "autonomous", "multi-agent", "orchestration", "langchain", "autogen"],
            "description": "AI agent frameworks and orchestration"
        },
        "machine_learning": {
            "keywords": ["machine learning", "deep learning", "neural network", "tensorflow", "pytorch"],
            "description": "Machine learning and deep learning tutorials"
        },
        "devops_automation": {
            "keywords": ["devops", "ci/cd", "automation", "github actions", "docker", "kubernetes"],
            "description": "DevOps and automation for development"
        },
        "testing_quality": {
            "keywords": ["testing", "test driven", "tdd", "unit test", "integration test", "quality"],
            "description": "Testing strategies and quality assurance"
        }
    }
    
    def categorize(self, title: str, channel: str = "") -> List[str]:
        """Categorize content based on title and channel."""
        categories = []
        combined = f"{title} {channel}".lower()
        
        for category, info in self.CATEGORIES.items():
            if any(keyword in combined for keyword in info["keywords"]):
                categories.append(category)
        
        return categories if categories else ["uncategorized"]
    
    def score_relevance(self, title: str, channel: str = "") -> float:
        """Score content relevance for AI/coding (0-1)."""
        categories = self.categorize(title, channel)
        
        # High-value categories for codebase analysis
        high_value = ["ai_coding", "codebase_analysis", "ai_agents", "llm_tutorials"]
        
        if any(cat in high_value for cat in categories):
            return 0.9
        elif categories and categories[0] != "uncategorized":
            return 0.6
        else:
            return 0.2


class YouTubeAIAnalyzer:
    """Main analyzer that orchestrates agents using Talk framework."""
    
    def __init__(self, takeout_path: str):
        """Initialize analyzer with YouTube takeout data."""
        self.takeout_path = takeout_path
        self.blackboard = Blackboard()
        self.categorizer = AIContentCategorizer()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "top_channels": [],
            "recommendations": [],
            "learning_paths": []
        }
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all required agents."""
        return {
            "youtube": YoutubeAgent(takeout_path=self.takeout_path),
            "search": WebSearchAgent(max_results=3),
            "planner": PlanningAgent(),
            "coder": CodeAgent(),
            "file": FileAgent()
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Run the full analysis pipeline."""
        log.info("Starting YouTube AI content analysis...")
        
        # Step 1: Extract YouTube data
        youtube_data = self._extract_youtube_data()
        
        # Step 2: Categorize content
        categorized_content = self._categorize_content(youtube_data)
        
        # Step 3: Identify top AI/coding channels
        top_channels = self._identify_top_channels(categorized_content)
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(categorized_content, top_channels)
        
        # Step 5: Create learning paths
        learning_paths = self._create_learning_paths(categorized_content)
        
        # Step 6: Export results
        self._export_results(recommendations, learning_paths)
        
        return self.results
    
    def _extract_youtube_data(self) -> Dict[str, Any]:
        """Extract data using YoutubeAgent."""
        log.info("Extracting YouTube data...")
        
        # Create execution plan
        steps = [
            Step(label="extract_watch_history", agent_key="youtube"),
            Step(label="extract_subscriptions", agent_key="youtube"),
            Step(label="extract_search_history", agent_key="youtube")
        ]
        
        # Run extraction
        runner = PlanRunner(steps, self.agents, self.blackboard)
        
        # Execute different prompts for each step
        prompts = {
            "extract_watch_history": "Extract and analyze my watch history, focusing on channels and topics",
            "extract_subscriptions": "List all my subscriptions with channel names",
            "extract_search_history": "Analyze my search history for programming and AI related searches"
        }
        
        extracted_data = {}
        for step in steps:
            result = self.agents["youtube"].run(prompts[step.label])
            self.blackboard.add(step.label, result)
            extracted_data[step.label] = result
        
        return extracted_data
    
    def _categorize_content(self, youtube_data: Dict[str, Any]) -> Dict[str, List]:
        """Categorize content by AI/coding relevance."""
        log.info("Categorizing content...")
        
        categorized = {cat: [] for cat in self.categorizer.CATEGORIES.keys()}
        categorized["uncategorized"] = []
        
        # Parse watch history from the extracted data
        watch_history_text = youtube_data.get("extract_watch_history", "")
        
        # Simple parsing - in production, would parse the actual data structure
        lines = watch_history_text.split("\n")
        for line in lines:
            if ":" in line and any(word in line.lower() for word in ["channel", "video", "title"]):
                # Extract title/channel info (simplified)
                content = line.strip()
                categories = self.categorizer.categorize(content)
                relevance = self.categorizer.score_relevance(content)
                
                item = {
                    "content": content,
                    "relevance_score": relevance,
                    "categories": categories
                }
                
                for category in categories:
                    if category in categorized:
                        categorized[category].append(item)
        
        # Sort by relevance
        for category in categorized:
            categorized[category].sort(key=lambda x: x["relevance_score"], reverse=True)
        
        self.results["categories"] = {
            cat: len(items) for cat, items in categorized.items() if items
        }
        
        return categorized
    
    def _identify_top_channels(self, categorized_content: Dict[str, List]) -> List[Dict]:
        """Identify top AI/coding channels."""
        log.info("Identifying top AI/coding channels...")
        
        # Aggregate channel mentions across categories
        channel_scores = {}
        
        for category, items in categorized_content.items():
            if category == "uncategorized":
                continue
            
            for item in items:
                # Extract channel name (simplified)
                content = item["content"]
                if "channel" in content.lower():
                    # Simple extraction - would be more robust in production
                    channel = content.split(":")[-1].strip()[:50]
                    
                    if channel not in channel_scores:
                        channel_scores[channel] = {
                            "name": channel,
                            "score": 0,
                            "categories": set(),
                            "count": 0
                        }
                    
                    channel_scores[channel]["score"] += item["relevance_score"]
                    channel_scores[channel]["categories"].add(category)
                    channel_scores[channel]["count"] += 1
        
        # Convert sets to lists for JSON serialization
        for channel in channel_scores.values():
            channel["categories"] = list(channel["categories"])
        
        # Sort by score
        top_channels = sorted(
            channel_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:10]
        
        self.results["top_channels"] = top_channels
        
        return top_channels
    
    def _generate_recommendations(self, categorized_content: Dict[str, List], 
                                 top_channels: List[Dict]) -> List[Dict]:
        """Generate content recommendations."""
        log.info("Generating recommendations...")
        
        recommendations = []
        
        # Recommend high-value categories with low watch count
        for category, info in self.categorizer.CATEGORIES.items():
            if category in ["ai_coding", "codebase_analysis", "ai_agents"]:
                count = len(categorized_content.get(category, []))
                if count < 5:  # Under-explored category
                    recommendations.append({
                        "type": "explore_category",
                        "category": category,
                        "reason": f"High-value category for codebase analysis with only {count} videos watched",
                        "description": info["description"],
                        "suggested_searches": info["keywords"][:3]
                    })
        
        # Recommend specific channels
        for channel in top_channels[:3]:
            if channel["score"] > 2.0:  # High relevance threshold
                recommendations.append({
                    "type": "follow_channel",
                    "channel": channel["name"],
                    "reason": f"High relevance score ({channel['score']:.2f}) in categories: {', '.join(channel['categories'][:3])}",
                    "watch_count": channel["count"]
                })
        
        # Research recommendations using WebSearchAgent
        for rec in recommendations[:2]:  # Limit API calls
            if rec["type"] == "explore_category":
                search_query = f"best YouTube channels for {rec['category']} tutorials 2024"
                search_result = self.agents["search"].run(search_query)
                rec["search_results"] = search_result[:500]  # Truncate for storage
        
        self.results["recommendations"] = recommendations
        
        return recommendations
    
    def _create_learning_paths(self, categorized_content: Dict[str, List]) -> List[Dict]:
        """Create suggested learning paths based on viewing history."""
        log.info("Creating learning paths...")
        
        learning_paths = []
        
        # Path 1: AI-Assisted Development
        ai_dev_path = {
            "name": "AI-Assisted Development Mastery",
            "description": "Master AI tools for software development",
            "current_progress": self._calculate_progress(categorized_content, 
                                                        ["ai_coding", "llm_tutorials"]),
            "steps": [
                {"order": 1, "topic": "LLM Fundamentals", "category": "llm_tutorials"},
                {"order": 2, "topic": "Prompt Engineering", "category": "llm_tutorials"},
                {"order": 3, "topic": "AI Coding Tools", "category": "ai_coding"},
                {"order": 4, "topic": "Multi-Agent Systems", "category": "ai_agents"},
                {"order": 5, "topic": "Codebase Analysis with AI", "category": "codebase_analysis"}
            ]
        }
        learning_paths.append(ai_dev_path)
        
        # Path 2: Automated Code Analysis
        analysis_path = {
            "name": "Automated Code Analysis Pipeline",
            "description": "Build automated systems for code quality and analysis",
            "current_progress": self._calculate_progress(categorized_content,
                                                        ["codebase_analysis", "testing_quality"]),
            "steps": [
                {"order": 1, "topic": "Static Analysis Tools", "category": "codebase_analysis"},
                {"order": 2, "topic": "AST and Code Parsing", "category": "codebase_analysis"},
                {"order": 3, "topic": "Automated Testing", "category": "testing_quality"},
                {"order": 4, "topic": "CI/CD Integration", "category": "devops_automation"},
                {"order": 5, "topic": "AI-Powered Code Review", "category": "ai_coding"}
            ]
        }
        learning_paths.append(analysis_path)
        
        self.results["learning_paths"] = learning_paths
        
        return learning_paths
    
    def _calculate_progress(self, categorized_content: Dict[str, List], 
                          categories: List[str]) -> float:
        """Calculate learning progress for given categories."""
        total_videos = sum(len(categorized_content.get(cat, [])) for cat in categories)
        
        # Simple heuristic: 10 videos per category = 100% progress
        target_videos = len(categories) * 10
        
        return min(1.0, total_videos / target_videos) if target_videos > 0 else 0.0
    
    def _export_results(self, recommendations: List[Dict], 
                       learning_paths: List[Dict]) -> None:
        """Export analysis results."""
        log.info("Exporting results...")
        
        # Create output directory
        output_dir = Path.cwd() / "miniapps" / "youtube_ai_analyzer" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ai_content_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        log.info(f"Results saved to: {output_file}")
        
        # Create markdown report
        report_file = output_dir / f"ai_content_report_{timestamp}.md"
        self._create_markdown_report(report_file)
        
        log.info(f"Report saved to: {report_file}")
    
    def _create_markdown_report(self, output_path: Path) -> None:
        """Create a markdown report of the analysis."""
        report = []
        report.append("# YouTube AI Content Analysis Report")
        report.append(f"\nGenerated: {self.results['timestamp']}")
        
        # Categories summary
        report.append("\n## Content Categories")
        for category, count in self.results["categories"].items():
            desc = self.categorizer.CATEGORIES.get(category, {}).get("description", "")
            report.append(f"- **{category}**: {count} videos - {desc}")
        
        # Top channels
        report.append("\n## Top AI/Coding Channels")
        for i, channel in enumerate(self.results["top_channels"][:5], 1):
            report.append(f"{i}. **{channel['name']}**")
            report.append(f"   - Score: {channel['score']:.2f}")
            report.append(f"   - Categories: {', '.join(channel['categories'])}")
            report.append(f"   - Videos watched: {channel['count']}")
        
        # Recommendations
        report.append("\n## Recommendations")
        for rec in self.results["recommendations"]:
            if rec["type"] == "explore_category":
                report.append(f"\n### Explore: {rec['category']}")
                report.append(f"- {rec['description']}")
                report.append(f"- Reason: {rec['reason']}")
                report.append(f"- Suggested searches: {', '.join(rec['suggested_searches'])}")
            elif rec["type"] == "follow_channel":
                report.append(f"\n### Follow Channel: {rec['channel']}")
                report.append(f"- Reason: {rec['reason']}")
        
        # Learning paths
        report.append("\n## Suggested Learning Paths")
        for path in self.results["learning_paths"]:
            report.append(f"\n### {path['name']}")
            report.append(f"{path['description']}")
            report.append(f"\nCurrent Progress: {path['current_progress']*100:.0f}%")
            report.append("\nSteps:")
            for step in path["steps"]:
                report.append(f"{step['order']}. {step['topic']}")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write("\n".join(report))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze YouTube history for AI/coding content")
    parser.add_argument(
        "--takeout",
        default="special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip",
        help="Path to YouTube takeout zip file"
    )
    parser.add_argument(
        "--output",
        default="miniapps/youtube_ai_analyzer/output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    takeout_path = Path(args.takeout)
    if not takeout_path.is_absolute():
        takeout_path = Path.cwd() / takeout_path
    
    if not takeout_path.exists():
        print(f"Error: Takeout file not found at {takeout_path}")
        return 1
    
    print("YouTube AI Content Analyzer")
    print("=" * 50)
    print(f"Takeout file: {takeout_path}")
    print(f"Output directory: {args.output}")
    print()
    
    # Run analysis
    analyzer = YouTubeAIAnalyzer(str(takeout_path))
    results = analyzer.analyze()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print(f"\nCategories found: {len(results['categories'])}")
    print(f"Top channels identified: {len(results['top_channels'])}")
    print(f"Recommendations generated: {len(results['recommendations'])}")
    print(f"Learning paths created: {len(results['learning_paths'])}")
    
    # Show top recommendations
    print("\nTop Recommendations:")
    for i, rec in enumerate(results['recommendations'][:3], 1):
        if rec['type'] == 'explore_category':
            print(f"{i}. Explore {rec['category']}: {rec['reason']}")
        else:
            print(f"{i}. Follow {rec['channel']}: {rec['reason']}")
    
    print(f"\nFull results saved to: miniapps/youtube_ai_analyzer/output/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())