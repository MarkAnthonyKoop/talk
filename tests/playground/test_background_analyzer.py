#!/usr/bin/env python3
"""
Background analyzer that processes YouTube data with progress updates.
Runs analysis in background thread to avoid timeouts.
"""

import sys
import time
import threading
import queue
import json
import zipfile
import csv
import io
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.agent import Agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'miniapps' / 'youtube_ai_analyzer'))
from youtube_ai_analyzer import AIContentCategorizer


class BackgroundYouTubeAnalyzer:
    """Analyzer that runs in background with progress updates."""
    
    def __init__(self, takeout_path: str):
        self.takeout_path = Path(takeout_path)
        self.categorizer = AIContentCategorizer()
        self.progress_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.agent = None
        
    def run_analysis(self):
        """Run analysis in background thread."""
        thread = threading.Thread(target=self._analyze_worker)
        thread.daemon = True
        thread.start()
        
        # Monitor progress
        print("Starting background analysis...")
        print("This will process your YouTube data and identify AI/coding content.")
        print("-" * 50)
        
        last_message_time = time.time()
        dots = 0
        
        while thread.is_alive() or not self.progress_queue.empty():
            # Get progress messages
            while not self.progress_queue.empty():
                msg = self.progress_queue.get()
                # Clear the dots line
                if dots > 0:
                    print("\r" + " " * (dots + 10), end="\r")
                    dots = 0
                print(f"► {msg}")
                last_message_time = time.time()
            
            # Show activity dots if no message for a while
            if time.time() - last_message_time > 2:
                dots = (dots + 1) % 4
                print(f"\r{'.' * dots}   ", end="", flush=True)
            
            time.sleep(1)
        
        # Clear any remaining dots
        if dots > 0:
            print("\r" + " " * (dots + 10), end="\r")
        
        # Get results
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def _analyze_worker(self):
        """Worker thread for analysis."""
        try:
            # Step 1: Initialize agent
            self.progress_queue.put("Initializing AI agent...")
            self.agent = Agent(
                roles=["You are an expert at analyzing YouTube content and identifying AI/coding related videos."],
                overrides={"llm": {"provider": "anthropic"}}
            )
            
            # Step 2: Extract data
            self.progress_queue.put("Extracting YouTube data from takeout...")
            data = self._extract_data()
            
            # Step 3: Categorize content
            self.progress_queue.put(f"Categorizing {len(data.get('videos', []))} videos...")
            categorized = self._categorize_videos(data.get('videos', []))
            
            # Step 4: Analyze channels
            self.progress_queue.put("Analyzing channel patterns...")
            channel_analysis = self._analyze_channels(data.get('videos', []))
            
            # Step 5: Get AI insights
            self.progress_queue.put("Generating AI insights...")
            insights = self._get_ai_insights(categorized, channel_analysis)
            
            # Step 6: Generate recommendations
            self.progress_queue.put("Creating recommendations...")
            recommendations = self._generate_recommendations(categorized, insights)
            
            # Compile results
            results = {
                "timestamp": datetime.now().isoformat(),
                "stats": {
                    "total_videos": len(data.get('videos', [])),
                    "ai_videos": len([v for v in categorized if v['score'] > 0.5]),
                    "subscriptions": len(data.get('subscriptions', [])),
                    "searches": len(data.get('searches', []))
                },
                "categories": self._count_categories(categorized),
                "top_channels": channel_analysis[:10],
                "insights": insights,
                "recommendations": recommendations
            }
            
            self.progress_queue.put("Analysis complete!")
            self.result_queue.put(results)
            
        except Exception as e:
            self.progress_queue.put(f"Error: {str(e)}")
            self.result_queue.put(None)
    
    def _extract_data(self):
        """Extract data from takeout zip."""
        data = {"videos": [], "searches": [], "subscriptions": []}
        
        with zipfile.ZipFile(self.takeout_path, 'r') as zf:
            files = zf.namelist()
            
            # Watch history (simplified extraction)
            watch_file = next((f for f in files if 'watch-history.html' in f), None)
            if watch_file:
                with zf.open(watch_file) as f:
                    content = f.read().decode('utf-8')
                    # Simple extraction without BeautifulSoup
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '<a href="https://www.youtube.com/watch' in line:
                            # Extract title (simplified)
                            if '>' in line and '</a>' in line:
                                title = line.split('>')[1].split('</a>')[0]
                                # Look for channel in next lines
                                channel = "Unknown"
                                for j in range(1, min(5, len(lines) - i)):
                                    next_line = lines[i + j]
                                    if '<a href="https://www.youtube.com/channel' in next_line:
                                        if '>' in next_line and '</a>' in next_line:
                                            channel = next_line.split('>')[1].split('</a>')[0]
                                            break
                                
                                data["videos"].append({
                                    "title": title[:200],
                                    "channel": channel[:100]
                                })
            
            # Subscriptions
            subs_file = next((f for f in files if 'subscriptions.csv' in f), None)
            if subs_file:
                with zf.open(subs_file) as f:
                    content = f.read().decode('utf-8')
                    reader = csv.DictReader(io.StringIO(content))
                    data["subscriptions"] = list(reader)
        
        return data
    
    def _categorize_videos(self, videos):
        """Categorize videos for AI/coding relevance."""
        categorized = []
        
        for video in videos:
            title = video.get("title", "")
            channel = video.get("channel", "")
            
            categories = self.categorizer.categorize(title, channel)
            score = self.categorizer.score_relevance(title, channel)
            
            categorized.append({
                "title": title,
                "channel": channel,
                "categories": categories,
                "score": score
            })
        
        return categorized
    
    def _analyze_channels(self, videos):
        """Analyze channel patterns."""
        channel_counts = Counter()
        channel_ai_scores = {}
        
        for video in videos:
            channel = video.get("channel", "Unknown")
            channel_counts[channel] += 1
            
            # Calculate AI relevance for channel
            if channel not in channel_ai_scores:
                score = self.categorizer.score_relevance("", channel)
                channel_ai_scores[channel] = score
        
        # Combine counts and scores
        channel_analysis = []
        for channel, count in channel_counts.most_common(50):
            channel_analysis.append({
                "channel": channel,
                "video_count": count,
                "ai_relevance": channel_ai_scores.get(channel, 0)
            })
        
        # Sort by AI relevance
        channel_analysis.sort(key=lambda x: x["ai_relevance"], reverse=True)
        
        return channel_analysis
    
    def _get_ai_insights(self, categorized_videos, channel_analysis):
        """Use AI agent to generate insights."""
        if not self.agent:
            return "Unable to generate insights"
        
        # Prepare summary for AI
        ai_videos = [v for v in categorized_videos if v['score'] > 0.5]
        
        # Count categories
        category_counts = Counter()
        for video in ai_videos:
            for cat in video['categories']:
                category_counts[cat] += 1
        
        prompt = f"""
        Analyze this YouTube viewing data and provide insights:
        
        Total videos analyzed: {len(categorized_videos)}
        AI/coding related videos: {len(ai_videos)}
        
        Top AI/coding categories watched:
        {json.dumps(dict(category_counts.most_common(5)), indent=2)}
        
        Top AI/coding channels:
        {json.dumps([c['channel'] for c in channel_analysis[:5]], indent=2)}
        
        Provide 3-5 key insights about the user's AI/coding learning journey and interests.
        Be specific and actionable.
        """
        
        try:
            insights = self.agent.run(prompt)
            return insights
        except Exception as e:
            return f"AI insights generation failed: {str(e)}"
    
    def _generate_recommendations(self, categorized_videos, insights):
        """Generate recommendations."""
        recommendations = []
        
        # Count categories
        category_counts = Counter()
        for video in categorized_videos:
            if video['score'] > 0.5:
                for cat in video['categories']:
                    category_counts[cat] += 1
        
        # Recommend under-explored categories
        high_value_categories = ["codebase_analysis", "ai_agents", "ai_coding"]
        for cat in high_value_categories:
            count = category_counts.get(cat, 0)
            if count < 5:
                recommendations.append({
                    "type": "explore_category",
                    "category": cat,
                    "current_count": count,
                    "description": self.categorizer.CATEGORIES[cat]["description"]
                })
        
        return recommendations
    
    def _count_categories(self, categorized_videos):
        """Count videos by category."""
        counts = Counter()
        for video in categorized_videos:
            if video['score'] > 0.5:
                for cat in video['categories']:
                    counts[cat] += 1
        return dict(counts)


def main():
    """Run the background analyzer."""
    print("YouTube AI Content Analyzer (Background Processing)")
    print("=" * 60)
    
    # Check for takeout file
    takeout_path = Path("/home/xx/code/special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip")
    
    if not takeout_path.exists():
        print(f"Error: Takeout file not found at {takeout_path}")
        return 1
    
    print(f"Takeout file: {takeout_path.name}")
    print(f"File size: {takeout_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Run analysis
    analyzer = BackgroundYouTubeAnalyzer(str(takeout_path))
    results = analyzer.run_analysis()
    
    if results:
        print("\n" + "=" * 60)
        print("Analysis Results")
        print("-" * 60)
        
        # Show stats
        stats = results.get("stats", {})
        print(f"\nStatistics:")
        print(f"  Total videos: {stats.get('total_videos', 0)}")
        print(f"  AI/coding videos: {stats.get('ai_videos', 0)}")
        print(f"  Subscriptions: {stats.get('subscriptions', 0)}")
        
        # Show categories
        categories = results.get("categories", {})
        if categories:
            print(f"\nTop AI/Coding Categories:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {cat}: {count} videos")
        
        # Show top channels
        channels = results.get("top_channels", [])
        if channels:
            print(f"\nTop AI/Coding Channels:")
            for ch in channels[:5]:
                print(f"  - {ch['channel']}: {ch['video_count']} videos (relevance: {ch['ai_relevance']:.2f})")
        
        # Show insights
        insights = results.get("insights", "")
        if insights:
            print(f"\nAI Insights:")
            print("-" * 40)
            print(insights[:500])
            print("-" * 40)
        
        # Save results
        output_dir = Path.cwd() / "tests" / "playground" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"youtube_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Full results saved to: {output_file}")
    else:
        print("\n✗ Analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())