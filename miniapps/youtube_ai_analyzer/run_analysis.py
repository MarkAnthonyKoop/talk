#!/usr/bin/env python3
"""
Simplified runner for YouTube AI Content Analyzer.

This version runs the analysis without requiring full agent initialization,
focusing on data extraction and categorization.
"""

import sys
import json
import zipfile
import csv
import io
from pathlib import Path
from datetime import datetime
from collections import Counter
from bs4 import BeautifulSoup

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from youtube_ai_analyzer import AIContentCategorizer


class SimpleYouTubeAnalyzer:
    """Simplified analyzer that works directly with takeout data."""
    
    def __init__(self, takeout_path: str):
        self.takeout_path = Path(takeout_path)
        self.categorizer = AIContentCategorizer()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "top_channels": [],
            "top_videos": [],
            "recommendations": [],
            "search_insights": []
        }
    
    def analyze(self):
        """Run the analysis."""
        print("Starting YouTube AI Content Analysis...")
        print("=" * 50)
        
        # Extract data from takeout
        data = self._extract_takeout_data()
        
        # Analyze watch history
        if data.get("watch_history"):
            self._analyze_watch_history(data["watch_history"])
        
        # Analyze search history
        if data.get("search_history"):
            self._analyze_search_history(data["search_history"])
        
        # Analyze subscriptions
        if data.get("subscriptions"):
            self._analyze_subscriptions(data["subscriptions"])
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _extract_takeout_data(self):
        """Extract data from takeout zip."""
        print("\nExtracting data from takeout file...")
        
        data = {}
        
        with zipfile.ZipFile(self.takeout_path, 'r') as zf:
            files = zf.namelist()
            
            # Extract watch history
            watch_file = next((f for f in files if 'watch-history.html' in f), None)
            if watch_file:
                print("  Extracting watch history...")
                with zf.open(watch_file) as f:
                    content = f.read().decode('utf-8')
                    data["watch_history"] = self._parse_watch_history(content)
                    print(f"    Found {len(data['watch_history'])} videos")
            
            # Extract search history
            search_file = next((f for f in files if 'search-history.html' in f), None)
            if search_file:
                print("  Extracting search history...")
                with zf.open(search_file) as f:
                    content = f.read().decode('utf-8')
                    data["search_history"] = self._parse_search_history(content)
                    print(f"    Found {len(data['search_history'])} searches")
            
            # Extract subscriptions
            subs_file = next((f for f in files if 'subscriptions.csv' in f), None)
            if subs_file:
                print("  Extracting subscriptions...")
                with zf.open(subs_file) as f:
                    content = f.read().decode('utf-8')
                    reader = csv.DictReader(io.StringIO(content))
                    data["subscriptions"] = list(reader)
                    print(f"    Found {len(data['subscriptions'])} subscriptions")
        
        return data
    
    def _parse_watch_history(self, html_content):
        """Parse watch history HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        videos = []
        
        for item in soup.find_all('div', class_='mdl-grid'):
            try:
                links = item.find_all('a')
                if links:
                    title = links[0].text.strip() if links else "Unknown"
                    channel = links[1].text.strip() if len(links) > 1 else "Unknown"
                    
                    videos.append({
                        "title": title,
                        "channel": channel,
                        "url": links[0].get('href', '') if links else ""
                    })
            except:
                continue
        
        return videos
    
    def _parse_search_history(self, html_content):
        """Parse search history HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        searches = []
        
        for item in soup.find_all('div', class_='mdl-grid'):
            try:
                search_link = item.find('a')
                if search_link:
                    query = search_link.text.strip()
                    searches.append(query)
            except:
                continue
        
        return searches
    
    def _analyze_watch_history(self, videos):
        """Analyze watch history for AI/coding content."""
        print("\nAnalyzing watch history...")
        
        # Categorize videos
        categorized = {}
        channel_stats = Counter()
        ai_videos = []
        
        for video in videos:
            title = video.get("title", "")
            channel = video.get("channel", "")
            
            # Get categories
            categories = self.categorizer.categorize(title, channel)
            score = self.categorizer.score_relevance(title, channel)
            
            # Track channel
            channel_stats[channel] += 1
            
            # Store if AI/coding related
            if score > 0.5:
                ai_videos.append({
                    "title": title[:100],
                    "channel": channel,
                    "categories": categories,
                    "score": score
                })
            
            # Count categories
            for cat in categories:
                if cat not in categorized:
                    categorized[cat] = 0
                categorized[cat] += 1
        
        # Store results
        self.results["categories"] = categorized
        
        # Top AI/coding videos
        ai_videos.sort(key=lambda x: x["score"], reverse=True)
        self.results["top_videos"] = ai_videos[:20]
        
        # Top channels by frequency
        top_channels = []
        for channel, count in channel_stats.most_common(20):
            # Check if it's AI/coding related
            score = self.categorizer.score_relevance("", channel)
            top_channels.append({
                "channel": channel,
                "video_count": count,
                "ai_relevance": score
            })
        
        # Sort by AI relevance
        top_channels.sort(key=lambda x: x["ai_relevance"], reverse=True)
        self.results["top_channels"] = top_channels[:10]
        
        # Print summary
        print(f"  Total videos: {len(videos)}")
        print(f"  AI/coding videos: {len(ai_videos)}")
        print(f"  Categories found: {len(categorized)}")
        
        # Show top categories
        print("\n  Top AI/Coding Categories:")
        for cat, count in sorted(categorized.items(), key=lambda x: x[1], reverse=True)[:5]:
            if cat != "uncategorized":
                desc = self.categorizer.CATEGORIES.get(cat, {}).get("description", "")
                print(f"    - {cat}: {count} videos")
                print(f"      {desc}")
    
    def _analyze_search_history(self, searches):
        """Analyze search history for AI/coding patterns."""
        print("\nAnalyzing search history...")
        
        # Extract AI/coding related searches
        ai_searches = []
        search_categories = Counter()
        
        for query in searches:
            categories = self.categorizer.categorize(query)
            score = self.categorizer.score_relevance(query)
            
            if score > 0.3:  # Lower threshold for searches
                ai_searches.append({
                    "query": query,
                    "categories": categories,
                    "score": score
                })
                
                for cat in categories:
                    search_categories[cat] += 1
        
        # Find common search terms
        all_terms = []
        for query in ai_searches:
            words = query["query"].lower().split()
            all_terms.extend([w for w in words if len(w) > 3])
        
        term_counts = Counter(all_terms)
        top_terms = term_counts.most_common(15)
        
        self.results["search_insights"] = {
            "total_searches": len(searches),
            "ai_searches": len(ai_searches),
            "top_terms": dict(top_terms),
            "search_categories": dict(search_categories)
        }
        
        print(f"  Total searches: {len(searches)}")
        print(f"  AI/coding searches: {len(ai_searches)}")
        print(f"  Top search terms: {', '.join([t[0] for t in top_terms[:5]])}")
    
    def _analyze_subscriptions(self, subscriptions):
        """Analyze subscriptions for AI/coding channels."""
        print("\nAnalyzing subscriptions...")
        
        ai_subs = []
        
        for sub in subscriptions:
            channel = sub.get("Channel Title", "")
            channel_id = sub.get("Channel Id", "")
            
            score = self.categorizer.score_relevance("", channel)
            
            if score > 0.3:
                ai_subs.append({
                    "channel": channel,
                    "channel_id": channel_id,
                    "ai_relevance": score
                })
        
        ai_subs.sort(key=lambda x: x["ai_relevance"], reverse=True)
        
        self.results["ai_subscriptions"] = ai_subs
        
        print(f"  Total subscriptions: {len(subscriptions)}")
        print(f"  AI/coding subscriptions: {len(ai_subs)}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print("\nGenerating recommendations...")
        
        recommendations = []
        
        # Check for under-explored categories
        high_value_cats = ["codebase_analysis", "ai_agents", "ai_coding"]
        
        for cat in high_value_cats:
            count = self.results["categories"].get(cat, 0)
            if count < 5:
                recommendations.append({
                    "type": "explore_category",
                    "category": cat,
                    "current_count": count,
                    "reason": f"High-value category with only {count} videos watched",
                    "description": self.categorizer.CATEGORIES[cat]["description"],
                    "keywords": self.categorizer.CATEGORIES[cat]["keywords"][:3]
                })
        
        # Recommend top AI channels not subscribed to
        if self.results.get("top_channels"):
            for channel_data in self.results["top_channels"][:3]:
                if channel_data["ai_relevance"] > 0.7:
                    recommendations.append({
                        "type": "channel",
                        "channel": channel_data["channel"],
                        "video_count": channel_data["video_count"],
                        "reason": f"High AI relevance ({channel_data['ai_relevance']:.2f}) with {channel_data['video_count']} videos watched"
                    })
        
        self.results["recommendations"] = recommendations
        
        print(f"  Generated {len(recommendations)} recommendations")
    
    def _save_results(self):
        """Save analysis results."""
        output_dir = Path.cwd() / "miniapps" / "youtube_ai_analyzer" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = output_dir / f"analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {json_file}")
        
        # Create markdown report
        self._create_report(output_dir / f"report_{timestamp}.md")
    
    def _create_report(self, output_path):
        """Create markdown report."""
        lines = []
        lines.append("# YouTube AI Content Analysis Report")
        lines.append(f"\nGenerated: {self.results['timestamp']}")
        
        # Categories
        lines.append("\n## AI/Coding Content Categories")
        for cat, count in sorted(self.results["categories"].items(), 
                                key=lambda x: x[1], reverse=True):
            if cat != "uncategorized" and count > 0:
                desc = self.categorizer.CATEGORIES.get(cat, {}).get("description", "")
                lines.append(f"\n### {cat.replace('_', ' ').title()} ({count} videos)")
                lines.append(f"{desc}")
        
        # Top channels
        lines.append("\n## Top AI/Coding Channels")
        for i, ch in enumerate(self.results["top_channels"][:5], 1):
            lines.append(f"{i}. **{ch['channel']}**")
            lines.append(f"   - Videos watched: {ch['video_count']}")
            lines.append(f"   - AI relevance: {ch['ai_relevance']:.2f}")
        
        # Top videos
        lines.append("\n## Top AI/Coding Videos")
        for i, video in enumerate(self.results["top_videos"][:10], 1):
            lines.append(f"{i}. {video['title']}")
            lines.append(f"   - Channel: {video['channel']}")
            lines.append(f"   - Categories: {', '.join(video['categories'])}")
        
        # Search insights
        if self.results.get("search_insights"):
            insights = self.results["search_insights"]
            lines.append("\n## Search Insights")
            lines.append(f"- Total searches: {insights['total_searches']}")
            lines.append(f"- AI/coding searches: {insights['ai_searches']}")
            
            if insights.get("top_terms"):
                lines.append("\n### Top Search Terms")
                for term, count in list(insights["top_terms"].items())[:10]:
                    lines.append(f"- {term}: {count} times")
        
        # Recommendations
        lines.append("\n## Recommendations")
        for rec in self.results["recommendations"]:
            if rec["type"] == "explore_category":
                lines.append(f"\n### Explore: {rec['category'].replace('_', ' ').title()}")
                lines.append(f"- {rec['description']}")
                lines.append(f"- Current videos: {rec['current_count']}")
                lines.append(f"- Search for: {', '.join(rec['keywords'])}")
            elif rec["type"] == "channel":
                lines.append(f"\n### Channel: {rec['channel']}")
                lines.append(f"- {rec['reason']}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"✓ Report saved to: {output_path}")


def main():
    """Run the analysis."""
    print("YouTube AI Content Analyzer (Simplified)")
    print("=" * 50)
    
    # Find takeout file
    takeout_path = project_root / "special_agents/research_agents/youtube/takeout_20250806T082512Z_1_001.zip"
    
    if not takeout_path.exists():
        print(f"Error: Takeout file not found at {takeout_path}")
        return 1
    
    print(f"Using takeout file: {takeout_path.name}")
    print(f"File size: {takeout_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Run analysis
    analyzer = SimpleYouTubeAnalyzer(takeout_path)
    results = analyzer.analyze()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("\nSummary:")
    
    # Category summary
    if results.get("categories"):
        print("\nTop Categories:")
        for cat, count in sorted(results["categories"].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
            if cat != "uncategorized":
                print(f"  - {cat}: {count} videos")
    
    # Top channels
    if results.get("top_channels"):
        print("\nTop AI/Coding Channels:")
        for ch in results["top_channels"][:3]:
            print(f"  - {ch['channel']}: {ch['video_count']} videos (relevance: {ch['ai_relevance']:.2f})")
    
    # Recommendations
    if results.get("recommendations"):
        print(f"\nGenerated {len(results['recommendations'])} recommendations")
        for rec in results["recommendations"][:3]:
            if rec["type"] == "explore_category":
                print(f"  ⚡ Explore {rec['category']}: {rec['current_count']} videos watched")
            else:
                print(f"  ⭐ {rec['channel']}: {rec['reason']}")
    
    print("\nOutput files saved in: miniapps/youtube_ai_analyzer/output/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())