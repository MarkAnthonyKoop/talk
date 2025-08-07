#!/usr/bin/env python3

"""
YoutubeAgent - Specialized agent for analyzing YouTube takeout data.

This agent processes YouTube takeout data including watch history, search history,
subscriptions, playlists, and other YouTube activity to provide insights and research.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from collections import Counter, defaultdict
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from agent.agent import Agent

from agent.agent import Agent
from agent.messages import Message, Role

log = logging.getLogger(__name__)

class YoutubeAgent(Agent):
    """
    Specialized agent for YouTube takeout data analysis.
    
    This agent can process YouTube takeout data to extract insights about:
    - Watch history patterns
    - Search history analysis
    - Subscription lists
    - Playlist contents
    - Channel interactions
    - Video metadata
    """
    
    def __init__(self, takeout_path: Optional[str] = None, **kwargs):
        """Initialize with specialized system prompt for YouTube analysis."""
        roles = kwargs.pop("roles", [])
        
        # Add YouTube analysis-specific system prompts
        youtube_system_prompts = [
            "You are a YouTube data analysis specialist.",
            "You help users understand their YouTube usage patterns and extract insights from their takeout data.",
            "You can analyze watch history, search patterns, subscriptions, and content preferences.",
            "Always provide clear summaries and actionable insights from the data.",
            "Respect user privacy and handle data securely."
        ]
        
        # Combine with any existing roles
        roles = youtube_system_prompts + roles
        
        # Initialize the base agent
        super().__init__(roles=roles, **kwargs)
        
        # Store takeout path
        self.takeout_path = takeout_path
        self.data_cache = {}
        
    def run(self, input_text: str) -> str:
        """
        Process YouTube data analysis request.
        
        Args:
            input_text: Analysis request or query about YouTube data
            
        Returns:
            Analysis results and insights
        """
        try:
            # Parse the request to understand what analysis is needed
            analysis_type = self._determine_analysis_type(input_text)
            
            # Load relevant data if not cached
            if not self.data_cache and self.takeout_path:
                self._load_takeout_data()
            
            # Perform the requested analysis
            if analysis_type == "watch_history":
                result = self._analyze_watch_history(input_text)
            elif analysis_type == "search_history":
                result = self._analyze_search_history(input_text)
            elif analysis_type == "subscriptions":
                result = self._analyze_subscriptions(input_text)
            elif analysis_type == "playlists":
                result = self._analyze_playlists(input_text)
            elif analysis_type == "general":
                result = self._general_analysis(input_text)
            else:
                result = self._comprehensive_analysis(input_text)
            
            # Generate insights and recommendations
            insights = self._generate_insights(result, input_text)
            
            # Save analysis results to scratch folder
            self._save_analysis_results(result, insights)
            
            return f"{insights}\n\n{result}"
            
        except Exception as e:
            log.error(f"YouTube analysis failed: {e}")
            return f"YOUTUBE_ANALYSIS_ERROR: {str(e)}\n\nPlease ensure the takeout file path is correct and the data is properly formatted."
    
    def _determine_analysis_type(self, input_text: str) -> str:
        """Determine what type of analysis is requested."""
        input_lower = input_text.lower()
        
        if any(term in input_lower for term in ['watch', 'history', 'viewed', 'watched']):
            return "watch_history"
        elif any(term in input_lower for term in ['search', 'searched', 'queries']):
            return "search_history"
        elif any(term in input_lower for term in ['subscription', 'subscribed', 'channels']):
            return "subscriptions"
        elif any(term in input_lower for term in ['playlist', 'lists', 'saved']):
            return "playlists"
        elif any(term in input_lower for term in ['general', 'overview', 'summary']):
            return "general"
        else:
            return "comprehensive"
    
    def _load_takeout_data(self):
        """Load and parse YouTube takeout data from zip file."""
        if not self.takeout_path or not os.path.exists(self.takeout_path):
            log.warning(f"Takeout file not found at: {self.takeout_path}")
            return
        
        try:
            with zipfile.ZipFile(self.takeout_path, 'r') as zip_ref:
                # List all files in the archive
                file_list = zip_ref.namelist()
                
                # Load watch history
                watch_history_file = next((f for f in file_list if 'watch-history.html' in f), None)
                if watch_history_file:
                    with zip_ref.open(watch_history_file) as f:
                        self.data_cache['watch_history'] = self._parse_watch_history(f.read().decode('utf-8'))
                
                # Load search history
                search_history_file = next((f for f in file_list if 'search-history.html' in f), None)
                if search_history_file:
                    with zip_ref.open(search_history_file) as f:
                        self.data_cache['search_history'] = self._parse_search_history(f.read().decode('utf-8'))
                
                # Load subscriptions
                subscriptions_file = next((f for f in file_list if 'subscriptions.csv' in f), None)
                if subscriptions_file:
                    with zip_ref.open(subscriptions_file) as f:
                        self.data_cache['subscriptions'] = self._parse_csv_data(f.read().decode('utf-8'))
                
                # Load playlists
                playlist_files = [f for f in file_list if 'playlists/' in f and f.endswith('.csv')]
                self.data_cache['playlists'] = {}
                for playlist_file in playlist_files:
                    playlist_name = os.path.basename(playlist_file).replace('-videos.csv', '')
                    with zip_ref.open(playlist_file) as f:
                        self.data_cache['playlists'][playlist_name] = self._parse_csv_data(f.read().decode('utf-8'))
                
                # Load other CSV data
                for csv_file in file_list:
                    if csv_file.endswith('.csv') and csv_file not in [subscriptions_file] + playlist_files:
                        key = os.path.basename(csv_file).replace('.csv', '').replace(' ', '_')
                        with zip_ref.open(csv_file) as f:
                            self.data_cache[key] = self._parse_csv_data(f.read().decode('utf-8'))
                
                log.info(f"Loaded YouTube takeout data: {list(self.data_cache.keys())}")
                
        except Exception as e:
            log.error(f"Error loading takeout data: {e}")
            raise
    
    def _parse_watch_history(self, html_content: str) -> List[Dict]:
        """Parse watch history from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        history = []
        
        for item in soup.find_all('div', class_='mdl-grid'):
            try:
                title_elem = item.find('a')
                time_elem = item.find('br')
                
                if title_elem:
                    video_data = {
                        'title': title_elem.text.strip(),
                        'url': title_elem.get('href', ''),
                        'timestamp': time_elem.next_sibling.strip() if time_elem else 'Unknown'
                    }
                    
                    # Extract channel if available
                    channel_elem = item.find_all('a')
                    if len(channel_elem) > 1:
                        video_data['channel'] = channel_elem[1].text.strip()
                    
                    history.append(video_data)
            except Exception as e:
                log.warning(f"Error parsing watch history item: {e}")
                continue
        
        return history
    
    def _parse_search_history(self, html_content: str) -> List[Dict]:
        """Parse search history from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        searches = []
        
        for item in soup.find_all('div', class_='mdl-grid'):
            try:
                search_elem = item.find('a')
                time_elem = item.find('br')
                
                if search_elem:
                    search_data = {
                        'query': search_elem.text.strip(),
                        'timestamp': time_elem.next_sibling.strip() if time_elem else 'Unknown'
                    }
                    searches.append(search_data)
            except Exception as e:
                log.warning(f"Error parsing search history item: {e}")
                continue
        
        return searches
    
    def _parse_csv_data(self, csv_content: str) -> List[Dict]:
        """Parse CSV data into list of dictionaries."""
        import io
        data = []
        
        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            for row in reader:
                data.append(dict(row))
        except Exception as e:
            log.warning(f"Error parsing CSV data: {e}")
        
        return data
    
    def _analyze_watch_history(self, input_text: str) -> str:
        """Analyze watch history patterns."""
        if 'watch_history' not in self.data_cache:
            return "Watch history data not available."
        
        history = self.data_cache['watch_history']
        
        # Basic statistics
        total_videos = len(history)
        
        # Channel frequency
        channel_counts = Counter(v.get('channel', 'Unknown') for v in history)
        top_channels = channel_counts.most_common(10)
        
        # Time patterns (simplified - would need proper date parsing for real analysis)
        time_distribution = defaultdict(int)
        for video in history:
            timestamp = video.get('timestamp', '')
            # Extract hour if possible (simplified)
            if 'AM' in timestamp or 'PM' in timestamp:
                parts = timestamp.split(',')
                if len(parts) > 1:
                    time_part = parts[-1].strip()
                    time_distribution[time_part[-2:]] += 1  # AM/PM
        
        # Format results
        result = f"WATCH_HISTORY_ANALYSIS:\n\n"
        result += f"Total videos watched: {total_videos}\n\n"
        
        result += "Top 10 Most Watched Channels:\n"
        for channel, count in top_channels:
            result += f"  - {channel}: {count} videos\n"
        
        if time_distribution:
            result += f"\nViewing time preference: "
            result += "Daytime" if time_distribution.get('PM', 0) > time_distribution.get('AM', 0) else "Morning"
        
        return result
    
    def _analyze_search_history(self, input_text: str) -> str:
        """Analyze search history patterns."""
        if 'search_history' not in self.data_cache:
            return "Search history data not available."
        
        searches = self.data_cache['search_history']
        
        # Basic statistics
        total_searches = len(searches)
        
        # Common search terms
        all_terms = []
        for search in searches:
            query = search.get('query', '').lower()
            # Split into words and filter
            words = [w for w in query.split() if len(w) > 2 and w not in ['the', 'and', 'for', 'how']]
            all_terms.extend(words)
        
        term_counts = Counter(all_terms)
        top_terms = term_counts.most_common(15)
        
        # Format results
        result = f"SEARCH_HISTORY_ANALYSIS:\n\n"
        result += f"Total searches: {total_searches}\n\n"
        
        result += "Top 15 Search Terms:\n"
        for term, count in top_terms:
            result += f"  - {term}: {count} occurrences\n"
        
        return result
    
    def _analyze_subscriptions(self, input_text: str) -> str:
        """Analyze subscription data."""
        if 'subscriptions' not in self.data_cache:
            return "Subscription data not available."
        
        subs = self.data_cache['subscriptions']
        
        # Basic statistics
        total_subs = len(subs)
        
        # Format results
        result = f"SUBSCRIPTION_ANALYSIS:\n\n"
        result += f"Total subscriptions: {total_subs}\n\n"
        
        if subs:
            result += "Subscribed Channels:\n"
            for sub in subs[:20]:  # Show first 20
                channel_name = sub.get('Channel Title', sub.get('Channel Id', 'Unknown'))
                result += f"  - {channel_name}\n"
            
            if total_subs > 20:
                result += f"  ... and {total_subs - 20} more\n"
        
        return result
    
    def _analyze_playlists(self, input_text: str) -> str:
        """Analyze playlist data."""
        if 'playlists' not in self.data_cache:
            return "Playlist data not available."
        
        playlists = self.data_cache['playlists']
        
        # Format results
        result = f"PLAYLIST_ANALYSIS:\n\n"
        result += f"Total playlists: {len(playlists)}\n\n"
        
        for playlist_name, videos in playlists.items():
            result += f"{playlist_name}: {len(videos)} videos\n"
        
        return result
    
    def _general_analysis(self, input_text: str) -> str:
        """Provide general overview of YouTube data."""
        result = "YOUTUBE_DATA_OVERVIEW:\n\n"
        
        # Summary of available data
        result += "Available Data:\n"
        for key in self.data_cache.keys():
            if isinstance(self.data_cache[key], list):
                count = len(self.data_cache[key])
                result += f"  - {key}: {count} items\n"
            elif isinstance(self.data_cache[key], dict):
                count = sum(len(v) if isinstance(v, list) else 1 for v in self.data_cache[key].values())
                result += f"  - {key}: {count} items across {len(self.data_cache[key])} categories\n"
        
        return result
    
    def _comprehensive_analysis(self, input_text: str) -> str:
        """Perform comprehensive analysis of all available data."""
        results = []
        
        # Run all analyses
        results.append(self._general_analysis(input_text))
        results.append(self._analyze_watch_history(input_text))
        results.append(self._analyze_search_history(input_text))
        results.append(self._analyze_subscriptions(input_text))
        results.append(self._analyze_playlists(input_text))
        
        return "\n\n".join(filter(lambda x: "not available" not in x, results))
    
    def _generate_insights(self, analysis_result: str, original_request: str) -> str:
        """Generate insights and recommendations based on analysis."""
        insight_prompt = f"""
Based on the following YouTube data analysis, provide insights and recommendations:

Original Request: {original_request}

Analysis Results:
{analysis_result}

Please provide:
1. Key insights about the user's YouTube usage
2. Interesting patterns or trends
3. Recommendations for content discovery or usage optimization

Keep it concise and actionable.

INSIGHTS:
"""
        
        try:
            # Use the base agent to generate insights
            insights = super().run(insight_prompt)
            return insights.strip()
        except Exception as e:
            log.warning(f"Insight generation failed: {e}")
            return "INSIGHTS: Analysis complete. Review the results above for YouTube usage patterns."
    
    def _save_analysis_results(self, result: str, insights: str):
        """Save analysis results to scratch folder for other agents."""
        try:
            # Create scratch directory if it doesn't exist
            scratch_dir = Path(".talk/scratch")
            scratch_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = scratch_dir / f"youtube_analysis_{timestamp}.json"
            
            # Save results
            analysis_data = {
                "timestamp": timestamp,
                "analysis_result": result,
                "insights": insights,
                "data_summary": {
                    "total_items": sum(
                        len(v) if isinstance(v, list) else 
                        sum(len(vv) if isinstance(vv, list) else 1 for vv in v.values()) if isinstance(v, dict) else 1
                        for v in self.data_cache.values()
                    ),
                    "data_types": list(self.data_cache.keys())
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            log.info(f"Analysis results saved to: {filename}")
            
        except Exception as e:
            log.warning(f"Failed to save analysis results: {e}")


class YoutubeAgentIntegration:
    """
    Integration helper for YoutubeAgent with Talk framework.
    """
    
    @staticmethod
    def should_analyze_youtube_data(task_description: str) -> bool:
        """
        Determine if a task involves YouTube data analysis.
        
        Args:
            task_description: The task description
            
        Returns:
            True if YouTube analysis is needed
        """
        task_lower = task_description.lower()
        
        youtube_keywords = ['youtube', 'watch history', 'search history', 'subscriptions',
                           'playlists', 'video', 'channel', 'takeout']
        
        return any(keyword in task_lower for keyword in youtube_keywords)
    
    @staticmethod
    def create_analysis_prompt(task_description: str, takeout_path: str) -> str:
        """
        Create an analysis prompt for YouTube data.
        
        Args:
            task_description: The analysis task
            takeout_path: Path to takeout file
            
        Returns:
            Formatted analysis prompt
        """
        return f"Analyze YouTube takeout data at {takeout_path} for: {task_description}"


# For backward compatibility and easy imports
__all__ = ['YoutubeAgent', 'YoutubeAgentIntegration']