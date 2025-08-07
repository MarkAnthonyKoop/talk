#!/usr/bin/env python3
"""
YouTube Analytics Engine - Advanced analytics and visualization

Features:
- Viewing velocity trends
- Topic evolution analysis
- Completion rate estimation
- Recommendation accuracy tracking
- Interactive visualizations
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import logging
import numpy as np
from scipy import stats

# Visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Install matplotlib and seaborn for visualizations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ViewingMetrics:
    """Container for viewing metrics."""
    total_videos: int
    total_hours: float
    unique_channels: int
    avg_videos_per_day: float
    peak_viewing_day: str
    peak_viewing_count: int
    ai_content_percentage: float
    favorite_channels: List[Tuple[str, int]]
    viewing_trends: Dict[str, Any]
    topic_distribution: Dict[str, int]


@dataclass
class TemporalAnalysis:
    """Container for temporal analysis results."""
    daily_counts: Dict[str, int]
    weekly_patterns: Dict[str, float]
    monthly_trends: Dict[str, int]
    yearly_growth: Dict[str, int]
    peak_periods: List[Dict]
    viewing_velocity: float  # Videos per week
    acceleration: float  # Change in velocity


class AnalyticsEngine:
    """Advanced analytics for YouTube viewing history."""
    
    def __init__(self, db_path: str):
        """Initialize analytics engine."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
    
    def get_overview_metrics(self) -> ViewingMetrics:
        """Get comprehensive viewing metrics."""
        cursor = self.conn.cursor()
        
        # Total videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        # Estimated total hours (assuming avg 10 min per video)
        total_hours = total_videos * 10 / 60
        
        # Unique channels
        cursor.execute("SELECT COUNT(DISTINCT channel) FROM videos WHERE channel != 'Unknown'")
        unique_channels = cursor.fetchone()[0]
        
        # Daily viewing stats
        cursor.execute("""
            SELECT DATE(watch_time) as date, COUNT(*) as count
            FROM videos
            WHERE watch_time IS NOT NULL
            GROUP BY DATE(watch_time)
            ORDER BY count DESC
        """)
        
        daily_stats = cursor.fetchall()
        
        if daily_stats:
            peak_day = daily_stats[0]
            peak_viewing_day = peak_day['date']
            peak_viewing_count = peak_day['count']
            
            # Calculate average videos per day
            total_days = len(daily_stats)
            avg_videos_per_day = total_videos / max(total_days, 1)
        else:
            peak_viewing_day = "Unknown"
            peak_viewing_count = 0
            avg_videos_per_day = 0
        
        # AI content percentage
        cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
        ai_videos = cursor.fetchone()[0]
        ai_percentage = (ai_videos / total_videos * 100) if total_videos > 0 else 0
        
        # Favorite channels
        cursor.execute("""
            SELECT channel, COUNT(*) as count
            FROM videos
            WHERE channel != 'Unknown'
            GROUP BY channel
            ORDER BY count DESC
            LIMIT 10
        """)
        favorite_channels = [(row['channel'], row['count']) for row in cursor.fetchall()]
        
        # Topic distribution
        topic_dist = self._get_topic_distribution()
        
        # Viewing trends
        trends = self._analyze_viewing_trends()
        
        return ViewingMetrics(
            total_videos=total_videos,
            total_hours=total_hours,
            unique_channels=unique_channels,
            avg_videos_per_day=avg_videos_per_day,
            peak_viewing_day=peak_viewing_day,
            peak_viewing_count=peak_viewing_count,
            ai_content_percentage=ai_percentage,
            favorite_channels=favorite_channels,
            viewing_trends=trends,
            topic_distribution=topic_dist
        )
    
    def analyze_temporal_patterns(self) -> TemporalAnalysis:
        """Analyze temporal viewing patterns."""
        cursor = self.conn.cursor()
        
        # Get all videos with timestamps
        cursor.execute("""
            SELECT DATE(watch_time) as date, COUNT(*) as count
            FROM videos
            WHERE watch_time IS NOT NULL
            GROUP BY DATE(watch_time)
            ORDER BY date
        """)
        
        daily_data = cursor.fetchall()
        
        if not daily_data:
            return TemporalAnalysis(
                daily_counts={},
                weekly_patterns={},
                monthly_trends={},
                yearly_growth={},
                peak_periods=[],
                viewing_velocity=0,
                acceleration=0
            )
        
        # Process daily counts
        daily_counts = {row['date']: row['count'] for row in daily_data}
        
        # Weekly patterns (day of week analysis)
        weekly_patterns = defaultdict(list)
        for date_str, count in daily_counts.items():
            try:
                date = datetime.fromisoformat(date_str)
                day_name = date.strftime('%A')
                weekly_patterns[day_name].append(count)
            except:
                continue
        
        # Average by day of week
        weekly_avg = {
            day: np.mean(counts) if counts else 0
            for day, counts in weekly_patterns.items()
        }
        
        # Monthly trends
        monthly_trends = defaultdict(int)
        for date_str, count in daily_counts.items():
            try:
                date = datetime.fromisoformat(date_str)
                month_key = date.strftime('%Y-%m')
                monthly_trends[month_key] += count
            except:
                continue
        
        # Yearly growth
        yearly_growth = defaultdict(int)
        for date_str, count in daily_counts.items():
            try:
                date = datetime.fromisoformat(date_str)
                year = str(date.year)
                yearly_growth[year] += count
            except:
                continue
        
        # Calculate viewing velocity (videos per week)
        if len(daily_counts) > 7:
            recent_week = list(daily_counts.values())[-7:]
            older_week = list(daily_counts.values())[-14:-7] if len(daily_counts) > 14 else [0]
            
            current_velocity = sum(recent_week)
            previous_velocity = sum(older_week)
            acceleration = current_velocity - previous_velocity
        else:
            current_velocity = sum(daily_counts.values()) / max(len(daily_counts) / 7, 1)
            acceleration = 0
        
        # Identify peak periods
        peak_periods = self._identify_peak_periods(daily_counts)
        
        return TemporalAnalysis(
            daily_counts=dict(daily_counts),
            weekly_patterns=dict(weekly_avg),
            monthly_trends=dict(monthly_trends),
            yearly_growth=dict(yearly_growth),
            peak_periods=peak_periods,
            viewing_velocity=current_velocity,
            acceleration=acceleration
        )
    
    def analyze_topic_evolution(self, months: int = 12) -> Dict[str, Any]:
        """Analyze how topic interests evolved over time."""
        cursor = self.conn.cursor()
        
        # Get videos with timestamps and categories
        cutoff = datetime.now() - timedelta(days=months * 30)
        
        cursor.execute("""
            SELECT watch_time, title, categories, ai_score
            FROM videos
            WHERE watch_time IS NOT NULL
            AND DATE(watch_time) >= ?
            ORDER BY watch_time
        """, (cutoff.isoformat(),))
        
        videos = cursor.fetchall()
        
        # Group by month and analyze topics
        monthly_topics = defaultdict(lambda: defaultdict(int))
        
        for video in videos:
            try:
                date = datetime.fromisoformat(video['watch_time'])
                month_key = date.strftime('%Y-%m')
                
                # Extract topics from title and categories
                topics = self._extract_video_topics(video)
                for topic in topics:
                    monthly_topics[month_key][topic] += 1
            except:
                continue
        
        # Calculate topic trends
        topic_trends = {}
        all_topics = set()
        for month_data in monthly_topics.values():
            all_topics.update(month_data.keys())
        
        for topic in all_topics:
            monthly_counts = []
            months_list = sorted(monthly_topics.keys())
            
            for month in months_list:
                monthly_counts.append(monthly_topics[month].get(topic, 0))
            
            if sum(monthly_counts) > 5:  # Only include topics with significant presence
                # Calculate trend (increasing, decreasing, stable)
                if len(monthly_counts) > 1:
                    trend = "increasing" if monthly_counts[-1] > monthly_counts[0] else "decreasing"
                else:
                    trend = "stable"
                
                topic_trends[topic] = {
                    "total": sum(monthly_counts),
                    "trend": trend,
                    "monthly": dict(zip(months_list, monthly_counts))
                }
        
        # Identify emerging topics
        emerging = []
        declining = []
        
        for topic, data in topic_trends.items():
            monthly = list(data['monthly'].values())
            if len(monthly) > 3:
                recent_avg = np.mean(monthly[-3:])
                older_avg = np.mean(monthly[:-3])
                
                if recent_avg > older_avg * 1.5:
                    emerging.append(topic)
                elif recent_avg < older_avg * 0.5:
                    declining.append(topic)
        
        return {
            "topic_trends": topic_trends,
            "emerging_topics": emerging,
            "declining_topics": declining,
            "topic_diversity": len(all_topics),
            "analysis_period_months": months
        }
    
    def estimate_completion_rates(self) -> Dict[str, Any]:
        """Estimate video completion rates based on patterns."""
        cursor = self.conn.cursor()
        
        # Get videos grouped by length indicators
        cursor.execute("""
            SELECT title, COUNT(*) as watch_count
            FROM videos
            GROUP BY title
            HAVING watch_count > 1
            ORDER BY watch_count DESC
        """)
        
        rewatched = cursor.fetchall()
        
        # Categorize by video type
        short_form = []  # < 10 min
        medium_form = []  # 10-30 min
        long_form = []  # > 30 min
        
        for video in rewatched:
            title = video['title'].lower()
            
            # Estimate video length from title patterns
            if any(word in title for word in ['short', 'quick', 'minute', 'tip']):
                short_form.append(video)
            elif any(word in title for word in ['documentary', 'lecture', 'course', 'full']):
                long_form.append(video)
            else:
                medium_form.append(video)
        
        # Calculate estimated completion rates
        # Assumption: rewatched videos = higher completion
        # Single watch = possibly lower completion
        
        total_unique = len(rewatched)
        total_rewatched = sum(v['watch_count'] for v in rewatched)
        
        estimated_rates = {
            "highly_engaged": len([v for v in rewatched if v['watch_count'] > 2]),
            "moderately_engaged": len([v for v in rewatched if v['watch_count'] == 2]),
            "single_view": total_unique - len(rewatched),
            "rewatch_rate": (total_rewatched / total_unique) if total_unique > 0 else 0,
            "by_length": {
                "short": len(short_form),
                "medium": len(medium_form),
                "long": len(long_form)
            }
        }
        
        return estimated_rates
    
    def track_recommendation_accuracy(self, recommendations: List[str], 
                                    watched_after: List[str]) -> Dict[str, float]:
        """Track how accurate recommendations were."""
        if not recommendations:
            return {"accuracy": 0, "precision": 0, "recall": 0}
        
        # Calculate metrics
        recommended_set = set(recommendations)
        watched_set = set(watched_after)
        
        true_positives = len(recommended_set & watched_set)
        false_positives = len(recommended_set - watched_set)
        false_negatives = len(watched_set - recommended_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": true_positives / len(recommendations),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "recommendations_followed": true_positives,
            "recommendations_ignored": false_positives
        }
    
    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report."""
        metrics = self.get_overview_metrics()
        temporal = self.analyze_temporal_patterns()
        evolution = self.analyze_topic_evolution(12)
        completion = self.estimate_completion_rates()
        
        report = f"""
# YouTube Viewing Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
- **Total Videos Watched**: {metrics.total_videos:,}
- **Estimated Hours**: {metrics.total_hours:.1f}
- **Unique Channels**: {metrics.unique_channels}
- **Average Videos/Day**: {metrics.avg_videos_per_day:.1f}
- **AI Content**: {metrics.ai_content_percentage:.1f}%

## Peak Activity
- **Most Active Day**: {metrics.peak_viewing_day} ({metrics.peak_viewing_count} videos)
- **Current Velocity**: {temporal.viewing_velocity:.1f} videos/week
- **Acceleration**: {temporal.acceleration:+.1f} videos/week²

## Top Channels
"""
        for channel, count in metrics.favorite_channels[:5]:
            report += f"- {channel}: {count} videos\n"
        
        report += f"""

## Topic Evolution (Last 12 Months)
- **Topic Diversity**: {evolution['topic_diversity']} unique topics
- **Emerging Topics**: {', '.join(evolution['emerging_topics'][:5]) if evolution['emerging_topics'] else 'None'}
- **Declining Topics**: {', '.join(evolution['declining_topics'][:5]) if evolution['declining_topics'] else 'None'}

## Engagement Metrics
- **Highly Engaged Videos**: {completion['highly_engaged']} (watched 3+ times)
- **Rewatch Rate**: {completion['rewatch_rate']:.2f}x average
- **Content Preference**:
  - Short-form: {completion['by_length']['short']} videos
  - Medium-form: {completion['by_length']['medium']} videos
  - Long-form: {completion['by_length']['long']} videos

## Weekly Patterns
"""
        if temporal.weekly_patterns:
            sorted_days = sorted(temporal.weekly_patterns.items(), 
                               key=lambda x: x[1], reverse=True)
            for day, avg in sorted_days[:3]:
                report += f"- {day}: {avg:.1f} videos average\n"
        
        report += """

## Insights & Recommendations

1. **Learning Momentum**: """
        
        if temporal.acceleration > 0:
            report += "Your viewing is accelerating - great time to start structured learning paths!"
        else:
            report += "Your viewing has slowed - consider setting learning goals to re-engage."
        
        report += """

2. **Content Diversity**: """
        
        if evolution['topic_diversity'] > 20:
            report += "You explore diverse topics - consider focusing on fewer areas for deeper expertise."
        else:
            report += "You have focused interests - good foundation for specialized learning."
        
        report += """

3. **Engagement Quality**: """
        
        if completion['rewatch_rate'] > 1.5:
            report += "High rewatch rate indicates deep engagement with content."
        else:
            report += "Low rewatch rate - consider reviewing important videos multiple times."
        
        report += """

---
*Use these insights to optimize your learning journey and make informed decisions about future content consumption.*
"""
        
        return report
    
    def visualize_analytics(self, output_dir: Path = None):
        """Generate visualization plots."""
        if not HAS_PLOTTING:
            logger.warning("Matplotlib not available for visualizations")
            return
        
        if output_dir is None:
            output_dir = Path("analytics_plots")
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Daily viewing trend
        temporal = self.analyze_temporal_patterns()
        if temporal.daily_counts:
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = list(temporal.daily_counts.keys())[-90:]  # Last 90 days
            counts = [temporal.daily_counts[d] for d in dates]
            
            ax.plot(range(len(dates)), counts, 'b-', alpha=0.7)
            ax.fill_between(range(len(dates)), counts, alpha=0.3)
            ax.set_title('YouTube Viewing Trend (Last 90 Days)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Videos Watched')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'viewing_trend.png', dpi=150)
            plt.close()
        
        # 2. Topic distribution pie chart
        metrics = self.get_overview_metrics()
        if metrics.topic_distribution:
            fig, ax = plt.subplots(figsize=(10, 8))
            topics = list(metrics.topic_distribution.keys())[:10]
            sizes = [metrics.topic_distribution[t] for t in topics]
            
            ax.pie(sizes, labels=topics, autopct='%1.1f%%', startangle=90)
            ax.set_title('Top 10 Topic Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'topic_distribution.png', dpi=150)
            plt.close()
        
        # 3. Weekly pattern heatmap
        if temporal.weekly_patterns:
            fig, ax = plt.subplots(figsize=(8, 6))
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            values = [temporal.weekly_patterns.get(d, 0) for d in days]
            
            bars = ax.bar(days, values, color='steelblue')
            ax.set_title('Average Videos by Day of Week')
            ax.set_ylabel('Average Videos')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'weekly_pattern.png', dpi=150)
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    # Helper methods
    
    def _get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of topics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT categories FROM videos WHERE categories IS NOT NULL")
        
        topic_counts = Counter()
        for row in cursor.fetchall():
            try:
                categories = json.loads(row['categories'])
                for cat in categories:
                    topic_counts[cat] += 1
            except:
                continue
        
        return dict(topic_counts)
    
    def _analyze_viewing_trends(self) -> Dict[str, Any]:
        """Analyze viewing trends."""
        cursor = self.conn.cursor()
        
        # Get recent vs older viewing
        recent_cutoff = datetime.now() - timedelta(days=30)
        older_cutoff = datetime.now() - timedelta(days=90)
        
        cursor.execute("""
            SELECT COUNT(*) FROM videos
            WHERE watch_time IS NOT NULL
            AND DATE(watch_time) >= ?
        """, (recent_cutoff.isoformat(),))
        recent_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM videos
            WHERE watch_time IS NOT NULL
            AND DATE(watch_time) < ? AND DATE(watch_time) >= ?
        """, (recent_cutoff.isoformat(), older_cutoff.isoformat()))
        older_count = cursor.fetchone()[0]
        
        trend = "increasing" if recent_count > older_count else "decreasing"
        
        return {
            "recent_30_days": recent_count,
            "previous_60_days": older_count,
            "trend": trend,
            "change_percentage": ((recent_count - older_count) / max(older_count, 1)) * 100
        }
    
    def _identify_peak_periods(self, daily_counts: Dict[str, int]) -> List[Dict]:
        """Identify peak viewing periods."""
        if not daily_counts:
            return []
        
        # Calculate mean and std
        counts = list(daily_counts.values())
        mean = np.mean(counts)
        std = np.std(counts)
        
        # Find peaks (2 std above mean)
        threshold = mean + 2 * std
        peaks = []
        
        for date, count in daily_counts.items():
            if count > threshold:
                peaks.append({
                    "date": date,
                    "count": count,
                    "deviation": (count - mean) / std
                })
        
        return sorted(peaks, key=lambda x: x['count'], reverse=True)[:10]
    
    def _extract_video_topics(self, video: sqlite3.Row) -> List[str]:
        """Extract topics from a video."""
        topics = []
        
        # From categories
        if video['categories']:
            try:
                cats = json.loads(video['categories'])
                topics.extend(cats)
            except:
                pass
        
        # From title keywords
        title = video['title'].lower()
        if 'python' in title:
            topics.append('Python')
        if 'javascript' in title or ' js ' in title:
            topics.append('JavaScript')
        if 'machine learning' in title or ' ml ' in title:
            topics.append('ML')
        if 'ai' in title or 'artificial' in title:
            topics.append('AI')
        
        return list(set(topics))


def main():
    """CLI for analytics engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Analytics Engine")
    parser.add_argument("--db", default="youtube_fast.db", help="Database path")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        db_path = Path.home() / "code" / "miniapps" / "youtube_database" / args.db
    
    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return
    
    # Initialize engine
    engine = AnalyticsEngine(str(db_path))
    
    if args.report:
        report = engine.generate_insights_report()
        print(report)
    
    if args.visualize:
        output_dir = Path(args.output) if args.output else None
        engine.visualize_analytics(output_dir)
        print(f"Visualizations saved to {output_dir or 'analytics_plots'}")
    
    if not args.report and not args.visualize:
        # Default to overview
        metrics = engine.get_overview_metrics()
        print(f"Total videos: {metrics.total_videos}")
        print(f"AI content: {metrics.ai_content_percentage:.1f}%")
        print(f"Top channel: {metrics.favorite_channels[0][0] if metrics.favorite_channels else 'Unknown'}")


if __name__ == "__main__":
    main()