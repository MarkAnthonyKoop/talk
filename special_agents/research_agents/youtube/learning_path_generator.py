#!/usr/bin/env python3
"""
Learning Path Generator - Creates personalized learning paths from YouTube history

Features:
- Analyzes viewing patterns to identify knowledge areas
- Detects knowledge gaps
- Suggests optimal viewing order
- Creates structured learning curricula
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningNode:
    """Represents a learning concept or video."""
    id: str
    title: str
    type: str  # 'video', 'concept', 'skill'
    level: str  # 'beginner', 'intermediate', 'advanced'
    prerequisites: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    duration_minutes: int = 0
    ai_score: float = 0.0
    watch_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class LearningPath:
    """Represents a complete learning path."""
    goal: str
    current_level: str
    target_level: str
    total_duration_hours: float
    nodes: List[LearningNode]
    phases: Dict[str, List[LearningNode]]
    knowledge_gaps: List[str]
    recommended_resources: List[Dict]


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from viewing history."""
    
    def __init__(self, db_path: str):
        """Initialize with database connection."""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.graph = nx.DiGraph()
        
        # Initialize AI agent for content analysis
        self.analyzer = Agent(
            roles=[
                "You are an expert at analyzing educational content and learning paths.",
                "You identify prerequisites, dependencies, and optimal learning sequences.",
                "You understand technical topics and their relationships.",
                "You can assess difficulty levels and learning progressions."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
    
    def build_graph(self, topic_filter: str = None) -> nx.DiGraph:
        """Build knowledge graph from viewing history."""
        cursor = self.conn.cursor()
        
        # Query videos based on topic
        if topic_filter:
            query = """
                SELECT video_id, title, channel, ai_score, categories, watch_time
                FROM videos
                WHERE LOWER(title) LIKE ? OR categories LIKE ?
                ORDER BY ai_score DESC
                LIMIT 500
            """
            cursor.execute(query, (f'%{topic_filter.lower()}%', f'%{topic_filter}%'))
        else:
            query = """
                SELECT video_id, title, channel, ai_score, categories, watch_time
                FROM videos
                WHERE ai_score > 0.3
                ORDER BY ai_score DESC
                LIMIT 500
            """
            cursor.execute(query)
        
        videos = cursor.fetchall()
        
        # Create nodes for each video
        for video in videos:
            node = LearningNode(
                id=video['video_id'],
                title=video['title'],
                type='video',
                level=self._assess_level(video['title']),
                topics=self._extract_topics(video['title'], video['categories']),
                ai_score=video['ai_score'],
                metadata={'channel': video['channel'], 'watch_time': video['watch_time']}
            )
            
            self.graph.add_node(node.id, data=node)
        
        # Infer relationships
        self._infer_prerequisites()
        
        return self.graph
    
    def _assess_level(self, title: str) -> str:
        """Assess difficulty level from title."""
        title_lower = title.lower()
        
        beginner_keywords = ['intro', 'introduction', 'beginner', 'basic', 'getting started',
                           'tutorial', 'for beginners', 'learn', 'first', 'simple', 'easy']
        advanced_keywords = ['advanced', 'expert', 'deep dive', 'master', 'pro', 'complex',
                           'architecture', 'optimization', 'internals', 'under the hood']
        
        for keyword in beginner_keywords:
            if keyword in title_lower:
                return 'beginner'
        
        for keyword in advanced_keywords:
            if keyword in title_lower:
                return 'advanced'
        
        return 'intermediate'
    
    def _extract_topics(self, title: str, categories: str) -> List[str]:
        """Extract topics from title and categories."""
        topics = []
        
        # Parse categories
        if categories:
            try:
                cat_list = json.loads(categories)
                topics.extend(cat_list)
            except:
                pass
        
        # Extract from title
        title_lower = title.lower()
        topic_keywords = {
            'python': ['python', 'pandas', 'numpy', 'django', 'flask'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
            'ai/ml': ['ai', 'machine learning', 'deep learning', 'neural', 'llm', 'gpt', 'claude'],
            'database': ['database', 'sql', 'mongodb', 'postgresql', 'redis'],
            'devops': ['docker', 'kubernetes', 'k8s', 'ci/cd', 'jenkins', 'aws'],
            'web': ['html', 'css', 'frontend', 'backend', 'api', 'rest']
        }
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    topics.append(topic)
                    break
        
        return list(set(topics))
    
    def _infer_prerequisites(self):
        """Infer prerequisite relationships between videos."""
        nodes = list(self.graph.nodes())
        
        for i, node1 in enumerate(nodes):
            data1 = self.graph.nodes[node1]['data']
            
            for j, node2 in enumerate(nodes):
                if i == j:
                    continue
                
                data2 = self.graph.nodes[node2]['data']
                
                # Simple heuristic: beginner -> intermediate -> advanced
                if data1.level == 'beginner' and data2.level in ['intermediate', 'advanced']:
                    # Check if same topic
                    common_topics = set(data1.topics) & set(data2.topics)
                    if common_topics:
                        self.graph.add_edge(node1, node2, weight=1.0)
                
                elif data1.level == 'intermediate' and data2.level == 'advanced':
                    common_topics = set(data1.topics) & set(data2.topics)
                    if common_topics:
                        self.graph.add_edge(node1, node2, weight=0.8)


class LearningPathGenerator:
    """Generates personalized learning paths."""
    
    def __init__(self, db_path: str):
        """Initialize generator."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.graph_builder = KnowledgeGraphBuilder(db_path)
        
        # AI agent for path generation
        self.path_agent = Agent(
            roles=[
                "You are an expert learning path designer.",
                "You create structured, progressive learning curricula.",
                "You identify knowledge gaps and suggest resources.",
                "You understand different learning styles and paces.",
                "You provide actionable, specific recommendations."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
    
    def generate_path(self, goal: str, current_knowledge: List[str] = None) -> LearningPath:
        """Generate a learning path for a specific goal."""
        logger.info(f"Generating learning path for: {goal}")
        
        # Build knowledge graph
        graph = self.graph_builder.build_graph(topic_filter=goal)
        
        # Analyze current knowledge
        current_level = self._assess_current_level(current_knowledge, graph)
        
        # Find relevant videos
        relevant_nodes = self._find_relevant_nodes(goal, graph)
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_gaps(goal, relevant_nodes, current_knowledge)
        
        # Create learning sequence
        learning_sequence = self._create_sequence(relevant_nodes, current_level)
        
        # Organize into phases
        phases = self._organize_phases(learning_sequence)
        
        # Get external recommendations
        recommendations = self._get_recommendations(goal, knowledge_gaps)
        
        # Calculate total duration
        total_duration = sum(node.duration_minutes for node in learning_sequence) / 60
        
        return LearningPath(
            goal=goal,
            current_level=current_level,
            target_level='advanced',
            total_duration_hours=total_duration,
            nodes=learning_sequence,
            phases=phases,
            knowledge_gaps=knowledge_gaps,
            recommended_resources=recommendations
        )
    
    def _assess_current_level(self, current_knowledge: List[str], graph: nx.DiGraph) -> str:
        """Assess current knowledge level."""
        if not current_knowledge:
            # Check viewing history
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count, AVG(ai_score) as avg_score
                FROM videos
                WHERE ai_score > 0.5
            """)
            
            result = cursor.fetchone()
            if result['count'] > 50 and result['avg_score'] > 0.7:
                return 'intermediate'
            elif result['count'] > 20:
                return 'beginner'
            else:
                return 'novice'
        
        # Analyze provided knowledge
        advanced_count = sum(1 for k in current_knowledge if 'advanced' in k.lower())
        if advanced_count > len(current_knowledge) / 2:
            return 'advanced'
        elif current_knowledge:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _find_relevant_nodes(self, goal: str, graph: nx.DiGraph) -> List[LearningNode]:
        """Find nodes relevant to the learning goal."""
        relevant = []
        goal_lower = goal.lower()
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]['data']
            title_lower = node_data.title.lower()
            
            # Calculate relevance score
            relevance = 0
            
            # Direct keyword match
            goal_words = goal_lower.split()
            for word in goal_words:
                if len(word) > 3 and word in title_lower:
                    relevance += 1
            
            # Topic match
            for topic in node_data.topics:
                if topic.lower() in goal_lower or goal_lower in topic.lower():
                    relevance += 2
            
            # AI score bonus
            relevance += node_data.ai_score
            
            if relevance > 0.5:
                node_data.metadata['relevance'] = relevance
                relevant.append(node_data)
        
        # Sort by relevance and level
        relevant.sort(key=lambda x: (x.metadata.get('relevance', 0), x.level), reverse=True)
        
        return relevant[:50]  # Top 50 most relevant
    
    def _identify_gaps(self, goal: str, available_nodes: List[LearningNode], 
                      current_knowledge: List[str]) -> List[str]:
        """Identify knowledge gaps."""
        # Use AI to identify gaps
        available_titles = [node.title for node in available_nodes[:20]]
        
        prompt = f"""
        Learning goal: {goal}
        
        Available videos in history:
        {json.dumps(available_titles, indent=2)}
        
        Current knowledge: {json.dumps(current_knowledge or [], indent=2)}
        
        Identify 5-10 specific knowledge gaps or missing topics that would be needed
        to achieve the learning goal. Focus on topics NOT covered by the available videos.
        
        Return as a JSON list of strings.
        """
        
        try:
            response = self.path_agent.run(prompt)
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                gaps = json.loads(json_match.group())
                return gaps[:10]
        except Exception as e:
            logger.error(f"Failed to identify gaps: {e}")
        
        # Fallback gaps
        return [
            f"Advanced {goal} concepts",
            f"{goal} best practices",
            f"{goal} real-world projects",
            f"{goal} troubleshooting",
            f"{goal} performance optimization"
        ]
    
    def _create_sequence(self, nodes: List[LearningNode], current_level: str) -> List[LearningNode]:
        """Create optimal learning sequence."""
        # Group by level
        by_level = defaultdict(list)
        for node in nodes:
            by_level[node.level].append(node)
        
        sequence = []
        
        # Start with appropriate level
        if current_level == 'novice':
            sequence.extend(by_level['beginner'][:10])
            sequence.extend(by_level['intermediate'][:10])
            sequence.extend(by_level['advanced'][:5])
        elif current_level == 'beginner':
            sequence.extend(by_level['beginner'][:5])
            sequence.extend(by_level['intermediate'][:15])
            sequence.extend(by_level['advanced'][:10])
        elif current_level == 'intermediate':
            sequence.extend(by_level['intermediate'][:10])
            sequence.extend(by_level['advanced'][:15])
        else:  # advanced
            sequence.extend(by_level['advanced'][:20])
        
        return sequence
    
    def _organize_phases(self, sequence: List[LearningNode]) -> Dict[str, List[LearningNode]]:
        """Organize learning sequence into phases."""
        phases = {
            'foundation': [],
            'core': [],
            'advanced': [],
            'mastery': []
        }
        
        total = len(sequence)
        if total == 0:
            return phases
        
        # Distribute nodes across phases
        foundation_end = int(total * 0.2)
        core_end = int(total * 0.5)
        advanced_end = int(total * 0.8)
        
        phases['foundation'] = sequence[:foundation_end]
        phases['core'] = sequence[foundation_end:core_end]
        phases['advanced'] = sequence[core_end:advanced_end]
        phases['mastery'] = sequence[advanced_end:]
        
        return phases
    
    def _get_recommendations(self, goal: str, gaps: List[str]) -> List[Dict]:
        """Get external resource recommendations."""
        recommendations = []
        
        # Use AI to suggest resources
        prompt = f"""
        Learning goal: {goal}
        Knowledge gaps: {json.dumps(gaps, indent=2)}
        
        Suggest 5-10 specific external resources (courses, books, tutorials, documentation)
        that would help fill these knowledge gaps. Focus on high-quality, practical resources.
        
        Return as JSON list with format:
        [
            {{"type": "course/book/tutorial/doc", "title": "...", "description": "...", "url": "optional"}}
        ]
        """
        
        try:
            response = self.path_agent.run(prompt)
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
        
        # Add default recommendations if none found
        if not recommendations:
            recommendations = [
                {
                    "type": "documentation",
                    "title": f"Official {goal} Documentation",
                    "description": "Primary reference for learning"
                },
                {
                    "type": "tutorial",
                    "title": f"{goal} Hands-on Tutorial",
                    "description": "Practical exercises and projects"
                }
            ]
        
        return recommendations
    
    def generate_report(self, path: LearningPath) -> str:
        """Generate a detailed learning path report."""
        report = f"""
# Learning Path: {path.goal}

## Your Current Level: {path.current_level}
## Target Level: {path.target_level}
## Estimated Duration: {path.total_duration_hours:.1f} hours

## Learning Phases

### Phase 1: Foundation ({len(path.phases.get('foundation', []))} items)
Build fundamental understanding of core concepts.

"""
        for node in path.phases.get('foundation', [])[:5]:
            report += f"- {node.title}\n"
        
        report += f"""

### Phase 2: Core Skills ({len(path.phases.get('core', []))} items)
Develop practical skills and deeper understanding.

"""
        for node in path.phases.get('core', [])[:5]:
            report += f"- {node.title}\n"
        
        report += f"""

### Phase 3: Advanced Topics ({len(path.phases.get('advanced', []))} items)
Master complex concepts and techniques.

"""
        for node in path.phases.get('advanced', [])[:5]:
            report += f"- {node.title}\n"
        
        report += f"""

### Phase 4: Mastery ({len(path.phases.get('mastery', []))} items)
Achieve expert-level understanding.

"""
        for node in path.phases.get('mastery', [])[:5]:
            report += f"- {node.title}\n"
        
        report += """

## Knowledge Gaps to Address

You should focus on these areas not covered in your viewing history:

"""
        for gap in path.knowledge_gaps:
            report += f"- {gap}\n"
        
        report += """

## Recommended External Resources

To fill knowledge gaps, consider these resources:

"""
        for rec in path.recommended_resources[:5]:
            report += f"- **{rec['title']}** ({rec['type']}): {rec['description']}\n"
        
        report += """

## Next Steps

1. Start with the foundation phase videos in your history
2. Supplement with external resources for gap areas
3. Practice with hands-on projects between phases
4. Track your progress and adjust pace as needed
5. Revisit complex topics multiple times

*Remember: Learning is not linear. Feel free to jump between phases based on your interests and needs.*
"""
        
        return report


def main():
    """CLI for learning path generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Learning Path Generator")
    parser.add_argument("goal", help="Learning goal (e.g., 'machine learning', 'React development')")
    parser.add_argument("--db", default="youtube_fast.db", help="Database path")
    parser.add_argument("--current-knowledge", nargs='+', help="Current knowledge areas")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        db_path = Path.home() / "code" / "miniapps" / "youtube_database" / args.db
    
    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return
    
    # Generate learning path
    generator = LearningPathGenerator(str(db_path))
    path = generator.generate_path(args.goal, args.current_knowledge)
    
    # Generate report
    report = generator.generate_report(path)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()