#!/usr/bin/env python3
"""
Category Tree Builder - Dynamic categorization and visualization of YouTube history

Features:
- AI-powered categorization at index time
- Runtime reorganization based on prompts
- Interactive tree visualization
- Multiple categorization schemes
- Cached category indexes
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import pickle
import logging

# Visualization libraries
try:
    import networkx as nx
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Install plotly for interactive visualizations: pip install plotly networkx")

# Add project root
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CategoryNode:
    """Represents a node in the category tree."""
    id: str
    name: str
    level: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    videos: List[Dict] = field(default_factory=list)
    video_count: int = 0
    total_watch_time: float = 0
    metadata: Dict = field(default_factory=dict)
    

@dataclass 
class CategoryScheme:
    """A complete categorization scheme."""
    name: str
    description: str
    root_categories: List[str]
    nodes: Dict[str, CategoryNode]
    created_at: datetime
    prompt_used: str
    

class CategoryTreeBuilder:
    """Builds and manages category trees for YouTube videos."""
    
    def __init__(self, db_path: str, cache_dir: Path = None):
        """Initialize the category tree builder."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Cache directory for category schemes
        self.cache_dir = cache_dir or Path.home() / ".cache" / "youtube_categories"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # AI agent for categorization
        self.categorizer = Agent(
            roles=[
                "You are an expert at categorizing educational and entertainment content.",
                "You create logical, hierarchical category structures.",
                "You understand technical topics and their relationships.",
                "You can reorganize content based on different perspectives.",
                "You identify patterns and group related content effectively."
            ],
            overrides={"llm": {"provider": "anthropic"}}
        )
        
        # Load cached schemes
        self.schemes = self._load_cached_schemes()
    
    def build_category_tree(self, prompt: str = None, scheme_name: str = None) -> CategoryScheme:
        """
        Build a category tree based on a prompt or use default categorization.
        
        Args:
            prompt: Custom categorization prompt (e.g., "organize by learning difficulty")
            scheme_name: Name for this categorization scheme
        """
        if not prompt:
            prompt = "Create a hierarchical category tree for these videos focusing on topic and purpose"
        
        if not scheme_name:
            scheme_name = f"scheme_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        
        # Check cache first
        if scheme_name in self.schemes:
            logger.info(f"Using cached scheme: {scheme_name}")
            return self.schemes[scheme_name]
        
        logger.info(f"Building new category tree with prompt: {prompt}")
        
        # Get all videos
        videos = self._load_videos()
        
        # Get AI categorization
        categories = self._get_ai_categories(videos, prompt)
        
        # Build tree structure
        scheme = self._build_tree_structure(categories, videos, scheme_name, prompt)
        
        # Cache the scheme
        self._cache_scheme(scheme)
        self.schemes[scheme_name] = scheme
        
        return scheme
    
    def _load_videos(self, limit: int = 5000) -> List[Dict]:
        """Load videos from database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT video_id, title, channel, url, watch_time, ai_score, categories
            FROM videos
            ORDER BY ai_score DESC, watch_time DESC
            LIMIT ?
        """, (limit,))
        
        videos = []
        for row in cursor.fetchall():
            videos.append({
                'video_id': row['video_id'],
                'title': row['title'],
                'channel': row['channel'],
                'url': row['url'],
                'watch_time': row['watch_time'],
                'ai_score': row['ai_score'],
                'categories': row['categories']
            })
        
        logger.info(f"Loaded {len(videos)} videos")
        return videos
    
    def _get_ai_categories(self, videos: List[Dict], prompt: str) -> Dict:
        """Get AI-generated categories for videos."""
        # Sample videos for categorization (can't send all 5000 to AI)
        sample_size = min(100, len(videos))
        sample_videos = videos[:sample_size]
        
        # Prepare titles for categorization
        titles = [v['title'] for v in sample_videos]
        
        categorization_prompt = f"""
        {prompt}
        
        Analyze these video titles and create a hierarchical category tree.
        
        Video titles (sample of {sample_size}):
        {json.dumps(titles[:50], indent=2)}
        
        Create a JSON structure with:
        1. Top-level categories (5-10 main categories)
        2. Subcategories (2-5 per main category)
        3. Rules for categorizing videos
        
        Return JSON like:
        {{
            "categories": {{
                "Technology": {{
                    "subcategories": ["Programming", "AI/ML", "Web Development"],
                    "keywords": ["code", "programming", "software", "development"]
                }},
                "Education": {{
                    "subcategories": ["Tutorials", "Courses", "Lectures"],
                    "keywords": ["learn", "tutorial", "course", "how to"]
                }}
            }},
            "default_category": "Miscellaneous"
        }}
        """
        
        try:
            response = self.categorizer.run(categorization_prompt)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                categories = json.loads(json_match.group())
                return categories
            else:
                raise ValueError("No JSON in response")
                
        except Exception as e:
            logger.error(f"AI categorization failed: {e}")
            # Fallback categories
            return self._get_default_categories()
    
    def _get_default_categories(self) -> Dict:
        """Get default category structure."""
        return {
            "categories": {
                "Technology": {
                    "subcategories": ["Programming", "AI/ML", "DevOps", "Web Dev"],
                    "keywords": ["code", "programming", "software", "python", "javascript", "ai", "ml", "docker", "kubernetes"]
                },
                "Education": {
                    "subcategories": ["Tutorials", "Courses", "Lectures", "Workshops"],
                    "keywords": ["learn", "tutorial", "course", "lecture", "teach", "lesson", "class"]
                },
                "Science": {
                    "subcategories": ["Physics", "Math", "Biology", "Space"],
                    "keywords": ["science", "physics", "math", "biology", "chemistry", "space", "quantum", "research"]
                },
                "Entertainment": {
                    "subcategories": ["Music", "Gaming", "Comedy", "Movies"],
                    "keywords": ["music", "song", "game", "gaming", "funny", "comedy", "movie", "film"]
                },
                "Professional": {
                    "subcategories": ["Career", "Business", "Productivity", "Leadership"],
                    "keywords": ["career", "job", "business", "productivity", "management", "leadership", "interview"]
                },
                "Creative": {
                    "subcategories": ["Art", "Design", "Writing", "Photography"],
                    "keywords": ["art", "design", "creative", "drawing", "writing", "photography", "animation"]
                },
                "Lifestyle": {
                    "subcategories": ["Health", "Fitness", "Travel", "Cooking"],
                    "keywords": ["health", "fitness", "travel", "cooking", "food", "recipe", "workout"]
                }
            },
            "default_category": "Miscellaneous"
        }
    
    def _build_tree_structure(self, categories: Dict, videos: List[Dict], 
                             scheme_name: str, prompt: str) -> CategoryScheme:
        """Build the tree structure from categories and videos."""
        nodes = {}
        root_categories = []
        
        # Create root node
        root_node = CategoryNode(
            id="root",
            name="All Videos",
            level=0
        )
        nodes["root"] = root_node
        
        # Create category nodes
        for cat_name, cat_info in categories.get("categories", {}).items():
            # Main category node
            cat_id = cat_name.lower().replace(" ", "_")
            cat_node = CategoryNode(
                id=cat_id,
                name=cat_name,
                level=1,
                parent="root"
            )
            nodes[cat_id] = cat_node
            root_node.children.append(cat_id)
            root_categories.append(cat_id)
            
            # Subcategory nodes
            for subcat_name in cat_info.get("subcategories", []):
                subcat_id = f"{cat_id}_{subcat_name.lower().replace(' ', '_')}"
                subcat_node = CategoryNode(
                    id=subcat_id,
                    name=subcat_name,
                    level=2,
                    parent=cat_id
                )
                nodes[subcat_id] = subcat_node
                cat_node.children.append(subcat_id)
        
        # Add miscellaneous category
        misc_node = CategoryNode(
            id="miscellaneous",
            name="Miscellaneous",
            level=1,
            parent="root"
        )
        nodes["miscellaneous"] = misc_node
        root_node.children.append("miscellaneous")
        
        # Categorize videos
        for video in videos:
            category_id = self._categorize_video(video, categories)
            if category_id in nodes:
                nodes[category_id].videos.append(video)
                nodes[category_id].video_count += 1
                
                # Update parent counts
                parent_id = nodes[category_id].parent
                while parent_id:
                    nodes[parent_id].video_count += 1
                    parent_id = nodes[parent_id].parent if parent_id in nodes else None
        
        # Create scheme
        scheme = CategoryScheme(
            name=scheme_name,
            description=f"Categorization based on: {prompt[:100]}",
            root_categories=root_categories,
            nodes=nodes,
            created_at=datetime.now(),
            prompt_used=prompt
        )
        
        return scheme
    
    def _categorize_video(self, video: Dict, categories: Dict) -> str:
        """Categorize a single video."""
        title_lower = video['title'].lower()
        
        # Check each category's keywords
        for cat_name, cat_info in categories.get("categories", {}).items():
            cat_id = cat_name.lower().replace(" ", "_")
            
            # Check main category keywords
            for keyword in cat_info.get("keywords", []):
                if keyword in title_lower:
                    # Try to find matching subcategory
                    for subcat_name in cat_info.get("subcategories", []):
                        subcat_keywords = self._get_subcat_keywords(subcat_name)
                        for subcat_kw in subcat_keywords:
                            if subcat_kw in title_lower:
                                return f"{cat_id}_{subcat_name.lower().replace(' ', '_')}"
                    
                    # Return main category if no subcategory matches
                    return cat_id
        
        # Default to miscellaneous
        return "miscellaneous"
    
    def _get_subcat_keywords(self, subcat_name: str) -> List[str]:
        """Get keywords for a subcategory."""
        keyword_map = {
            "Programming": ["python", "javascript", "java", "code", "programming", "function", "class"],
            "AI/ML": ["ai", "ml", "machine learning", "neural", "gpt", "claude", "llm"],
            "Tutorials": ["tutorial", "how to", "guide", "learn", "beginner"],
            "Music": ["music", "song", "album", "concert", "live", "official"],
            "Gaming": ["game", "gaming", "gameplay", "walkthrough", "stream"]
        }
        
        return keyword_map.get(subcat_name, [subcat_name.lower()])
    
    def reorganize_tree(self, scheme: CategoryScheme, new_prompt: str) -> CategoryScheme:
        """Reorganize an existing tree based on a new prompt."""
        logger.info(f"Reorganizing tree with prompt: {new_prompt}")
        
        # Get all videos from the current scheme
        all_videos = []
        for node in scheme.nodes.values():
            all_videos.extend(node.videos)
        
        # Remove duplicates
        seen = set()
        unique_videos = []
        for v in all_videos:
            if v['video_id'] not in seen:
                seen.add(v['video_id'])
                unique_videos.append(v)
        
        # Build new tree with new prompt
        return self.build_category_tree(new_prompt, f"reorg_{hashlib.md5(new_prompt.encode()).hexdigest()[:8]}")
    
    def visualize_tree(self, scheme: CategoryScheme, output_file: str = None, 
                       show_in_browser: bool = True) -> Optional[str]:
        """Create an interactive visualization of the category tree."""
        if not HAS_PLOTLY:
            logger.error("Plotly not installed. Cannot create visualization.")
            return None
        
        # Build network graph
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in scheme.nodes.items():
            G.add_node(node_id, 
                      name=node.name,
                      level=node.level,
                      count=node.video_count)
        
        # Add edges
        for node_id, node in scheme.nodes.items():
            if node.parent:
                G.add_edge(node.parent, node_id)
        
        # Calculate positions using hierarchical layout
        pos = self._hierarchical_layout(G, "root")
        
        # Create Plotly figure
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            ))
        
        # Node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Video Count',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            node_data = scheme.nodes[node_id]
            node_trace['text'] += tuple([node_data.name])
            node_trace['marker']['color'] += tuple([node_data.video_count])
            node_trace['marker']['size'] += tuple([min(50, 10 + node_data.video_count/10)])
            
            # Hover text
            hover_text = f"{node_data.name}<br>"
            hover_text += f"Videos: {node_data.video_count}<br>"
            if node_data.level > 0:
                hover_text += f"Level: {node_data.level}<br>"
            if node_data.videos and len(node_data.videos) > 0:
                hover_text += f"Sample: {node_data.videos[0]['title'][:50]}..."
            
            node_trace['hovertext'] = node_trace.get('hovertext', ()) + tuple([hover_text])
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title=f"YouTube Category Tree: {scheme.name}",
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text=f"Created: {scheme.created_at.strftime('%Y-%m-%d')}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=800
                       ))
        
        # Save or show
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Visualization saved to {output_file}")
            return output_file
        
        if show_in_browser:
            fig.show()
        
        return None
    
    def _hierarchical_layout(self, G: nx.DiGraph, root: str) -> Dict:
        """Create hierarchical layout for tree."""
        pos = {}
        levels = {}
        
        # BFS to assign levels
        queue = [(root, 0)]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            for child in G.successors(node):
                queue.append((child, level + 1))
        
        # Assign positions
        y_gap = 1.0
        for level, nodes in levels.items():
            x_gap = 2.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                pos[node] = ((i + 1) * x_gap - 1.0, -level * y_gap)
        
        return pos
    
    def export_tree(self, scheme: CategoryScheme, format: str = "json") -> str:
        """Export tree in various formats."""
        if format == "json":
            return self._export_json(scheme)
        elif format == "markdown":
            return self._export_markdown(scheme)
        elif format == "csv":
            return self._export_csv(scheme)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self, scheme: CategoryScheme) -> str:
        """Export tree as JSON."""
        export_data = {
            "name": scheme.name,
            "description": scheme.description,
            "created_at": scheme.created_at.isoformat(),
            "prompt_used": scheme.prompt_used,
            "categories": {}
        }
        
        # Build hierarchical structure
        root = scheme.nodes.get("root")
        if root:
            for child_id in root.children:
                child = scheme.nodes[child_id]
                export_data["categories"][child.name] = self._build_json_node(child, scheme)
        
        return json.dumps(export_data, indent=2)
    
    def _build_json_node(self, node: CategoryNode, scheme: CategoryScheme) -> Dict:
        """Build JSON representation of a node and its children."""
        node_data = {
            "video_count": node.video_count,
            "videos": [v['title'] for v in node.videos[:5]],  # Sample
            "subcategories": {}
        }
        
        for child_id in node.children:
            child = scheme.nodes[child_id]
            node_data["subcategories"][child.name] = self._build_json_node(child, scheme)
        
        return node_data
    
    def _export_markdown(self, scheme: CategoryScheme) -> str:
        """Export tree as Markdown."""
        md = f"# {scheme.name}\n\n"
        md += f"*{scheme.description}*\n\n"
        md += f"Created: {scheme.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        root = scheme.nodes.get("root")
        if root:
            for child_id in root.children:
                child = scheme.nodes[child_id]
                md += self._build_markdown_node(child, scheme, level=2)
        
        return md
    
    def _build_markdown_node(self, node: CategoryNode, scheme: CategoryScheme, level: int) -> str:
        """Build Markdown representation of a node."""
        prefix = "#" * level
        md = f"{prefix} {node.name} ({node.video_count} videos)\n\n"
        
        # Add sample videos
        if node.videos and level < 4:
            md += "Sample videos:\n"
            for video in node.videos[:3]:
                md += f"- {video['title']}\n"
            md += "\n"
        
        # Add children
        for child_id in node.children:
            child = scheme.nodes[child_id]
            md += self._build_markdown_node(child, scheme, level + 1)
        
        return md
    
    def _export_csv(self, scheme: CategoryScheme) -> str:
        """Export tree as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["Category", "Subcategory", "Video Title", "Video ID", "Channel"])
        
        # Data
        for node in scheme.nodes.values():
            if node.level == 2:  # Subcategories with videos
                parent = scheme.nodes[node.parent]
                for video in node.videos:
                    writer.writerow([
                        parent.name,
                        node.name,
                        video['title'],
                        video['video_id'],
                        video['channel']
                    ])
        
        return output.getvalue()
    
    def _load_cached_schemes(self) -> Dict[str, CategoryScheme]:
        """Load cached categorization schemes."""
        schemes = {}
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    scheme = pickle.load(f)
                    schemes[scheme.name] = scheme
                    logger.info(f"Loaded cached scheme: {scheme.name}")
            except Exception as e:
                logger.error(f"Failed to load cache {cache_file}: {e}")
        
        return schemes
    
    def _cache_scheme(self, scheme: CategoryScheme):
        """Cache a categorization scheme."""
        cache_file = self.cache_dir / f"{scheme.name}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(scheme, f)
        
        logger.info(f"Cached scheme: {scheme.name}")
    
    def get_statistics(self, scheme: CategoryScheme) -> Dict:
        """Get statistics about the categorization."""
        stats = {
            "total_categories": 0,
            "total_videos": 0,
            "max_depth": 0,
            "category_distribution": {},
            "empty_categories": [],
            "largest_categories": []
        }
        
        for node in scheme.nodes.values():
            if node.level > 0:  # Skip root
                stats["total_categories"] += 1
                stats["max_depth"] = max(stats["max_depth"], node.level)
                
                if node.video_count == 0:
                    stats["empty_categories"].append(node.name)
                
                if node.level == 1:  # Top-level categories
                    stats["category_distribution"][node.name] = node.video_count
        
        # Sort categories by size
        sorted_cats = sorted(
            [(name, count) for name, count in stats["category_distribution"].items()],
            key=lambda x: x[1],
            reverse=True
        )
        stats["largest_categories"] = sorted_cats[:5]
        
        # Total videos (from root node)
        root = scheme.nodes.get("root")
        if root:
            stats["total_videos"] = root.video_count
        
        return stats


def main():
    """CLI for category tree builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Category Tree Builder")
    parser.add_argument("--db", default="youtube_fast.db", help="Database path")
    parser.add_argument("--prompt", help="Categorization prompt")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--export", choices=["json", "markdown", "csv"], help="Export format")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--reorganize", help="Reorganize with new prompt")
    
    args = parser.parse_args()
    
    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        db_path = Path.home() / "code" / "miniapps" / "youtube_database" / args.db
    
    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return
    
    # Build tree
    builder = CategoryTreeBuilder(str(db_path))
    
    # Build or reorganize
    if args.reorganize and "default" in builder.schemes:
        scheme = builder.reorganize_tree(builder.schemes["default"], args.reorganize)
    else:
        scheme = builder.build_category_tree(args.prompt)
    
    # Show statistics
    if args.stats:
        stats = builder.get_statistics(scheme)
        print("\nCategory Tree Statistics:")
        print(f"  Total categories: {stats['total_categories']}")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Empty categories: {len(stats['empty_categories'])}")
        print("\n  Largest categories:")
        for name, count in stats['largest_categories']:
            print(f"    - {name}: {count} videos")
    
    # Export
    if args.export:
        content = builder.export_tree(scheme, args.export)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(content)
            print(f"Exported to {args.output}")
        else:
            print(content)
    
    # Visualize
    if args.visualize:
        output_file = args.output or "category_tree.html"
        builder.visualize_tree(scheme, output_file)
        print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    main()