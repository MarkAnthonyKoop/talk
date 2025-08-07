#!/usr/bin/env python3
"""
Category Tree Server - Flask server for interactive visualization

Serves the category tree visualization and handles:
- Real-time reorganization
- Data updates when takeout changes
- Multiple categorization schemes
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_cors import CORS
import logging

from category_tree_builder import CategoryTreeBuilder, CategoryScheme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global tree builder instance
builder = None
current_scheme = None


def init_builder(db_path: str = None):
    """Initialize the tree builder."""
    global builder, current_scheme
    
    if db_path is None:
        # Try to find database
        db_locations = [
            Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_fast.db",
            Path.home() / "code" / "miniapps" / "youtube_database" / "youtube_enhanced.db",
            Path("youtube_fast.db"),
            Path("youtube_enhanced.db")
        ]
        
        for loc in db_locations:
            if loc.exists():
                db_path = str(loc)
                break
    
    if db_path and Path(db_path).exists():
        builder = CategoryTreeBuilder(db_path)
        # Build default scheme
        current_scheme = builder.build_category_tree(
            prompt="Create a hierarchical category tree focusing on educational value and technical depth",
            scheme_name="default"
        )
        logger.info(f"Initialized with database: {db_path}")
    else:
        logger.error("No database found")


def scheme_to_d3_format(scheme: CategoryScheme) -> dict:
    """Convert CategoryScheme to D3.js tree format."""
    
    def build_node(node_id: str) -> dict:
        node = scheme.nodes[node_id]
        
        d3_node = {
            "name": node.name,
            "value": node.video_count,
            "id": node.id,
            "level": node.level
        }
        
        # Add sample videos
        if node.videos:
            d3_node["videos"] = [v['title'] for v in node.videos[:5]]
        
        # Add children
        if node.children:
            d3_node["children"] = [build_node(child_id) for child_id in node.children]
        
        return d3_node
    
    return build_node("root")


@app.route('/')
def index():
    """Serve the main visualization page."""
    with open('category_tree_visualizer.html', 'r') as f:
        html = f.read()
    return render_template_string(html)


@app.route('/get_tree_data')
def get_tree_data():
    """Get current tree data in D3 format."""
    if current_scheme:
        return jsonify(scheme_to_d3_format(current_scheme))
    else:
        return jsonify({"error": "No data available"}), 404


@app.route('/reorganize', methods=['POST'])
def reorganize():
    """Reorganize tree based on new prompt."""
    global current_scheme
    
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    if not builder:
        return jsonify({"error": "Builder not initialized"}), 500
    
    try:
        # Create new categorization
        scheme_name = f"custom_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        current_scheme = builder.build_category_tree(prompt, scheme_name)
        
        # Return new tree
        return jsonify(scheme_to_d3_format(current_scheme))
        
    except Exception as e:
        logger.error(f"Reorganization failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_schemes')
def get_schemes():
    """Get list of available categorization schemes."""
    if builder:
        schemes = [
            {
                "name": name,
                "description": scheme.description,
                "created": scheme.created_at.isoformat()
            }
            for name, scheme in builder.schemes.items()
        ]
        return jsonify(schemes)
    return jsonify([])


@app.route('/load_scheme/<scheme_name>')
def load_scheme(scheme_name):
    """Load a specific categorization scheme."""
    global current_scheme
    
    if builder and scheme_name in builder.schemes:
        current_scheme = builder.schemes[scheme_name]
        return jsonify(scheme_to_d3_format(current_scheme))
    
    return jsonify({"error": "Scheme not found"}), 404


@app.route('/get_statistics')
def get_statistics():
    """Get statistics about current scheme."""
    if current_scheme and builder:
        stats = builder.get_statistics(current_scheme)
        return jsonify(stats)
    return jsonify({})


@app.route('/search_videos')
def search_videos():
    """Search for videos in the tree."""
    query = request.args.get('q', '').lower()
    
    if not current_scheme or not query:
        return jsonify([])
    
    results = []
    for node in current_scheme.nodes.values():
        if query in node.name.lower():
            results.append({
                "id": node.id,
                "name": node.name,
                "path": _get_node_path(node.id, current_scheme),
                "video_count": node.video_count
            })
        
        # Also search in video titles
        for video in node.videos:
            if query in video['title'].lower():
                results.append({
                    "id": node.id,
                    "name": f"{video['title']} (in {node.name})",
                    "path": _get_node_path(node.id, current_scheme),
                    "video_count": 1
                })
                break
    
    return jsonify(results[:20])  # Limit to 20 results


def _get_node_path(node_id: str, scheme: CategoryScheme) -> str:
    """Get the path to a node."""
    path = []
    current = scheme.nodes.get(node_id)
    
    while current and current.parent:
        path.insert(0, current.name)
        current = scheme.nodes.get(current.parent)
    
    return " > ".join(path)


@app.route('/export/<format>')
def export(format):
    """Export tree in various formats."""
    if not current_scheme or not builder:
        return jsonify({"error": "No data available"}), 404
    
    if format not in ['json', 'markdown', 'csv']:
        return jsonify({"error": "Invalid format"}), 400
    
    content = builder.export_tree(current_scheme, format)
    
    # Create temporary file
    from tempfile import NamedTemporaryFile
    suffix = f".{format}" if format != 'markdown' else ".md"
    
    with NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f"youtube_categories_{datetime.now().strftime('%Y%m%d')}{suffix}"
    )


@app.route('/watch_for_changes', methods=['POST'])
def watch_for_changes():
    """Set up watching for takeout data changes."""
    data = request.json
    takeout_path = data.get('path', '')
    
    if not takeout_path or not Path(takeout_path).exists():
        return jsonify({"error": "Invalid path"}), 400
    
    # Set up file watcher (simplified version)
    # In production, use watchdog or similar
    import threading
    
    def check_for_updates():
        """Check for takeout updates periodically."""
        import time
        last_modified = Path(takeout_path).stat().st_mtime
        
        while True:
            time.sleep(60)  # Check every minute
            current_modified = Path(takeout_path).stat().st_mtime
            
            if current_modified > last_modified:
                logger.info("Takeout data changed, rebuilding...")
                # Rebuild database and categories
                # This would trigger the build_db_enhanced.py script
                last_modified = current_modified
    
    thread = threading.Thread(target=check_for_updates, daemon=True)
    thread.start()
    
    return jsonify({"status": "Watching for changes"})


@app.route('/get_video_details/<node_id>')
def get_video_details(node_id):
    """Get detailed information about videos in a node."""
    if not current_scheme or node_id not in current_scheme.nodes:
        return jsonify({"error": "Node not found"}), 404
    
    node = current_scheme.nodes[node_id]
    
    details = {
        "id": node.id,
        "name": node.name,
        "level": node.level,
        "video_count": node.video_count,
        "videos": [
            {
                "title": v['title'],
                "channel": v['channel'],
                "url": v['url'],
                "ai_score": v.get('ai_score', 0)
            }
            for v in node.videos[:50]  # Limit to 50 videos
        ],
        "path": _get_node_path(node_id, current_scheme)
    }
    
    return jsonify(details)


@app.route('/suggest_categories', methods=['POST'])
def suggest_categories():
    """Suggest new categorization schemes based on viewing patterns."""
    if not builder:
        return jsonify({"error": "Builder not initialized"}), 500
    
    suggestions = [
        {
            "name": "by_learning_progression",
            "prompt": "Organize videos by learning progression from beginner to expert, considering prerequisites and dependencies",
            "description": "Progressive learning path"
        },
        {
            "name": "by_project_relevance", 
            "prompt": "Categorize videos by their relevance to practical projects and real-world applications",
            "description": "Project-based organization"
        },
        {
            "name": "by_time_investment",
            "prompt": "Group videos by time investment required: quick tips (< 10 min), tutorials (10-30 min), deep dives (30+ min)",
            "description": "Time-based categorization"
        },
        {
            "name": "by_skill_area",
            "prompt": "Organize by specific skill areas: frontend, backend, databases, DevOps, AI/ML, system design",
            "description": "Skill-focused tree"
        },
        {
            "name": "by_content_quality",
            "prompt": "Categorize by content quality indicators: official docs, verified creators, community content, experimental",
            "description": "Quality-based hierarchy"
        }
    ]
    
    # Check which ones already exist
    for suggestion in suggestions:
        suggestion['exists'] = suggestion['name'] in builder.schemes
    
    return jsonify(suggestions)


def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Category Tree Visualization Server")
    parser.add_argument("--db", help="Database path")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Initialize builder
    init_builder(args.db)
    
    if not builder:
        print("Error: Could not initialize tree builder. Check database path.")
        return
    
    print(f"🌳 Category Tree Server starting on http://localhost:{args.port}")
    print(f"📊 Database: {builder.db_path}")
    print(f"🎯 Schemes available: {list(builder.schemes.keys())}")
    print("\nOpen your browser to see the interactive visualization!")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()