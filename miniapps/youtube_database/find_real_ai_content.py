#!/usr/bin/env python3
"""
Find real AI/coding content in YouTube database using better keyword matching.
"""

import sys
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

def find_real_ai_content(db_path: str):
    """Find actual AI/coding content with better filtering."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Finding Real AI/Coding Content in YouTube History")
    print("=" * 60)
    
    # Better keywords for actual AI/coding content
    ai_coding_keywords = {
        'programming': ['python', 'javascript', 'java ', 'c++', 'golang', 'rust', 'typescript', 'react', 'vue', 'angular'],
        'ai_ml': ['machine learning', 'deep learning', 'neural network', 'tensorflow', 'pytorch', 'ai ', 'artificial intelligence'],
        'llm': ['gpt', 'claude', 'llm', 'large language', 'prompt engineering', 'chatgpt', 'openai', 'anthropic'],
        'coding_tools': ['github', 'copilot', 'cursor', 'vscode', 'ide ', 'debugg', 'git '],
        'software_eng': ['software', 'architect', 'design pattern', 'clean code', 'refactor', 'algorithm', 'data structure'],
        'devops': ['docker', 'kubernetes', 'k8s', 'aws', 'cloud', 'ci/cd', 'jenkins', 'terraform'],
        'web_dev': ['frontend', 'backend', 'full stack', 'api', 'rest', 'graphql', 'database', 'sql'],
        'code_analysis': ['code review', 'static analysis', 'code quality', 'testing', 'unit test', 'tdd']
    }
    
    # Find videos matching keywords
    ai_videos = defaultdict(list)
    
    for category, keywords in ai_coding_keywords.items():
        for keyword in keywords:
            cursor.execute("""
                SELECT title, channel, url
                FROM videos
                WHERE LOWER(title) LIKE ? OR LOWER(channel) LIKE ?
            """, (f'%{keyword.lower()}%', f'%{keyword.lower()}%'))
            
            results = cursor.fetchall()
            for title, channel, url in results:
                # Filter out obvious non-tech content
                title_lower = title.lower()
                if any(exclude in title_lower for exclude in ['music', 'song', 'lyrics', 'official video', 'trailer', 'movie']):
                    continue
                    
                ai_videos[category].append({
                    'title': title,
                    'channel': channel,
                    'url': url,
                    'keyword': keyword
                })
    
    # Deduplicate videos
    seen = set()
    unique_videos = defaultdict(list)
    
    for category, videos in ai_videos.items():
        for video in videos:
            video_key = (video['title'], video['channel'])
            if video_key not in seen:
                seen.add(video_key)
                unique_videos[category].append(video)
    
    # Display results
    total_ai_videos = sum(len(videos) for videos in unique_videos.values())
    print(f"\nFound {total_ai_videos} actual AI/coding videos\n")
    
    for category, videos in unique_videos.items():
        if videos:
            print(f"\n{category.upper()} ({len(videos)} videos)")
            print("-" * 40)
            
            # Show top 5 for each category
            for video in videos[:5]:
                print(f"• {video['title'][:70]}")
                print(f"  Channel: {video['channel']}")
                print(f"  Matched: {video['keyword']}")
                print()
    
    # Find top AI/coding channels
    print("\n" + "=" * 60)
    print("TOP AI/CODING CHANNELS")
    print("-" * 60)
    
    channel_counts = Counter()
    for videos in unique_videos.values():
        for video in videos:
            if video['channel'] != 'Unknown':
                channel_counts[video['channel']] += 1
    
    for channel, count in channel_counts.most_common(20):
        print(f"{channel}: {count} videos")
    
    # Export results
    output = {
        'total_ai_videos': total_ai_videos,
        'categories': {cat: len(videos) for cat, videos in unique_videos.items()},
        'top_channels': dict(channel_counts.most_common(20)),
        'videos_by_category': {
            cat: [{'title': v['title'], 'channel': v['channel'], 'url': v['url']} 
                  for v in videos[:20]]  # Top 20 per category
            for cat, videos in unique_videos.items()
        }
    }
    
    output_file = "real_ai_content.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults exported to: {output_file}")
    
    # Specific search for codebase analysis tools
    print("\n" + "=" * 60)
    print("CODEBASE ANALYSIS SPECIFIC")
    print("-" * 60)
    
    codebase_tools = [
        'ast', 'abstract syntax', 'tree-sitter', 'language server', 'lsp',
        'semantic', 'code navigation', 'code intelligence', 'sourcegraph',
        'code search', 'grep', 'ripgrep', 'ag ', 'ack'
    ]
    
    codebase_videos = []
    for tool in codebase_tools:
        cursor.execute("""
            SELECT title, channel, url
            FROM videos
            WHERE LOWER(title) LIKE ?
        """, (f'%{tool}%',))
        
        for title, channel, url in cursor.fetchall():
            codebase_videos.append({
                'title': title,
                'channel': channel,
                'tool': tool
            })
    
    if codebase_videos:
        print(f"\nFound {len(codebase_videos)} videos about codebase analysis tools:\n")
        for video in codebase_videos[:10]:
            print(f"• {video['title'][:70]}")
            print(f"  Channel: {video['channel']}")
            print(f"  Tool: {video['tool']}\n")
    else:
        print("\nNo specific codebase analysis tool videos found.")
        print("Recommendation: Search for content about:")
        print("  - Tree-sitter for parsing")
        print("  - Language Server Protocol")
        print("  - Static analysis tools")
        print("  - Code intelligence platforms")
    
    conn.close()
    
    return unique_videos


if __name__ == "__main__":
    db_path = "youtube_fast.db"
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found")
        sys.exit(1)
    
    videos = find_real_ai_content(db_path)