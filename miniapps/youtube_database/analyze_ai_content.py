#!/usr/bin/env python3
"""
Analyze AI/coding content in YouTube database.
Find videos that would help with codebase analysis and design.
"""

import sys
import sqlite3
import json
from pathlib import Path
from collections import Counter

def analyze_ai_content(db_path: str):
    """Analyze AI/coding content in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("YouTube AI/Coding Content Analysis")
    print("=" * 60)
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM videos")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM videos WHERE ai_score > 0.5")
    ai_count = cursor.fetchone()[0]
    
    print(f"Total videos: {total:,}")
    print(f"AI/coding videos: {ai_count:,} ({ai_count/total*100:.1f}%)")
    
    # Top AI/coding videos
    print("\n" + "=" * 60)
    print("TOP AI/CODING VIDEOS FOR CODEBASE ANALYSIS")
    print("-" * 60)
    
    # Categories that are most relevant for codebase analysis
    codebase_keywords = [
        'codebase_analysis', 'ai_agents', 'ai_coding', 'llm_tutorials',
        'software_architecture', 'testing_quality'
    ]
    
    cursor.execute("""
        SELECT title, channel, url, ai_score, categories
        FROM videos
        WHERE ai_score > 0.8
        ORDER BY ai_score DESC
        LIMIT 20
    """)
    
    print("\nHighest AI Relevance (>0.8 score):")
    for i, (title, channel, url, score, cats) in enumerate(cursor.fetchall(), 1):
        categories = json.loads(cats) if cats else []
        print(f"\n{i}. {title[:80]}")
        print(f"   Channel: {channel}")
        print(f"   Score: {score:.2f}")
        print(f"   Categories: {', '.join(categories)}")
        if any(cat in codebase_keywords for cat in categories):
            print(f"   ⭐ HIGHLY RELEVANT for codebase analysis")
    
    # Find specific codebase analysis content
    print("\n" + "=" * 60)
    print("CODEBASE ANALYSIS SPECIFIC CONTENT")
    print("-" * 60)
    
    analysis_terms = [
        'static analysis', 'code analysis', 'ast', 'parsing', 'tree-sitter',
        'language server', 'lsp', 'code review', 'architecture', 'refactor',
        'code quality', 'technical debt', 'code smell', 'design pattern'
    ]
    
    for term in analysis_terms:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM videos 
            WHERE LOWER(title) LIKE ?
        """, (f'%{term}%',))
        
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"\n'{term}': {count} videos")
            
            # Show top videos for this term
            cursor.execute("""
                SELECT title, channel, ai_score
                FROM videos 
                WHERE LOWER(title) LIKE ?
                ORDER BY ai_score DESC
                LIMIT 3
            """, (f'%{term}%',))
            
            for title, channel, score in cursor.fetchall():
                print(f"  - {title[:60]} ({channel}) [score: {score:.2f}]")
    
    # AI tool specific content
    print("\n" + "=" * 60)
    print("AI TOOLS FOR DEVELOPMENT")
    print("-" * 60)
    
    ai_tools = [
        'cursor', 'copilot', 'claude', 'chatgpt', 'gpt', 'llm',
        'langchain', 'autogen', 'crew ai', 'agent', 'prompt'
    ]
    
    tool_counts = {}
    for tool in ai_tools:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM videos 
            WHERE LOWER(title) LIKE ?
        """, (f'%{tool}%',))
        
        count = cursor.fetchone()[0]
        if count > 0:
            tool_counts[tool] = count
    
    print("\nAI Tool Coverage:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool}: {count} videos")
    
    # Channels focusing on AI/coding
    print("\n" + "=" * 60)
    print("TOP CHANNELS FOR AI/CODING CONTENT")
    print("-" * 60)
    
    cursor.execute("""
        SELECT channel, COUNT(*) as count, AVG(ai_score) as avg_score
        FROM videos
        WHERE ai_score > 0.5
        AND channel != 'Unknown'
        GROUP BY channel
        HAVING count >= 5
        ORDER BY avg_score DESC
        LIMIT 15
    """)
    
    print("\nChannels with 5+ AI/coding videos:")
    for channel, count, avg_score in cursor.fetchall():
        print(f"  {channel}: {count} videos (avg score: {avg_score:.2f})")
    
    # Learning progression
    print("\n" + "=" * 60)
    print("RECOMMENDED LEARNING PATH")
    print("-" * 60)
    
    learning_path = [
        ("LLM Fundamentals", "llm_tutorials"),
        ("Prompt Engineering", "llm_tutorials"),
        ("AI Coding Tools", "ai_coding"),
        ("Code Analysis Techniques", "codebase_analysis"),
        ("Multi-Agent Systems", "ai_agents"),
        ("Software Architecture", "software_architecture")
    ]
    
    print("\nBased on your viewing history, here's a learning path:")
    for i, (topic, category) in enumerate(learning_path, 1):
        # Check how many videos watched in this category
        cursor.execute("""
            SELECT COUNT(*)
            FROM videos
            WHERE categories LIKE ?
        """, (f'%"{category}"%',))
        
        count = cursor.fetchone()[0]
        status = "✓ Good coverage" if count > 10 else "⚠ Need more content" if count > 0 else "❌ Not explored"
        
        print(f"\n{i}. {topic}")
        print(f"   Videos watched: {count}")
        print(f"   Status: {status}")
        
        if count < 10:
            # Suggest searches
            if category == "codebase_analysis":
                print("   Suggested searches: 'AST parsing', 'static analysis', 'code quality'")
            elif category == "ai_agents":
                print("   Suggested searches: 'LangChain agents', 'AutoGPT', 'multi-agent systems'")
            elif category == "ai_coding":
                print("   Suggested searches: 'Cursor IDE', 'GitHub Copilot', 'AI pair programming'")
    
    # Export recommendations
    print("\n" + "=" * 60)
    print("EXPORT RECOMMENDATIONS")
    print("-" * 60)
    
    # Get top videos for export
    cursor.execute("""
        SELECT title, channel, url, ai_score, categories
        FROM videos
        WHERE ai_score > 0.7
        ORDER BY ai_score DESC
        LIMIT 100
    """)
    
    recommendations = []
    for title, channel, url, score, cats in cursor.fetchall():
        categories = json.loads(cats) if cats else []
        recommendations.append({
            'title': title,
            'channel': channel,
            'url': url,
            'score': score,
            'categories': categories
        })
    
    # Save to file
    output_file = "ai_codebase_videos.json"
    with open(output_file, 'w') as f:
        json.dump({
            'generated': str(Path.cwd()),
            'total_videos': total,
            'ai_videos': ai_count,
            'recommendations': recommendations
        }, f, indent=2)
    
    print(f"\nTop 100 AI/coding videos exported to: {output_file}")
    
    conn.close()
    
    return recommendations


if __name__ == "__main__":
    db_path = "youtube_fast.db"
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found")
        print("Run build_db_fast.py first")
        sys.exit(1)
    
    recommendations = analyze_ai_content(db_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review ai_codebase_videos.json for specific videos to watch")
    print("2. Use youtube_enrichment.py to fetch transcripts for top videos")
    print("3. Use youtube_query.py for interactive exploration")
    print("\nThe database contains your complete YouTube history with AI/coding categorization!")