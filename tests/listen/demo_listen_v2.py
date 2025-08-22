#!/usr/bin/env python3
"""
Demo of Listen v2 showing what it hears and how it processes information.

This demo simulates a conversation and shows:
1. What was heard (transcribed)
2. Who said it (speaker identification)
3. How it was categorized
4. What interjections were made
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from special_agents.conversation_manager import ConversationManager
from special_agents.information_organizer import InformationOrganizer
from special_agents.interjection_agent import InterjectionAgent


def demo_conversation():
    """Demonstrate conversation tracking and processing."""
    
    print("\n" + "=" * 60)
    print("LISTEN V2 DEMO - What It Hears and Does")
    print("=" * 60)
    
    # Initialize components
    conv_manager = ConversationManager(conversation_id="demo")
    info_organizer = InformationOrganizer()
    
    # Simulate a conversation
    conversation = [
        ("user", "I need to finish the API implementation by tomorrow", {"pitch": 150}),
        ("assistant", "I can help with that. What endpoints are you working on?", {"pitch": 180}),
        ("user", "The user authentication and profile management endpoints", {"pitch": 150}),
        ("user", "Also, I have a doctor appointment at 3pm today", {"pitch": 150}),
        ("colleague", "Don't forget we have the team meeting at 2pm", {"pitch": 120}),
        ("user", "Thanks for the reminder. Can someone review my code?", {"pitch": 150}),
    ]
    
    print("\n1. WHAT IT HEARS (Conversation Tracking):")
    print("-" * 40)
    
    # Process each turn
    for speaker_id, text, audio_features in conversation:
        # Add speaker if new
        if speaker_id not in conv_manager.speakers:
            conv_manager.add_speaker(speaker_id, speaker_id.capitalize())
        
        # Add turn
        turn = conv_manager.add_turn(
            text=text,
            speaker_id=speaker_id,
            audio_features=audio_features
        )
        
        # Show what was heard
        speaker_name = conv_manager.speakers[speaker_id].name
        print(f"[{speaker_name}]: {text}")
        
        # Process through information organizer
        category, confidence = info_organizer.categorize(text, source=speaker_id)
        item = info_organizer.organize(
            content=text,
            source=speaker_id,
            metadata={"speaker": speaker_name},
            auto_categorize=True
        )
    
    print("\n2. SPEAKER IDENTIFICATION:")
    print("-" * 40)
    
    # Analyze conversation
    analysis = conv_manager.analyze_conversation()
    
    for speaker_name, stats in analysis["speakers"].items():
        print(f"\n{speaker_name}:")
        print(f"  - Said {stats['utterance_count']} things")
        print(f"  - Average length: {stats['avg_utterance_length']:.0f} characters")
        print(f"  - Asked {stats['questions_asked']} questions")
        print(f"  - Participation: {stats['participation_rate']*100:.1f}%")
    
    print(f"\nDominant speaker: {analysis['dominant_speaker']}")
    
    print("\n3. INFORMATION CATEGORIZATION:")
    print("-" * 40)
    
    # Show how information was organized
    summary = info_organizer.get_summary()
    
    print(f"Total items organized: {summary['total_items']}")
    print("\nCategories:")
    for category, data in summary["categories"].items():
        if data["count"] > 0:
            print(f"  - {category}: {data['count']} items ({data['percentage']:.0f}%)")
    
    # Show specific categorizations
    print("\nDetailed categorization:")
    for i in range(min(3, len(info_organizer.items))):
        item_id = list(info_organizer.items.keys())[i]
        item = info_organizer.items[item_id]
        print(f"  '{item.content[:40]}...' -> {item.category}")
    
    print("\n4. INTELLIGENT INTERJECTIONS:")
    print("-" * 40)
    
    # Simulate interjection decisions
    interjection_agent = InterjectionAgent(confidence_threshold=0.6)
    
    # Check recent turns for interjection opportunities
    recent_context = conv_manager.get_context(num_turns=3)
    
    for turn in recent_context[-2:]:  # Check last 2 turns
        # Get relevant information
        relevant_info = []
        query_text = turn["text"]
        
        # Search for relevant organized information
        items = info_organizer.retrieve(query=query_text, limit=3)
        for item in items:
            relevant_info.append({
                "content": item.content,
                "relevance_score": 0.8,
                "category": item.category
            })
        
        # Check if should interject
        should_interject, confidence, int_type = interjection_agent.should_interject(
            conversation_turn=turn,
            available_info=relevant_info
        )
        
        if should_interject:
            print(f"\n[Would Interject] After: '{turn['text'][:50]}...'")
            print(f"  Type: {int_type}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Reason: Found {len(relevant_info)} relevant items")
    
    print("\n5. ACTIONABLE INSIGHTS:")
    print("-" * 40)
    
    # Extract actionable items
    actionable = []
    for item_id, item in info_organizer.items.items():
        text = item.content.lower()
        if any(word in text for word in ["need", "must", "should", "appointment", "meeting", "review"]):
            actionable.append(item.content)
    
    print("Identified action items:")
    for i, action in enumerate(actionable, 1):
        print(f"  {i}. {action}")
    
    # Save outputs for inspection
    output_dir = Path("tests/output/demo/listen_v2_demo/2025_08")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save conversation
    with open(output_dir / "conversation_transcript.txt", "w") as f:
        f.write(conv_manager.export_transcript())
    
    # Save analysis
    with open(output_dir / "analysis.json", "w") as f:
        json.dump({
            "conversation_analysis": analysis,
            "information_summary": summary,
            "actionable_items": actionable
        }, f, indent=2)
    
    print(f"\n[Demo outputs saved to: {output_dir}]")
    
    print("\n" + "=" * 60)
    print("SUMMARY: Listen v2 Personal Assistant")
    print("=" * 60)
    print("""
What it does:
1. Listens and transcribes audio (simulated here)
2. Identifies different speakers by voice characteristics
3. Tracks the full conversation with timestamps
4. Automatically categorizes information (work, health, scheduling, etc.)
5. Identifies when to interject with helpful information
6. Extracts actionable items from conversations
7. Builds a knowledge base about you over time

The goal: A personal assistant that knows your context and helps
manage your digital life by understanding everything you say and do.
    """)


if __name__ == "__main__":
    demo_conversation()