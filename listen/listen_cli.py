#!/usr/bin/env python3
"""
Listen CLI - Universal launcher for all Listen versions.

Provides a clean command-line interface to launch any Listen version
with appropriate configuration and graceful fallback.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main entry point for Listen assistant."""
    parser = argparse.ArgumentParser(
        description="Listen - Intelligent voice assistant with multiple versions"
    )
    
    parser.add_argument(
        "--db",
        default="speakers.db",
        help="Database path for speaker profiles"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Confidence threshold for speaker identification (0.0-1.0)"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech responses"
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="TTS voice to use"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode"
    )
    parser.add_argument(
        "--version",
        choices=["4", "5", "6", "7"],
        default="7",
        help="Listen version to use (default: 7 - Full agentic architecture)"
    )
    parser.add_argument(
        "--service-tier",
        choices=["economy", "standard", "premium"],
        default="standard",
        help="Service tier for v6/v7 (affects AI model selection)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo mode
        print(f"\n🎭 Demo mode for Listen v{args.version}")
        print("\n📋 Core Features (all versions):")
        print("  • Speaker identification & enrollment")
        print("  • Context-aware responses")
        print("  • Voice synthesis replies")
        print("  • Intelligent conversation management")
        
        if args.version == "5":
            print("\n🔧 v5 - System Automation:")
            print("  • Command execution (bash, git, etc.)")
            print("  • File system operations")
            print("  • Development tools integration")
            
        elif args.version == "6":
            print("\n🌟 v6 - Premium Services:")
            print("  • State-of-the-art AI models")
            print("  • Deepgram Nova-3 transcription")
            print("  • Multi-model intelligence (Claude, GPT-4, Gemini)")
            print("  • Enterprise MCP ecosystem")
            print("  • Cost-optimized service orchestration")
            
        elif args.version == "7":
            print("\n🚀 v7 - Agentic Architecture:")
            print("  • PlanRunner orchestration")
            print("  • Dynamic execution plans")
            print("  • Blackboard inter-agent communication")
            print("  • Parallel agent execution")
            print("  • Modular, reusable agents")
            print("  • Async-to-sync bridging")
            
        print("\n💡 Try: python listen/listen_cli.py --version 7")
        return
    
    # Import and run the appropriate version
    try:
        if args.version == "7":
            print("\n🚀 Starting Listen v7 - Full Agentic Architecture")
            from listen.versions.listen_v7 import ListenV7
            assistant = ListenV7(
                name="Listen v7",
                service_tier=args.service_tier
            )
            print("📋 Dynamic plans loaded: voice_command, conversation, quick_action")
            print("🤖 Agents initialized: voice, intent, security, execution, response")
            
        elif args.version == "6":
            print("\n🌟 Starting Listen v6 - Premium AI Services")
            from listen.versions.listen_v6 import create_listen_v6
            assistant = create_listen_v6(tier=args.service_tier)
            
        elif args.version == "5":
            print("\n🔧 Starting Listen v5 - System Automation")
            from listen.versions.listen_v5 import ListenV5
            assistant = ListenV5(
                name="Listen v5",
                db_path=args.db,
                confidence_threshold=args.confidence,
                use_tts=not args.no_tts,
                tts_voice=args.voice
            )
            
        else:  # v4
            print("\n🎤 Starting Listen v4 - Voice Conversations")
            from listen.versions.listen_v4 import ListenV4
            assistant = ListenV4(
                name="Listen v4",
                db_path=args.db,
                confidence_threshold=args.confidence,
                use_tts=not args.no_tts,
                tts_voice=args.voice
            )
        
        # Run the assistant
        print("✅ Ready! Speak into your microphone...")
        print("   (Press Ctrl+C to stop)\n")
        
        asyncio.run(assistant.start())
        
    except ImportError as e:
        print(f"\n⚠️  Error: {e}")
        print(f"Failed to import Listen v{args.version}")
        print("\nAvailable versions:")
        
        # Check what's available
        try:
            from listen.versions import listen_v4
            print("  ✓ v4 - Basic voice assistant")
        except:
            pass
            
        try:
            from listen.versions import listen_v5
            print("  ✓ v5 - System automation")
        except:
            pass
            
        try:
            from listen.versions import listen_v6
            print("  ✓ v6 - Premium services")
        except:
            pass
            
        try:
            from listen.versions import listen_v7
            print("  ✓ v7 - Agentic architecture")
        except:
            pass
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()