#!/usr/bin/env python3
"""
Listen v2 - Advanced personal assistant with multi-source context management.

This version adds:
- Multiple input sources (audio, email, social media, files)
- Conversation tracking with speaker identification
- Information organization and categorization
- Confident interjections when relevant
- Extensible architecture for custom applications
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Listen v2 components
from special_agents.input_sources import MultiSourceOrchestrator
from special_agents.input_sources.audio_source import AudioSource
from special_agents.conversation_manager import ConversationManager
from special_agents.information_organizer import InformationOrganizer
from special_agents.interjection_agent import InterjectionAgent

# Import Talk framework components
from plan_runner.blackboard import Blackboard
from plan_runner.plan_runner import PlanRunner
from plan_runner.step import Step
from special_agents.planning_agent import PlanningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("listen_v2")


class PersonalAssistant:
    """
    Advanced personal assistant that manages your digital life.
    
    This assistant:
    - Monitors multiple input sources
    - Tracks conversations with speaker identification
    - Organizes information automatically
    - Provides timely, relevant interjections
    - Can build custom applications on demand
    """
    
    def __init__(self,
                 name: str = "Assistant",
                 workspace: Optional[Path] = None,
                 enable_interjections: bool = True,
                 auto_organize: bool = True,
                 model: str = "gemini-2.0-flash"):
        """
        Initialize the personal assistant.
        
        Args:
            name: Name of the assistant
            workspace: Directory for storing data
            enable_interjections: Whether to enable proactive interjections
            auto_organize: Whether to automatically organize information
            model: LLM model to use
        """
        self.name = name
        self.workspace = workspace or Path(".talk_scratch/assistant")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.enable_interjections = enable_interjections
        self.auto_organize = auto_organize
        self.model = model
        
        # Core components
        self.orchestrator = MultiSourceOrchestrator()
        self.conversation_manager = ConversationManager(
            save_path=self.workspace / "conversations"
        )
        self.information_organizer = InformationOrganizer(
            save_path=self.workspace / "organized_info"
        )
        self.interjection_agent = InterjectionAgent(
            confidence_threshold=0.7,
            cooldown_seconds=30
        )
        
        # Planning and execution
        self.blackboard = Blackboard()
        self.planning_agent = PlanningAgent()
        
        # State
        self.is_running = False
        self.current_task = None
        self.user_context = {}
        self.active_applications = {}
        
        # Statistics
        self.stats = {
            "start_time": None,
            "items_processed": 0,
            "interjections_made": 0,
            "tasks_executed": 0,
            "conversations_tracked": 0
        }
        
        log.info(f"Initialized {self.name} personal assistant")
    
    def add_audio_source(self, device_index: Optional[int] = None):
        """Add audio input source."""
        audio_source = AudioSource(device_index=device_index)
        if audio_source.validate():
            self.orchestrator.add_source(audio_source)
            log.info("Added audio input source")
        else:
            log.warning("Audio source not available")
    
    def add_custom_source(self, source):
        """Add a custom input source."""
        self.orchestrator.add_source(source)
        log.info(f"Added custom source: {source.name}")
    
    async def start(self):
        """Start the personal assistant."""
        if self.is_running:
            log.warning("Assistant already running")
            return
        
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        # Load previous state
        self._load_state()
        
        # Start input sources
        await self.orchestrator.start()
        
        # Start processing loop
        asyncio.create_task(self._process_loop())
        
        # Start interjection monitoring
        if self.enable_interjections:
            asyncio.create_task(self._interjection_loop())
        
        log.info(f"{self.name} started successfully")
        self._display_welcome()
    
    async def stop(self):
        """Stop the personal assistant."""
        log.info(f"Stopping {self.name}...")
        
        self.is_running = False
        
        # Stop input sources
        await self.orchestrator.stop()
        
        # Save state
        self._save_state()
        
        # Save conversation
        self.conversation_manager.save_conversation()
        
        # Save organized information
        self.information_organizer.save_state()
        
        log.info(f"{self.name} stopped")
        self._display_summary()
    
    async def _process_loop(self):
        """Main processing loop for incoming data."""
        while self.is_running:
            try:
                # Get next data from any source
                data = await self.orchestrator.get_next_data()
                
                # Update statistics
                self.stats["items_processed"] += 1
                
                # Process based on source
                source = data.get("source")
                
                if source == "audio":
                    await self._process_audio(data)
                else:
                    await self._process_generic(data)
                
                # Organize information if enabled
                if self.auto_organize:
                    self._organize_information(data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in process loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_audio(self, data: Dict[str, Any]):
        """Process audio input."""
        text = data.get("data", "")
        metadata = data.get("metadata", {})
        
        # Extract speaker hint
        speaker_hint = metadata.get("speaker_hint", "unknown")
        audio_features = metadata.get("audio_features", {})
        
        # Add to conversation
        turn = self.conversation_manager.add_turn(
            text=text,
            speaker_id=speaker_hint,
            audio_features=audio_features,
            confidence=data.get("confidence", 1.0),
            metadata=metadata
        )
        
        self.stats["conversations_tracked"] += 1
        
        # Update interjection agent context
        self.interjection_agent.update_context({
            "text": text,
            "speaker": speaker_hint,
            "timestamp": turn.timestamp.isoformat()
        })
        
        # Display in console
        speaker = self.conversation_manager.speakers.get(speaker_hint, None)
        speaker_name = speaker.name if speaker else speaker_hint
        print(f"\n[{speaker_name}]: {text}")
        
        # Check for commands
        await self._check_for_commands(text)
    
    async def _process_generic(self, data: Dict[str, Any]):
        """Process generic input from other sources."""
        content = data.get("data", "")
        source = data.get("source", "unknown")
        
        log.debug(f"Processing from {source}: {str(content)[:100]}...")
        
        # Add to information context
        self.interjection_agent.add_information(data)
    
    def _organize_information(self, data: Dict[str, Any]):
        """Organize incoming information."""
        try:
            item = self.information_organizer.organize(
                content=data.get("data"),
                source=data.get("source"),
                metadata=data.get("metadata"),
                auto_categorize=True
            )
            
            # Add organized item to interjection context
            self.interjection_agent.add_information(item.to_dict())
            
        except Exception as e:
            log.error(f"Error organizing information: {e}")
    
    async def _interjection_loop(self):
        """Monitor for interjection opportunities."""
        while self.is_running:
            try:
                # Wait a bit between checks
                await asyncio.sleep(2)
                
                # Get recent conversation context
                recent_turns = self.conversation_manager.get_context(num_turns=3)
                if not recent_turns:
                    continue
                
                last_turn = recent_turns[-1]
                
                # Get relevant information
                relevant_info = self._get_relevant_information(last_turn["text"])
                
                # Check if should interject
                should_interject, confidence, int_type = self.interjection_agent.should_interject(
                    conversation_turn=last_turn,
                    available_info=relevant_info
                )
                
                if should_interject:
                    # Generate interjection
                    interjection = self.interjection_agent.generate_interjection(
                        interjection_type=int_type,
                        context=last_turn,
                        available_info=relevant_info
                    )
                    
                    # Display interjection
                    print(f"\n[{self.name}]: {interjection}")
                    
                    # Add to conversation
                    self.conversation_manager.add_turn(
                        text=interjection,
                        speaker_id="assistant",
                        confidence=confidence
                    )
                    
                    self.stats["interjections_made"] += 1
                
            except Exception as e:
                log.error(f"Error in interjection loop: {e}")
                await asyncio.sleep(5)
    
    def _get_relevant_information(self, query: str) -> List[Dict[str, Any]]:
        """Get information relevant to the query."""
        try:
            # Search organized information
            items = self.information_organizer.retrieve(
                query=query,
                limit=5
            )
            
            # Convert to dictionaries
            return [item.to_dict() for item in items]
        
        except Exception as e:
            log.error(f"Error retrieving information: {e}")
            return []
    
    async def _check_for_commands(self, text: str):
        """Check for special commands in text."""
        text_lower = text.lower()
        
        # Check for task execution request
        if any(trigger in text_lower for trigger in ["create", "build", "make", "implement"]):
            if "application" in text_lower or "app" in text_lower or "tool" in text_lower:
                await self._create_application(text)
        
        # Check for summary request
        elif "summary" in text_lower or "summarize" in text_lower:
            self._display_summary()
        
        # Check for organization request
        elif "organize" in text_lower or "categorize" in text_lower:
            self._display_organization_summary()
        
        # Check for conversation analysis
        elif "analyze" in text_lower and "conversation" in text_lower:
            self._display_conversation_analysis()
    
    async def _create_application(self, request: str):
        """Create a custom application based on request."""
        log.info(f"Creating application: {request}")
        
        try:
            # Use planning agent to create plan
            plan_response = self.planning_agent.run(request)
            
            # Store in blackboard
            self.blackboard.add_sync(
                label="application_request",
                content=request,
                section="task",
                role="user"
            )
            
            # Convert to steps
            steps = self.planning_agent.create_steps_from_plan(plan_response)
            
            if steps:
                print(f"\n[{self.name}]: I'll create that application for you.")
                print(f"Generated {len(steps)} implementation steps.")
                
                # In v2, we just show the plan
                # In future versions, this would execute automatically
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step.description}")
                
                self.stats["tasks_executed"] += 1
            else:
                print(f"\n[{self.name}]: I need more details to create that application.")
        
        except Exception as e:
            log.error(f"Error creating application: {e}")
            print(f"\n[{self.name}]: I encountered an error creating the application.")
    
    def _display_welcome(self):
        """Display welcome message."""
        print("\n" + "=" * 60)
        print(f"{self.name.upper()} - Personal Assistant v2")
        print("=" * 60)
        print(f"\nHello! I'm {self.name}, your personal assistant.")
        print("I'm monitoring your conversations and organizing information.")
        
        if self.enable_interjections:
            print("I'll interject when I have relevant information to share.")
        
        print("\nCapabilities:")
        print("- Track conversations with speaker identification")
        print("- Organize information into categories automatically")
        print("- Provide timely, relevant information")
        print("- Create custom applications on request")
        
        print("\nActive input sources:")
        for source_name in self.orchestrator.sources.keys():
            print(f"  • {source_name}")
        
        print("\nYou can ask me to:")
        print("- 'Create an application for...'")
        print("- 'Show me a summary'")
        print("- 'Analyze our conversation'")
        print("=" * 60 + "\n")
    
    def _display_summary(self):
        """Display session summary."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        
        if self.stats["start_time"]:
            duration = (datetime.now() - self.stats["start_time"]).total_seconds()
            print(f"\nDuration: {duration/60:.1f} minutes")
        
        print(f"\nActivity:")
        print(f"  Items processed: {self.stats['items_processed']}")
        print(f"  Conversations tracked: {self.stats['conversations_tracked']}")
        print(f"  Interjections made: {self.stats['interjections_made']}")
        print(f"  Tasks executed: {self.stats['tasks_executed']}")
        
        # Conversation analysis
        conv_analysis = self.conversation_manager.analyze_conversation()
        if conv_analysis["total_turns"] > 0:
            print(f"\nConversation:")
            print(f"  Total turns: {conv_analysis['total_turns']}")
            print(f"  Speakers: {conv_analysis['speaker_count']}")
            
            if conv_analysis.get("dominant_speaker"):
                print(f"  Most active: {conv_analysis['dominant_speaker']}")
        
        # Information summary
        info_summary = self.information_organizer.get_summary()
        if info_summary["total_items"] > 0:
            print(f"\nOrganized Information:")
            print(f"  Total items: {info_summary['total_items']}")
            print(f"  Categories: {len(info_summary['categories'])}")
            
            # Top categories
            top_cats = sorted(
                info_summary["categories"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:3]
            
            if top_cats:
                print("  Top categories:")
                for cat, data in top_cats:
                    print(f"    • {cat}: {data['count']} items")
        
        # Interjection statistics
        if self.enable_interjections:
            int_stats = self.interjection_agent.get_statistics()
            if int_stats["total_interjections"] > 0:
                print(f"\nInterjections:")
                print(f"  Total: {int_stats['total_interjections']}")
                print(f"  Confidence threshold: {int_stats['current_threshold']:.2f}")
        
        print("=" * 60)
    
    def _display_organization_summary(self):
        """Display information organization summary."""
        summary = self.information_organizer.get_summary(
            time_range=timedelta(hours=1)
        )
        
        print("\n" + "=" * 60)
        print("INFORMATION ORGANIZATION")
        print("=" * 60)
        
        print(f"\nTotal items: {summary['total_items']}")
        
        print("\nCategories:")
        for cat, data in summary["categories"].items():
            print(f"  {cat}: {data['count']} items ({data['percentage']:.1f}%)")
        
        if summary["top_tags"]:
            print("\nTop tags:")
            for tag, count in summary["top_tags"][:5]:
                print(f"  {tag}: {count}")
        
        if summary["recent_activity"]:
            print("\nRecent activity:")
            for item in summary["recent_activity"][:5]:
                print(f"  [{item['category']}] {item['preview'][:50]}...")
        
        print("=" * 60)
    
    def _display_conversation_analysis(self):
        """Display conversation analysis."""
        analysis = self.conversation_manager.analyze_conversation()
        
        print("\n" + "=" * 60)
        print("CONVERSATION ANALYSIS")
        print("=" * 60)
        
        if analysis.get("status") == "no_conversation":
            print("\nNo conversation to analyze yet.")
        else:
            print(f"\nDuration: {analysis['duration']/60:.1f} minutes")
            print(f"Total turns: {analysis['total_turns']}")
            print(f"Speakers: {analysis['speaker_count']}")
            
            print("\nSpeaker breakdown:")
            for speaker_name, stats in analysis["speakers"].items():
                print(f"\n  {speaker_name}:")
                print(f"    Utterances: {stats['utterance_count']}")
                print(f"    Avg length: {stats['avg_utterance_length']:.0f} chars")
                print(f"    Questions: {stats['questions_asked']}")
                print(f"    Participation: {stats['participation_rate']*100:.1f}%")
            
            if analysis.get("recent_topics"):
                print(f"\nRecent topics: {', '.join(analysis['recent_topics'])}")
        
        print("=" * 60)
    
    def _save_state(self):
        """Save assistant state."""
        try:
            state_file = self.workspace / "assistant_state.json"
            with open(state_file, "w") as f:
                json.dump({
                    "name": self.name,
                    "stats": self.stats,
                    "user_context": self.user_context,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            log.debug("Saved assistant state")
        
        except Exception as e:
            log.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load previous assistant state."""
        state_file = self.workspace / "assistant_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.user_context = state.get("user_context", {})
                    log.info("Loaded previous assistant state")
            except Exception as e:
                log.error(f"Failed to load state: {e}")
        
        # Load organized information
        self.information_organizer.load_state()


async def main():
    """Main entry point for Listen v2."""
    parser = argparse.ArgumentParser(
        description="Listen v2 - Advanced personal assistant"
    )
    
    parser.add_argument(
        "--name",
        default="Assistant",
        help="Name of your assistant"
    )
    
    parser.add_argument(
        "--workspace",
        default=".talk_scratch/assistant",
        help="Directory for storing assistant data"
    )
    
    parser.add_argument(
        "--no-interjections",
        action="store_true",
        help="Disable proactive interjections"
    )
    
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="Disable automatic information organization"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create assistant
    assistant = PersonalAssistant(
        name=args.name,
        workspace=Path(args.workspace),
        enable_interjections=not args.no_interjections,
        auto_organize=not args.no_organize,
        model=args.model
    )
    
    # Add audio source
    assistant.add_audio_source()
    
    # Note: Future versions would add other sources here
    # assistant.add_email_source()
    # assistant.add_social_media_source()
    # assistant.add_file_monitor_source()
    
    # Handle shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start assistant
        await assistant.start()
        
        # Wait for shutdown
        await shutdown_event.wait()
        
        # Stop assistant
        await assistant.stop()
    
    except Exception as e:
        log.error(f"Fatal error: {e}")
        await assistant.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())