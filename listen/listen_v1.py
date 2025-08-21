#!/usr/bin/env python3
"""
Listen v1 - Real-time audio listening and task execution system.

This application listens to audio input, transcribes it in real-time,
filters for task-relevant content, and executes plans based on the
audio context using the Talk framework.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Listen components
from listen.audio_listener_agent import AudioListenerAgent
from listen.relevance_agent import RelevanceAgent

# Import Talk framework components
from plan_runner.blackboard import Blackboard
from plan_runner.plan_runner import PlanRunner
from plan_runner.step import Step

# Import special agents
from special_agents.planning_agent import PlanningAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.branching_agent import BranchingAgent
# Removed import - IntelligentTalkOrchestrator not needed for Listen

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("listen_v1")


class ListenOrchestrator:
    """
    Orchestrates real-time audio listening with task execution.
    
    This orchestrator:
    1. Listens to audio continuously
    2. Transcribes and filters for relevance
    3. Creates execution plans based on relevant content
    4. Executes plans when triggered
    """
    
    def __init__(self,
                 task_description: str,
                 working_dir: Optional[str] = None,
                 relevance_threshold: float = 0.4,
                 action_threshold: float = 0.6,
                 model: str = "gemini-2.0-flash",
                 continuous: bool = True,
                 auto_execute: bool = False):
        """
        Initialize the Listen orchestrator.
        
        Args:
            task_description: The task to listen for and execute
            working_dir: Directory for file operations
            relevance_threshold: Minimum score to consider content relevant
            action_threshold: Minimum score to trigger plan execution
            model: LLM model to use for agents
            continuous: Whether to continuously listen
            auto_execute: Whether to automatically execute plans
        """
        self.task_description = task_description
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.relevance_threshold = relevance_threshold
        self.action_threshold = action_threshold
        self.model = model
        self.continuous = continuous
        self.auto_execute = auto_execute
        
        # Initialize blackboard for communication
        self.blackboard = Blackboard()
        
        # Store task description in blackboard
        self.blackboard.add_sync(
            label="task_description",
            content=task_description,
            section="task",
            role="user"
        )
        
        # Initialize agents
        self._initialize_agents()
        
        # State tracking
        self.is_running = False
        self.execution_in_progress = False
        self.recent_executions = []
        self.pending_actions = []
        
        # Background threads
        self.listener_thread = None
        self.processor_thread = None
    
    def _initialize_agents(self):
        """Initialize all required agents."""
        log.info("Initializing agents...")
        
        # Audio listening agent
        self.audio_agent = AudioListenerAgent(
            task_description=self.task_description,
            continuous=self.continuous
        )
        
        # Relevance filtering agent
        self.relevance_agent = RelevanceAgent(
            task_description=self.task_description,
            relevance_threshold=self.relevance_threshold
        )
        
        # Planning agent for creating execution plans
        self.planning_agent = PlanningAgent()
        
        # Code generation and file management agents
        self.code_agent = CodeAgent()
        self.file_agent = FileAgent(base_dir=self.working_dir)
        self.test_agent = TestAgent(base_dir=self.working_dir)
        
        # Note: BranchingAgent requires step and plan, so we'll skip it for now
        # It would be initialized during plan execution with proper context
        
        # Store agents for plan execution
        self.agents = {
            "audio": self.audio_agent,
            "relevance": self.relevance_agent,
            "planning": self.planning_agent,
            "code": self.code_agent,
            "file": self.file_agent,
            "test": self.test_agent,
            # "branching": self.branching_agent  # Would be added during execution
        }
        
        log.info(f"Initialized {len(self.agents)} agents")
    
    def start(self):
        """Start the listening and processing system."""
        if self.is_running:
            log.warning("System already running")
            return
        
        log.info("Starting Listen orchestrator...")
        log.info(f"Task: {self.task_description}")
        log.info(f"Relevance threshold: {self.relevance_threshold}")
        log.info(f"Action threshold: {self.action_threshold}")
        log.info(f"Auto-execute: {self.auto_execute}")
        
        self.is_running = True
        
        # Start audio listening
        self.audio_agent.start_listening()
        
        # Start processing thread
        self.processor_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self.processor_thread.start()
        
        log.info("System started. Listening for audio...")
        
        # Display initial instructions
        self._display_instructions()
    
    def stop(self):
        """Stop the listening and processing system."""
        log.info("Stopping Listen orchestrator...")
        
        self.is_running = False
        
        # Stop audio listening
        self.audio_agent.stop_listening()
        
        # Wait for threads to finish
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        
        log.info("System stopped")
        
        # Display summary
        self._display_summary()
    
    def _process_loop(self):
        """Background loop that processes transcriptions."""
        while self.is_running:
            try:
                # Get recent transcriptions
                recent = self.audio_agent.get_recent_transcriptions(10)
                
                if recent:
                    # Filter for relevance
                    relevant = self.relevance_agent.filter_transcriptions(recent)
                    
                    if relevant:
                        self._process_relevant_content(relevant)
                
                # Small delay to avoid busy waiting
                time.sleep(1)
            
            except Exception as e:
                log.error(f"Error in processing loop: {e}")
                time.sleep(2)
    
    def _process_relevant_content(self, relevant_transcriptions: List[Dict[str, Any]]):
        """Process relevant transcriptions and potentially trigger execution."""
        # Check for high-relevance content that should trigger action
        for trans in relevant_transcriptions:
            score = trans.get('overall_score', 0)
            
            if score >= self.action_threshold:
                log.info(f"HIGH RELEVANCE ({score:.2f}): {trans['text'][:100]}...")
                
                # Store in blackboard
                self.blackboard.add_sync(
                    label="relevant_audio",
                    content=json.dumps(trans),
                    section="input",
                    role="audio"
                )
                
                # Check for action triggers
                if trans.get('actionable_items'):
                    self._handle_actionable_items(trans['actionable_items'])
                
                # Check for direct execution triggers
                if self._should_execute(trans):
                    self._trigger_execution(trans)
    
    def _handle_actionable_items(self, items: List[str]):
        """Handle identified actionable items."""
        for item in items:
            if item not in self.pending_actions:
                self.pending_actions.append(item)
                log.info(f"Added action item: {item}")
        
        # Display pending actions
        if self.pending_actions:
            print("\n=== PENDING ACTIONS ===")
            for i, action in enumerate(self.pending_actions[-5:], 1):
                print(f"{i}. {action}")
            print("=" * 23)
    
    def _should_execute(self, transcription: Dict[str, Any]) -> bool:
        """Determine if execution should be triggered."""
        if self.execution_in_progress:
            return False
        
        if not self.auto_execute:
            # Check for explicit execution triggers
            triggers = transcription.get('triggers', [])
            execution_triggers = [
                "go ahead", "start", "begin", "proceed",
                "yes", "do it", "execute", "run it"
            ]
            
            return any(trigger in execution_triggers for trigger in triggers)
        
        # Auto-execute if score is high enough and has actionable items
        return (
            transcription.get('overall_score', 0) >= self.action_threshold and
            transcription.get('actionable_items')
        )
    
    def _trigger_execution(self, transcription: Dict[str, Any]):
        """Trigger plan execution based on transcription."""
        if self.execution_in_progress:
            log.warning("Execution already in progress")
            return
        
        log.info("Triggering plan execution...")
        self.execution_in_progress = True
        
        # Create execution thread to avoid blocking
        execution_thread = threading.Thread(
            target=self._execute_plan,
            args=(transcription,),
            daemon=True
        )
        execution_thread.start()
    
    def _execute_plan(self, transcription: Dict[str, Any]):
        """Execute a plan based on the transcription context."""
        try:
            # Combine original task with audio context
            enhanced_task = self._create_enhanced_task(transcription)
            
            log.info(f"Creating execution plan for: {enhanced_task[:200]}...")
            
            # Use planning agent to create plan
            plan_response = self.planning_agent.run(enhanced_task)
            
            # Store plan in blackboard
            self.blackboard.add_sync(
                label="execution_plan",
                content=plan_response,
                section="planning",
                role="system"
            )
            
            # Convert plan to steps
            steps = self.planning_agent.create_steps_from_plan(plan_response)
            
            if not steps:
                log.warning("No execution steps generated")
                return
            
            log.info(f"Executing {len(steps)} steps...")
            
            # Create plan runner
            runner = PlanRunner(
                steps=steps,
                agents=self.agents,
                blackboard=self.blackboard
            )
            
            # Execute the plan
            result = runner.run(enhanced_task)
            
            # Store execution result
            self.recent_executions.append({
                "timestamp": datetime.now().isoformat(),
                "trigger": transcription['text'][:100],
                "plan_type": json.loads(plan_response).get('plan_type', 'unknown'),
                "steps_count": len(steps),
                "result_summary": result[:200] if result else "No result"
            })
            
            log.info("Plan execution completed")
            
            # Clear executed actions from pending
            self.pending_actions.clear()
        
        except Exception as e:
            log.error(f"Error executing plan: {e}")
        
        finally:
            self.execution_in_progress = False
    
    def _create_enhanced_task(self, transcription: Dict[str, Any]) -> str:
        """Create an enhanced task description with audio context."""
        base_task = self.task_description
        audio_text = transcription.get('text', '')
        action_items = transcription.get('actionable_items', [])
        
        enhanced = f"""
Original Task: {base_task}

Audio Context: {audio_text}

Identified Actions:
"""
        
        for item in action_items:
            enhanced += f"- {item}\n"
        
        enhanced += "\nPlease create a plan to address these requirements."
        
        return enhanced
    
    def _display_instructions(self):
        """Display initial instructions to the user."""
        print("\n" + "=" * 50)
        print("LISTEN v1 - Audio-Driven Task Execution")
        print("=" * 50)
        print(f"\nTask: {self.task_description}")
        print(f"\nListening for audio input...")
        print("\nSpeak clearly about your task to provide context.")
        
        if self.auto_execute:
            print("Auto-execution is ENABLED - Plans will execute automatically")
        else:
            print("Say 'go ahead' or 'start' to trigger execution")
        
        print("\nPress Ctrl+C to stop listening")
        print("=" * 50 + "\n")
    
    def _display_summary(self):
        """Display summary of the session."""
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        
        # Transcription summary
        total_trans = len(self.audio_agent.transcription_history)
        relevant_trans = len(self.relevance_agent.processed_content)
        print(f"\nTotal transcriptions: {total_trans}")
        print(f"Relevant transcriptions: {relevant_trans}")
        
        # Action summary
        action_summary = self.relevance_agent.get_action_summary()
        print(f"\nAction items identified: {action_summary['total_action_items']}")
        
        if action_summary['unique_actions']:
            print("\nUnique actions:")
            for action in action_summary['unique_actions'][:5]:
                print(f"  - {action}")
        
        # Execution summary
        print(f"\nExecutions triggered: {len(self.recent_executions)}")
        
        if self.recent_executions:
            print("\nRecent executions:")
            for exec_data in self.recent_executions[-3:]:
                print(f"  - {exec_data['timestamp']}: {exec_data['plan_type']} ({exec_data['steps_count']} steps)")
        
        print("=" * 50 + "\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "execution_in_progress": self.execution_in_progress,
            "total_transcriptions": len(self.audio_agent.transcription_history),
            "relevant_count": len(self.relevance_agent.processed_content),
            "pending_actions": len(self.pending_actions),
            "executions_count": len(self.recent_executions),
            "audio_status": self.audio_agent._get_status()
        }


def main():
    """Main entry point for Listen v1."""
    parser = argparse.ArgumentParser(
        description="Listen v1 - Real-time audio listening and task execution"
    )
    
    parser.add_argument(
        "task",
        help="The task description to listen for and execute"
    )
    
    parser.add_argument(
        "--dir",
        default=".",
        help="Working directory for file operations (default: current directory)"
    )
    
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.4,
        help="Minimum relevance score to consider content (0.0-1.0, default: 0.4)"
    )
    
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.6,
        help="Minimum relevance score to trigger actions (0.0-1.0, default: 0.6)"
    )
    
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model to use for agents (default: gemini-2.0-flash)"
    )
    
    parser.add_argument(
        "--auto-execute",
        action="store_true",
        help="Automatically execute plans when relevant content is detected"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single capture mode (capture once and exit)"
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
    
    # Create orchestrator
    orchestrator = ListenOrchestrator(
        task_description=args.task,
        working_dir=args.dir,
        relevance_threshold=args.relevance_threshold,
        action_threshold=args.action_threshold,
        model=args.model,
        continuous=not args.single,
        auto_execute=args.auto_execute
    )
    
    # Handle interrupt signal
    def signal_handler(sig, frame):
        print("\n\nStopping...")
        orchestrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start the system
        orchestrator.start()
        
        if args.single:
            # Single capture mode
            log.info("Running in single capture mode")
            time.sleep(10)  # Wait for single capture
            orchestrator.stop()
        else:
            # Continuous mode - keep running
            while orchestrator.is_running:
                time.sleep(1)
                
                # Periodically display status
                if int(time.time()) % 30 == 0:
                    status = orchestrator.get_status()
                    log.debug(f"Status: {json.dumps(status, indent=2)}")
    
    except KeyboardInterrupt:
        log.info("Received interrupt, stopping...")
        orchestrator.stop()
    
    except Exception as e:
        log.error(f"Fatal error: {e}")
        orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()