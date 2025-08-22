#!/usr/bin/env python3
"""
Demo script for Listen v1 - Simulates audio input for testing.

This script demonstrates the Listen application by simulating
audio transcriptions without requiring actual microphone input.
"""

import json
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from listen.listen_v1 import ListenOrchestrator
from plan_runner.blackboard import Blackboard


class SimulatedListenDemo:
    """Demonstrates Listen functionality with simulated audio."""
    
    def __init__(self, task: str):
        """Initialize the demo with a task."""
        self.task = task
        self.orchestrator = None
        self.simulation_thread = None
        self.is_running = False
        
        # Simulated transcriptions to inject
        self.simulated_transcriptions = [
            {
                "delay": 2,
                "text": "Let me explain what I need for this project",
                "relevance": 0.3
            },
            {
                "delay": 3,
                "text": f"I need you to {task.lower()}",
                "relevance": 0.8
            },
            {
                "delay": 2,
                "text": "Make sure it has proper error handling and logging",
                "relevance": 0.6
            },
            {
                "delay": 2,
                "text": "The weather is nice today, isn't it?",
                "relevance": 0.1
            },
            {
                "delay": 3,
                "text": "Also add comprehensive unit tests for all the functionality",
                "relevance": 0.7
            },
            {
                "delay": 2,
                "text": "Use modern Python best practices and type hints",
                "relevance": 0.6
            },
            {
                "delay": 3,
                "text": "Alright, go ahead and start building it now",
                "relevance": 0.9
            }
        ]
    
    def start(self):
        """Start the demo."""
        print("\n" + "=" * 60)
        print("LISTEN v1 - DEMO MODE")
        print("=" * 60)
        print(f"\nTask: {self.task}")
        print("\nThis demo simulates audio input to demonstrate")
        print("how Listen processes speech and triggers actions.")
        print("\n" + "=" * 60 + "\n")
        
        # Create orchestrator (without actual audio listening)
        self.orchestrator = ListenOrchestrator(
            task_description=self.task,
            working_dir="./demo_workspace",
            relevance_threshold=0.4,
            action_threshold=0.7,
            auto_execute=False
        )
        
        # Start simulation
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulate_transcriptions,
            daemon=True
        )
        self.simulation_thread.start()
        
        print("Starting simulated audio stream...\n")
    
    def _simulate_transcriptions(self):
        """Simulate audio transcriptions being received."""
        time.sleep(2)  # Initial delay
        
        for sim_trans in self.simulated_transcriptions:
            if not self.is_running:
                break
            
            # Wait for specified delay
            time.sleep(sim_trans['delay'])
            
            # Create transcription data
            transcription = {
                "text": sim_trans['text'],
                "timestamp": datetime.now().isoformat(),
                "relevance_score": sim_trans['relevance']
            }
            
            # Display simulated speech
            print(f"[SIMULATED AUDIO]: \"{sim_trans['text']}\"")
            
            # Process through relevance agent
            evaluation = self.orchestrator.relevance_agent.evaluate_relevance(
                sim_trans['text']
            )
            
            # Display relevance
            score = evaluation['overall_score']
            if score >= self.orchestrator.action_threshold:
                print(f"  → HIGH RELEVANCE ({score:.2f}) - Action threshold met!")
            elif score >= self.orchestrator.relevance_threshold:
                print(f"  → Relevant ({score:.2f})")
            else:
                print(f"  → Not relevant ({score:.2f})")
            
            # Check for triggers
            if evaluation.get('triggers'):
                print(f"  → Triggers detected: {', '.join(evaluation['triggers'])}")
            
            # Display any actionable items
            if evaluation.get('actionable_items'):
                print(f"  → Actionable items found:")
                for item in evaluation['actionable_items'][:3]:
                    print(f"     • {item}")
            
            print()  # Blank line for readability
        
        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("=" * 60)
        
        # Display summary
        self._display_summary()
    
    def _display_summary(self):
        """Display summary of the simulation."""
        print("\nSUMMARY:")
        print("-" * 30)
        
        # Get action summary from relevance agent
        action_summary = self.orchestrator.relevance_agent.get_action_summary()
        
        print(f"Total transcriptions processed: {len(self.simulated_transcriptions)}")
        print(f"Relevant transcriptions: {action_summary['processed_count']}")
        print(f"Average relevance score: {action_summary['average_relevance']:.2f}")
        
        if action_summary['unique_actions']:
            print(f"\nIdentified actions ({len(action_summary['unique_actions'])}):")
            for i, action in enumerate(action_summary['unique_actions'], 1):
                print(f"  {i}. {action}")
        
        print("\nIn a real scenario, saying 'go ahead' would trigger")
        print("the PlanningAgent to create and execute a development plan.")
        print("-" * 30)
    
    def stop(self):
        """Stop the demo."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5)
        print("\nDemo stopped.")


def run_interactive_demo():
    """Run an interactive demonstration."""
    print("\n" + "=" * 60)
    print("LISTEN v1 - Interactive Demo")
    print("=" * 60)
    
    print("\nThis demo shows how Listen would process audio input")
    print("and identify relevant content for task execution.")
    
    print("\nExample tasks you can try:")
    print("1. Create a REST API for user management")
    print("2. Build a web scraper for news articles")
    print("3. Implement a chat application")
    print("4. Fix authentication bugs in the system")
    print("5. Optimize database query performance")
    
    task = input("\nEnter a task (or press Enter for default): ").strip()
    
    if not task:
        task = "Create a REST API for user management with authentication"
        print(f"Using default task: {task}")
    
    # Run the demo
    demo = SimulatedListenDemo(task)
    demo.start()
    
    # Wait for simulation to complete
    if demo.simulation_thread:
        demo.simulation_thread.join()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    print("\nTo use Listen with real audio input, run:")
    print(f'  python3 listen/listen_v1.py "{task}"')
    print("\nNote: Requires microphone and audio libraries installed.")


def run_batch_demo():
    """Run demos for multiple tasks to show versatility."""
    print("\n" + "=" * 60)
    print("LISTEN v1 - Batch Demo")
    print("=" * 60)
    
    tasks = [
        "Create a Python package for data validation",
        "Debug the memory leak in the application",
        "Add OAuth2 authentication to the API"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'=' * 60}")
        print(f"DEMO {i}/{len(tasks)}")
        print(f"{'=' * 60}")
        
        demo = SimulatedListenDemo(task)
        demo.start()
        
        # Run shorter simulation for batch mode
        time.sleep(5)
        demo.stop()
        
        if i < len(tasks):
            print("\nMoving to next demo in 3 seconds...")
            time.sleep(3)
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


def main():
    """Main entry point for the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo script for Listen v1"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch demo with multiple tasks"
    )
    
    parser.add_argument(
        "--task",
        help="Specific task to demonstrate"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            run_batch_demo()
        elif args.task:
            demo = SimulatedListenDemo(args.task)
            demo.start()
            if demo.simulation_thread:
                demo.simulation_thread.join()
        else:
            run_interactive_demo()
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()