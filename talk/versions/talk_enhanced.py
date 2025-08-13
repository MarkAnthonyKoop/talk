#!/usr/bin/env python3
"""
Enhanced Talk with dynamic workflow selection.

This is a practical integration that enhances the existing Talk
with intelligent task routing while maintaining compatibility.
"""

import re
import sys
import argparse
from typing import List, Optional

# Add Talk directory to path
sys.path.insert(0, '/home/xx/code')

# Import from the talk.py file directly
import importlib.util
spec = importlib.util.spec_from_file_location("talk", "/home/xx/code/talk/talk.py")
talk_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(talk_module)
TalkOrchestrator = talk_module.TalkOrchestrator


class EnhancedTalk:
    """
    Enhanced Talk that routes tasks intelligently.
    
    This wrapper analyzes tasks and either:
    1. Routes simple tasks directly to appropriate agents
    2. Uses full Talk orchestration for complex tasks
    """
    
    def __init__(self):
        self.simple_patterns = [
            (r"^(ls|list|show)\s+(files?|dir|directory)", "shell"),
            (r"^(cat|read|show)\s+[\w./]+", "shell"),
            (r"^(mkdir|create\s+directory)", "shell"),
            (r"^(rm|delete|remove)\s+[\w./]+", "shell"),
            (r"^run\s+[\w\s]+", "shell"),
            (r"^execute\s+[\w\s]+", "shell"),
        ]
    
    def analyze_task(self, task: str) -> tuple[str, str]:
        """
        Analyze task and return (complexity, suggested_agent).
        
        Returns:
            (complexity, agent): complexity is 'simple' or 'complex'
        """
        task_lower = task.lower()
        
        # Check for simple patterns
        for pattern, agent in self.simple_patterns:
            if re.match(pattern, task_lower):
                return "simple", agent
        
        # Everything else is complex
        return "complex", "orchestrator"
    
    def execute_simple_task(self, task: str, agent_type: str) -> int:
        """Execute a simple task directly with minimal overhead."""
        print(f"\n⚡ [FAST MODE] Executing simple {agent_type} task...")
        
        if agent_type == "shell":
            # Direct shell execution
            import subprocess
            try:
                # Extract the actual command
                cmd_match = re.search(r"(?:run|execute|ls|list|show|cat|read|mkdir|rm|delete)\s+(.+)", task, re.I)
                if cmd_match:
                    command = cmd_match.group(1)
                    # Map common requests to actual commands
                    if "files" in task or "directory" in task:
                        command = "ls -la"
                    
                    print(f"🔧 Executing: {command}")
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    
                    if result.stdout:
                        print(f"\n📤 Output:\n{result.stdout}")
                    if result.stderr:
                        print(f"\n⚠️  Error:\n{result.stderr}")
                    
                    return result.returncode
                else:
                    print(f"❌ Could not parse command from: {task}")
                    return 1
            except Exception as e:
                print(f"❌ Error: {e}")
                return 1
        
        # Fallback to full orchestration
        return self.execute_complex_task(task)
    
    def execute_complex_task(self, task: str) -> int:
        """Execute a complex task using full Talk orchestration."""
        print(f"\n🚀 [FULL MODE] Executing complex task with Talk orchestration...")
        
        # Use existing Talk orchestrator
        orchestrator = TalkOrchestrator(
            task=task,
            model="gemini-2.0-flash",  # Use free model by default
            timeout_minutes=30,
            interactive=False
        )
        
        return orchestrator.run()
    
    def run(self, task: str) -> int:
        """
        Main entry point - routes task to appropriate execution path.
        """
        print(f"🧠 [ENHANCED TALK] Analyzing task...")
        print(f"📋 Task: '{task}'")
        
        complexity, agent = self.analyze_task(task)
        
        print(f"📊 Analysis: {complexity.upper()} task, suggested agent: {agent}")
        
        if complexity == "simple":
            return self.execute_simple_task(task, agent)
        else:
            return self.execute_complex_task(task)


def main():
    """CLI entry point for Enhanced Talk."""
    parser = argparse.ArgumentParser(
        description="Enhanced Talk - Intelligent task routing"
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Task description"
    )
    parser.add_argument(
        "--force-full", "-f",
        action="store_true",
        help="Force full orchestration mode"
    )
    
    args = parser.parse_args()
    
    # Join task words
    task = " ".join(args.task)
    
    # Create enhanced Talk
    enhanced = EnhancedTalk()
    
    # Force full mode if requested
    if args.force_full:
        exit_code = enhanced.execute_complex_task(task)
    else:
        exit_code = enhanced.run(task)
    
    exit(exit_code)


if __name__ == "__main__":
    main()