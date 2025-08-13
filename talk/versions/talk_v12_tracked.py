#!/usr/bin/env python3
"""
Talk v12 - Comprehensive Code Generation with Full Conversation Tracking

Key improvements over v11:
1. Fixes code block extraction issues
2. Tracks all agent conversations with prompts and completions
3. Exports conversation history for analysis
4. Better error handling and recovery
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys
import re
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import Agent
from plan_runner.blackboard import Blackboard
from plan_runner.step import Step
from agent.output_manager import OutputManager

from special_agents.planning_agent import PlanningAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.refinement_agent import RefinementAgent
from special_agents.research_agents.web_search_agent import WebSearchAgent

log = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """Track a single conversation exchange."""
    timestamp: str
    agent: str
    role: str  # 'user' or 'assistant'
    content: str
    tokens: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)


class ConversationTracker:
    """Track all conversations across agents."""
    
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.conversations = {}  # agent_name -> List[ConversationEntry]
        self.flow = []  # List of (agent, action, summary) tuples
        self.stats = {
            "total_prompts": 0,
            "total_completions": 0,
            "total_tokens": 0,
            "agent_calls": {}
        }
    
    def add_entry(self, agent_name: str, role: str, content: str, tokens: Optional[int] = None):
        """Add a conversation entry."""
        if agent_name not in self.conversations:
            self.conversations[agent_name] = []
        
        entry = ConversationEntry(
            timestamp=datetime.now().isoformat(),
            agent=agent_name,
            role=role,
            content=content,
            tokens=tokens
        )
        
        self.conversations[agent_name].append(entry)
        
        # Update stats
        if role == "user":
            self.stats["total_prompts"] += 1
        else:
            self.stats["total_completions"] += 1
        
        if tokens:
            self.stats["total_tokens"] += tokens
        
        if agent_name not in self.stats["agent_calls"]:
            self.stats["agent_calls"][agent_name] = 0
        self.stats["agent_calls"][agent_name] += 1
    
    def add_flow(self, agent: str, action: str, summary: str):
        """Track the high-level flow."""
        self.flow.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "summary": summary[:200]  # Truncate long summaries
        })
    
    def export_conversations(self):
        """Export all conversations to files."""
        conv_dir = self.session_dir / "conversations"
        conv_dir.mkdir(exist_ok=True)
        
        # Export individual agent conversations
        for agent_name, entries in self.conversations.items():
            agent_file = conv_dir / f"{agent_name}_conversation.json"
            with open(agent_file, "w") as f:
                json.dump([e.to_dict() for e in entries], f, indent=2)
        
        # Export flow
        flow_file = conv_dir / "execution_flow.json"
        with open(flow_file, "w") as f:
            json.dump(self.flow, f, indent=2)
        
        # Export stats
        stats_file = conv_dir / "conversation_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        # Generate markdown report
        self._generate_report()
    
    def _generate_report(self):
        """Generate a human-readable conversation report."""
        report_file = self.session_dir / "conversation_report.md"
        
        with open(report_file, "w") as f:
            f.write("# Talk v12 Conversation Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Stats
            f.write("## Statistics\n\n")
            f.write(f"- Total Prompts: {self.stats['total_prompts']}\n")
            f.write(f"- Total Completions: {self.stats['total_completions']}\n")
            f.write(f"- Total Tokens: {self.stats['total_tokens']:,}\n")
            f.write(f"- Agents Called: {len(self.stats['agent_calls'])}\n\n")
            
            # Agent breakdown
            f.write("### Agent Call Counts\n\n")
            for agent, count in self.stats['agent_calls'].items():
                f.write(f"- {agent}: {count} calls\n")
            f.write("\n")
            
            # Execution flow
            f.write("## Execution Flow\n\n")
            for i, step in enumerate(self.flow, 1):
                f.write(f"{i}. **{step['agent']}** - {step['action']}\n")
                f.write(f"   - {step['summary']}\n\n")
            
            # Detailed conversations
            f.write("## Detailed Conversations\n\n")
            for agent_name, entries in self.conversations.items():
                f.write(f"### {agent_name}\n\n")
                for entry in entries[:5]:  # Show first 5 exchanges
                    f.write(f"**{entry.role}** ({entry.timestamp}):\n")
                    content_preview = entry.content[:500]
                    if len(entry.content) > 500:
                        content_preview += "..."
                    f.write(f"```\n{content_preview}\n```\n\n")
                if len(entries) > 5:
                    f.write(f"*... and {len(entries) - 5} more exchanges*\n\n")


class TrackedAgent(Agent):
    """Wrapper for agents that tracks conversations."""
    
    def __init__(self, agent: Agent, tracker: ConversationTracker):
        self.agent = agent
        self.tracker = tracker
        self.agent_name = agent.__class__.__name__
    
    def run(self, input_text: str) -> str:
        """Run agent and track conversation."""
        # Track prompt
        self.tracker.add_entry(self.agent_name, "user", input_text)
        
        # Run agent
        output = self.agent.run(input_text)
        
        # Track completion
        self.tracker.add_entry(self.agent_name, "assistant", output)
        
        # Track flow
        summary = output[:200] if output else "No output"
        self.tracker.add_flow(self.agent_name, "completed", summary)
        
        return output


class ComprehensivePlanningAgentV12(PlanningAgent):
    """Enhanced planning agent with better prompt handling."""
    
    def __init__(self, **kwargs):
        """Initialize with enhanced planning capabilities."""
        super().__init__(**kwargs)
        # Override roles for comprehensive planning
        self.roles = [
            "You are a comprehensive planning agent for large-scale code generation.",
            "Generate a detailed plan with MULTIPLE specific code generation prompts.",
            "Each prompt should target 200-500 lines of production code.",
            "",
            "CRITICAL: Return ONLY valid JSON, no markdown formatting.",
            "Do NOT wrap the JSON in ```json``` blocks.",
            "Just return the raw JSON object."
        ]
        self.messages = []  # Reset conversation
    
    def run(self, input_text: str) -> str:
        """Generate comprehensive planning with multiple code prompts."""
        try:
            task_info = self._parse_input(input_text)
            task = task_info.get("task", task_info.get("task_description", ""))
            
            prompt = f"""TASK: {task}

Generate a comprehensive implementation plan with 10-15 specific code generation prompts.

Each prompt should be detailed enough to generate 200-500 lines of production code.

Return ONLY a JSON object (no markdown, no code blocks) with this EXACT structure:

{{
    "component_breakdown": {{
        "core": [...],
        "data": [...],
        "api": [...],
        "utils": [...],
        "tests": [...]
    }},
    "code_generation_prompts": [
        {{
            "prompt": "Detailed implementation instructions...",
            "component": "module.name",
            "estimated_lines": 300,
            "dependencies": []
        }}
    ],
    "estimated_total_lines": 3000,
    "implementation_order": [...],
    "rationale": "..."
}}"""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Clean up completion to ensure it's valid JSON
            completion = completion.strip()
            if completion.startswith("```"):
                # Remove markdown if present
                completion = re.sub(r'^```(?:json)?\s*\n', '', completion)
                completion = re.sub(r'\n```\s*$', '', completion)
            
            # Validate it's JSON
            try:
                json.loads(completion)
            except:
                # If not valid JSON, wrap in a minimal structure
                log.warning("Planning output was not valid JSON, creating fallback")
                completion = json.dumps({
                    "component_breakdown": {"core": ["main"]},
                    "code_generation_prompts": [{
                        "prompt": f"Implement: {task}",
                        "component": "main",
                        "estimated_lines": 500,
                        "dependencies": []
                    }],
                    "estimated_total_lines": 500
                })
            
            return completion
            
        except Exception as e:
            log.error(f"Planning error: {e}")
            return json.dumps({
                "error": str(e),
                "component_breakdown": {"core": ["main"]},
                "code_generation_prompts": [{
                    "prompt": f"Implement: {input_text}",
                    "component": "main", 
                    "estimated_lines": 500,
                    "dependencies": []
                }],
                "estimated_total_lines": 500
            })


class EnhancedCodeAgentV12(CodeAgent):
    """Enhanced code agent with better code block handling."""
    
    def __init__(self, working_dir=None, **kwargs):
        """Initialize with enhanced code generation."""
        super().__init__(**kwargs)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.messages = []  # Reset conversation
    
    def run(self, input_text: str) -> str:
        """Generate comprehensive code with proper formatting."""
        try:
            # Parse input
            if input_text.startswith("{"):
                task_info = json.loads(input_text)
                prompt = task_info.get("prompt", input_text)
                component = task_info.get("component", "unknown")
                estimated_lines = task_info.get("estimated_lines", 300)
            else:
                prompt = input_text
                component = "main"
                estimated_lines = 300
            
            # Build code generation prompt that encourages markdown code blocks
            code_prompt = f"""Component: {component}
Target: {estimated_lines}+ lines of production-ready code

TASK: {prompt}

Generate COMPLETE, COMPREHENSIVE implementation with:
- All necessary imports
- Full class and function implementations  
- Error handling and validation
- Logging and docstrings
- Type hints

IMPORTANT: Format your response EXACTLY like this:

## Implementation

Here's the complete implementation for {component}:

```python
# filename: {component.replace('.', '/')}.py
import ...

[YOUR {estimated_lines}+ LINES OF CODE HERE]
```

Dependencies: [list any pip packages needed]

The code should be production-ready and fully functional."""

            self._append("user", code_prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Save code to scratch
            self._save_code_v12(completion, component)
            
            return completion
            
        except Exception as e:
            log.error(f"Code generation error: {e}")
            return f"Error generating code: {e}"
    
    def _save_code_v12(self, completion: str, component: str):
        """Enhanced code extraction and saving."""
        try:
            scratch_dir = self.working_dir / ".talk_scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            
            # Try multiple patterns to extract code
            code_blocks = []
            
            # Pattern 1: Standard markdown code blocks
            pattern1 = re.findall(r'```(?:python|py)?\n(.*?)\n```', completion, re.DOTALL)
            code_blocks.extend(pattern1)
            
            # Pattern 2: Code blocks without language specifier
            if not code_blocks:
                pattern2 = re.findall(r'```\n(.*?)\n```', completion, re.DOTALL)
                code_blocks.extend(pattern2)
            
            # Pattern 3: Indented code blocks (4 spaces)
            if not code_blocks:
                lines = completion.split('\n')
                code_lines = []
                in_code = False
                for line in lines:
                    if line.startswith('    ') or line.startswith('\t'):
                        in_code = True
                        code_lines.append(line[4:] if line.startswith('    ') else line[1:])
                    elif in_code and line.strip() == '':
                        code_lines.append('')
                    elif in_code:
                        if len(code_lines) > 10:  # Only save if substantial
                            code_blocks.append('\n'.join(code_lines))
                        code_lines = []
                        in_code = False
                if code_lines and len(code_lines) > 10:
                    code_blocks.append('\n'.join(code_lines))
            
            log.info(f"Found {len(code_blocks)} code blocks for {component}")
            
            saved_files = []
            for i, code in enumerate(code_blocks):
                if not code.strip():
                    continue
                
                # Extract filename from comment
                filename_match = re.search(r'#\s*filename:\s*(.+)', code)
                if filename_match:
                    filename = filename_match.group(1).strip()
                    # Remove the filename comment
                    code = re.sub(r'^#\s*filename:.*\n', '', code)
                else:
                    filename = f"{component.replace('.', '/')}.py" if i == 0 else f"{component.replace('.', '_')}_{i}.py"
                
                # Save the code
                file_path = scratch_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code)
                
                lines = len(code.splitlines())
                log.info(f"Saved {lines} lines to {filename}")
                saved_files.append((filename, lines))
            
            # If no code blocks found, try to save the entire completion as code
            if not saved_files and "import" in completion and "def " in completion:
                log.warning(f"No code blocks found, saving entire completion for {component}")
                filename = f"{component.replace('.', '/')}.py"
                file_path = scratch_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(completion)
                log.info(f"Saved entire completion as {filename}")
                
        except Exception as e:
            log.error(f"Could not save code: {e}")


class TalkV12Orchestrator:
    """Talk v12 orchestrator with comprehensive tracking."""
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_prompts: int = 15):
        """Initialize v12 orchestrator."""
        self.task = task
        self.max_prompts = max_prompts
        self.start_time = time.time()
        
        # Set model
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager and directories
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Initialize conversation tracker
        self.tracker = ConversationTracker(self.session_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize blackboard
        self.blackboard = Blackboard()
        self.blackboard.add_sync(
            label="task_description",
            content=task,
            section="input",
            role="user"
        )
        
        # Initialize agents with tracking
        self.agents = self._create_agents(model)
        
        log.info(f"Talk v12 initialized - Model: {model}, Task: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("talk_v12_tracked", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .talk_scratch
        scratch_dir = work_dir / ".talk_scratch"
        scratch_dir.mkdir(exist_ok=True)
        
        # Save session info
        session_info = {
            "task": self.task,
            "working_directory": str(work_dir),
            "model": os.environ.get("TALK_FORCE_MODEL", "gemini-2.0-flash"),
            "created": datetime.now().isoformat(),
            "version": "v12_tracked"
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.session_dir / "talk_v12.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ],
            force=True
        )
    
    def _create_agents(self, model: str) -> Dict[str, Agent]:
        """Create agents with tracking."""
        # Provider config
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif "claude" in model.lower() or "sonnet" in model.lower() or "opus" in model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": model}}}
        else:  # Gemini
            provider_config = {"provider": {"google": {"model_name": model}}}
        
        agents = {}
        
        # Create base agents
        planning_agent = ComprehensivePlanningAgentV12(
            overrides=provider_config,
            name="ComprehensivePlanner"
        )
        
        code_agent = EnhancedCodeAgentV12(
            working_dir=self.working_dir,
            overrides=provider_config,
            name="ComprehensiveCodeGenerator"
        )
        
        file_agent = FileAgent(
            base_dir=str(self.working_dir),
            overrides=provider_config,
            name="FileOperator"
        )
        
        # Wrap with tracking
        agents["planning"] = TrackedAgent(planning_agent, self.tracker)
        agents["code"] = TrackedAgent(code_agent, self.tracker)
        agents["file"] = TrackedAgent(file_agent, self.tracker)
        
        return agents
    
    def run(self) -> int:
        """Run comprehensive code generation with tracking."""
        try:
            print(f"\n[TALK v12] Comprehensive Code Generation with Tracking")
            print(f"[TASK] {self.task}")
            print(f"[MODEL] {os.environ.get('TALK_FORCE_MODEL', 'gemini-2.0-flash')}")
            print(f"[SESSION] {self.session_dir}")
            print(f"[WORKSPACE] {self.working_dir}\n")
            
            # Step 1: Planning
            print("[STEP 1] Generating comprehensive plan...")
            self.tracker.add_flow("Orchestrator", "start_planning", f"Planning for: {self.task}")
            
            planning_input = json.dumps({
                "task": self.task,
                "max_prompts": self.max_prompts
            })
            
            plan_output = self.agents["planning"].run(planning_input)
            
            # Parse plan
            try:
                plan = json.loads(plan_output)
            except Exception as e:
                log.error(f"Failed to parse plan: {e}")
                print(f"[ERROR] Planning failed: {e}")
                self.tracker.add_flow("Orchestrator", "planning_failed", str(e))
                return 1
            
            code_prompts = plan.get("code_generation_prompts", [])
            total_prompts = min(len(code_prompts), self.max_prompts)
            estimated_lines = plan.get("estimated_total_lines", 0)
            
            print(f"[PLAN] Generated {len(code_prompts)} code generation tasks")
            print(f"[ESTIMATE] ~{estimated_lines} lines of code")
            print(f"[EXECUTING] {total_prompts} prompts\n")
            
            self.tracker.add_flow("Orchestrator", "plan_complete", 
                                f"Plan with {len(code_prompts)} prompts, ~{estimated_lines} lines")
            
            # Step 2: Code Generation
            print("[STEP 2] Generating comprehensive codebase...")
            total_lines = 0
            
            for i, prompt_info in enumerate(code_prompts[:total_prompts], 1):
                component = prompt_info.get("component", "unknown")
                prompt = prompt_info.get("prompt", "")
                est_lines = prompt_info.get("estimated_lines", 200)
                
                print(f"\n[GENERATION {i}/{total_prompts}] {component}")
                print(f"  Target: ~{est_lines} lines")
                print(f"  Prompt: {prompt[:100]}...")
                
                self.tracker.add_flow("Orchestrator", f"generate_{component}", 
                                    f"Generating {component} (~{est_lines} lines)")
                
                # Generate code
                code_input = json.dumps(prompt_info)
                code_output = self.agents["code"].run(code_input)
                
                lines = code_output.count('\n')
                total_lines += lines
                print(f"  Generated: {lines} lines")
                
                # Small delay to avoid rate limits
                if i < total_prompts:
                    time.sleep(1)
            
            # Step 3: Persist files
            print(f"\n[STEP 3] Persisting files to workspace...")
            self.tracker.add_flow("Orchestrator", "persist_files", "Moving files from scratch to workspace")
            
            persisted = self._persist_all_files()
            print(f"  Persisted {persisted} files")
            
            # Step 4: Export conversations
            print(f"\n[STEP 4] Exporting conversation history...")
            self.tracker.export_conversations()
            print(f"  Saved to {self.session_dir / 'conversations'}")
            
            # Summary
            print(f"\n[COMPLETE] Code generation finished")
            print(f"Total time: {(time.time() - self.start_time) / 60:.1f} minutes")
            print(f"Total lines generated: ~{total_lines}")
            
            # List generated files
            self._list_generated_files()
            
            # Final tracking
            self.tracker.add_flow("Orchestrator", "complete", 
                                f"Generated ~{total_lines} lines in {(time.time() - self.start_time) / 60:.1f} minutes")
            
            print(f"\n[TRACKING] Full conversation history saved to:")
            print(f"  {self.session_dir / 'conversation_report.md'}")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Execution stopped by user")
            self.tracker.add_flow("Orchestrator", "interrupted", "User interrupted execution")
            self.tracker.export_conversations()
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] {str(e)}")
            self.tracker.add_flow("Orchestrator", "error", str(e))
            self.tracker.export_conversations()
            return 1
    
    def _persist_all_files(self) -> int:
        """Persist all files from scratch to workspace."""
        scratch_dir = self.working_dir / ".talk_scratch"
        if not scratch_dir.exists():
            return 0
        
        persisted = 0
        for py_file in scratch_dir.rglob("*.py"):
            try:
                # Skip if already processed
                if py_file.suffix == ".processed":
                    continue
                
                # Determine target path
                rel_path = py_file.relative_to(scratch_dir)
                target_path = self.working_dir / rel_path
                
                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                target_path.write_text(py_file.read_text())
                
                # Mark as processed
                py_file.rename(py_file.with_suffix('.processed'))
                
                persisted += 1
                log.info(f"Persisted {rel_path}")
                
            except Exception as e:
                log.error(f"Failed to persist {py_file}: {e}")
        
        return persisted
    
    def _list_generated_files(self):
        """List all generated Python files."""
        py_files = list(self.working_dir.rglob("*.py"))
        py_files = [f for f in py_files if ".talk_scratch" not in str(f)]
        
        if py_files:
            print(f"\nGenerated {len(py_files)} Python files:")
            total_lines = 0
            for f in sorted(py_files)[:20]:
                try:
                    lines = f.read_text().count('\n')
                    size = f.stat().st_size
                    total_lines += lines
                    print(f"  - {f.relative_to(self.working_dir)} ({lines} lines, {size:,} bytes)")
                except:
                    pass
            if len(py_files) > 20:
                print(f"  ... and {len(py_files) - 20} more")
            print(f"\nTotal: {total_lines:,} lines across all files")


def main():
    """Run Talk v12 from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Talk v12 - Tracked Comprehensive Code Generation")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--max-prompts", type=int, default=15, help="Maximum code prompts to execute")
    
    args = parser.parse_args()
    
    orchestrator = TalkV12Orchestrator(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        max_prompts=args.max_prompts
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    exit(main())