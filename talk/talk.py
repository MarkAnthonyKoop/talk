#!/usr/bin/env python3
"""
Talk v11 - Comprehensive Code Generation

Key improvements over v10:
1. Planning agent generates MULTIPLE specific code generation prompts
2. Code agent gets called repeatedly with focused tasks
3. Iterative building of large systems
4. Better prompt decomposition for comprehensive output
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


class ComprehensivePlanningAgent(PlanningAgent):
    """
    Enhanced planning agent that generates multiple specific code prompts.
    """
    
    def __init__(self, **kwargs):
        """Initialize with enhanced planning capabilities."""
        roles = [
            "You are a comprehensive planning agent for large-scale code generation.",
            "You break down complex tasks into MULTIPLE specific code generation prompts.",
            "Each prompt should be focused on a single component or module.",
            "",
            "CRITICAL: Instead of recommending one next_action, you must generate:",
            "1. A complete hierarchical breakdown of ALL components needed",
            "2. A LIST of specific code generation prompts (5-20 prompts typically)",
            "3. Each prompt should be self-contained and generate 100-500 lines of code",
            "",
            "For example, if asked to 'build a database system', generate prompts like:",
            "- 'Create the core database engine with B-tree indexing'",
            "- 'Implement the SQL query parser and AST'",
            "- 'Build the transaction manager with ACID properties'",
            "- 'Create the storage layer with page management'",
            "- 'Implement the query optimizer with cost-based optimization'",
            "- 'Build the connection pool and client handler'",
            "- 'Create the backup and recovery system'",
            "- 'Implement the replication module'",
            "- 'Build the monitoring and metrics system'",
            "- 'Create comprehensive tests for all modules'",
            "",
            "Your output must be JSON with:",
            "1. component_breakdown: Detailed hierarchy of all components",
            "2. code_generation_prompts: List of specific prompts for CodeAgent",
            "3. dependencies: Order in which components should be built",
            "4. estimated_total_lines: Rough estimate of total code to generate"
        ]
        
        # Replace parent's roles with our enhanced ones
        super().__init__(**kwargs)
        self.roles = roles
        self.messages = []  # Reset conversation history
    
    def run(self, input_text: str) -> str:
        """Generate comprehensive planning with multiple code prompts."""
        try:
            # Parse input
            task_info = self._parse_input(input_text)
            task = task_info.get("task", task_info.get("task_description", ""))
            
            # Build comprehensive planning prompt
            prompt = f"""TASK: {task}

Generate a COMPREHENSIVE plan for building this system. Break it down into multiple specific components.

For EACH component, create a specific code generation prompt that will produce 100-500 lines of focused code.

Think about a production-ready system with:
- Core functionality modules
- Data models and schemas
- API/interface layers
- Business logic
- Error handling and validation
- Testing infrastructure
- Configuration management
- Monitoring and logging
- Documentation
- CLI/UI components (if applicable)

Return JSON with this structure:
{{
    "component_breakdown": {{
        "core": ["component1", "component2"],
        "data": ["models", "schemas", "migrations"],
        "api": ["routes", "handlers", "middleware"],
        "utils": ["helpers", "validators", "formatters"],
        "tests": ["unit", "integration", "e2e"],
        "config": ["settings", "environment"],
        "docs": ["api_docs", "readme", "examples"]
    }},
    "code_generation_prompts": [
        {{
            "prompt": "Create the core database engine with B-tree indexing, page management, and buffer pool",
            "component": "core.engine",
            "estimated_lines": 400,
            "dependencies": []
        }},
        {{
            "prompt": "Implement the SQL parser with full SELECT, INSERT, UPDATE, DELETE support",
            "component": "core.parser", 
            "estimated_lines": 350,
            "dependencies": ["core.engine"]
        }}
        // ... 10-20 more prompts ...
    ],
    "estimated_total_lines": 3000,
    "implementation_order": ["core.engine", "core.parser", "..."],
    "rationale": "Explanation of the architecture and approach"
}}

Generate AT LEAST 10 specific code generation prompts for a comprehensive system."""

            # Get comprehensive plan from LLM
            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Save plan for reference
            self._save_comprehensive_plan(completion)
            
            return completion
            
        except Exception as e:
            log.error(f"Comprehensive planning error: {e}")
            # Return a fallback plan
            return json.dumps({
                "component_breakdown": {"core": ["main"]},
                "code_generation_prompts": [
                    {
                        "prompt": f"Generate complete implementation for: {input_text}",
                        "component": "main",
                        "estimated_lines": 500,
                        "dependencies": []
                    }
                ],
                "estimated_total_lines": 500,
                "error": str(e)
            })
    
    def _save_comprehensive_plan(self, plan_json: str):
        """Save the comprehensive plan for other agents."""
        try:
            scratch_dir = Path.cwd() / ".talk_scratch"
            scratch_dir.mkdir(exist_ok=True)
            
            plan_file = scratch_dir / "comprehensive_plan.json"
            with open(plan_file, "w") as f:
                f.write(plan_json)
            
            # Also save as readable markdown
            try:
                plan = json.loads(plan_json)
                md_file = scratch_dir / "comprehensive_plan.md"
                with open(md_file, "w") as f:
                    f.write("# Comprehensive Implementation Plan\n\n")
                    f.write(f"**Estimated Total Lines:** {plan.get('estimated_total_lines', 'Unknown')}\n\n")
                    
                    f.write("## Component Breakdown\n\n")
                    for category, components in plan.get("component_breakdown", {}).items():
                        f.write(f"### {category.title()}\n")
                        for comp in components:
                            f.write(f"- {comp}\n")
                        f.write("\n")
                    
                    f.write("## Code Generation Tasks\n\n")
                    for i, prompt_info in enumerate(plan.get("code_generation_prompts", []), 1):
                        f.write(f"### Task {i}: {prompt_info.get('component', 'Unknown')}\n")
                        f.write(f"**Estimated Lines:** {prompt_info.get('estimated_lines', 'Unknown')}\n")
                        f.write(f"**Prompt:** {prompt_info.get('prompt', 'No prompt')}\n\n")
                        
            except:
                pass  # Markdown generation is optional
                
        except Exception as e:
            log.error(f"Could not save comprehensive plan: {e}")


class EnhancedCodeAgent(CodeAgent):
    """
    Enhanced code agent that generates more comprehensive implementations.
    """
    
    def __init__(self, working_dir=None, **kwargs):
        """Initialize with enhanced code generation."""
        super().__init__(**kwargs)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        
        # Update roles for more comprehensive generation
        self.roles = [
            "You are an expert code generator creating COMPREHENSIVE implementations.",
            "Generate COMPLETE, PRODUCTION-READY code with ALL features.",
            "Each code block should be 100-500+ lines of functional code.",
            "",
            "Guidelines:",
            "1. Include ALL necessary imports and dependencies",
            "2. Implement FULL functionality, not just stubs",
            "3. Add comprehensive error handling",
            "4. Include logging and monitoring hooks",
            "5. Write detailed docstrings",
            "6. Consider edge cases and validation",
            "7. Make code production-ready",
            "",
            "Generate as much code as needed to fully implement the requested component.",
            "Do not use placeholders like 'TODO' or 'implement later'.",
            "Write the COMPLETE implementation."
        ]
        self.messages = []  # Reset conversation history
    
    def run(self, input_text: str) -> str:
        """Generate comprehensive code for the specific prompt."""
        try:
            # Parse input - could be JSON or plain text
            if input_text.startswith("{"):
                task_info = json.loads(input_text)
                prompt = task_info.get("prompt", input_text)
                component = task_info.get("component", "unknown")
                estimated_lines = task_info.get("estimated_lines", 200)
            else:
                prompt = input_text
                component = "main"
                estimated_lines = 200
            
            # Build comprehensive code generation prompt
            code_prompt = f"""Component: {component}
Target Lines: {estimated_lines}+ lines of production code

TASK: {prompt}

Generate a COMPLETE, COMPREHENSIVE implementation. Include:
1. All imports and dependencies
2. Full class/function implementations
3. Error handling and validation
4. Logging statements
5. Docstrings and type hints
6. Helper functions as needed
7. Configuration handling
8. Edge case handling

Write {estimated_lines}+ lines of production-ready code.
Use descriptive names and follow best practices.
This should be deployable code, not a prototype.

Start with the filename comment, then provide the complete implementation:

```python
# filename: {component.replace('.', '/')}.py
```"""

            # Get comprehensive code from LLM
            self._append("user", code_prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Save to scratch
            self._save_comprehensive_code(completion, component)
            
            return completion
            
        except Exception as e:
            log.error(f"Enhanced code generation error: {e}")
            return f"# Error generating code: {e}\n\n# Retrying with basic implementation..."
    
    def _save_comprehensive_code(self, completion: str, component: str):
        """Save generated code with proper structure."""
        try:
            scratch_dir = self.working_dir / ".talk_scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract code blocks
            import re
            code_blocks = re.findall(r'```(?:python|py)?\n(.*?)\n```', completion, re.DOTALL)
            log.info(f"Found {len(code_blocks)} code blocks in completion for {component}")
            
            for i, code in enumerate(code_blocks):
                # Try to extract filename from comment
                filename_match = re.search(r'#\s*filename:\s*(.+)', code)
                if filename_match:
                    filename = filename_match.group(1).strip()
                else:
                    filename = f"{component.replace('.', '_')}_{i}.py"
                
                # Create subdirectories if needed
                file_path = scratch_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the code
                with open(file_path, "w") as f:
                    # Remove filename comment
                    code = re.sub(r'^#\s*filename:.*\n', '', code)
                    f.write(code)
                
                log.info(f"Saved {len(code.splitlines())} lines to {filename}")
                
        except Exception as e:
            log.error(f"Could not save comprehensive code: {e}")


class TalkV11Orchestrator:
    """
    Talk v11 orchestrator for comprehensive code generation.
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_prompts: int = 20):
        """Initialize v11 orchestrator."""
        self.task = task
        self.max_prompts = max_prompts
        self.start_time = time.time()
        
        # Set model
        if model:
            os.environ["TALK_FORCE_MODEL"] = model
        
        # Initialize output manager and directories
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
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
        
        # Initialize agents
        self.agents = self._create_agents(model)
        
        log.info(f"Talk v11 initialized - Model: {model}, Task: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("talk_v11_comprehensive", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .talk_scratch
        scratch_dir = work_dir / ".talk_scratch"
        scratch_dir.mkdir(exist_ok=True)
        
        return session_dir, work_dir
    
    def _setup_logging(self):
        """Configure logging."""
        log_file = self.session_dir / "talk_v11.log"
        
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
        """Create agents with appropriate configurations."""
        # Provider config based on model
        if "gpt" in model.lower():
            provider_config = {"provider": {"openai": {"model_name": model}}}
        elif "claude" in model.lower() or "sonnet" in model.lower() or "opus" in model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": model}}}
        else:  # Gemini
            provider_config = {"provider": {"google": {"model_name": model}}}
        
        agents = {}
        
        # Use enhanced agents
        agents["planning"] = ComprehensivePlanningAgent(
            overrides=provider_config,
            name="ComprehensivePlanner"
        )
        
        agents["code"] = EnhancedCodeAgent(
            working_dir=self.working_dir,
            overrides=provider_config,
            name="ComprehensiveCodeGenerator"
        )
        
        agents["file"] = FileAgent(
            base_dir=str(self.working_dir),
            overrides=provider_config,
            name="FileOperator"
        )
        
        return agents
    
    def run(self) -> int:
        """Run comprehensive code generation."""
        try:
            print(f"\n[TALK v11] Comprehensive Code Generation")
            print(f"[TASK] {self.task}")
            print(f"[MODEL] {os.environ.get('TALK_FORCE_MODEL', 'gemini-2.0-flash')}")
            print(f"[WORKSPACE] {self.working_dir}\n")
            
            # Step 1: Comprehensive Planning
            print("[STEP 1] Generating comprehensive plan...")
            planning_input = json.dumps({
                "task": self.task,
                "max_prompts": self.max_prompts
            })
            
            plan_output = self.agents["planning"].run(planning_input)
            
            # Parse the plan
            try:
                # Extract JSON from markdown if needed
                import re
                json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', plan_output, re.DOTALL)
                if json_match:
                    plan_json = json_match.group(1)
                else:
                    # Try direct JSON parsing if no markdown blocks
                    plan_json = plan_output
                
                # Clean up any potential issues
                plan_json = plan_json.strip()
                if plan_json.startswith('```'):
                    # Remove incomplete markdown
                    plan_json = plan_json[3:]
                    if plan_json.startswith('json'):
                        plan_json = plan_json[4:]
                    plan_json = plan_json.strip()
                    
                plan = json.loads(plan_json)
            except Exception as e:
                log.error(f"Failed to parse plan as JSON: {e}")
                log.error(f"Plan output was: {plan_output[:500]}")
                print(f"[ERROR] Planning failed to generate valid JSON: {e}")
                print(f"[DEBUG] Plan output: {plan_output[:500]}...")
                return 1
            
            code_prompts = plan.get("code_generation_prompts", [])
            total_prompts = len(code_prompts)
            estimated_lines = plan.get("estimated_total_lines", 0)
            
            print(f"\n[PLAN] Generated {total_prompts} code generation tasks")
            print(f"[ESTIMATE] ~{estimated_lines} lines of code\n")
            
            # Step 2: Iterative Code Generation
            print("[STEP 2] Generating comprehensive codebase...")
            
            generated_files = []
            total_lines = 0
            
            for i, prompt_info in enumerate(code_prompts[:self.max_prompts], 1):
                print(f"\n[GENERATION {i}/{total_prompts}] {prompt_info.get('component', 'unknown')}")
                print(f"  Prompt: {prompt_info.get('prompt', 'No prompt')[:100]}...")
                
                # Generate code for this component
                code_input = json.dumps(prompt_info)
                code_output = self.agents["code"].run(code_input)
                
                # Track what was generated
                lines = code_output.count('\n')
                total_lines += lines
                print(f"  Generated: {lines} lines")
                
                # Small delay to avoid rate limits
                if i < total_prompts:
                    time.sleep(1)
            
            # Step 3: Persist all files
            print(f"\n[STEP 3] Persisting files to workspace...")
            self._persist_all_files()
            
            # Step 4: Summary
            print(f"\n[COMPLETE] Code generation finished")
            print(f"Total time: {(time.time() - self.start_time) / 60:.1f} minutes")
            print(f"Total lines generated: ~{total_lines}")
            
            # List generated files
            py_files = list(self.working_dir.rglob("*.py"))
            if py_files:
                print(f"\nGenerated {len(py_files)} Python files:")
                for f in sorted(py_files)[:20]:
                    if not str(f).startswith('.'):
                        size = f.stat().st_size
                        lines = f.read_text().count('\n')
                        print(f"  - {f.relative_to(self.working_dir)} ({lines} lines, {size} bytes)")
                if len(py_files) > 20:
                    print(f"  ... and {len(py_files) - 20} more")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Execution stopped by user")
            return 130
        except Exception as e:
            log.exception("Unhandled exception")
            print(f"\n[ERROR] {str(e)}")
            return 1
    
    def _persist_all_files(self):
        """Persist all files from scratch to workspace."""
        scratch_dir = self.working_dir / ".talk_scratch"
        if not scratch_dir.exists():
            return
        
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
                
            except Exception as e:
                log.error(f"Failed to persist {py_file}: {e}")
        
        print(f"  Persisted {persisted} files from scratch to workspace")


def main():
    """Run Talk v11 from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Talk v11 - Comprehensive Code Generation")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--max-prompts", type=int, default=20, help="Maximum code prompts to execute")
    
    args = parser.parse_args()
    
    orchestrator = TalkV11Orchestrator(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        max_prompts=args.max_prompts
    )
    
    return orchestrator.run()


if __name__ == "__main__":
    exit(main())