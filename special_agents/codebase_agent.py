#!/usr/bin/env python3
"""
CodebaseAgent - Orchestrates the generation of complete codebases.

This agent creates and manages a sophisticated execution plan that loops
until a complete codebase is generated. It leverages PlanningAgent,
BranchingAgent, and RefinementAgent at key positions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from agent.agent import Agent
from plan_runner.blackboard import Blackboard
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner
from agent.output_manager import OutputManager

from special_agents.planning_agent import PlanningAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.refinement_agent import RefinementAgent

log = logging.getLogger(__name__)


@dataclass
class CodebaseState:
    """Tracks the state of codebase generation."""
    task_description: str
    components_planned: List[Dict[str, Any]] = None
    components_completed: Set[str] = None
    components_in_progress: Set[str] = None
    current_component: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 50
    refinement_cycles: int = 0
    max_refinement_per_component: int = 3
    errors: List[str] = None
    
    def __post_init__(self):
        if self.components_planned is None:
            self.components_planned = []
        if self.components_completed is None:
            self.components_completed = set()
        if self.components_in_progress is None:
            self.components_in_progress = set()
        if self.errors is None:
            self.errors = []
    
    def is_complete(self) -> bool:
        """Check if all planned components are completed."""
        if not self.components_planned:
            return False
        planned_names = {c.get("name", "") for c in self.components_planned}
        return planned_names.issubset(self.components_completed)
    
    def get_next_component(self) -> Optional[Dict[str, Any]]:
        """Get the next component to work on."""
        for component in self.components_planned:
            name = component.get("name", "")
            if name not in self.components_completed and name not in self.components_in_progress:
                return component
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "task_description": self.task_description,
            "components_planned": self.components_planned,
            "components_completed": list(self.components_completed),
            "components_in_progress": list(self.components_in_progress),
            "current_component": self.current_component,
            "iteration_count": self.iteration_count,
            "refinement_cycles": self.refinement_cycles,
            "errors": self.errors[-10:] if self.errors else []  # Last 10 errors
        }


class CodebasePlanningAgent(PlanningAgent):
    """Extended planning agent for comprehensive codebase planning."""
    
    def __init__(self, state: CodebaseState, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.roles = [
            "You are a comprehensive codebase planning agent.",
            "You analyze tasks and create detailed multi-component implementation plans.",
            "You track progress and adjust plans based on what's been completed.",
            "",
            "Your output must be JSON with:",
            "1. components: List of components to build",
            "2. next_action: What to do next",
            "3. reasoning: Why this action"
        ]
        self.messages = []
    
    def run(self, input_text: str) -> str:
        """Generate or update the codebase plan."""
        try:
            # Debug logging
            log.info(f"[CodebasePlanningAgent] Received input: {input_text[:100]}...")
            
            # Always build context from state, ignore input_text
            # The input_text is just what the previous step returned
            context = {
                "task": self.state.task_description,
                "iteration": self.state.iteration_count,
                "completed": list(self.state.components_completed),
                "in_progress": list(self.state.components_in_progress),
                "current": self.state.current_component,
                "errors": self.state.errors[-5:] if self.state.errors else []
            }
            
            prompt = f"""Current codebase generation state:
{json.dumps(context, indent=2)}

Analyze the task and current progress. If this is the first iteration, create a comprehensive plan
with 10-20 specific components. If we're mid-generation, assess progress and adjust.

Return JSON with:
{{
    "components": [
        {{
            "name": "core.storage_engine",
            "description": "Storage engine with file I/O and indexing",
            "estimated_lines": 400,
            "dependencies": [],
            "prompt": "Create a storage engine class that..."
        }},
        // ... more components
    ],
    "next_action": "generate_code|refine_code|run_tests|integrate|complete",
    "reasoning": "Explanation of decision",
    "is_complete": false,
    "confidence": 0.8
}}

Focus on generating a COMPLETE system with all necessary components."""

            # Clear old messages to prevent context explosion
            if len(self.messages) > 10:
                self.messages = self.messages[-4:]  # Keep last 2 exchanges
            
            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Debug what we got back
            log.info(f"[CodebasePlanningAgent] Got completion: {completion[:200]}...")
            
            # Extract JSON from markdown code blocks if present
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
                log.info("Extracted JSON from markdown code block")
            else:
                json_text = completion
            
            # Parse and update state
            try:
                plan = json.loads(json_text)
                if not self.state.components_planned and plan.get("components"):
                    self.state.components_planned = plan["components"]
                    log.info(f"Created plan with {len(plan['components'])} components")
                # Always return the parsed JSON
                completion = json.dumps(plan)
            except json.JSONDecodeError as e:
                log.warning(f"Failed to parse planning output as JSON: {e}")
                
                # If we don't have any components planned yet, create a basic plan
                if not self.state.components_planned:
                    log.info("Creating fallback plan for REST API")
                    self.state.components_planned = [
                        {"name": "models.user", "description": "User model", "estimated_lines": 100, 
                         "prompt": "Create a User model with username, password hash, email"},
                        {"name": "auth.jwt", "description": "JWT authentication", "estimated_lines": 150,
                         "prompt": "Create JWT token generation and verification functions"},
                        {"name": "api.auth_routes", "description": "Authentication routes", "estimated_lines": 200,
                         "prompt": "Create login, register, and logout API endpoints"},
                        {"name": "api.user_routes", "description": "User management routes", "estimated_lines": 150,
                         "prompt": "Create CRUD endpoints for user management"},
                        {"name": "middleware.auth", "description": "Auth middleware", "estimated_lines": 100,
                         "prompt": "Create authentication middleware to protect routes"},
                        {"name": "main", "description": "Main application", "estimated_lines": 100,
                         "prompt": "Create main FastAPI/Flask application with all routes"}
                    ]
                
                # Return a minimal valid JSON response
                completion = json.dumps({
                    "next_action": "generate_code",
                    "reasoning": "Continuing with code generation",
                    "is_complete": False
                })
            
            return completion
            
        except Exception as e:
            log.error(f"Planning error: {e}")
            return json.dumps({
                "next_action": "error_recovery",
                "reasoning": f"Planning failed: {e}",
                "error": str(e)
            })


class CodebaseBranchingAgent(BranchingAgent):
    """Extended branching agent for codebase generation flow control."""
    
    def __init__(self, state: CodebaseState, step: Step, plan: List[Step], **kwargs):
        super().__init__(step=step, plan=plan, **kwargs)
        self.state = state
    
    def run(self, input_text: str) -> str:
        """Decide which step to execute next based on planning output."""
        try:
            # Parse planning output
            planning_output = {}
            if input_text.startswith("{"):
                try:
                    planning_output = json.loads(input_text)
                except:
                    log.debug(f"Branching received non-JSON input: {input_text[:100]}")
            else:
                log.debug(f"Branching received text input: {input_text[:100]}")
            
            next_action = planning_output.get("next_action", "generate_code")
            
            # Map actions to step labels
            action_map = {
                "generate_code": "generate_component",
                "refine_code": "refine_component",
                "run_tests": "test_component",
                "integrate": "integrate_components",
                "complete": "finalize_codebase",
                "error_recovery": "handle_error"
            }
            
            target_label = action_map.get(next_action, "generate_component")
            
            # Find the target step
            target_step = None
            for step in self.plan:
                if step.label == target_label:
                    target_step = step
                    break
            
            if target_step:
                log.info(f"Branching to: {target_label}")
                # Modify the step's on_success to branch to target
                self.step.on_success = target_label
                return json.dumps({
                    "branch_to": target_label,
                    "reason": planning_output.get("reasoning", "Based on plan")
                })
            
            return json.dumps({
                "branch_to": "generate_component",
                "reason": "Default to code generation"
            })
            
        except Exception as e:
            log.error(f"Branching error: {e}")
            return json.dumps({
                "branch_to": "handle_error",
                "error": str(e)
            })


class ComponentCodeAgent(CodeAgent):
    """Extended code agent for generating individual components."""
    
    def __init__(self, state: CodebaseState, working_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.working_dir = working_dir
    
    def run(self, input_text: str) -> str:
        """Generate code for a specific component."""
        try:
            # Get next component to generate
            component = self.state.get_next_component()
            if not component:
                return "No components left to generate"
            
            # Mark as in progress
            component_name = component.get("name", "unknown")
            self.state.components_in_progress.add(component_name)
            self.state.current_component = component_name
            
            # Build comprehensive prompt
            existing_files = list(self.state.components_completed)
            
            prompt = f"""Generate code for component: {component_name}

Description: {component.get('description', '')}
Target lines: {component.get('estimated_lines', 200)}
Dependencies: {component.get('dependencies', [])}

Existing components in project:
{json.dumps(existing_files, indent=2)}

Detailed requirements:
{component.get('prompt', 'Implement this component with full functionality')}

Generate COMPLETE, PRODUCTION-READY code with:
- All necessary imports
- Full implementation (no TODOs or placeholders)
- Error handling
- Logging
- Type hints
- Docstrings

Format as:
```python
# filename: {component_name.replace('.', '/')}.py
[YOUR CODE HERE]
```"""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Save to scratch
            self._save_component(completion, component_name)
            
            # Mark as completed
            self.state.components_in_progress.discard(component_name)
            self.state.components_completed.add(component_name)
            
            return completion
            
        except Exception as e:
            log.error(f"Code generation error: {e}")
            self.state.errors.append(str(e))
            if self.state.current_component:
                self.state.components_in_progress.discard(self.state.current_component)
            return f"Error generating code: {e}"
    
    def _save_component(self, code: str, component_name: str):
        """Save generated component to scratch."""
        try:
            scratch_dir = self.working_dir / ".talk_scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract code from markdown
            import re
            code_match = re.search(r'```(?:python)?\n(.*?)\n```', code, re.DOTALL)
            if code_match:
                code_content = code_match.group(1)
                
                # Remove filename comment if present
                code_content = re.sub(r'^#\s*filename:.*\n', '', code_content)
                
                # Save file
                filename = f"{component_name.replace('.', '/')}.py"
                file_path = scratch_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code_content)
                
                log.info(f"Saved {component_name} to {filename}")
        except Exception as e:
            log.error(f"Failed to save component: {e}")


class CodebaseAgent(Agent):
    """
    Master agent that orchestrates complete codebase generation.
    
    This agent:
    1. Creates a comprehensive plan with multiple components
    2. Loops through generation, refinement, and testing
    3. Uses planning and branching at each iteration
    4. Continues until all components are complete
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_iterations: int = 50,
                 **kwargs):
        """Initialize CodebaseAgent."""
        super().__init__(**kwargs)
        
        self.task = task
        self.model = model
        self.max_iterations = max_iterations
        
        # Setup working directory
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Initialize state
        self.state = CodebaseState(
            task_description=task,
            max_iterations=max_iterations
        )
        
        # Initialize blackboard
        self.blackboard = Blackboard()
        self.blackboard.add_sync(
            label="task_description",
            content=task,
            section="input",
            role="user"
        )
        
        # Create agents
        self.agents = self._create_agents()
        
        # Create execution plan
        self.plan = self._create_codebase_plan()
        
        # Initialize plan runner
        self.runner = PlanRunner(
            steps=self.plan,
            agents=self.agents,
            blackboard=self.blackboard
        )
        
        log.info(f"CodebaseAgent initialized for task: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None):
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("codebase_agent", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .talk_scratch
        scratch_dir = work_dir / ".talk_scratch"
        scratch_dir.mkdir(exist_ok=True)
        
        return session_dir, work_dir
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all necessary agents."""
        # Provider config
        if "gpt" in self.model.lower():
            provider_config = {"provider": {"openai": {"model_name": self.model}}}
        elif "claude" in self.model.lower():
            provider_config = {"provider": {"anthropic": {"model_name": self.model}}}
        else:
            provider_config = {"provider": {"google": {"model_name": self.model}}}
        
        agents = {}
        
        # Core orchestration agents
        agents["planner"] = CodebasePlanningAgent(
            state=self.state,
            overrides=provider_config,
            name="CodebasePlanner"
        )
        
        # Branching agent needs the plan, will set it later
        agents["brancher"] = None  # Will be set after plan creation
        
        # Work agents
        agents["coder"] = ComponentCodeAgent(
            state=self.state,
            working_dir=self.working_dir,
            overrides=provider_config,
            name="ComponentGenerator"
        )
        
        agents["refiner"] = RefinementAgent(
            base_dir=str(self.working_dir),
            max_iterations=3,
            overrides=provider_config,
            name="ComponentRefiner"
        )
        
        agents["filer"] = FileAgent(
            base_dir=str(self.working_dir),
            overrides=provider_config,
            name="FileManager"
        )
        
        agents["tester"] = TestAgent(
            base_dir=str(self.working_dir),
            name="ComponentTester"
        )
        
        return agents
    
    def _create_codebase_plan(self) -> List[Step]:
        """
        Create a sophisticated looping plan for codebase generation.
        
        The plan loops through:
        1. Planning phase (assess what needs to be done)
        2. Branching phase (decide which action to take)
        3. Action phase (generate, refine, test, or integrate)
        4. Check phase (are we done?)
        5. Loop or complete
        """
        steps = []
        
        # Step 1: Initial planning
        steps.append(Step(
            label="initial_planning",
            agent_key="planner",
            on_success="decide_action",
            on_fail="handle_error"
        ))
        
        # Step 2: Branching decision point (skip for now, go direct to generate)
        steps.append(Step(
            label="decide_action",
            agent_key=None,  # Skip branching agent for now
            on_success="generate_component",  # Go straight to generation
            on_fail="handle_error"
        ))
        
        # Step 3: Generate a component
        steps.append(Step(
            label="generate_component",
            agent_key="coder",
            on_success="persist_component",  # Skip testing for now
            on_fail="log_error"
        ))
        
        # Step 4: Check generated component (skip for now)
        steps.append(Step(
            label="check_component",
            agent_key="tester",
            on_success="persist_component",
            on_fail="refine_component"
        ))
        
        # Step 5: Refine component if needed
        steps.append(Step(
            label="refine_component",
            agent_key="refiner",
            on_success="persist_component",
            on_fail="log_error"
        ))
        
        # Step 6: Persist component to workspace
        steps.append(Step(
            label="persist_component",
            agent_key="filer",
            on_success="update_plan",
            on_fail="log_error"
        ))
        
        # Step 7: Update plan and check if complete
        steps.append(Step(
            label="update_plan",
            agent_key=None,  # Don't call planner every time, handle in check_complete
            on_success="check_complete",
            on_fail="handle_error"
        ))
        
        # Step 8: Check if codebase is complete
        steps.append(Step(
            label="check_complete",
            agent_key=None,  # Internal check
            on_success="finalize_codebase",
            on_fail="decide_action"  # Loop back if not complete
        ))
        
        # Step 9: Integration phase (optional)
        steps.append(Step(
            label="integrate_components",
            agent_key="refiner",
            on_success="test_integration",
            on_fail="log_error"
        ))
        
        # Step 10: Test integrated system
        steps.append(Step(
            label="test_integration",
            agent_key="tester",
            on_success="finalize_codebase",
            on_fail="refine_integration"
        ))
        
        # Step 11: Refine integration
        steps.append(Step(
            label="refine_integration",
            agent_key="refiner",
            on_success="finalize_codebase",
            on_fail="log_error"
        ))
        
        # Step 12: Finalize and complete
        steps.append(Step(
            label="finalize_codebase",
            agent_key=None,
            on_success=None,
            on_fail=None
        ))
        
        # Error handling steps
        steps.append(Step(
            label="log_error",
            agent_key=None,
            on_success="update_plan",  # Continue after logging
            on_fail="handle_error"
        ))
        
        steps.append(Step(
            label="handle_error",
            agent_key="planner",  # Let planner figure out recovery
            on_success="decide_action",
            on_fail="emergency_stop"
        ))
        
        steps.append(Step(
            label="emergency_stop",
            agent_key=None,
            on_success=None,
            on_fail=None
        ))
        
        # Now create branching agent with the plan
        # Find the decide_action step to pass to branching agent
        decide_step = None
        for step in steps:
            if step.label == "decide_action":
                decide_step = step
                break
        
        if decide_step:
            self.agents["brancher"] = CodebaseBranchingAgent(
                state=self.state,
                step=decide_step,
                plan=steps,
                overrides={"provider": {"google": {"model_name": self.model}}},
                name="CodebaseBrancher"
            )
        
        return steps
    
    def run(self, input_text: str = "") -> str:
        """
        Run the codebase generation process.
        
        This executes the plan which loops until all components are generated.
        """
        try:
            log.info(f"Starting codebase generation: {self.task}")
            print(f"\n[CODEBASE AGENT] Starting generation for: {self.task}")
            print(f"[SESSION] {self.session_dir}")
            print(f"[WORKSPACE] {self.working_dir}\n")
            
            # Override check_complete step to actually check state
            original_run_single = self.runner._run_single
            
            def custom_run_single(step: Step, prompt: str) -> str:
                # Handle special internal steps
                if step.label == "check_complete":
                    # Check if codebase is complete
                    self.state.iteration_count += 1
                    
                    if self.state.is_complete():
                        log.info("Codebase generation complete!")
                        print(f"\n[COMPLETE] All {len(self.state.components_completed)} components generated")
                        step.on_success = "finalize_codebase"  # Force to finalize
                        return "complete"
                    
                    if self.state.iteration_count >= self.state.max_iterations:
                        log.warning("Max iterations reached")
                        print(f"\n[TIMEOUT] Max iterations ({self.state.max_iterations}) reached")
                        step.on_success = "finalize_codebase"
                        return "max_iterations"
                    
                    # Continue looping
                    remaining = len(self.state.components_planned) - len(self.state.components_completed)
                    print(f"\n[PROGRESS] Iteration {self.state.iteration_count}: {len(self.state.components_completed)} completed, {remaining} remaining")
                    
                    # Only replan every 5 iterations or if no components planned
                    if self.state.iteration_count % 5 == 0 or not self.state.components_planned:
                        step.on_success = "initial_planning"  # Replan
                    else:
                        step.on_success = "generate_component"  # Skip decide_action, go straight to generation
                    return "continue"
                
                elif step.label == "decide_action":
                    # For now, just pass through to generate_component
                    return "deciding"
                
                elif step.label == "update_plan":
                    # Just pass through, actual checking happens in check_complete
                    return "updated"
                
                elif step.label == "persist_component":
                    # Directly persist files from .talk_scratch to workspace
                    scratch_dir = self.working_dir / ".talk_scratch"
                    if scratch_dir.exists():
                        py_files = list(scratch_dir.rglob("*.py"))
                        for src_file in py_files:
                            rel_path = src_file.relative_to(scratch_dir)
                            dst_file = self.working_dir / rel_path
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            if not dst_file.exists():  # Only copy if not already there
                                dst_file.write_text(src_file.read_text())
                                log.info(f"Persisted {rel_path}")
                    return "persisted"
                
                elif step.label == "finalize_codebase":
                    # Set on_success to None to stop execution
                    step.on_success = None
                    return "finalized"
                
                elif step.label == "log_error":
                    if self.state.errors:
                        log.error(f"Error logged: {self.state.errors[-1]}")
                    return "error_logged"
                
                elif step.label == "emergency_stop":
                    return "stopped"
                
                # For steps with agents, use original
                if step.agent_key:
                    return original_run_single(step, prompt)
                else:
                    # No agent, just return prompt
                    return prompt
            
            self.runner._run_single = custom_run_single
            
            # Run the plan with initial input
            result = self.runner.run(input_text or self.task)
            
            # Summary
            print(f"\n[SUMMARY] Codebase generation completed")
            print(f"  Components planned: {len(self.state.components_planned)}")
            print(f"  Components completed: {len(self.state.components_completed)}")
            print(f"  Iterations: {self.state.iteration_count}")
            print(f"  Errors: {len(self.state.errors)}")
            
            # List generated files
            self._list_generated_files()
            
            # Save state
            self._save_state()
            
            return json.dumps({
                "status": "complete",
                "components": list(self.state.components_completed),
                "iterations": self.state.iteration_count,
                "workspace": str(self.working_dir)
            })
            
        except Exception as e:
            log.exception("Codebase generation failed")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "components_completed": list(self.state.components_completed)
            })
    
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
                    total_lines += lines
                    print(f"  - {f.relative_to(self.working_dir)} ({lines} lines)")
                except:
                    pass
            if len(py_files) > 20:
                print(f"  ... and {len(py_files) - 20} more")
            print(f"\nTotal: {total_lines:,} lines of code")
    
    def _save_state(self):
        """Save the final state for analysis."""
        state_file = self.session_dir / "codebase_state.json"
        with open(state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        log.info(f"State saved to {state_file}")


def main():
    """Test the CodebaseAgent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CodebaseAgent - Generate complete codebases")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum iterations")
    
    args = parser.parse_args()
    
    agent = CodebaseAgent(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        max_iterations=args.max_iterations
    )
    
    result = agent.run()
    print(f"\nResult: {result}")
    return 0


if __name__ == "__main__":
    exit(main())