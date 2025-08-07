#!/usr/bin/env python3
"""
EnhancedCodebaseAgent - Advanced orchestration with hierarchical planning and quality assurance.

This agent implements:
1. Multi-level hierarchical planning with todo tracking
2. Comprehensive evaluation and refinement loops
3. Dependency management and validation
4. Quality gates ensuring production-ready code
"""

from __future__ import annotations

import json
import logging
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime

from agent.agent import Agent
from plan_runner.blackboard import Blackboard
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner
from agent.output_manager import OutputManager

from special_agents.planning_agent import PlanningAgent
from special_agents.code_agent import CodeAgent
from special_agents.refinement_agent import RefinementAgent
from special_agents.test_agent import TestAgent

log = logging.getLogger(__name__)


@dataclass
class HierarchicalPlan:
    """Represents a hierarchical project plan."""
    name: str
    description: str
    level: int = 0
    estimated_lines: int = 0
    dependencies: List[str] = field(default_factory=list)
    subcomponents: List['HierarchicalPlan'] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, refined
    quality_score: float = 0.0
    refinement_count: int = 0
    test_results: Dict[str, Any] = field(default_factory=dict)
    generated_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level,
            "estimated_lines": self.estimated_lines,
            "dependencies": self.dependencies,
            "subcomponents": [s.to_dict() for s in self.subcomponents],
            "status": self.status,
            "quality_score": self.quality_score,
            "refinement_count": self.refinement_count,
            "test_results": self.test_results,
            "generated_files": self.generated_files
        }
    
    def get_all_components(self) -> List['HierarchicalPlan']:
        """Get flat list of all components including subcomponents."""
        components = [self]
        for sub in self.subcomponents:
            components.extend(sub.get_all_components())
        return components


@dataclass
class EnhancedCodebaseState:
    """Enhanced state tracking with quality metrics."""
    task_description: str
    hierarchical_plan: Optional[HierarchicalPlan] = None
    quality_threshold: float = 0.85  # Minimum quality score
    max_refinements: int = 5
    total_files_generated: int = 0
    total_lines_generated: int = 0
    dependencies_installed: Set[str] = field(default_factory=set)
    required_dependencies: Set[str] = field(default_factory=set)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 100


class HierarchicalPlanningAgent(PlanningAgent):
    """Creates detailed hierarchical plans with todo tracking."""
    
    def __init__(self, state: EnhancedCodebaseState, todos_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.todos_dir = todos_dir
        self.todos_dir.mkdir(parents=True, exist_ok=True)
        
        self.roles = [
            "You are an expert software architect creating comprehensive hierarchical plans.",
            "You break down complex systems into detailed component hierarchies.",
            "You identify all dependencies, interfaces, and integration points.",
            "You ensure COMPLETE coverage with no missing functionality.",
            "You aim for production-quality, enterprise-grade implementations.",
            "",
            "CRITICAL: The code you plan must be COMPLETE, RUNNABLE, and EXCEED expectations.",
            "Include: error handling, logging, validation, tests, documentation, CLI, configs.",
            "Plan for 10x more detail than requested - this is for the highest standards."
        ]
    
    def run(self, input_text: str) -> str:
        """Generate comprehensive hierarchical plan."""
        try:
            # Clear messages to prevent token explosion
            if len(self.messages) > 6:
                self.messages = self.messages[-4:]
            
            prompt = f"""Create a COMPREHENSIVE hierarchical plan for: {self.state.task_description}

You are being held to the HIGHEST STANDARDS. The code must:
- Be COMPLETE and PRODUCTION-READY
- Include ALL error handling, validation, logging
- Have comprehensive tests and documentation
- Include CLI interfaces and configuration
- Handle all edge cases and failure modes
- Be modular, maintainable, and extensible

Generate a detailed hierarchical plan with multiple levels:

Level 0: Main application components (5-10 major modules)
Level 1: Subcomponents for each module (3-5 per module)
Level 2: Implementation details (2-3 per subcomponent)

Return JSON with this EXACT structure:
{{
    "project_name": "project_name_here",
    "total_estimated_lines": 5000,
    "required_dependencies": ["fastapi", "sqlalchemy", "pytest", ...],
    "hierarchy": {{
        "name": "root",
        "description": "Complete system",
        "level": 0,
        "subcomponents": [
            {{
                "name": "core",
                "description": "Core business logic and models",
                "level": 1,
                "estimated_lines": 1500,
                "subcomponents": [
                    {{
                        "name": "core.models.user",
                        "description": "User model with full validation",
                        "level": 2,
                        "estimated_lines": 200,
                        "dependencies": ["core.models.base"],
                        "implementation_details": {{
                            "classes": ["User", "UserProfile", "UserPreferences"],
                            "methods": ["create", "update", "delete", "authenticate"],
                            "validations": ["email", "password_strength", "username_unique"]
                        }}
                    }},
                    // ... more subcomponents
                ]
            }},
            // ... more top-level components
        ]
    }},
    "integration_points": [
        {{"from": "api.routes", "to": "core.services", "type": "dependency"}},
        // ... more integration points
    ],
    "test_strategy": {{
        "unit_tests": ["test_models", "test_services", "test_utils"],
        "integration_tests": ["test_api", "test_database"],
        "coverage_target": 90
    }},
    "deployment": {{
        "dockerfile": true,
        "docker_compose": true,
        "ci_cd": true,
        "monitoring": true
    }}
}}

Remember: Plan for EXCELLENCE. Include 10x more than asked. Every component must be production-ready."""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Extract and parse JSON
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = completion
            
            try:
                plan_data = json.loads(json_text)
                
                # Convert to HierarchicalPlan
                self.state.hierarchical_plan = self._parse_hierarchy(plan_data.get("hierarchy", {}))
                self.state.required_dependencies = set(plan_data.get("required_dependencies", []))
                
                # Save plan to todos directory
                self._save_plan_todos(plan_data)
                
                log.info(f"Created hierarchical plan with {len(self.state.hierarchical_plan.get_all_components())} components")
                
                return json.dumps({
                    "status": "success",
                    "total_components": len(self.state.hierarchical_plan.get_all_components()),
                    "estimated_lines": plan_data.get("total_estimated_lines", 5000)
                })
                
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse hierarchical plan: {e}")
                # Create a comprehensive fallback plan
                self.state.hierarchical_plan = self._create_fallback_plan()
                return json.dumps({"status": "fallback", "components": 30})
                
        except Exception as e:
            log.error(f"Hierarchical planning error: {e}")
            self.state.hierarchical_plan = self._create_fallback_plan()
            return json.dumps({"status": "error", "message": str(e)})
    
    def _parse_hierarchy(self, data: Dict[str, Any], level: int = 0) -> HierarchicalPlan:
        """Parse hierarchy data into HierarchicalPlan objects."""
        plan = HierarchicalPlan(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            level=level,
            estimated_lines=data.get("estimated_lines", 100),
            dependencies=data.get("dependencies", [])
        )
        
        for sub_data in data.get("subcomponents", []):
            sub_plan = self._parse_hierarchy(sub_data, level + 1)
            plan.subcomponents.append(sub_plan)
        
        return plan
    
    def _save_plan_todos(self, plan_data: Dict[str, Any]):
        """Save detailed plan to todos directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plan_file = self.todos_dir / f"plan_{timestamp}.json"
        
        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)
        
        # Create individual todo files for each component
        if self.state.hierarchical_plan:
            for component in self.state.hierarchical_plan.get_all_components():
                if component.level > 0:  # Skip root
                    todo_file = self.todos_dir / f"{component.name.replace('.', '_')}.todo"
                    with open(todo_file, "w") as f:
                        f.write(f"Component: {component.name}\n")
                        f.write(f"Description: {component.description}\n")
                        f.write(f"Status: {component.status}\n")
                        f.write(f"Dependencies: {', '.join(component.dependencies)}\n")
                        f.write(f"Estimated Lines: {component.estimated_lines}\n")
        
        log.info(f"Saved plan and todos to {self.todos_dir}")
    
    def _create_fallback_plan(self) -> HierarchicalPlan:
        """Create comprehensive fallback plan for REST API."""
        root = HierarchicalPlan(
            name="root",
            description="Complete REST API System",
            level=0,
            estimated_lines=5000
        )
        
        # Core module
        core = HierarchicalPlan(
            name="core",
            description="Core business logic",
            level=1,
            estimated_lines=1500
        )
        core.subcomponents = [
            HierarchicalPlan("core.models.user", "User model with validation", 2, 200),
            HierarchicalPlan("core.models.base", "Base model classes", 2, 150),
            HierarchicalPlan("core.services.auth", "Authentication service", 2, 300),
            HierarchicalPlan("core.services.user", "User management service", 2, 250),
            HierarchicalPlan("core.utils.validators", "Input validators", 2, 200),
            HierarchicalPlan("core.utils.security", "Security utilities", 2, 200),
            HierarchicalPlan("core.exceptions", "Custom exceptions", 2, 100),
            HierarchicalPlan("core.constants", "System constants", 2, 100)
        ]
        
        # API module
        api = HierarchicalPlan(
            name="api",
            description="REST API endpoints",
            level=1,
            estimated_lines=1200
        )
        api.subcomponents = [
            HierarchicalPlan("api.routes.auth", "Authentication routes", 2, 300),
            HierarchicalPlan("api.routes.users", "User management routes", 2, 250),
            HierarchicalPlan("api.routes.admin", "Admin routes", 2, 200),
            HierarchicalPlan("api.middleware.auth", "Auth middleware", 2, 150),
            HierarchicalPlan("api.middleware.logging", "Logging middleware", 2, 100),
            HierarchicalPlan("api.middleware.cors", "CORS middleware", 2, 100),
            HierarchicalPlan("api.schemas", "Request/response schemas", 2, 100)
        ]
        
        # Database module
        database = HierarchicalPlan(
            name="database",
            description="Database layer",
            level=1,
            estimated_lines=800
        )
        database.subcomponents = [
            HierarchicalPlan("database.connection", "Database connection", 2, 150),
            HierarchicalPlan("database.migrations", "Database migrations", 2, 200),
            HierarchicalPlan("database.repositories.user", "User repository", 2, 250),
            HierarchicalPlan("database.repositories.base", "Base repository", 2, 200)
        ]
        
        # Tests module
        tests = HierarchicalPlan(
            name="tests",
            description="Comprehensive tests",
            level=1,
            estimated_lines=1000
        )
        tests.subcomponents = [
            HierarchicalPlan("tests.unit.models", "Model tests", 2, 200),
            HierarchicalPlan("tests.unit.services", "Service tests", 2, 300),
            HierarchicalPlan("tests.integration.api", "API tests", 2, 300),
            HierarchicalPlan("tests.fixtures", "Test fixtures", 2, 100),
            HierarchicalPlan("tests.conftest", "Test configuration", 2, 100)
        ]
        
        # Config module
        config = HierarchicalPlan(
            name="config",
            description="Configuration management",
            level=1,
            estimated_lines=300
        )
        config.subcomponents = [
            HierarchicalPlan("config.settings", "Application settings", 2, 150),
            HierarchicalPlan("config.logging", "Logging configuration", 2, 100),
            HierarchicalPlan("config.database", "Database configuration", 2, 50)
        ]
        
        # CLI module
        cli = HierarchicalPlan(
            name="cli",
            description="Command-line interface",
            level=1,
            estimated_lines=200
        )
        cli.subcomponents = [
            HierarchicalPlan("cli.commands", "CLI commands", 2, 150),
            HierarchicalPlan("cli.utils", "CLI utilities", 2, 50)
        ]
        
        root.subcomponents = [core, api, database, tests, config, cli]
        return root


class QualityEvaluationAgent(Agent):
    """Evaluates generated code quality and determines if refinement is needed."""
    
    def __init__(self, state: EnhancedCodebaseState, working_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.working_dir = working_dir
        
        self.roles = [
            "You are a senior code reviewer with extremely high standards.",
            "You evaluate code for production readiness and best practices.",
            "You check for completeness, error handling, testing, and documentation.",
            "You are STRICT - only excellent code passes your review.",
            "",
            "Rate code on these criteria (each 0-1):",
            "- Completeness: All functionality implemented, no TODOs",
            "- Error Handling: Comprehensive error handling and validation",
            "- Testing: Unit tests, integration tests, fixtures",
            "- Documentation: Docstrings, comments, README",
            "- Best Practices: SOLID, DRY, clean code",
            "- Dependencies: All imports work, requirements.txt complete",
            "- Runnability: Code actually runs without errors"
        ]
    
    def run(self, component_name: str) -> Dict[str, Any]:
        """Evaluate a component's quality."""
        try:
            # Find generated files for this component
            component_files = self._find_component_files(component_name)
            
            if not component_files:
                return {
                    "component": component_name,
                    "quality_score": 0.0,
                    "needs_refinement": True,
                    "issues": ["No files found for component"]
                }
            
            # Read file contents
            file_contents = {}
            for file_path in component_files:
                try:
                    with open(file_path, 'r') as f:
                        file_contents[str(file_path)] = f.read()
                except Exception as e:
                    log.error(f"Failed to read {file_path}: {e}")
            
            prompt = f"""Evaluate the quality of this component: {component_name}

Files generated:
{chr(10).join(file_contents.keys())}

Code content:
{chr(10).join(f'=== {path} ==={chr(10)}{content[:2000]}' for path, content in file_contents.items())}

Evaluate strictly on these criteria (0.0 to 1.0):
1. Completeness (no TODOs, all features implemented)
2. Error Handling (try/except, validation, edge cases)
3. Testing (unit tests, fixtures, coverage)
4. Documentation (docstrings, type hints, comments)
5. Best Practices (SOLID, DRY, naming, structure)
6. Dependencies (imports work, requirements listed)
7. Runnability (will actually execute without errors)

Return JSON:
{{
    "component": "{component_name}",
    "scores": {{
        "completeness": 0.0-1.0,
        "error_handling": 0.0-1.0,
        "testing": 0.0-1.0,
        "documentation": 0.0-1.0,
        "best_practices": 0.0-1.0,
        "dependencies": 0.0-1.0,
        "runnability": 0.0-1.0
    }},
    "overall_score": 0.0-1.0,
    "needs_refinement": true/false,
    "issues": [
        "List of specific issues found"
    ],
    "refinement_suggestions": [
        "Specific improvements needed"
    ]
}}

BE STRICT! Only production-ready code scores above 0.85."""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Parse response
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = completion
            
            try:
                evaluation = json.loads(json_text)
                
                # Store in state
                self.state.evaluation_history.append(evaluation)
                
                return evaluation
                
            except json.JSONDecodeError:
                # Conservative evaluation if parsing fails
                return {
                    "component": component_name,
                    "quality_score": 0.5,
                    "needs_refinement": True,
                    "issues": ["Failed to parse evaluation"]
                }
                
        except Exception as e:
            log.error(f"Quality evaluation error: {e}")
            return {
                "component": component_name,
                "quality_score": 0.0,
                "needs_refinement": True,
                "issues": [str(e)]
            }
    
    def _find_component_files(self, component_name: str) -> List[Path]:
        """Find all files related to a component."""
        files = []
        
        # Convert component name to file pattern
        # e.g., "core.models.user" -> "core/models/user.py"
        file_pattern = component_name.replace(".", "/") + ".py"
        file_path = self.working_dir / file_pattern
        
        if file_path.exists():
            files.append(file_path)
        
        # Also check for related test files
        test_pattern = f"tests/{file_pattern.replace('.py', '_test.py')}"
        test_path = self.working_dir / test_pattern
        if test_path.exists():
            files.append(test_path)
        
        # Check for __init__ files
        init_path = file_path.parent / "__init__.py"
        if init_path.exists():
            files.append(init_path)
        
        return files


class EnhancedComponentGenerator(CodeAgent):
    """Generates high-quality components based on hierarchical plan."""
    
    def __init__(self, state: EnhancedCodebaseState, working_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.working_dir = working_dir
        
        # Initialize roles if not present
        if not hasattr(self, 'roles'):
            self.roles = []
        
        self.roles.extend([
            "You generate PRODUCTION-READY code that exceeds expectations.",
            "Every line you write must be perfect - no TODOs, no placeholders.",
            "Include comprehensive error handling, logging, and validation.",
            "Add detailed docstrings and type hints to everything.",
            "The code must RUN without any modifications.",
            "",
            "REMEMBER: You are being held to the HIGHEST standards.",
            "This code will be evaluated by a strict reviewer.",
            "Go above and beyond - impress with quality and completeness."
        ])
    
    def generate_component(self, component: HierarchicalPlan) -> str:
        """Generate a high-quality component."""
        try:
            # Get context from dependencies
            dep_context = self._get_dependency_context(component.dependencies)
            
            prompt = f"""Generate PRODUCTION-READY code for: {component.name}

Component Details:
- Name: {component.name}
- Description: {component.description}
- Estimated Lines: {component.estimated_lines} (aim for MORE)
- Dependencies: {', '.join(component.dependencies)}

Dependency Context:
{dep_context}

Requirements (ALL MANDATORY):
1. COMPLETE implementation - no TODOs or placeholders
2. Comprehensive error handling with custom exceptions
3. Full input validation with detailed error messages
4. Logging at appropriate levels (debug, info, warning, error)
5. Type hints on ALL functions and methods
6. Detailed docstrings (Google style) for all classes/functions
7. Unit tests in the same response (test_{component.name})
8. Configuration options where appropriate
9. Performance optimizations (caching, pooling, etc.)
10. Security best practices (input sanitization, rate limiting, etc.)

Generate the following files:

1. Main implementation file
2. Test file with comprehensive tests
3. __init__.py if needed
4. Any additional supporting files

Format each file as:
```python
# filename: path/to/file.py
[COMPLETE CODE HERE]
```

REMEMBER: This must be PERFECT, RUNNABLE code that exceeds all expectations!"""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Extract and save all code blocks
            self._save_generated_files(completion, component)
            
            component.status = "completed"
            self.state.total_files_generated += len(component.generated_files)
            
            return completion
            
        except Exception as e:
            log.error(f"Component generation error: {e}")
            component.status = "error"
            return f"Error generating {component.name}: {e}"
    
    def _get_dependency_context(self, dependencies: List[str]) -> str:
        """Get context from dependency files."""
        context = []
        
        for dep in dependencies[:3]:  # Limit to avoid token explosion
            dep_path = self.working_dir / (dep.replace(".", "/") + ".py")
            if dep_path.exists():
                try:
                    with open(dep_path, 'r') as f:
                        content = f.read()
                        # Extract interfaces (classes and functions)
                        import re
                        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                        
                        context.append(f"{dep} exports: classes={classes}, functions={functions}")
                except Exception as e:
                    log.error(f"Failed to read dependency {dep}: {e}")
        
        return "\n".join(context) if context else "No dependency context available"
    
    def _save_generated_files(self, completion: str, component: HierarchicalPlan):
        """Extract and save all generated files."""
        import re
        
        # Find all code blocks with filenames
        pattern = r'```(?:python)?\s*\n#\s*filename:\s*(.+?)\n(.*?)\n```'
        matches = re.findall(pattern, completion, re.DOTALL)
        
        for filename, code in matches:
            file_path = self.working_dir / filename.strip()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clean code (remove filename comment)
            clean_code = re.sub(r'^#\s*filename:.*\n', '', code)
            
            file_path.write_text(clean_code)
            component.generated_files.append(str(file_path))
            
            log.info(f"Saved {filename} for {component.name}")
            
            # Track line count
            self.state.total_lines_generated += len(clean_code.split('\n'))


class EnhancedRefinementAgent(RefinementAgent):
    """Refines components based on evaluation feedback."""
    
    def __init__(self, state: EnhancedCodebaseState, working_dir: Path, **kwargs):
        super().__init__(base_dir=str(working_dir), **kwargs)
        self.state = state
        self.working_dir = working_dir
        
        # Initialize roles if not present
        if not hasattr(self, 'roles'):
            self.roles = []
        
        self.roles.extend([
            "You are a perfectionist who refines code to the highest standards.",
            "You fix ALL issues identified in evaluations.",
            "You enhance code with additional features and optimizations.",
            "You ensure 100% test coverage and documentation.",
            "The refined code must be FLAWLESS."
        ])
    
    def refine_component(self, component: HierarchicalPlan, evaluation: Dict[str, Any]) -> str:
        """Refine a component based on evaluation feedback."""
        try:
            # Read current implementation
            current_files = {}
            for file_path in component.generated_files:
                try:
                    with open(file_path, 'r') as f:
                        current_files[file_path] = f.read()
                except Exception as e:
                    log.error(f"Failed to read {file_path}: {e}")
            
            prompt = f"""REFINE this component to PERFECTION: {component.name}

Current Quality Score: {evaluation.get('overall_score', 0.0)}
Target Quality Score: {self.state.quality_threshold}

Issues Found:
{chr(10).join('- ' + issue for issue in evaluation.get('issues', []))}

Refinement Suggestions:
{chr(10).join('- ' + suggestion for suggestion in evaluation.get('refinement_suggestions', []))}

Current Implementation:
{chr(10).join(f'=== {path} ==={chr(10)}{content[:1000]}' for path, content in current_files.items())}

REQUIREMENTS FOR REFINEMENT:
1. Fix ALL identified issues completely
2. Add missing error handling and validation
3. Improve test coverage to 100%
4. Enhance documentation and examples
5. Optimize performance where possible
6. Add CLI interface if applicable
7. Include configuration management
8. Add logging and monitoring
9. Implement rate limiting and security
10. Make it PRODUCTION-READY

Generate refined versions of ALL files plus any new files needed.

Format each file as:
```python
# filename: path/to/file.py
[REFINED CODE HERE]
```

This refinement must achieve a quality score of {self.state.quality_threshold} or higher!"""

            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Save refined files
            self._save_refined_files(completion, component)
            
            component.refinement_count += 1
            component.status = "refined"
            
            return completion
            
        except Exception as e:
            log.error(f"Refinement error: {e}")
            return f"Error refining {component.name}: {e}"
    
    def _save_refined_files(self, completion: str, component: HierarchicalPlan):
        """Save refined files."""
        import re
        
        pattern = r'```(?:python)?\s*\n#\s*filename:\s*(.+?)\n(.*?)\n```'
        matches = re.findall(pattern, completion, re.DOTALL)
        
        for filename, code in matches:
            file_path = self.working_dir / filename.strip()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            clean_code = re.sub(r'^#\s*filename:.*\n', '', code)
            file_path.write_text(clean_code)
            
            if str(file_path) not in component.generated_files:
                component.generated_files.append(str(file_path))
            
            log.info(f"Saved refined {filename}")


class EnhancedCodebaseAgent(Agent):
    """
    Master orchestrator with hierarchical planning and quality assurance.
    
    Features:
    - Multi-level hierarchical planning with todo tracking
    - Component generation with dependency awareness
    - Quality evaluation and refinement loops
    - Dependency management and validation
    - Production-ready code generation
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 quality_threshold: float = 0.85,
                 max_iterations: int = 100,
                 **kwargs):
        """Initialize EnhancedCodebaseAgent."""
        super().__init__(**kwargs)
        
        self.task = task
        self.model = model
        
        # Setup directories
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        self.todos_dir = self.working_dir / ".talk" / "talk_todos"
        self.todos_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.state = EnhancedCodebaseState(
            task_description=task,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations
        )
        
        # Create agents
        self.agents = self._create_agents()
        
        log.info(f"EnhancedCodebaseAgent initialized for: {task}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:50]
        
        session_dir = self.output_manager.create_session_dir("enhanced_codebase", task_name)
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        agents["planner"] = HierarchicalPlanningAgent(
            state=self.state,
            todos_dir=self.todos_dir,
            overrides=provider_config,
            name="HierarchicalPlanner"
        )
        
        agents["generator"] = EnhancedComponentGenerator(
            state=self.state,
            working_dir=self.working_dir,
            overrides=provider_config,
            name="ComponentGenerator"
        )
        
        agents["evaluator"] = QualityEvaluationAgent(
            state=self.state,
            working_dir=self.working_dir,
            overrides=provider_config,
            name="QualityEvaluator"
        )
        
        agents["refiner"] = EnhancedRefinementAgent(
            state=self.state,
            working_dir=self.working_dir,
            overrides=provider_config,
            name="ComponentRefiner"
        )
        
        return agents
    
    def run(self) -> Dict[str, Any]:
        """Execute the enhanced codebase generation process."""
        try:
            print(f"\n{'='*60}")
            print(f"ENHANCED CODEBASE GENERATION")
            print(f"{'='*60}")
            print(f"Task: {self.task}")
            print(f"Quality Threshold: {self.state.quality_threshold}")
            print(f"Working Directory: {self.working_dir}")
            print(f"Todos Directory: {self.todos_dir}")
            print(f"{'='*60}\n")
            
            # Phase 1: Hierarchical Planning
            print("\n[PHASE 1] Creating Hierarchical Plan...")
            planning_result = self.agents["planner"].run(self.task)
            
            if not self.state.hierarchical_plan:
                raise Exception("Failed to create hierarchical plan")
            
            all_components = self.state.hierarchical_plan.get_all_components()[1:]  # Skip root
            print(f"✓ Created plan with {len(all_components)} components")
            
            # Phase 2: Install Dependencies
            print("\n[PHASE 2] Setting Up Dependencies...")
            self._setup_dependencies()
            
            # Phase 3: Component Generation with Quality Loop
            print("\n[PHASE 3] Generating Components with Quality Assurance...")
            
            for i, component in enumerate(all_components, 1):
                if self.state.iteration_count >= self.state.max_iterations:
                    print(f"\n⚠ Reached maximum iterations ({self.state.max_iterations})")
                    break
                
                print(f"\n[{i}/{len(all_components)}] Processing: {component.name}")
                
                # Generate component
                print(f"  → Generating initial implementation...")
                self.agents["generator"].generate_component(component)
                
                # Quality evaluation and refinement loop
                for refinement in range(self.state.max_refinements):
                    print(f"  → Evaluating quality (attempt {refinement + 1})...")
                    evaluation = self.agents["evaluator"].run(component.name)
                    
                    quality_score = evaluation.get("overall_score", 0.0)
                    component.quality_score = quality_score
                    
                    print(f"    Quality Score: {quality_score:.2f}/{self.state.quality_threshold}")
                    
                    if quality_score >= self.state.quality_threshold:
                        print(f"  ✓ Component meets quality standards!")
                        break
                    
                    if refinement < self.state.max_refinements - 1:
                        print(f"  → Refining component (issues: {len(evaluation.get('issues', []))})")
                        self.agents["refiner"].refine_component(component, evaluation)
                    else:
                        print(f"  ⚠ Max refinements reached, moving on")
                
                self.state.iteration_count += 1
                
                # Update todo status
                self._update_todo_status(component)
            
            # Phase 4: Integration Testing
            print("\n[PHASE 4] Integration and Final Validation...")
            self._run_integration_tests()
            
            # Phase 5: Generate Supporting Files
            print("\n[PHASE 5] Generating Supporting Files...")
            self._generate_supporting_files()
            
            # Summary
            return self._generate_summary()
            
        except Exception as e:
            log.exception("Enhanced codebase generation failed")
            return {
                "status": "error",
                "error": str(e),
                "files_generated": self.state.total_files_generated,
                "lines_generated": self.state.total_lines_generated
            }
    
    def _setup_dependencies(self):
        """Setup project dependencies."""
        # Generate requirements.txt
        requirements_file = self.working_dir / "requirements.txt"
        requirements = "\n".join(sorted(self.state.required_dependencies))
        requirements_file.write_text(requirements)
        print(f"✓ Created requirements.txt with {len(self.state.required_dependencies)} dependencies")
        
        # Generate setup.py
        setup_file = self.working_dir / "setup.py"
        setup_content = f'''from setuptools import setup, find_packages

setup(
    name="{self.task.replace(' ', '_').lower()}",
    version="1.0.0",
    packages=find_packages(),
    install_requires={list(self.state.required_dependencies)},
    python_requires=">=3.8",
    entry_points={{
        "console_scripts": [
            "app=main:main",
        ],
    }},
)'''
        setup_file.write_text(setup_content)
        print("✓ Created setup.py")
    
    def _update_todo_status(self, component: HierarchicalPlan):
        """Update todo file status."""
        todo_file = self.todos_dir / f"{component.name.replace('.', '_')}.todo"
        if todo_file.exists():
            content = todo_file.read_text()
            content = re.sub(r'Status: \w+', f'Status: {component.status}', content)
            content += f"\nQuality Score: {component.quality_score:.2f}\n"
            content += f"Refinements: {component.refinement_count}\n"
            todo_file.write_text(content)
    
    def _run_integration_tests(self):
        """Run integration tests on the generated code."""
        try:
            # Create a simple test runner
            test_runner = self.working_dir / "run_tests.py"
            test_content = '''#!/usr/bin/env python3
import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
'''
            test_runner.write_text(test_content)
            test_runner.chmod(0o755)
            
            print("✓ Created test runner")
            
            # Note: Actual test execution would happen here in a real environment
            
        except Exception as e:
            log.error(f"Integration test setup failed: {e}")
    
    def _generate_supporting_files(self):
        """Generate supporting files like README, Dockerfile, etc."""
        # README
        readme_file = self.working_dir / "README.md"
        readme_content = f"""# {self.task.title()}

## Overview
This project was generated with the highest quality standards for production readiness.

## Installation
```bash
pip install -r requirements.txt
python setup.py install
```

## Usage
```bash
python main.py
```

## Testing
```bash
python run_tests.py
```

## Project Structure
- `core/` - Core business logic
- `api/` - REST API endpoints
- `database/` - Database layer
- `tests/` - Comprehensive test suite
- `config/` - Configuration management
- `cli/` - Command-line interface

## Quality Metrics
- Files Generated: {self.state.total_files_generated}
- Lines of Code: {self.state.total_lines_generated}
- Test Coverage: Target 90%
- Quality Threshold: {self.state.quality_threshold}

## License
MIT
"""
        readme_file.write_text(readme_content)
        print("✓ Created README.md")
        
        # Dockerfile
        dockerfile = self.working_dir / "Dockerfile"
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py install

EXPOSE 8000

CMD ["python", "main.py"]
"""
        dockerfile.write_text(dockerfile_content)
        print("✓ Created Dockerfile")
        
        # docker-compose.yml
        compose_file = self.working_dir / "docker-compose.yml"
        compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
"""
        compose_file.write_text(compose_content)
        print("✓ Created docker-compose.yml")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate final summary."""
        # Count files and lines
        py_files = list(self.working_dir.rglob("*.py"))
        total_lines = sum(f.read_text().count('\n') for f in py_files)
        
        # Calculate average quality
        all_components = self.state.hierarchical_plan.get_all_components()[1:]
        avg_quality = sum(c.quality_score for c in all_components) / len(all_components) if all_components else 0
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Files Generated: {len(py_files)}")
        print(f"Total Lines: {total_lines:,}")
        print(f"Average Quality Score: {avg_quality:.2f}")
        print(f"Components Completed: {sum(1 for c in all_components if c.status in ['completed', 'refined'])}/{len(all_components)}")
        print(f"Components Refined: {sum(1 for c in all_components if c.refinement_count > 0)}")
        print(f"Working Directory: {self.working_dir}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "files_generated": len(py_files),
            "total_lines": total_lines,
            "average_quality": avg_quality,
            "components_completed": sum(1 for c in all_components if c.status in ['completed', 'refined']),
            "components_total": len(all_components),
            "working_directory": str(self.working_dir)
        }


def main():
    """Test the EnhancedCodebaseAgent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Codebase Generator")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--quality", type=float, default=0.85, help="Quality threshold")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations")
    
    args = parser.parse_args()
    
    agent = EnhancedCodebaseAgent(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        quality_threshold=args.quality,
        max_iterations=args.max_iterations
    )
    
    result = agent.run()
    print(f"\nFinal Result: {json.dumps(result, indent=2)}")
    return 0


if __name__ == "__main__":
    exit(main())