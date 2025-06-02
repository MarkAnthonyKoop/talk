#!/usr/bin/env python3
"""
Planner module for Talk CLI.

This module takes the output from the analyzer and generates a plan of code
improvements. It prioritizes improvements, estimates difficulty, and generates
detailed descriptions of the changes to be made.
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from talk.analyzer import (
    AnalysisResult, 
    Improvement, 
    ImprovementType, 
    CodeLocation,
    Language
)

logger = logging.getLogger("talk.planner")


class DifficultyLevel(Enum):
    """Difficulty levels for implementing improvements."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


class PriorityLevel(Enum):
    """Priority levels for improvements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class PlanningStrategy(Enum):
    """Planning strategies for generating improvement plans."""
    CONSERVATIVE = "conservative"  # Focus on safe, high-confidence improvements
    BALANCED = "balanced"          # Balance between safety and impact
    AGGRESSIVE = "aggressive"      # Prioritize high-impact improvements


@dataclass
class PlannedImprovement:
    """Represents a planned improvement with additional planning metadata."""
    improvement: Improvement
    priority: PriorityLevel
    difficulty: DifficultyLevel
    estimated_time_minutes: int
    detailed_description: str
    implementation_steps: List[str]
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the planned improvement."""
        return (
            f"{self.priority.value.capitalize()} priority, {self.difficulty.value} difficulty: "
            f"{self.improvement.description} (~{self.estimated_time_minutes} min)"
        )


@dataclass
class Plan:
    """Represents a plan of code improvements."""
    improvements: List[PlannedImprovement]
    strategy: PlanningStrategy
    total_estimated_time_minutes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_improvements_by_priority(self, priority: PriorityLevel) -> List[PlannedImprovement]:
        """Filter improvements by priority level."""
        return [i for i in self.improvements if i.priority == priority]

    def get_improvements_by_difficulty(self, difficulty: DifficultyLevel) -> List[PlannedImprovement]:
        """Filter improvements by difficulty level."""
        return [i for i in self.improvements if i.difficulty == difficulty]

    def get_improvements_by_type(self, improvement_type: ImprovementType) -> List[PlannedImprovement]:
        """Filter improvements by type."""
        return [i for i in self.improvements if i.improvement.type == improvement_type]

    def get_next_improvement(self) -> Optional[PlannedImprovement]:
        """Get the next improvement to implement based on priority and difficulty."""
        if not self.improvements:
            return None
        
        # Sort by priority (highest first) then by difficulty (easiest first)
        sorted_improvements = sorted(
            self.improvements,
            key=lambda i: (
                list(PriorityLevel).index(i.priority),
                list(DifficultyLevel).index(i.difficulty)
            )
        )
        
        return sorted_improvements[0]


class Planner:
    """Plans code improvements based on analysis results."""
    
    def __init__(
        self, 
        strategy: PlanningStrategy = PlanningStrategy.BALANCED,
        max_improvements: int = 10,
        min_confidence: float = 0.6
    ):
        """Initialize the planner with a strategy and limits."""
        self.strategy = strategy
        self.max_improvements = max_improvements
        self.min_confidence = min_confidence
    
    def create_plan(self, analysis_result: AnalysisResult) -> Plan:
        """Create a plan of code improvements based on analysis results."""
        logger.info(f"Creating improvement plan using {self.strategy.value} strategy")
        
        # Filter improvements by confidence
        confidence_threshold = self._get_confidence_threshold()
        filtered_improvements = [
            i for i in analysis_result.improvements 
            if i.confidence >= confidence_threshold
        ]
        
        logger.info(f"Found {len(filtered_improvements)} improvements above confidence threshold {confidence_threshold}")
        
        # Prioritize improvements
        prioritized_improvements = self._prioritize_improvements(filtered_improvements)
        
        # Limit the number of improvements based on strategy and max_improvements
        limit = min(self.max_improvements, len(prioritized_improvements))
        selected_improvements = prioritized_improvements[:limit]
        
        logger.info(f"Selected {len(selected_improvements)} improvements for the plan")
        
        # Create planned improvements with additional metadata
        planned_improvements = []
        total_time = 0
        
        for improvement in selected_improvements:
            planned = self._create_planned_improvement(improvement, analysis_result)
            planned_improvements.append(planned)
            total_time += planned.estimated_time_minutes
        
        # Create and return the plan
        plan = Plan(
            improvements=planned_improvements,
            strategy=self.strategy,
            total_estimated_time_minutes=total_time,
            metadata={
                'language': analysis_result.language.value,
                'files_analyzed': len(analysis_result.files_analyzed),
                'total_improvements_found': len(analysis_result.improvements),
                'improvements_above_threshold': len(filtered_improvements),
                'improvements_selected': len(planned_improvements),
            }
        )
        
        return plan
    
    def _get_confidence_threshold(self) -> float:
        """Get the confidence threshold based on the strategy."""
        if self.strategy == PlanningStrategy.CONSERVATIVE:
            return max(0.8, self.min_confidence)
        elif self.strategy == PlanningStrategy.BALANCED:
            return max(0.7, self.min_confidence)
        elif self.strategy == PlanningStrategy.AGGRESSIVE:
            return max(0.6, self.min_confidence)
        else:
            return self.min_confidence
    
    def _prioritize_improvements(self, improvements: List[Improvement]) -> List[Improvement]:
        """Prioritize improvements based on type, confidence, and strategy."""
        # Define type weights based on strategy
        type_weights = self._get_type_weights()
        
        # Score each improvement
        scored_improvements = []
        for improvement in improvements:
            # Base score is the confidence
            score = improvement.confidence
            
            # Adjust score based on improvement type
            type_weight = type_weights.get(improvement.type, 1.0)
            score *= type_weight
            
            scored_improvements.append((improvement, score))
        
        # Sort by score (descending)
        scored_improvements.sort(key=lambda x: x[1], reverse=True)
        
        # Return the sorted improvements
        return [i[0] for i in scored_improvements]
    
    def _get_type_weights(self) -> Dict[ImprovementType, float]:
        """Get weights for improvement types based on strategy."""
        if self.strategy == PlanningStrategy.CONSERVATIVE:
            return {
                ImprovementType.UNUSED_CODE: 1.5,      # Safe to remove
                ImprovementType.STYLE: 1.3,            # Safe to fix
                ImprovementType.DOCUMENTATION: 1.2,    # Safe to add
                ImprovementType.COMPLEXITY: 0.8,       # Risky to refactor
                ImprovementType.PERFORMANCE: 0.7,      # Risky to optimize
                ImprovementType.SECURITY: 1.4,         # Important but careful
                ImprovementType.TESTING: 1.1,          # Safe to add
                ImprovementType.DUPLICATION: 0.9       # Risky to deduplicate
            }
        elif self.strategy == PlanningStrategy.BALANCED:
            return {
                ImprovementType.UNUSED_CODE: 1.2,
                ImprovementType.STYLE: 1.0,
                ImprovementType.DOCUMENTATION: 1.0,
                ImprovementType.COMPLEXITY: 1.1,
                ImprovementType.PERFORMANCE: 1.1,
                ImprovementType.SECURITY: 1.5,         # Security is always important
                ImprovementType.TESTING: 1.2,
                ImprovementType.DUPLICATION: 1.1
            }
        elif self.strategy == PlanningStrategy.AGGRESSIVE:
            return {
                ImprovementType.UNUSED_CODE: 1.0,
                ImprovementType.STYLE: 0.8,            # Less important
                ImprovementType.DOCUMENTATION: 0.7,    # Less important
                ImprovementType.COMPLEXITY: 1.4,       # Focus on refactoring
                ImprovementType.PERFORMANCE: 1.5,      # Focus on optimization
                ImprovementType.SECURITY: 1.5,         # Security is always important
                ImprovementType.TESTING: 1.0,
                ImprovementType.DUPLICATION: 1.3       # Focus on deduplication
            }
        else:
            return {t: 1.0 for t in ImprovementType}
    
    def _estimate_difficulty(self, improvement: Improvement) -> DifficultyLevel:
        """Estimate the difficulty of implementing an improvement."""
        # Base difficulty on improvement type
        if improvement.type == ImprovementType.STYLE:
            base_difficulty = DifficultyLevel.TRIVIAL
        elif improvement.type == ImprovementType.DOCUMENTATION:
            base_difficulty = DifficultyLevel.EASY
        elif improvement.type == ImprovementType.UNUSED_CODE:
            base_difficulty = DifficultyLevel.EASY
        elif improvement.type == ImprovementType.TESTING:
            base_difficulty = DifficultyLevel.MEDIUM
        elif improvement.type == ImprovementType.DUPLICATION:
            base_difficulty = DifficultyLevel.MEDIUM
        elif improvement.type == ImprovementType.COMPLEXITY:
            base_difficulty = DifficultyLevel.HARD
        elif improvement.type == ImprovementType.PERFORMANCE:
            base_difficulty = DifficultyLevel.HARD
        elif improvement.type == ImprovementType.SECURITY:
            base_difficulty = DifficultyLevel.HARD
        else:
            base_difficulty = DifficultyLevel.MEDIUM
        
        # Adjust based on confidence (higher confidence = easier)
        confidence_adjustment = 0
        if improvement.confidence > 0.9:
            confidence_adjustment = -1  # Easier
        elif improvement.confidence < 0.7:
            confidence_adjustment = 1   # Harder
        
        # Calculate adjusted difficulty level
        difficulty_levels = list(DifficultyLevel)
        base_index = difficulty_levels.index(base_difficulty)
        adjusted_index = max(0, min(len(difficulty_levels) - 1, base_index + confidence_adjustment))
        
        return difficulty_levels[adjusted_index]
    
    def _estimate_time(self, improvement: Improvement, difficulty: DifficultyLevel) -> int:
        """Estimate the time (in minutes) to implement an improvement."""
        # Base time on difficulty
        if difficulty == DifficultyLevel.TRIVIAL:
            base_time = 5
        elif difficulty == DifficultyLevel.EASY:
            base_time = 15
        elif difficulty == DifficultyLevel.MEDIUM:
            base_time = 30
        elif difficulty == DifficultyLevel.HARD:
            base_time = 60
        elif difficulty == DifficultyLevel.VERY_HARD:
            base_time = 120
        else:
            base_time = 30
        
        # Adjust based on improvement type
        type_multiplier = 1.0
        if improvement.type == ImprovementType.COMPLEXITY:
            type_multiplier = 1.5  # Complex refactorings take longer
        elif improvement.type == ImprovementType.SECURITY:
            type_multiplier = 1.3  # Security fixes require careful testing
        
        # Calculate final time estimate
        return int(base_time * type_multiplier)
    
    def _determine_priority(self, improvement: Improvement) -> PriorityLevel:
        """Determine the priority of an improvement."""
        # Security issues are always high priority
        if improvement.type == ImprovementType.SECURITY:
            return PriorityLevel.CRITICAL
        
        # Base priority on confidence and type
        if improvement.confidence > 0.9:
            base_priority = PriorityLevel.HIGH
        elif improvement.confidence > 0.8:
            base_priority = PriorityLevel.MEDIUM
        elif improvement.confidence > 0.7:
            base_priority = PriorityLevel.LOW
        else:
            base_priority = PriorityLevel.OPTIONAL
        
        # Adjust based on improvement type and strategy
        if self.strategy == PlanningStrategy.CONSERVATIVE:
            # In conservative mode, prioritize safe improvements
            if improvement.type in [ImprovementType.UNUSED_CODE, ImprovementType.STYLE]:
                return self._increase_priority(base_priority)
            elif improvement.type in [ImprovementType.COMPLEXITY, ImprovementType.PERFORMANCE]:
                return self._decrease_priority(base_priority)
        
        elif self.strategy == PlanningStrategy.AGGRESSIVE:
            # In aggressive mode, prioritize impactful improvements
            if improvement.type in [ImprovementType.COMPLEXITY, ImprovementType.PERFORMANCE]:
                return self._increase_priority(base_priority)
            elif improvement.type in [ImprovementType.DOCUMENTATION, ImprovementType.STYLE]:
                return self._decrease_priority(base_priority)
        
        return base_priority
    
    def _increase_priority(self, priority: PriorityLevel) -> PriorityLevel:
        """Increase a priority level by one step."""
        priorities = list(PriorityLevel)
        current_index = priorities.index(priority)
        new_index = max(0, current_index - 1)  # Lower index = higher priority
        return priorities[new_index]
    
    def _decrease_priority(self, priority: PriorityLevel) -> PriorityLevel:
        """Decrease a priority level by one step."""
        priorities = list(PriorityLevel)
        current_index = priorities.index(priority)
        new_index = min(len(priorities) - 1, current_index + 1)  # Higher index = lower priority
        return priorities[new_index]
    
    def _create_planned_improvement(
        self, 
        improvement: Improvement, 
        analysis_result: AnalysisResult
    ) -> PlannedImprovement:
        """Create a planned improvement with additional metadata."""
        # Determine difficulty and priority
        difficulty = self._estimate_difficulty(improvement)
        priority = self._determine_priority(improvement)
        
        # Estimate time
        estimated_time = self._estimate_time(improvement, difficulty)
        
        # Generate detailed description and implementation steps
        detailed_description = self._generate_detailed_description(improvement, analysis_result)
        implementation_steps = self._generate_implementation_steps(improvement, analysis_result)
        
        # Generate before/after code if possible
        before_code = self._extract_code_snippet(improvement.location)
        after_code = self._generate_after_code(improvement, before_code, analysis_result)
        
        # Create and return the planned improvement
        return PlannedImprovement(
            improvement=improvement,
            priority=priority,
            difficulty=difficulty,
            estimated_time_minutes=estimated_time,
            detailed_description=detailed_description,
            implementation_steps=implementation_steps,
            before_code=before_code,
            after_code=after_code
        )
    
    def _extract_code_snippet(self, location: CodeLocation) -> Optional[str]:
        """Extract the code snippet from a file location."""
        try:
            with open(location.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract the relevant lines
            start_line = max(0, location.start_line - 1)  # Convert to 0-based indexing
            end_line = min(len(lines), location.end_line)
            
            return ''.join(lines[start_line:end_line])
        except Exception as e:
            logger.debug(f"Failed to extract code snippet: {e}")
            return None
    
    def _generate_detailed_description(
        self, 
        improvement: Improvement, 
        analysis_result: AnalysisResult
    ) -> str:
        """Generate a detailed description of the improvement."""
        base_description = improvement.description
        
        if improvement.type == ImprovementType.COMPLEXITY:
            return (
                f"{base_description}\n\n"
                f"This code is too complex and should be refactored to improve readability and maintainability. "
                f"Consider breaking it down into smaller, more focused functions or methods. "
                f"Complex code is harder to understand, test, and maintain."
            )
        
        elif improvement.type == ImprovementType.DOCUMENTATION:
            return (
                f"{base_description}\n\n"
                f"Adding proper documentation will improve code readability and help other developers "
                f"understand the purpose and usage of this code. Documentation should include a clear "
                f"description of what the code does, parameters, return values, and any exceptions that might be raised."
            )
        
        elif improvement.type == ImprovementType.UNUSED_CODE:
            return (
                f"{base_description}\n\n"
                f"Removing unused code improves codebase clarity and reduces maintenance burden. "
                f"Dead code can confuse developers and lead to bugs when partially updated. "
                f"It also increases the size of the codebase unnecessarily."
            )
        
        elif improvement.type == ImprovementType.PERFORMANCE:
            return (
                f"{base_description}\n\n"
                f"This code could be optimized for better performance. The current implementation "
                f"may be inefficient and could lead to performance issues, especially with larger inputs or "
                f"in performance-critical sections of the application."
            )
        
        elif improvement.type == ImprovementType.STYLE:
            return (
                f"{base_description}\n\n"
                f"Adhering to consistent code style improves readability and maintainability. "
                f"It makes the codebase look professional and helps developers quickly understand the code."
            )
        
        elif improvement.type == ImprovementType.SECURITY:
            return (
                f"{base_description}\n\n"
                f"This code may have security implications and should be fixed to prevent potential vulnerabilities. "
                f"Security issues can lead to data breaches, unauthorized access, or other security incidents."
            )
        
        elif improvement.type == ImprovementType.TESTING:
            return (
                f"{base_description}\n\n"
                f"Adding tests will improve code reliability and make it easier to refactor in the future. "
                f"Tests help catch bugs early and provide documentation of expected behavior."
            )
        
        elif improvement.type == ImprovementType.DUPLICATION:
            return (
                f"{base_description}\n\n"
                f"Duplicate code increases maintenance burden and the risk of inconsistent updates. "
                f"It should be refactored into reusable functions or methods to improve maintainability."
            )
        
        else:
            return base_description
    
    def _generate_implementation_steps(
        self, 
        improvement: Improvement, 
        analysis_result: AnalysisResult
    ) -> List[str]:
        """Generate implementation steps for the improvement."""
        if improvement.type == ImprovementType.COMPLEXITY:
            return [
                "Identify the complex parts of the code and understand their purpose",
                "Break down complex logic into smaller, focused functions",
                "Rename variables and functions to clarify their purpose",
                "Add comments to explain complex algorithms or business logic",
                "Ensure tests pass after refactoring"
            ]
        
        elif improvement.type == ImprovementType.DOCUMENTATION:
            return [
                "Understand the purpose and behavior of the code",
                "Add docstrings with descriptions, parameters, return values, and exceptions",
                "Include examples if appropriate",
                "Ensure documentation follows project conventions"
            ]
        
        elif improvement.type == ImprovementType.UNUSED_CODE:
            return [
                "Verify that the code is truly unused (not referenced elsewhere)",
                "Remove the unused code",
                "Run tests to ensure no regressions"
            ]
        
        elif improvement.type == ImprovementType.PERFORMANCE:
            return [
                "Profile the code to identify bottlenecks",
                "Research more efficient algorithms or data structures",
                "Implement the optimization",
                "Benchmark before and after to verify improvement",
                "Ensure tests pass after optimization"
            ]
        
        elif improvement.type == ImprovementType.STYLE:
            return [
                "Apply style fixes according to project conventions",
                "Ensure consistent indentation, naming, and formatting",
                "Run linters or formatters if available"
            ]
        
        elif improvement.type == ImprovementType.SECURITY:
            return [
                "Understand the security implications of the current code",
                "Research best practices for addressing the security issue",
                "Implement the fix with careful attention to edge cases",
                "Add tests to verify the security issue is resolved",
                "Consider adding security-focused comments to prevent future issues"
            ]
        
        elif improvement.type == ImprovementType.TESTING:
            return [
                "Understand the code's expected behavior",
                "Write unit tests covering normal cases, edge cases, and error cases",
                "Ensure tests are meaningful and verify actual behavior",
                "Run tests to verify they pass with the current implementation"
            ]
        
        elif improvement.type == ImprovementType.DUPLICATION:
            return [
                "Identify the duplicated code sections",
                "Extract common functionality into reusable functions or methods",
                "Update all occurrences to use the new shared code",
                "Ensure tests pass after refactoring"
            ]
        
        else:
            return ["Implement the improvement as described"]
    
    def _generate_after_code(
        self, 
        improvement: Improvement, 
        before_code: Optional[str], 
        analysis_result: AnalysisResult
    ) -> Optional[str]:
        """Generate the improved code after applying the improvement."""
        if not before_code:
            return None
        
        # This is a simplified implementation. In a real-world scenario,
        # this would use more sophisticated code generation techniques,
        # possibly with the help of AI models or code transformation tools.
        
        if improvement.type == ImprovementType.UNUSED_CODE:
            # For unused imports, we can generate a simple fix by removing the import
            if "Unused import" in improvement.description:
                import_name = improvement.metadata.get('name', '')
                if import_name:
                    # Simple regex to remove the import
                    pattern = rf'(from\s+\S+\s+import\s+.*{re.escape(import_name)}.*|import\s+.*{re.escape(import_name)}.*)\n'
                    after_code = re.sub(pattern, '', before_code)
                    return after_code
        
        elif improvement.type == ImprovementType.DOCUMENTATION:
            # For missing docstrings, we can generate a simple template
            if "Missing docstring" in improvement.description:
                node_type = improvement.metadata.get('node_type', '')
                name = improvement.metadata.get('name', '')
                
                if node_type == 'function':
                    # Add a simple function docstring template
                    docstring = f'    """\n    {name} function.\n    \n    Args:\n        # TODO: Add parameters\n    \n    Returns:\n        # TODO: Add return value\n    """\n'
                    # Insert after the function definition line
                    lines = before_code.splitlines()
                    for i, line in enumerate(lines):
                        if f"def {name}" in line and i < len(lines) - 1:
                            indentation = re.match(r'(\s*)', lines[i+1]).group(1)
                            lines.insert(i+1, f"{indentation}{docstring}")
                            break
                    return '\n'.join(lines)
                
                elif node_type == 'class':
                    # Add a simple class docstring template
                    docstring = f'    """\n    {name} class.\n    \n    Attributes:\n        # TODO: Add attributes\n    """\n'
                    # Insert after the class definition line
                    lines = before_code.splitlines()
                    for i, line in enumerate(lines):
                        if f"class {name}" in line and i < len(lines) - 1:
                            indentation = re.match(r'(\s*)', lines[i+1]).group(1)
                            lines.insert(i+1, f"{indentation}{docstring}")
                            break
                    return '\n'.join(lines)
        
        # For other improvement types, we would need more sophisticated code generation
        # techniques, which are beyond the scope of this simplified implementation.
        return None


def create_planner(
    strategy_name: str = "balanced",
    max_improvements: int = 10,
    min_confidence: float = 0.6
) -> Planner:
    """Factory function to create a planner with the specified strategy."""
    # Map strategy name to enum
    strategy_map = {
        "conservative": PlanningStrategy.CONSERVATIVE,
        "balanced": PlanningStrategy.BALANCED,
        "aggressive": PlanningStrategy.AGGRESSIVE
    }
    
    strategy = strategy_map.get(strategy_name.lower(), PlanningStrategy.BALANCED)
    
    return Planner(
        strategy=strategy,
        max_improvements=max_improvements,
        min_confidence=min_confidence
    )


def plan_improvements(
    analysis_result: AnalysisResult,
    strategy_name: str = "balanced",
    max_improvements: int = 10,
    min_confidence: float = 0.6
) -> Plan:
    """Plan improvements based on analysis results."""
    planner = create_planner(
        strategy_name=strategy_name,
        max_improvements=max_improvements,
        min_confidence=min_confidence
    )
    
    return planner.create_plan(analysis_result)
