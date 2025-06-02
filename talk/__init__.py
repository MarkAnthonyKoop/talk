"""
Talk - Autonomous CLI for continuous, unattended improvement of any codebase.

This package provides tools for analyzing, planning, and executing
code improvements automatically without supervision.
"""

__version__ = "0.1.0"
__author__ = "Mark Anthony Koop"
__license__ = "MIT"

# Import main components from submodules
from talk.analyzer import (
    analyze_codebase,
    Analyzer,
    PythonAnalyzer,
    AnalysisResult,
    Improvement,
    ImprovementType,
    CodeLocation,
    Language,
)

from talk.planner import (
    plan_improvements,
    Planner,
    Plan,
    PlannedImprovement,
    PlanningStrategy,
    PriorityLevel,
    DifficultyLevel,
)

from talk.executor import (
    execute_plan,
    Executor,
    ExecutionResult,
    ExecutedImprovement,
    ExecutionStatus,
    BackupStrategy,
)

# Import main CLI function
from talk.cli import main as cli_main

# Convenience function to run the CLI
def main():
    """Run the Talk CLI."""
    return cli_main()

# Define what's available when using `from talk import *`
__all__ = [
    # Main functions
    "analyze_codebase",
    "plan_improvements",
    "execute_plan",
    "cli_main",
    "main",
    
    # Main classes
    "Analyzer",
    "PythonAnalyzer",
    "Planner",
    "Executor",
    
    # Result classes
    "AnalysisResult",
    "Plan",
    "ExecutionResult",
    
    # Improvement classes
    "Improvement",
    "PlannedImprovement",
    "ExecutedImprovement",
    
    # Enums
    "ImprovementType",
    "PlanningStrategy",
    "PriorityLevel",
    "DifficultyLevel",
    "ExecutionStatus",
    "BackupStrategy",
    "Language",
    
    # Utility classes
    "CodeLocation",
]
