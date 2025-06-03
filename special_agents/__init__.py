"""
Special Agents package for the Talk framework.

This package contains specialized agent implementations that extend the base Agent class
with specific capabilities like code generation, file operations, shell command execution,
test running, and control flow management.
"""

from special_agents.branching_agent import BranchingAgent
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.shell_agent import ShellAgent
from special_agents.test_agent import TestAgent

__all__ = [
    "BranchingAgent",
    "CodeAgent",
    "FileAgent",
    "ShellAgent",
    "TestAgent",
]
