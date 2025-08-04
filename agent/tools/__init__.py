# agent/tools/__init__.py
"""Unified tool definitions for Talk agents."""

from .file_ops import FILE_TOOLS
from .shell_ops import SHELL_TOOLS

ALL_TOOLS = FILE_TOOLS + SHELL_TOOLS

__all__ = ["FILE_TOOLS", "SHELL_TOOLS", "ALL_TOOLS"]