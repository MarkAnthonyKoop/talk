"""
Agentic Orchestration System

A comprehensive orchestration framework for managing multi-agent systems
with dynamic spawning, load balancing, and intelligent task distribution.
"""

from .core import AgentOrchestrator
from .registry import AgentRegistry
from .dispatcher import TaskDispatcher
from .monitor import OrchestrationMonitor
from .lifecycle import AgentLifecycleManager

__all__ = [
    'AgentOrchestrator',
    'AgentRegistry',
    'TaskDispatcher',
    'OrchestrationMonitor',
    'AgentLifecycleManager'
]