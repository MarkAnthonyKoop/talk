"""
Agent Collaboration System

Real-time collaboration framework for multi-agent systems working on shared codebases.

This package provides:
- AgentMessageBus: Real-time communication hub for agents
- SharedWorkspace: Coordinated file system access with locking
- VotingSystem: Collaborative decision making with multiple voting strategies

Components:
- agent_message_bus: Async pub/sub messaging system
- shared_workspace: File coordination with version control integration
- collaborative_decision_making: Voting and consensus mechanisms

Example usage:
    from special_agents.collaboration import AgentMessageBus, SharedWorkspace, VotingSystem
    
    # Initialize systems
    bus = AgentMessageBus()
    workspace = SharedWorkspace("/path/to/workspace")
    voting = VotingSystem()
    
    # Start collaboration
    await bus.start()
    await bus.register_agent("agent1", "coder", ["python"])
    await workspace.create_file("test.py", "print('hello')", "agent1")
    decision_id = await voting.create_decision("Merge PR", "...", DecisionType.CODE_MERGE, "agent1")
"""

from .agent_message_bus import AgentMessageBus, MessageType, Message
from .shared_workspace import SharedWorkspace, LockType
from .collaborative_decision_making import VotingSystem, DecisionType, VoteType, VotingStrategy

__all__ = [
    "AgentMessageBus",
    "MessageType", 
    "Message",
    "SharedWorkspace",
    "LockType",
    "VotingSystem",
    "DecisionType",
    "VoteType", 
    "VotingStrategy"
]

__version__ = "1.0.0"