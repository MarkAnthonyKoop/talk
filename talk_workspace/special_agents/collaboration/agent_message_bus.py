#!/usr/bin/env python3
"""
AgentMessageBus - Real-time communication hub for agent collaboration.

This module provides a centralized message bus that enables real-time communication
between multiple agents working on the same codebase. It supports:

- Asynchronous message publishing and subscription
- Topic-based message routing
- Message persistence and replay
- Agent presence tracking
- Event-driven architecture for coordination

Features:
- High-performance async/await messaging
- Type-safe message handling
- Automatic agent registration and discovery
- Message history and debugging support
- Configurable message retention policies
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum

log = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in the agent collaboration system."""
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update" 
    CODE_CHANGE = "code_change"
    TEST_RESULT = "test_result"
    DECISION_REQUEST = "decision_request"
    VOTE = "vote"
    CONFLICT_ALERT = "conflict_alert"
    COORDINATION = "coordination"
    HEARTBEAT = "heartbeat"
    AGENT_JOIN = "agent_join"
    AGENT_LEAVE = "agent_leave"

@dataclass
class Message:
    """A message in the agent collaboration system."""
    id: str
    sender_id: str
    topic: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    reply_to: Optional[str] = None
    expiry: Optional[float] = None
    priority: int = 0  # Higher = more important
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

@dataclass
class AgentInfo:
    """Information about an agent in the collaboration system."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: str  # "active", "busy", "idle", "offline"
    last_heartbeat: float
    metadata: Dict[str, Any]
    
    def is_online(self, timeout: float = 30.0) -> bool:
        """Check if agent is considered online."""
        return time.time() - self.last_heartbeat < timeout

class AgentMessageBus:
    """
    Real-time message bus for agent collaboration.
    
    Provides pub/sub messaging, agent discovery, and coordination features
    for multi-agent systems working on shared codebases.
    """
    
    def __init__(self, max_history: int = 1000, heartbeat_interval: float = 10.0):
        """
        Initialize the message bus.
        
        Args:
            max_history: Maximum number of messages to keep in history
            heartbeat_interval: How often agents should send heartbeats (seconds)
        """
        self.max_history = max_history
        self.heartbeat_interval = heartbeat_interval
        
        # Message storage and routing
        self.message_history: List[Message] = []
        self.subscribers: Dict[str, List[Callable]] = {}  # topic -> [callbacks]
        self.agent_subscribers: Dict[str, Dict[str, Callable]] = {}  # agent_id -> {topic: callback}
        
        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_last_seen: Dict[str, float] = {}
        
        # Performance tracking
        self.message_count = 0
        self.start_time = time.time()
        
        # Task coordination
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, List[str]] = {}  # agent_id -> [task_ids]
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the message bus services."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        log.info("AgentMessageBus started")
    
    async def stop(self):
        """Stop the message bus services."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        log.info("AgentMessageBus stopped")
    
    async def _cleanup_loop(self):
        """Background task to clean up expired messages and offline agents."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
                # Remove expired messages
                current_time = time.time()
                self.message_history = [
                    msg for msg in self.message_history 
                    if not msg.is_expired()
                ]
                
                # Trim message history if too large
                if len(self.message_history) > self.max_history:
                    self.message_history = self.message_history[-self.max_history:]
                
                # Mark offline agents
                for agent_id, agent_info in self.agents.items():
                    if not agent_info.is_online():
                        agent_info.status = "offline"
                        await self._publish_system_message(
                            "agent.offline",
                            MessageType.AGENT_LEAVE,
                            {"agent_id": agent_id, "reason": "timeout"}
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in cleanup loop: {e}")
    
    async def register_agent(
        self, 
        agent_id: str, 
        agent_type: str, 
        capabilities: List[str],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Register an agent with the message bus.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., "coder", "tester", "file")
            capabilities: List of capabilities this agent provides
            metadata: Additional agent metadata
            
        Returns:
            True if registration successful
        """
        if metadata is None:
            metadata = {}
        
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type, 
            capabilities=capabilities,
            status="active",
            last_heartbeat=time.time(),
            metadata=metadata
        )
        
        self.agents[agent_id] = agent_info
        self.agent_subscribers[agent_id] = {}
        
        # Announce agent join
        await self._publish_system_message(
            "agent.join",
            MessageType.AGENT_JOIN,
            {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": capabilities,
                "metadata": metadata
            }
        )
        
        log.info(f"Agent registered: {agent_id} ({agent_type})")
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the message bus.
        
        Args:
            agent_id: Agent to unregister
            
        Returns:
            True if unregistration successful
        """
        if agent_id not in self.agents:
            return False
        
        # Remove agent subscriptions
        if agent_id in self.agent_subscribers:
            for topic in self.agent_subscribers[agent_id]:
                if topic in self.subscribers:
                    self.subscribers[topic] = [
                        cb for cb in self.subscribers[topic] 
                        if cb != self.agent_subscribers[agent_id][topic]
                    ]
            del self.agent_subscribers[agent_id]
        
        # Remove from assignments
        if agent_id in self.agent_assignments:
            del self.agent_assignments[agent_id]
        
        # Announce agent leave
        await self._publish_system_message(
            "agent.leave", 
            MessageType.AGENT_LEAVE,
            {"agent_id": agent_id, "reason": "unregistered"}
        )
        
        del self.agents[agent_id]
        log.info(f"Agent unregistered: {agent_id}")
        return True
    
    async def heartbeat(self, agent_id: str, status: str = "active", metadata: Dict[str, Any] = None):
        """
        Send heartbeat from an agent.
        
        Args:
            agent_id: Agent sending heartbeat
            status: Current agent status
            metadata: Additional status metadata
        """
        if agent_id not in self.agents:
            log.warning(f"Heartbeat from unknown agent: {agent_id}")
            return
        
        agent_info = self.agents[agent_id]
        agent_info.last_heartbeat = time.time()
        agent_info.status = status
        
        if metadata:
            agent_info.metadata.update(metadata)
        
        # Publish heartbeat message
        await self._publish_system_message(
            "agent.heartbeat",
            MessageType.HEARTBEAT,
            {
                "agent_id": agent_id,
                "status": status,
                "metadata": metadata or {}
            }
        )
    
    async def subscribe(self, topic: str, callback: Callable, agent_id: str = None) -> bool:
        """
        Subscribe to messages on a topic.
        
        Args:
            topic: Topic to subscribe to (supports wildcards like "task.*")
            callback: Async function to call when messages arrive
            agent_id: Optional agent ID for tracking subscriptions
            
        Returns:
            True if subscription successful
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        
        if agent_id and agent_id in self.agent_subscribers:
            self.agent_subscribers[agent_id][topic] = callback
        
        log.debug(f"Subscription added: {topic} (agent: {agent_id})")
        return True
    
    async def unsubscribe(self, topic: str, callback: Callable, agent_id: str = None) -> bool:
        """
        Unsubscribe from messages on a topic.
        
        Args:
            topic: Topic to unsubscribe from
            callback: Callback to remove
            agent_id: Optional agent ID
            
        Returns:
            True if unsubscription successful
        """
        if topic not in self.subscribers:
            return False
        
        try:
            self.subscribers[topic].remove(callback)
            
            if agent_id and agent_id in self.agent_subscribers:
                if topic in self.agent_subscribers[agent_id]:
                    del self.agent_subscribers[agent_id][topic]
            
            log.debug(f"Subscription removed: {topic} (agent: {agent_id})")
            return True
        except ValueError:
            return False
    
    async def publish(self, topic: str, message_type: MessageType, content: Dict[str, Any], 
                     sender_id: str = "system", priority: int = 0, expiry_seconds: float = None) -> str:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            message_type: Type of message
            content: Message content
            sender_id: ID of sender
            priority: Message priority (higher = more important)
            expiry_seconds: Message expiry time in seconds
            
        Returns:
            Message ID
        """
        expiry = None
        if expiry_seconds:
            expiry = time.time() + expiry_seconds
        
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            topic=topic,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            priority=priority,
            expiry=expiry
        )
        
        # Store in history
        self.message_history.append(message)
        self.message_count += 1
        
        # Deliver to subscribers
        await self._deliver_message(message)
        
        log.debug(f"Message published: {topic} from {sender_id}")
        return message.id
    
    async def _deliver_message(self, message: Message):
        """Deliver a message to all matching subscribers."""
        matching_topics = self._get_matching_topics(message.topic)
        
        for topic in matching_topics:
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message.topic, message.to_dict())
                        else:
                            callback(message.topic, message.to_dict())
                    except Exception as e:
                        log.error(f"Error delivering message to subscriber: {e}")
    
    def _get_matching_topics(self, message_topic: str) -> List[str]:
        """Get all subscriber topics that match a message topic."""
        matching = []
        
        for subscriber_topic in self.subscribers:
            if self._topic_matches(message_topic, subscriber_topic):
                matching.append(subscriber_topic)
        
        return matching
    
    def _topic_matches(self, message_topic: str, subscriber_topic: str) -> bool:
        """Check if a message topic matches a subscriber pattern."""
        # Exact match
        if message_topic == subscriber_topic:
            return True
        
        # Wildcard match (basic implementation)
        if subscriber_topic.endswith("*"):
            prefix = subscriber_topic[:-1]
            return message_topic.startswith(prefix)
        
        return False
    
    async def _publish_system_message(self, topic: str, message_type: MessageType, content: Dict[str, Any]):
        """Publish a system message."""
        await self.publish(topic, message_type, content, "system")
    
    def get_agents(self) -> Dict[str, AgentInfo]:
        """Get all registered agents."""
        return self.agents.copy()
    
    def get_online_agents(self) -> Dict[str, AgentInfo]:
        """Get all online agents."""
        return {
            agent_id: info for agent_id, info in self.agents.items()
            if info.is_online()
        }
    
    def get_message_history(self, topic: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get message history.
        
        Args:
            topic: Optional topic filter
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = self.message_history
        
        if topic:
            messages = [msg for msg in messages if self._topic_matches(msg.topic, topic)]
        
        # Sort by timestamp (newest first) and limit
        messages = sorted(messages, key=lambda m: m.timestamp, reverse=True)[:limit]
        
        return [msg.to_dict() for msg in messages]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_messages": self.message_count,
            "messages_in_history": len(self.message_history),
            "registered_agents": len(self.agents),
            "online_agents": len(self.get_online_agents()),
            "active_subscriptions": sum(len(subs) for subs in self.subscribers.values()),
            "messages_per_second": self.message_count / uptime if uptime > 0 else 0
        }

# Example usage and testing
async def main():
    """Example usage of the AgentMessageBus."""
    bus = AgentMessageBus()
    await bus.start()
    
    # Example subscriber
    async def example_callback(topic, message):
        print(f"Received on {topic}: {message}")
    
    # Register agents
    await bus.register_agent("agent1", "coder", ["python", "javascript"])
    await bus.register_agent("agent2", "tester", ["pytest", "selenium"])
    
    # Subscribe to messages
    await bus.subscribe("code.changes", example_callback, "agent1")
    await bus.subscribe("test.results", example_callback, "agent2")
    
    # Publish some messages
    await bus.publish("code.changes", MessageType.CODE_CHANGE, 
                     {"file": "main.py", "changes": "Added new function"}, "agent1")
    
    await bus.publish("test.results", MessageType.TEST_RESULT,
                     {"status": "passed", "tests_run": 15}, "agent2")
    
    # Send heartbeats
    await bus.heartbeat("agent1", "busy", {"current_task": "implementing feature X"})
    await bus.heartbeat("agent2", "active")
    
    # Wait a bit for message processing
    await asyncio.sleep(1)
    
    # Get stats
    stats = bus.get_stats()
    print("Bus stats:", stats)
    
    # Get message history
    history = bus.get_message_history(limit=10)
    print(f"Message history ({len(history)} messages):")
    for msg in history:
        print(f"  {msg['timestamp']}: {msg['topic']} from {msg['sender_id']}")
    
    await bus.stop()

if __name__ == "__main__":
    asyncio.run(main())