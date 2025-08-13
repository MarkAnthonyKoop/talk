"""
Inter-Agent Communication System

Provides message passing, event bus, and coordination mechanisms
for agents to communicate and collaborate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Union

log = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages agents can exchange."""
    REQUEST = auto()
    RESPONSE = auto()
    NOTIFICATION = auto()
    BROADCAST = auto()
    QUERY = auto()
    COMMAND = auto()
    DATA = auto()
    ERROR = auto()
    HEARTBEAT = auto()


class MessagePriority(Enum):
    """Message priority levels."""
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Message:
    """Inter-agent message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    sender: str = ""
    recipient: Optional[str] = None  # None for broadcasts
    topic: Optional[str] = None
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[float] = None  # Time to live in seconds
    requires_ack: bool = False
    correlation_id: Optional[str] = None  # For request-response pairing
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl:
            age = (datetime.now() - self.timestamp).total_seconds()
            return age > self.ttl
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.name,
            "priority": self.priority.name,
            "sender": self.sender,
            "recipient": self.recipient,
            "topic": self.topic,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "requires_ack": self.requires_ack,
            "correlation_id": self.correlation_id
        }


@dataclass
class MessageAck:
    """Message acknowledgment."""
    message_id: str
    acknowledged_by: str
    acknowledged_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None


class MessageBus:
    """
    Central message bus for inter-agent communication.
    
    Features:
    - Point-to-point messaging
    - Broadcast messaging
    - Topic-based pub/sub
    - Message routing
    - Delivery guarantees
    - Message persistence
    """
    
    def __init__(self,
                 enable_persistence: bool = False,
                 max_queue_size: int = 1000,
                 delivery_timeout: float = 30.0):
        """
        Initialize message bus.
        
        Args:
            enable_persistence: Enable message persistence
            max_queue_size: Maximum messages per agent queue
            delivery_timeout: Timeout for message delivery
        """
        self.enable_persistence = enable_persistence
        self.max_queue_size = max_queue_size
        self.delivery_timeout = delivery_timeout
        
        # Agent message queues
        self.agent_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_queue_size))
        
        # Topic subscriptions
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Message tracking
        self.pending_acks: Dict[str, Message] = {}
        self.message_history = deque(maxlen=10000)
        
        # Request-response tracking
        self.pending_responses: Dict[str, threading.Event] = {}
        self.response_data: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "broadcasts_sent": 0,
            "acknowledgments_received": 0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Start message processor
        self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processor_thread.start()
        
        log.info("Message bus initialized")
    
    def send(self, message: Message) -> bool:
        """
        Send a message.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        with self._lock:
            # Validate message
            if message.is_expired():
                log.warning(f"Message {message.id} has expired")
                return False
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            # Record in history
            self.message_history.append(message)
            
            # Route message based on type
            if message.type == MessageType.BROADCAST:
                return self._broadcast(message)
            elif message.recipient:
                return self._send_to_agent(message)
            elif message.topic:
                return self._publish_to_topic(message)
            else:
                log.error(f"Message {message.id} has no recipient or topic")
                return False
    
    def _send_to_agent(self, message: Message) -> bool:
        """Send message to specific agent."""
        recipient = message.recipient
        
        if recipient not in self.agent_queues:
            log.warning(f"Agent {recipient} not registered")
            self.stats["messages_failed"] += 1
            return False
        
        # Add to agent queue
        self.agent_queues[recipient].append(message)
        
        # Track acknowledgment if required
        if message.requires_ack:
            self.pending_acks[message.id] = message
        
        self.stats["messages_delivered"] += 1
        log.debug(f"Message {message.id} sent to {recipient}")
        
        return True
    
    def _broadcast(self, message: Message) -> bool:
        """Broadcast message to all agents."""
        broadcast_count = 0
        
        for agent_id in list(self.agent_queues.keys()):
            if agent_id != message.sender:
                self.agent_queues[agent_id].append(message)
                broadcast_count += 1
        
        self.stats["broadcasts_sent"] += 1
        log.debug(f"Broadcast message {message.id} sent to {broadcast_count} agents")
        
        return broadcast_count > 0
    
    def _publish_to_topic(self, message: Message) -> bool:
        """Publish message to topic subscribers."""
        topic = message.topic
        subscribers = self.topic_subscribers.get(topic, set())
        
        if not subscribers:
            log.warning(f"No subscribers for topic {topic}")
            return False
        
        for subscriber in subscribers:
            if subscriber != message.sender:
                self.agent_queues[subscriber].append(message)
        
        log.debug(f"Message {message.id} published to topic {topic} ({len(subscribers)} subscribers)")
        
        return True
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Receive next message for an agent.
        
        Args:
            agent_id: Agent ID
            timeout: Optional timeout in seconds
            
        Returns:
            Next message or None
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if agent_id in self.agent_queues and self.agent_queues[agent_id]:
                    message = self.agent_queues[agent_id].popleft()
                    
                    # Check if message is still valid
                    if not message.is_expired():
                        return message
            
            # Check timeout
            if timeout:
                if time.time() - start_time > timeout:
                    return None
            else:
                return None
            
            time.sleep(0.01)
    
    def receive_all(self, agent_id: str) -> List[Message]:
        """
        Receive all pending messages for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of messages
        """
        with self._lock:
            if agent_id not in self.agent_queues:
                return []
            
            messages = []
            while self.agent_queues[agent_id]:
                message = self.agent_queues[agent_id].popleft()
                if not message.is_expired():
                    messages.append(message)
            
            return messages
    
    def acknowledge(self, ack: MessageAck):
        """
        Acknowledge message receipt.
        
        Args:
            ack: Message acknowledgment
        """
        with self._lock:
            if ack.message_id in self.pending_acks:
                del self.pending_acks[ack.message_id]
                self.stats["acknowledgments_received"] += 1
                log.debug(f"Message {ack.message_id} acknowledged by {ack.acknowledged_by}")
    
    def request(self, request_msg: Message, timeout: float = 30.0) -> Optional[Message]:
        """
        Send request and wait for response.
        
        Args:
            request_msg: Request message
            timeout: Response timeout
            
        Returns:
            Response message or None
        """
        # Set correlation ID
        request_msg.correlation_id = request_msg.id
        
        # Create response event
        response_event = threading.Event()
        
        with self._lock:
            self.pending_responses[request_msg.id] = response_event
        
        # Send request
        if not self.send(request_msg):
            with self._lock:
                del self.pending_responses[request_msg.id]
            return None
        
        # Wait for response
        if response_event.wait(timeout):
            with self._lock:
                response = self.response_data.get(request_msg.id)
                # Clean up
                del self.pending_responses[request_msg.id]
                self.response_data.pop(request_msg.id, None)
                return response
        else:
            with self._lock:
                del self.pending_responses[request_msg.id]
            log.warning(f"Request {request_msg.id} timed out")
            return None
    
    def respond(self, response_msg: Message):
        """
        Send response to a request.
        
        Args:
            response_msg: Response message
        """
        if not response_msg.correlation_id:
            log.error("Response message missing correlation_id")
            return
        
        with self._lock:
            if response_msg.correlation_id in self.pending_responses:
                # Store response and signal event
                self.response_data[response_msg.correlation_id] = response_msg
                self.pending_responses[response_msg.correlation_id].set()
            else:
                # Regular send if no pending request
                self.send(response_msg)
    
    def subscribe(self, agent_id: str, topic: str):
        """
        Subscribe agent to topic.
        
        Args:
            agent_id: Agent ID
            topic: Topic to subscribe to
        """
        with self._lock:
            self.topic_subscribers[topic].add(agent_id)
            log.info(f"Agent {agent_id} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """
        Unsubscribe agent from topic.
        
        Args:
            agent_id: Agent ID
            topic: Topic to unsubscribe from
        """
        with self._lock:
            self.topic_subscribers[topic].discard(agent_id)
            log.info(f"Agent {agent_id} unsubscribed from topic {topic}")
    
    def register_agent(self, agent_id: str):
        """
        Register an agent with the message bus.
        
        Args:
            agent_id: Agent ID
        """
        with self._lock:
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = deque(maxlen=self.max_queue_size)
                log.info(f"Agent {agent_id} registered with message bus")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the message bus.
        
        Args:
            agent_id: Agent ID
        """
        with self._lock:
            # Remove from queues
            self.agent_queues.pop(agent_id, None)
            
            # Remove from all topic subscriptions
            for subscribers in self.topic_subscribers.values():
                subscribers.discard(agent_id)
            
            log.info(f"Agent {agent_id} unregistered from message bus")
    
    def _process_messages(self):
        """Background message processor."""
        while not self._shutdown:
            try:
                # Check for expired acknowledgments
                self._check_pending_acks()
                
                time.sleep(1.0)
                
            except Exception as e:
                log.error(f"Message processor error: {e}")
    
    def _check_pending_acks(self):
        """Check for messages requiring acknowledgment."""
        with self._lock:
            expired = []
            
            for msg_id, message in self.pending_acks.items():
                age = (datetime.now() - message.timestamp).total_seconds()
                
                if age > self.delivery_timeout:
                    expired.append(msg_id)
                    log.warning(f"Message {msg_id} not acknowledged within timeout")
            
            for msg_id in expired:
                del self.pending_acks[msg_id]
                self.stats["messages_failed"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get message bus status."""
        with self._lock:
            return {
                "registered_agents": len(self.agent_queues),
                "total_topics": len(self.topic_subscribers),
                "pending_acks": len(self.pending_acks),
                "statistics": dict(self.stats),
                "queue_sizes": {
                    agent_id: len(queue)
                    for agent_id, queue in self.agent_queues.items()
                }
            }
    
    def shutdown(self):
        """Shutdown message bus."""
        log.info("Shutting down message bus...")
        self._shutdown = True
        self.processor_thread.join(timeout=5)
        log.info("Message bus shutdown complete")


class Event:
    """Event for the event bus."""
    
    def __init__(self,
                 name: str,
                 data: Any = None,
                 source: Optional[str] = None):
        """
        Initialize event.
        
        Args:
            name: Event name
            data: Event data
            source: Event source
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.propagate = True  # Can be set to False to stop propagation


class EventBus:
    """
    Event-driven communication system.
    
    Features:
    - Event publishing and subscription
    - Event filtering
    - Async and sync handlers
    - Event history
    """
    
    def __init__(self, enable_async: bool = True):
        """
        Initialize event bus.
        
        Args:
            enable_async: Enable async event handling
        """
        self.enable_async = enable_async
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history = deque(maxlen=1000)
        self.filters: List[Callable[[Event], bool]] = []
        
        self._lock = threading.RLock()
        
        if enable_async:
            self.loop = asyncio.new_event_loop()
            self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.async_thread.start()
        
        log.info("Event bus initialized")
    
    def _run_async_loop(self):
        """Run async event loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def subscribe(self, event_name: str, handler: Callable, is_async: bool = False):
        """
        Subscribe to an event.
        
        Args:
            event_name: Event name or pattern (* for all)
            handler: Event handler function
            is_async: Whether handler is async
        """
        with self._lock:
            if is_async and self.enable_async:
                self.async_handlers[event_name].append(handler)
            else:
                self.handlers[event_name].append(handler)
            
            log.debug(f"Handler subscribed to event {event_name}")
    
    def unsubscribe(self, event_name: str, handler: Callable):
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Event name
            handler: Handler to remove
        """
        with self._lock:
            if handler in self.handlers.get(event_name, []):
                self.handlers[event_name].remove(handler)
            
            if handler in self.async_handlers.get(event_name, []):
                self.async_handlers[event_name].remove(handler)
    
    def publish(self, event: Event):
        """
        Publish an event.
        
        Args:
            event: Event to publish
        """
        # Apply filters
        for filter_fn in self.filters:
            if not filter_fn(event):
                log.debug(f"Event {event.name} filtered out")
                return
        
        # Record in history
        self.event_history.append(event)
        
        # Get handlers
        with self._lock:
            sync_handlers = (
                self.handlers.get(event.name, []) +
                self.handlers.get("*", [])  # Wildcard handlers
            )
            
            async_handlers = (
                self.async_handlers.get(event.name, []) +
                self.async_handlers.get("*", [])
            )
        
        # Execute sync handlers
        for handler in sync_handlers:
            try:
                handler(event)
                if not event.propagate:
                    break
            except Exception as e:
                log.error(f"Error in event handler: {e}")
        
        # Execute async handlers
        if async_handlers and self.enable_async:
            asyncio.run_coroutine_threadsafe(
                self._handle_async(event, async_handlers),
                self.loop
            )
    
    async def _handle_async(self, event: Event, handlers: List[Callable]):
        """Handle async event handlers."""
        for handler in handlers:
            try:
                await handler(event)
                if not event.propagate:
                    break
            except Exception as e:
                log.error(f"Error in async event handler: {e}")
    
    def add_filter(self, filter_fn: Callable[[Event], bool]):
        """
        Add event filter.
        
        Args:
            filter_fn: Filter function (returns True to allow event)
        """
        self.filters.append(filter_fn)
    
    def emit(self, event_name: str, data: Any = None, source: Optional[str] = None):
        """
        Convenience method to create and publish an event.
        
        Args:
            event_name: Event name
            data: Event data
            source: Event source
        """
        event = Event(event_name, data, source)
        self.publish(event)
    
    def get_history(self, event_name: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_name: Filter by event name
            limit: Maximum events to return
            
        Returns:
            List of events
        """
        events = list(self.event_history)
        
        if event_name:
            events = [e for e in events if e.name == event_name]
        
        return events[-limit:]
    
    def shutdown(self):
        """Shutdown event bus."""
        if self.enable_async:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.async_thread.join(timeout=5)
        
        log.info("Event bus shutdown complete")


# Singleton instances
_global_message_bus = None
_global_event_bus = None


def get_message_bus() -> MessageBus:
    """Get or create global message bus."""
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = MessageBus()
    return _global_message_bus


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


# Export main components
__all__ = [
    'MessageBus',
    'Message',
    'MessageType',
    'MessagePriority',
    'MessageAck',
    'EventBus',
    'Event',
    'get_message_bus',
    'get_event_bus'
]