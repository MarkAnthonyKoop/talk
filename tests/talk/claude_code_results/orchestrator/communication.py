"""
Message bus and communication system for agent orchestration.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from queue import Queue, PriorityQueue
import pickle

log = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system"""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    COMMAND = "command"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""  # Empty string for broadcast
    type: MessageType = MessageType.TASK
    priority: MessagePriority = MessagePriority.NORMAL
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # For response messages
    ttl: Optional[int] = None  # Time to live in seconds
    requires_ack: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'type': self.type.value,
            'priority': self.priority.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'reply_to': self.reply_to,
            'ttl': self.ttl,
            'requires_ack': self.requires_ack
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            sender=data.get('sender', ''),
            recipient=data.get('recipient', ''),
            type=MessageType(data.get('type', 'task')),
            priority=MessagePriority(data.get('priority', 3)),
            content=data.get('content'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            reply_to=data.get('reply_to'),
            ttl=data.get('ttl'),
            requires_ack=data.get('requires_ack', False)
        )


class MessageRouter:
    """Routes messages between agents based on various strategies"""
    
    def __init__(self):
        self.routing_table: Dict[str, str] = {}  # pattern -> agent_id
        self.topic_subscriptions: Dict[str, Set[str]] = {}  # topic -> subscribers
        self.agent_capabilities: Dict[str, Set[str]] = {}  # agent_id -> capabilities
        
    def register_route(self, pattern: str, agent_id: str):
        """Register a routing pattern"""
        self.routing_table[pattern] = agent_id
        
    def subscribe_to_topic(self, topic: str, agent_id: str):
        """Subscribe an agent to a topic"""
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = set()
        self.topic_subscriptions[topic].add(agent_id)
    
    def unsubscribe_from_topic(self, topic: str, agent_id: str):
        """Unsubscribe an agent from a topic"""
        if topic in self.topic_subscriptions:
            self.topic_subscriptions[topic].discard(agent_id)
    
    def register_capabilities(self, agent_id: str, capabilities: List[str]):
        """Register agent capabilities for routing"""
        self.agent_capabilities[agent_id] = set(capabilities)
    
    def route_message(self, message: Message) -> List[str]:
        """Determine recipients for a message"""
        recipients = []
        
        # Direct recipient
        if message.recipient:
            recipients.append(message.recipient)
        
        # Topic-based routing
        elif 'topic' in message.metadata:
            topic = message.metadata['topic']
            recipients.extend(self.topic_subscriptions.get(topic, []))
        
        # Capability-based routing
        elif 'required_capabilities' in message.metadata:
            required = set(message.metadata['required_capabilities'])
            for agent_id, capabilities in self.agent_capabilities.items():
                if required.issubset(capabilities):
                    recipients.append(agent_id)
        
        # Pattern-based routing
        else:
            for pattern, agent_id in self.routing_table.items():
                if self._matches_pattern(message, pattern):
                    recipients.append(agent_id)
        
        return list(set(recipients))  # Remove duplicates
    
    def _matches_pattern(self, message: Message, pattern: str) -> bool:
        """Check if message matches routing pattern"""
        # Simple pattern matching - could be extended
        if pattern == "*":
            return True
        if pattern.startswith("type:"):
            return message.type.value == pattern[5:]
        if pattern.startswith("sender:"):
            return message.sender == pattern[7:]
        return False


class MessageBus:
    """
    Central message bus for agent communication.
    
    Features:
    - Asynchronous message passing
    - Priority-based message handling
    - Message persistence and replay
    - Dead letter queue for failed messages
    - Request-response patterns
    - Pub-sub patterns
    """
    
    def __init__(self, persist_messages: bool = False):
        self.persist_messages = persist_messages
        
        # Message queues
        self.agent_queues: Dict[str, PriorityQueue] = {}
        self.broadcast_queue = Queue()
        self.dead_letter_queue = Queue()
        
        # Message tracking
        self.message_history: List[Message] = []
        self.pending_responses: Dict[str, threading.Event] = {}
        self.response_messages: Dict[str, Message] = {}
        
        # Routing
        self.router = MessageRouter()
        
        # Handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'messages_expired': 0
        }
        
        # Background processing
        self.shutdown_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()
        
        log.info("Message bus initialized")
    
    def register_agent(self, agent_id: str, capabilities: List[str] = None):
        """Register an agent with the message bus"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = PriorityQueue()
            
        if capabilities:
            self.router.register_capabilities(agent_id, capabilities)
        
        log.debug(f"Registered agent {agent_id} with message bus")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
        
        # Clean up subscriptions
        for topic in list(self.router.topic_subscriptions.keys()):
            self.router.unsubscribe_from_topic(topic, agent_id)
        
        # Clean up capabilities
        if agent_id in self.router.agent_capabilities:
            del self.router.agent_capabilities[agent_id]
        
        log.debug(f"Unregistered agent {agent_id} from message bus")
    
    def send(self, message: Message) -> bool:
        """Send a message"""
        try:
            # Validate message
            if not message.sender:
                log.warning("Message missing sender")
                return False
            
            # Add to history
            if self.persist_messages:
                self.message_history.append(message)
            
            # Route message
            if message.recipient:
                # Direct message
                if message.recipient in self.agent_queues:
                    self.agent_queues[message.recipient].put(
                        (message.priority.value, message.timestamp, message)
                    )
                else:
                    log.warning(f"Recipient {message.recipient} not found")
                    self.dead_letter_queue.put(message)
                    self.stats['messages_failed'] += 1
                    return False
            else:
                # Broadcast or routed message
                recipients = self.router.route_message(message)
                if not recipients:
                    # True broadcast to all
                    self.broadcast_queue.put(message)
                else:
                    # Routed to specific agents
                    for recipient in recipients:
                        if recipient in self.agent_queues:
                            self.agent_queues[recipient].put(
                                (message.priority.value, message.timestamp, message)
                            )
            
            self.stats['messages_sent'] += 1
            
            # Trigger handlers
            self._trigger_handlers(message)
            
            return True
            
        except Exception as e:
            log.error(f"Error sending message: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    def send_and_wait(self, message: Message, timeout: float = 30) -> Optional[Message]:
        """Send a message and wait for response"""
        response_event = threading.Event()
        self.pending_responses[message.id] = response_event
        
        # Send the message
        if not self.send(message):
            del self.pending_responses[message.id]
            return None
        
        # Wait for response
        if response_event.wait(timeout):
            response = self.response_messages.get(message.id)
            
            # Clean up
            del self.pending_responses[message.id]
            if message.id in self.response_messages:
                del self.response_messages[message.id]
            
            return response
        else:
            # Timeout
            del self.pending_responses[message.id]
            log.warning(f"Timeout waiting for response to message {message.id}")
            return None
    
    def reply(self, original_message: Message, content: Any) -> bool:
        """Send a reply to a message"""
        reply = Message(
            sender=original_message.recipient,
            recipient=original_message.sender,
            type=MessageType.RESPONSE,
            priority=original_message.priority,
            content=content,
            reply_to=original_message.id,
            metadata={'original_type': original_message.type.value}
        )
        
        return self.send(reply)
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive a message for an agent"""
        if agent_id not in self.agent_queues:
            return None
        
        try:
            if timeout:
                # Non-blocking with timeout
                import queue
                priority, timestamp, message = self.agent_queues[agent_id].get(timeout=timeout)
            else:
                # Blocking
                priority, timestamp, message = self.agent_queues[agent_id].get()
            
            # Check TTL
            if message.ttl:
                age = (datetime.now() - message.timestamp).total_seconds()
                if age > message.ttl:
                    self.stats['messages_expired'] += 1
                    log.debug(f"Message {message.id} expired (TTL: {message.ttl}s)")
                    return None
            
            self.stats['messages_delivered'] += 1
            
            # Send acknowledgment if required
            if message.requires_ack:
                self._send_ack(message)
            
            return message
            
        except:
            return None
    
    def broadcast(self, sender: str, content: Any, type: MessageType = MessageType.BROADCAST):
        """Broadcast a message to all agents"""
        message = Message(
            sender=sender,
            recipient="",  # Empty for broadcast
            type=type,
            content=content
        )
        
        return self.send(message)
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic"""
        self.router.subscribe_to_topic(topic, agent_id)
        log.debug(f"Agent {agent_id} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic"""
        self.router.unsubscribe_from_topic(topic, agent_id)
        log.debug(f"Agent {agent_id} unsubscribed from topic {topic}")
    
    def publish(self, sender: str, topic: str, content: Any):
        """Publish a message to a topic"""
        message = Message(
            sender=sender,
            type=MessageType.EVENT,
            content=content,
            metadata={'topic': topic}
        )
        
        return self.send(message)
    
    def register_handler(self, agent_id: str, handler: Callable):
        """Register a message handler for an agent"""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = []
        self.message_handlers[agent_id].append(handler)
    
    def register_global_handler(self, handler: Callable):
        """Register a global message handler"""
        self.global_handlers.append(handler)
    
    def _trigger_handlers(self, message: Message):
        """Trigger message handlers"""
        # Global handlers
        for handler in self.global_handlers:
            try:
                handler(message)
            except Exception as e:
                log.error(f"Error in global handler: {e}")
        
        # Agent-specific handlers
        if message.recipient in self.message_handlers:
            for handler in self.message_handlers[message.recipient]:
                try:
                    handler(message)
                except Exception as e:
                    log.error(f"Error in agent handler: {e}")
    
    def _send_ack(self, message: Message):
        """Send acknowledgment for a message"""
        ack = Message(
            sender="message_bus",
            recipient=message.sender,
            type=MessageType.RESPONSE,
            content={'ack': True, 'message_id': message.id},
            reply_to=message.id,
            priority=MessagePriority.HIGH
        )
        self.send(ack)
    
    def _process_messages(self):
        """Background message processing"""
        while not self.shutdown_event.is_set():
            try:
                # Process broadcast messages
                try:
                    message = self.broadcast_queue.get(timeout=0.1)
                    for agent_id in self.agent_queues:
                        self.agent_queues[agent_id].put(
                            (message.priority.value, message.timestamp, message)
                        )
                    self.stats['messages_delivered'] += len(self.agent_queues)
                except:
                    pass
                
                # Check for response messages
                for agent_id, queue in self.agent_queues.items():
                    try:
                        # Peek at messages without removing
                        if not queue.empty():
                            priority, timestamp, message = queue.queue[0]
                            
                            if message.type == MessageType.RESPONSE and message.reply_to:
                                if message.reply_to in self.pending_responses:
                                    # This is a waited response
                                    queue.get()  # Remove from queue
                                    self.response_messages[message.reply_to] = message
                                    self.pending_responses[message.reply_to].set()
                    except:
                        pass
                
                # Process dead letter queue periodically
                if self.dead_letter_queue.qsize() > 100:
                    log.warning(f"Dead letter queue size: {self.dead_letter_queue.qsize()}")
                
            except Exception as e:
                log.error(f"Error in message processing: {e}")
                time.sleep(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        queue_sizes = {
            agent_id: queue.qsize() 
            for agent_id, queue in self.agent_queues.items()
        }
        
        return {
            **self.stats,
            'registered_agents': len(self.agent_queues),
            'queue_sizes': queue_sizes,
            'dead_letter_queue_size': self.dead_letter_queue.qsize(),
            'broadcast_queue_size': self.broadcast_queue.qsize(),
            'pending_responses': len(self.pending_responses),
            'message_history_size': len(self.message_history)
        }
    
    def replay_messages(self, filter_func: Optional[Callable] = None) -> List[Message]:
        """Replay messages from history with optional filter"""
        if not self.persist_messages:
            log.warning("Message persistence not enabled")
            return []
        
        messages = self.message_history
        if filter_func:
            messages = [m for m in messages if filter_func(m)]
        
        return messages
    
    def clear_dead_letters(self) -> List[Message]:
        """Clear and return dead letter queue"""
        dead_letters = []
        while not self.dead_letter_queue.empty():
            dead_letters.append(self.dead_letter_queue.get())
        return dead_letters
    
    def shutdown(self):
        """Shutdown the message bus"""
        log.info("Shutting down message bus")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Save message history if persistence is enabled
        if self.persist_messages and self.message_history:
            try:
                with open(f"message_history_{int(time.time())}.pkl", 'wb') as f:
                    pickle.dump(self.message_history, f)
                log.info(f"Saved {len(self.message_history)} messages to history")
            except Exception as e:
                log.error(f"Failed to save message history: {e}")
        
        log.info("Message bus shutdown complete")