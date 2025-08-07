"""
Agent registry for managing agent types and instances.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Set
from enum import Enum

log = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class AgentMetadata:
    """Metadata for registered agent types"""
    agent_type: str
    agent_class: Type
    capabilities: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    max_instances: int = 10
    min_instances: int = 0
    auto_scale: bool = True
    version: str = "1.0.0"
    description: str = ""
    registered_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentInstance:
    """Information about an active agent instance"""
    id: str
    type: str
    instance: Any
    state: AgentState
    capabilities: List[str]
    config: Dict[str, Any]
    created_at: datetime
    last_active: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Central registry for agent types and instances.
    
    Manages:
    - Agent type registration
    - Instance tracking
    - Capability mapping
    - Resource allocation
    """
    
    def __init__(self):
        self.registered_types: Dict[str, AgentMetadata] = {}
        self.active_agents: Dict[str, AgentInstance] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self.type_instances: Dict[str, Set[str]] = {}  # agent_type -> agent_ids
        self.lock = threading.RLock()
        
        # Statistics
        self.total_agents_created = 0
        self.total_agents_terminated = 0
        
        log.info("Agent registry initialized")
    
    def register(self, 
                 agent_type: str, 
                 agent_class: Type,
                 capabilities: List[str] = None,
                 **metadata) -> bool:
        """Register a new agent type"""
        with self.lock:
            if agent_type in self.registered_types:
                log.warning(f"Agent type {agent_type} already registered")
                return False
            
            agent_metadata = AgentMetadata(
                agent_type=agent_type,
                agent_class=agent_class,
                capabilities=capabilities or [],
                **metadata
            )
            
            self.registered_types[agent_type] = agent_metadata
            self.type_instances[agent_type] = set()
            
            log.info(f"Registered agent type: {agent_type} with capabilities: {capabilities}")
            return True
    
    def unregister(self, agent_type: str) -> bool:
        """Unregister an agent type"""
        with self.lock:
            if agent_type not in self.registered_types:
                return False
            
            # Check if there are active instances
            if self.type_instances.get(agent_type):
                log.warning(f"Cannot unregister {agent_type}: active instances exist")
                return False
            
            del self.registered_types[agent_type]
            del self.type_instances[agent_type]
            
            log.info(f"Unregistered agent type: {agent_type}")
            return True
    
    def create_instance(self, agent_type: str, config: Dict[str, Any] = None) -> Optional[str]:
        """Create a new agent instance"""
        with self.lock:
            if agent_type not in self.registered_types:
                log.error(f"Agent type {agent_type} not registered")
                return None
            
            metadata = self.registered_types[agent_type]
            
            # Check instance limits
            current_instances = len(self.type_instances[agent_type])
            if current_instances >= metadata.max_instances:
                log.warning(f"Maximum instances ({metadata.max_instances}) reached for {agent_type}")
                return None
            
            # Generate unique ID
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            
            try:
                # Create agent instance
                agent_class = metadata.agent_class
                agent_config = config or {}
                
                # Instantiate the agent
                if hasattr(agent_class, '__init__'):
                    agent_instance = agent_class(**agent_config)
                else:
                    agent_instance = agent_class()
                
                # Create instance record
                instance = AgentInstance(
                    id=agent_id,
                    type=agent_type,
                    instance=agent_instance,
                    state=AgentState.INITIALIZING,
                    capabilities=metadata.capabilities.copy(),
                    config=agent_config,
                    created_at=datetime.now(),
                    last_active=datetime.now()
                )
                
                # Register instance
                self.active_agents[agent_id] = instance
                self.type_instances[agent_type].add(agent_id)
                
                # Update capability index
                for capability in metadata.capabilities:
                    if capability not in self.capability_index:
                        self.capability_index[capability] = set()
                    self.capability_index[capability].add(agent_id)
                
                # Update statistics
                self.total_agents_created += 1
                
                # Set state to ready
                instance.state = AgentState.READY
                
                log.info(f"Created agent instance: {agent_id}")
                return agent_id
                
            except Exception as e:
                log.error(f"Failed to create agent instance: {e}")
                return None
    
    def destroy_instance(self, agent_id: str) -> bool:
        """Destroy an agent instance"""
        with self.lock:
            if agent_id not in self.active_agents:
                return False
            
            instance = self.active_agents[agent_id]
            instance.state = AgentState.TERMINATING
            
            # Remove from indices
            self.type_instances[instance.type].discard(agent_id)
            
            # Remove from capability index
            for capability in instance.capabilities:
                if capability in self.capability_index:
                    self.capability_index[capability].discard(agent_id)
                    if not self.capability_index[capability]:
                        del self.capability_index[capability]
            
            # Clean up the actual instance
            if hasattr(instance.instance, 'cleanup'):
                try:
                    instance.instance.cleanup()
                except Exception as e:
                    log.error(f"Error during agent cleanup: {e}")
            
            # Remove from registry
            del self.active_agents[agent_id]
            
            # Update statistics
            self.total_agents_terminated += 1
            
            log.info(f"Destroyed agent instance: {agent_id}")
            return True
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent instance information"""
        with self.lock:
            if agent_id not in self.active_agents:
                return None
            
            instance = self.active_agents[agent_id]
            return {
                'id': instance.id,
                'type': instance.type,
                'state': instance.state.value,
                'capabilities': instance.capabilities,
                'config': instance.config,
                'instance': instance.instance,
                'created_at': instance.created_at,
                'last_active': instance.last_active,
                'tasks_completed': instance.tasks_completed,
                'tasks_failed': instance.tasks_failed,
                'metadata': instance.metadata
            }
    
    def update_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """Update agent state"""
        with self.lock:
            if agent_id not in self.active_agents:
                return False
            
            self.active_agents[agent_id].state = state
            self.active_agents[agent_id].last_active = datetime.now()
            return True
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find all agents with a specific capability"""
        with self.lock:
            return list(self.capability_index.get(capability, set()))
    
    def find_agents_by_type(self, agent_type: str) -> List[str]:
        """Find all agents of a specific type"""
        with self.lock:
            return list(self.type_instances.get(agent_type, set()))
    
    def find_available_agents(self, capabilities: List[str] = None) -> List[str]:
        """Find available agents with optional capability filter"""
        with self.lock:
            available = []
            
            for agent_id, instance in self.active_agents.items():
                if instance.state != AgentState.READY:
                    continue
                
                if capabilities:
                    # Check if agent has all required capabilities
                    if not all(cap in instance.capabilities for cap in capabilities):
                        continue
                
                available.append(agent_id)
            
            return available
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self.lock:
            type_stats = {}
            for agent_type in self.registered_types:
                instances = self.type_instances[agent_type]
                type_stats[agent_type] = {
                    'count': len(instances),
                    'instances': list(instances)
                }
            
            state_distribution = {}
            for instance in self.active_agents.values():
                state = instance.state.value
                state_distribution[state] = state_distribution.get(state, 0) + 1
            
            return {
                'registered_types': list(self.registered_types.keys()),
                'total_active_agents': len(self.active_agents),
                'total_agents_created': self.total_agents_created,
                'total_agents_terminated': self.total_agents_terminated,
                'type_distribution': type_stats,
                'state_distribution': state_distribution,
                'capabilities': list(self.capability_index.keys())
            }
    
    def validate_resources(self, agent_type: str) -> bool:
        """Validate if resources are available for agent type"""
        with self.lock:
            if agent_type not in self.registered_types:
                return False
            
            metadata = self.registered_types[agent_type]
            required_resources = metadata.required_resources
            
            # This would integrate with actual resource management
            # For now, we'll do basic checks
            import psutil
            
            if 'min_memory_mb' in required_resources:
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                if available_memory < required_resources['min_memory_mb']:
                    log.warning(f"Insufficient memory for {agent_type}")
                    return False
            
            if 'min_cpu_cores' in required_resources:
                available_cores = psutil.cpu_count()
                if available_cores < required_resources['min_cpu_cores']:
                    log.warning(f"Insufficient CPU cores for {agent_type}")
                    return False
            
            return True
    
    def bulk_register(self, agents: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Bulk register multiple agent types"""
        results = {}
        for agent_type, agent_info in agents.items():
            agent_class = agent_info.pop('agent_class')
            results[agent_type] = self.register(agent_type, agent_class, **agent_info)
        return results
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry configuration"""
        with self.lock:
            return {
                'registered_types': {
                    agent_type: {
                        'capabilities': metadata.capabilities,
                        'max_instances': metadata.max_instances,
                        'min_instances': metadata.min_instances,
                        'auto_scale': metadata.auto_scale,
                        'version': metadata.version,
                        'description': metadata.description
                    }
                    for agent_type, metadata in self.registered_types.items()
                },
                'statistics': self.get_statistics()
            }