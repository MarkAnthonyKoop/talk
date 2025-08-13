"""
Agent Registry and Discovery System

Provides centralized registration, discovery, and management of agents
with capability matching and load balancing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Type

from agent.agent import Agent

log = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    REGISTERED = auto()
    INITIALIZING = auto()
    READY = auto()
    BUSY = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class AgentMetadata:
    """Metadata about a registered agent."""
    agent_id: str
    agent_class: str
    agent_type: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    supported_tasks: Set[str] = field(default_factory=set)
    input_formats: Set[str] = field(default_factory=set)
    output_formats: Set[str] = field(default_factory=set)
    
    # Performance
    max_concurrent: int = 1
    average_response_time: float = 0.0
    success_rate: float = 1.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    
    # Runtime
    state: AgentState = AgentState.REGISTERED
    instance: Optional[Agent] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    current_load: int = 0
    
    # Health
    health_score: float = 1.0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_class": self.agent_class,
            "agent_type": self.agent_type,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "state": self.state.name,
            "capabilities": list(self.capabilities),
            "supported_tasks": list(self.supported_tasks),
            "performance": {
                "average_response_time": self.average_response_time,
                "success_rate": self.success_rate,
                "total_completed": self.total_tasks_completed,
                "total_failed": self.total_tasks_failed
            },
            "health": {
                "score": self.health_score,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error
            }
        }


@dataclass
class CapabilityRequirement:
    """Defines requirements for agent capabilities."""
    required: Set[str] = field(default_factory=set)
    preferred: Set[str] = field(default_factory=set)
    excluded: Set[str] = field(default_factory=set)
    
    def matches(self, capabilities: Set[str]) -> Tuple[bool, float]:
        """
        Check if capabilities match requirements.
        Returns (matches, score) where score is 0-1.
        """
        # Check required capabilities
        if not self.required.issubset(capabilities):
            return False, 0.0
        
        # Check excluded capabilities
        if self.excluded.intersection(capabilities):
            return False, 0.0
        
        # Calculate preference score
        if self.preferred:
            matched_preferred = len(self.preferred.intersection(capabilities))
            score = matched_preferred / len(self.preferred)
        else:
            score = 1.0
        
        return True, score


class AgentRegistry:
    """
    Central registry for agent discovery and management.
    
    Features:
    - Agent registration and discovery
    - Capability-based matching
    - Load balancing
    - Health monitoring
    - Dynamic agent creation
    """
    
    def __init__(self, enable_health_check: bool = True, health_check_interval: float = 30.0):
        """
        Initialize the agent registry.
        
        Args:
            enable_health_check: Enable periodic health checking
            health_check_interval: Seconds between health checks
        """
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_classes: Dict[str, Type[Agent]] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> agent_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> agent_ids
        self.task_index: Dict[str, Set[str]] = defaultdict(set)  # task -> agent_ids
        
        self._lock = threading.RLock()
        self.enable_health_check = enable_health_check
        self.health_check_interval = health_check_interval
        
        # Statistics
        self.stats = {
            "total_registered": 0,
            "total_queries": 0,
            "successful_matches": 0,
            "failed_matches": 0
        }
        
        # Start health check thread
        if enable_health_check:
            self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.health_thread.start()
        
        log.info("Agent Registry initialized")
    
    def register_agent_class(self, agent_class: Type[Agent], metadata: Dict[str, Any]) -> str:
        """Register an agent class for dynamic instantiation."""
        class_name = agent_class.__name__
        self.agent_classes[class_name] = agent_class
        log.info(f"Registered agent class: {class_name}")
        return class_name
    
    def register_agent(self, agent: Agent, metadata: AgentMetadata) -> str:
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance
            metadata: Agent metadata
            
        Returns:
            Agent ID
        """
        with self._lock:
            agent_id = metadata.agent_id
            
            # Store agent
            metadata.instance = agent
            metadata.state = AgentState.READY
            self.agents[agent_id] = metadata
            
            # Update indices
            for capability in metadata.capabilities:
                self.capability_index[capability].add(agent_id)
            
            self.type_index[metadata.agent_type].add(agent_id)
            
            for task in metadata.supported_tasks:
                self.task_index[task].add(agent_id)
            
            self.stats["total_registered"] += 1
            
            log.info(f"Registered agent {agent_id} ({metadata.name}) with capabilities {metadata.capabilities}")
            
            return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent to unregister
            
        Returns:
            Success status
        """
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            metadata = self.agents[agent_id]
            
            # Remove from indices
            for capability in metadata.capabilities:
                self.capability_index[capability].discard(agent_id)
            
            self.type_index[metadata.agent_type].discard(agent_id)
            
            for task in metadata.supported_tasks:
                self.task_index[task].discard(agent_id)
            
            # Mark as shutdown
            metadata.state = AgentState.SHUTDOWN
            
            # Remove from registry
            del self.agents[agent_id]
            
            log.info(f"Unregistered agent {agent_id}")
            
            return True
    
    def find_agents_by_capability(self, 
                                  requirements: CapabilityRequirement,
                                  max_results: int = 10,
                                  only_available: bool = True) -> List[Tuple[str, float]]:
        """
        Find agents matching capability requirements.
        
        Args:
            requirements: Capability requirements
            max_results: Maximum agents to return
            only_available: Only return available agents
            
        Returns:
            List of (agent_id, match_score) tuples
        """
        with self._lock:
            self.stats["total_queries"] += 1
            matches = []
            
            # Get candidates from capability index
            if requirements.required:
                # Start with agents having the first required capability
                candidates = set(self.capability_index.get(next(iter(requirements.required)), set()))
                
                # Intersect with agents having all required capabilities
                for cap in requirements.required:
                    candidates &= self.capability_index.get(cap, set())
            else:
                # No required capabilities, consider all agents
                candidates = set(self.agents.keys())
            
            # Score and filter candidates
            for agent_id in candidates:
                metadata = self.agents[agent_id]
                
                # Check availability
                if only_available and metadata.state != AgentState.READY:
                    continue
                
                # Check capability match
                is_match, score = requirements.matches(metadata.capabilities)
                
                if is_match:
                    # Adjust score based on agent health and performance
                    adjusted_score = score * metadata.health_score * metadata.success_rate
                    
                    # Consider current load
                    if metadata.max_concurrent > 0:
                        load_factor = 1.0 - (metadata.current_load / metadata.max_concurrent)
                        adjusted_score *= load_factor
                    
                    matches.append((agent_id, adjusted_score))
            
            # Sort by score and return top results
            matches.sort(key=lambda x: x[1], reverse=True)
            
            if matches:
                self.stats["successful_matches"] += 1
            else:
                self.stats["failed_matches"] += 1
            
            return matches[:max_results]
    
    def find_agents_by_type(self, agent_type: str, only_available: bool = True) -> List[str]:
        """Find agents by type."""
        with self._lock:
            agents = list(self.type_index.get(agent_type, set()))
            
            if only_available:
                agents = [
                    aid for aid in agents
                    if self.agents[aid].state == AgentState.READY
                ]
            
            return agents
    
    def find_agents_for_task(self, task_type: str, only_available: bool = True) -> List[str]:
        """Find agents that support a specific task type."""
        with self._lock:
            agents = list(self.task_index.get(task_type, set()))
            
            if only_available:
                agents = [
                    aid for aid in agents
                    if self.agents[aid].state == AgentState.READY
                ]
            
            return agents
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent instance by ID."""
        with self._lock:
            metadata = self.agents.get(agent_id)
            return metadata.instance if metadata else None
    
    def get_agent_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID."""
        with self._lock:
            return self.agents.get(agent_id)
    
    def update_agent_state(self, agent_id: str, state: AgentState):
        """Update agent state."""
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].state = state
                self.agents[agent_id].last_active = datetime.now()
    
    def update_agent_performance(self, 
                                 agent_id: str,
                                 task_success: bool,
                                 response_time: float):
        """Update agent performance metrics."""
        with self._lock:
            if agent_id not in self.agents:
                return
            
            metadata = self.agents[agent_id]
            
            # Update task counts
            if task_success:
                metadata.total_tasks_completed += 1
                metadata.consecutive_failures = 0
            else:
                metadata.total_tasks_failed += 1
                metadata.consecutive_failures += 1
            
            # Update success rate
            total_tasks = metadata.total_tasks_completed + metadata.total_tasks_failed
            if total_tasks > 0:
                metadata.success_rate = metadata.total_tasks_completed / total_tasks
            
            # Update average response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            metadata.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * metadata.average_response_time
            )
            
            # Update health score
            self._update_health_score(metadata)
    
    def _update_health_score(self, metadata: AgentMetadata):
        """Calculate and update agent health score."""
        # Base score from success rate
        score = metadata.success_rate
        
        # Penalty for consecutive failures
        if metadata.consecutive_failures > 0:
            penalty = min(0.5, metadata.consecutive_failures * 0.1)
            score *= (1 - penalty)
        
        # Penalty for slow response
        if metadata.average_response_time > 10.0:  # 10 seconds threshold
            time_penalty = min(0.3, (metadata.average_response_time - 10) / 100)
            score *= (1 - time_penalty)
        
        metadata.health_score = max(0.1, min(1.0, score))
    
    def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                time.sleep(self.health_check_interval)
                self._perform_health_checks()
            except Exception as e:
                log.error(f"Health check error: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on all agents."""
        with self._lock:
            now = datetime.now()
            
            for agent_id, metadata in self.agents.items():
                # Check for stale agents
                if metadata.state == AgentState.READY:
                    time_since_active = (now - metadata.last_active).total_seconds()
                    
                    # Mark as paused if inactive for too long
                    if time_since_active > 300:  # 5 minutes
                        metadata.state = AgentState.PAUSED
                        log.warning(f"Agent {agent_id} marked as paused due to inactivity")
                
                # Check for error recovery
                elif metadata.state == AgentState.ERROR:
                    # Try to recover after some time
                    if metadata.consecutive_failures < 3:
                        metadata.state = AgentState.READY
                        log.info(f"Agent {agent_id} recovered from error state")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get current registry status."""
        with self._lock:
            type_counts = {t: len(agents) for t, agents in self.type_index.items()}
            
            state_counts = defaultdict(int)
            for metadata in self.agents.values():
                state_counts[metadata.state.name] += 1
            
            return {
                "total_agents": len(self.agents),
                "agents_by_state": dict(state_counts),
                "agents_by_type": type_counts,
                "total_capabilities": len(self.capability_index),
                "statistics": self.stats,
                "health": {
                    "average_health_score": sum(m.health_score for m in self.agents.values()) / max(1, len(self.agents)),
                    "agents_with_errors": sum(1 for m in self.agents.values() if m.state == AgentState.ERROR)
                }
            }
    
    def export_registry(self, filepath: Path):
        """Export registry to JSON file."""
        with self._lock:
            data = {
                "timestamp": datetime.now().isoformat(),
                "agents": {
                    agent_id: metadata.to_dict()
                    for agent_id, metadata in self.agents.items()
                },
                "statistics": self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            log.info(f"Registry exported to {filepath}")


# Singleton registry instance
_global_registry = None


def get_global_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


# Export main components
__all__ = [
    'AgentRegistry',
    'AgentMetadata',
    'AgentState',
    'CapabilityRequirement',
    'get_global_registry'
]