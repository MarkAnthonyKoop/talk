"""
Agent lifecycle management system.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from queue import Queue

from .registry import AgentRegistry, AgentState

log = logging.getLogger(__name__)


class LifecycleEvent(Enum):
    """Agent lifecycle events"""
    SPAWNED = "spawned"
    INITIALIZED = "initialized"
    STARTED = "started"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    HEALTH_CHECK = "health_check"
    RECONFIGURED = "reconfigured"
    TERMINATED = "terminated"
    CRASHED = "crashed"


@dataclass
class LifecyclePolicy:
    """Policy for agent lifecycle management"""
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_delay_seconds: int = 5
    health_check_interval_seconds: int = 30
    idle_timeout_seconds: int = 300
    graceful_shutdown_timeout: int = 30
    enable_suspension: bool = True
    enable_migration: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)


class AgentLifecycleManager:
    """
    Manages the complete lifecycle of agents.
    
    Features:
    - Agent spawning and initialization
    - Health monitoring
    - Automatic restart on failure
    - Resource management
    - Graceful shutdown
    - State transitions
    """
    
    def __init__(self, registry: AgentRegistry, policy: Optional[LifecyclePolicy] = None):
        self.registry = registry
        self.policy = policy or LifecyclePolicy()
        
        # Lifecycle tracking
        self.agent_lifecycles: Dict[str, Dict[str, Any]] = {}
        self.restart_counts: Dict[str, int] = {}
        self.health_status: Dict[str, bool] = {}
        
        # Event handling
        self.event_handlers: Dict[LifecycleEvent, List[Callable]] = {
            event: [] for event in LifecycleEvent
        }
        self.event_queue = Queue()
        
        # Background threads
        self.monitoring_thread = None
        self.event_thread = None
        self.shutdown_event = threading.Event()
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Start background processes
        self._start_background_processes()
        
        log.info("Agent lifecycle manager initialized")
    
    def _start_background_processes(self):
        """Start background monitoring and event processing"""
        # Health monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Event processing thread
        self.event_thread = threading.Thread(
            target=self._event_processing_loop,
            daemon=True
        )
        self.event_thread.start()
    
    def spawn_agent(self, agent_type: str, config: Dict[str, Any] = None) -> Optional[str]:
        """Spawn a new agent with full lifecycle management"""
        with self.lock:
            # Create agent instance
            agent_id = self.registry.create_instance(agent_type, config)
            if not agent_id:
                return None
            
            # Initialize lifecycle tracking
            self.agent_lifecycles[agent_id] = {
                'spawned_at': datetime.now(),
                'state_history': [(datetime.now(), AgentState.INITIALIZING)],
                'events': [(datetime.now(), LifecycleEvent.SPAWNED)],
                'health_checks': [],
                'config': config or {},
                'restart_count': 0
            }
            
            self.restart_counts[agent_id] = 0
            self.health_status[agent_id] = True
            
            # Trigger spawned event
            self._emit_event(LifecycleEvent.SPAWNED, agent_id)
            
            # Initialize the agent
            if self._initialize_agent(agent_id):
                self.registry.update_agent_state(agent_id, AgentState.READY)
                self._emit_event(LifecycleEvent.INITIALIZED, agent_id)
                
                # Start the agent
                if self._start_agent(agent_id):
                    self._emit_event(LifecycleEvent.STARTED, agent_id)
                    log.info(f"Agent {agent_id} successfully spawned and started")
                    return agent_id
            
            # Cleanup on failure
            self.registry.destroy_instance(agent_id)
            return None
    
    def _initialize_agent(self, agent_id: str) -> bool:
        """Initialize an agent"""
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return False
        
        agent = agent_info['instance']
        
        # Call agent's initialization method if it exists
        if hasattr(agent, 'initialize'):
            try:
                agent.initialize()
                return True
            except Exception as e:
                log.error(f"Failed to initialize agent {agent_id}: {e}")
                return False
        
        return True
    
    def _start_agent(self, agent_id: str) -> bool:
        """Start an agent"""
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return False
        
        agent = agent_info['instance']
        
        # Call agent's start method if it exists
        if hasattr(agent, 'start'):
            try:
                agent.start()
                return True
            except Exception as e:
                log.error(f"Failed to start agent {agent_id}: {e}")
                return False
        
        return True
    
    def terminate_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Terminate an agent"""
        with self.lock:
            if agent_id not in self.agent_lifecycles:
                return False
            
            # Update state
            self.registry.update_agent_state(agent_id, AgentState.TERMINATING)
            
            # Graceful shutdown if requested
            if graceful:
                if not self._graceful_shutdown(agent_id):
                    log.warning(f"Graceful shutdown failed for {agent_id}, forcing termination")
            
            # Clean up lifecycle tracking
            if agent_id in self.agent_lifecycles:
                del self.agent_lifecycles[agent_id]
            if agent_id in self.restart_counts:
                del self.restart_counts[agent_id]
            if agent_id in self.health_status:
                del self.health_status[agent_id]
            
            # Destroy the instance
            success = self.registry.destroy_instance(agent_id)
            
            if success:
                self._emit_event(LifecycleEvent.TERMINATED, agent_id)
                log.info(f"Agent {agent_id} terminated")
            
            return success
    
    def _graceful_shutdown(self, agent_id: str) -> bool:
        """Perform graceful shutdown of an agent"""
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return False
        
        agent = agent_info['instance']
        
        # Call agent's shutdown method if it exists
        if hasattr(agent, 'shutdown'):
            try:
                # Set timeout for graceful shutdown
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(agent.shutdown)
                    try:
                        future.result(timeout=self.policy.graceful_shutdown_timeout)
                        return True
                    except concurrent.futures.TimeoutError:
                        log.warning(f"Graceful shutdown timeout for {agent_id}")
                        return False
            except Exception as e:
                log.error(f"Error during graceful shutdown of {agent_id}: {e}")
                return False
        
        return True
    
    def suspend_agent(self, agent_id: str) -> bool:
        """Suspend an agent"""
        with self.lock:
            if not self.policy.enable_suspension:
                return False
            
            agent_info = self.registry.get_agent(agent_id)
            if not agent_info:
                return False
            
            if agent_info['state'] != AgentState.READY.value:
                return False
            
            # Update state
            self.registry.update_agent_state(agent_id, AgentState.SUSPENDED)
            
            # Call agent's suspend method if it exists
            agent = agent_info['instance']
            if hasattr(agent, 'suspend'):
                try:
                    agent.suspend()
                except Exception as e:
                    log.error(f"Error suspending agent {agent_id}: {e}")
            
            self._emit_event(LifecycleEvent.SUSPENDED, agent_id)
            log.info(f"Agent {agent_id} suspended")
            return True
    
    def resume_agent(self, agent_id: str) -> bool:
        """Resume a suspended agent"""
        with self.lock:
            agent_info = self.registry.get_agent(agent_id)
            if not agent_info:
                return False
            
            if agent_info['state'] != AgentState.SUSPENDED.value:
                return False
            
            # Call agent's resume method if it exists
            agent = agent_info['instance']
            if hasattr(agent, 'resume'):
                try:
                    agent.resume()
                except Exception as e:
                    log.error(f"Error resuming agent {agent_id}: {e}")
                    return False
            
            # Update state
            self.registry.update_agent_state(agent_id, AgentState.READY)
            
            self._emit_event(LifecycleEvent.RESUMED, agent_id)
            log.info(f"Agent {agent_id} resumed")
            return True
    
    def health_check(self, agent_id: str) -> bool:
        """Perform health check on an agent"""
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return False
        
        agent = agent_info['instance']
        is_healthy = True
        
        # Call agent's health check method if it exists
        if hasattr(agent, 'health_check'):
            try:
                is_healthy = agent.health_check()
            except Exception as e:
                log.error(f"Health check failed for {agent_id}: {e}")
                is_healthy = False
        
        # Update tracking
        with self.lock:
            self.health_status[agent_id] = is_healthy
            
            if agent_id in self.agent_lifecycles:
                self.agent_lifecycles[agent_id]['health_checks'].append(
                    (datetime.now(), is_healthy)
                )
        
        self._emit_event(LifecycleEvent.HEALTH_CHECK, agent_id, {'healthy': is_healthy})
        
        return is_healthy
    
    def reconfigure_agent(self, agent_id: str, new_config: Dict[str, Any]) -> bool:
        """Reconfigure an agent"""
        with self.lock:
            agent_info = self.registry.get_agent(agent_id)
            if not agent_info:
                return False
            
            agent = agent_info['instance']
            
            # Call agent's reconfigure method if it exists
            if hasattr(agent, 'reconfigure'):
                try:
                    agent.reconfigure(new_config)
                    
                    # Update tracking
                    if agent_id in self.agent_lifecycles:
                        self.agent_lifecycles[agent_id]['config'] = new_config
                    
                    self._emit_event(LifecycleEvent.RECONFIGURED, agent_id, new_config)
                    log.info(f"Agent {agent_id} reconfigured")
                    return True
                    
                except Exception as e:
                    log.error(f"Failed to reconfigure agent {agent_id}: {e}")
                    return False
            
            return False
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Get all active agents
                with self.lock:
                    agent_ids = list(self.agent_lifecycles.keys())
                
                # Perform health checks
                for agent_id in agent_ids:
                    if self.shutdown_event.is_set():
                        break
                    
                    # Check if agent exists
                    agent_info = self.registry.get_agent(agent_id)
                    if not agent_info:
                        continue
                    
                    # Skip suspended agents
                    if agent_info['state'] == AgentState.SUSPENDED.value:
                        continue
                    
                    # Perform health check
                    is_healthy = self.health_check(agent_id)
                    
                    # Handle unhealthy agents
                    if not is_healthy:
                        self._handle_unhealthy_agent(agent_id)
                    
                    # Check for idle timeout
                    if self.policy.idle_timeout_seconds > 0:
                        last_active = agent_info['last_active']
                        idle_time = (datetime.now() - last_active).total_seconds()
                        
                        if idle_time > self.policy.idle_timeout_seconds:
                            log.info(f"Agent {agent_id} idle for {idle_time}s, suspending")
                            self.suspend_agent(agent_id)
                
                # Sleep before next check
                time.sleep(self.policy.health_check_interval_seconds)
                
            except Exception as e:
                log.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)
    
    def _handle_unhealthy_agent(self, agent_id: str):
        """Handle an unhealthy agent"""
        with self.lock:
            if not self.policy.auto_restart_on_failure:
                return
            
            # Check restart count
            restart_count = self.restart_counts.get(agent_id, 0)
            if restart_count >= self.policy.max_restart_attempts:
                log.error(f"Agent {agent_id} exceeded max restart attempts, terminating")
                self.terminate_agent(agent_id, graceful=False)
                return
            
            # Attempt restart
            log.warning(f"Agent {agent_id} unhealthy, attempting restart ({restart_count + 1}/{self.policy.max_restart_attempts})")
            
            # Get agent info before termination
            agent_info = self.registry.get_agent(agent_id)
            if not agent_info:
                return
            
            agent_type = agent_info['type']
            config = agent_info['config']
            
            # Terminate the unhealthy agent
            self.terminate_agent(agent_id, graceful=False)
            
            # Wait before restart
            time.sleep(self.policy.restart_delay_seconds)
            
            # Spawn replacement
            new_agent_id = self.spawn_agent(agent_type, config)
            if new_agent_id:
                self.restart_counts[new_agent_id] = restart_count + 1
                log.info(f"Restarted agent as {new_agent_id}")
    
    def _event_processing_loop(self):
        """Process lifecycle events"""
        while not self.shutdown_event.is_set():
            try:
                # Get event from queue (with timeout to check shutdown)
                try:
                    event, agent_id, data = self.event_queue.get(timeout=1)
                except:
                    continue
                
                # Process event handlers
                handlers = self.event_handlers.get(event, [])
                for handler in handlers:
                    try:
                        handler(agent_id, event, data)
                    except Exception as e:
                        log.error(f"Error in event handler: {e}")
                
            except Exception as e:
                log.error(f"Error in event processing loop: {e}")
    
    def _emit_event(self, event: LifecycleEvent, agent_id: str, data: Any = None):
        """Emit a lifecycle event"""
        self.event_queue.put((event, agent_id, data))
        
        # Update lifecycle tracking
        with self.lock:
            if agent_id in self.agent_lifecycles:
                self.agent_lifecycles[agent_id]['events'].append(
                    (datetime.now(), event)
                )
    
    def register_event_handler(self, event: LifecycleEvent, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event].append(handler)
    
    def get_agent_lifecycle(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get complete lifecycle information for an agent"""
        with self.lock:
            return self.agent_lifecycles.get(agent_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lifecycle management statistics"""
        with self.lock:
            healthy_count = sum(1 for h in self.health_status.values() if h)
            unhealthy_count = len(self.health_status) - healthy_count
            
            return {
                'total_managed_agents': len(self.agent_lifecycles),
                'healthy_agents': healthy_count,
                'unhealthy_agents': unhealthy_count,
                'total_restarts': sum(self.restart_counts.values()),
                'policy': {
                    'auto_restart': self.policy.auto_restart_on_failure,
                    'max_restarts': self.policy.max_restart_attempts,
                    'health_check_interval': self.policy.health_check_interval_seconds,
                    'idle_timeout': self.policy.idle_timeout_seconds
                }
            }
    
    def shutdown(self):
        """Shutdown the lifecycle manager"""
        log.info("Shutting down lifecycle manager")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Terminate all agents
        with self.lock:
            agent_ids = list(self.agent_lifecycles.keys())
        
        for agent_id in agent_ids:
            self.terminate_agent(agent_id, graceful=True)
        
        # Wait for threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        if self.event_thread:
            self.event_thread.join(timeout=5)
        
        log.info("Lifecycle manager shutdown complete")