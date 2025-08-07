"""
Core orchestrator implementation with advanced agent management capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from queue import Queue, PriorityQueue
import threading

from .registry import AgentRegistry
from .dispatcher import TaskDispatcher
from .monitor import OrchestrationMonitor
from .lifecycle import AgentLifecycleManager
from .communication import MessageBus, Message
from .policies import LoadBalancingPolicy, FailoverPolicy

log = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Orchestration execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestrator"""
    max_agents: int = 50
    max_concurrent_tasks: int = 10
    enable_auto_scaling: bool = True
    enable_monitoring: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 60  # seconds
    task_timeout: int = 300  # seconds
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2,
        "max_backoff": 60
    })
    load_balancing_policy: str = "round_robin"
    failover_enabled: bool = True
    health_check_interval: int = 30  # seconds


class AgentOrchestrator:
    """
    Advanced orchestrator for managing multi-agent systems.
    
    Features:
    - Dynamic agent spawning and termination
    - Intelligent task distribution
    - Load balancing and failover
    - Real-time monitoring and metrics
    - Checkpoint and recovery
    - Hierarchical and swarm intelligence modes
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        self.orchestrator_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Core components
        self.registry = AgentRegistry()
        self.dispatcher = TaskDispatcher(self.config)
        self.monitor = OrchestrationMonitor() if self.config.enable_monitoring else None
        self.lifecycle_manager = AgentLifecycleManager(self.registry)
        self.message_bus = MessageBus()
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_concurrent_tasks // 2)
        
        # Task management
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: List[Any] = []
        self.failed_tasks: List[Any] = []
        
        # Agent pools
        self.agent_pools: Dict[str, List[Any]] = {}
        self.agent_load: Dict[str, int] = {}
        
        # Synchronization
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Background tasks
        self._start_background_tasks()
        
        log.info(f"Orchestrator {self.orchestrator_id} initialized with config: {self.config}")
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        if self.config.enable_monitoring:
            self.monitor.start_monitoring(self)
        
        if self.config.enable_checkpointing:
            self._start_checkpointing()
        
        if self.config.health_check_interval > 0:
            self._start_health_checking()
    
    def _start_checkpointing(self):
        """Start periodic checkpointing"""
        def checkpoint_loop():
            while not self.shutdown_event.is_set():
                time.sleep(self.config.checkpoint_interval)
                self.create_checkpoint()
        
        threading.Thread(target=checkpoint_loop, daemon=True).start()
    
    def _start_health_checking(self):
        """Start periodic health checks"""
        def health_check_loop():
            while not self.shutdown_event.is_set():
                time.sleep(self.config.health_check_interval)
                self.perform_health_check()
        
        threading.Thread(target=health_check_loop, daemon=True).start()
    
    def register_agent(self, agent_type: str, agent_class: type, capabilities: List[str] = None):
        """Register a new agent type"""
        return self.registry.register(agent_type, agent_class, capabilities)
    
    def spawn_agent(self, agent_type: str, config: Dict[str, Any] = None) -> str:
        """Dynamically spawn a new agent instance"""
        with self.lock:
            if len(self.registry.active_agents) >= self.config.max_agents:
                if self.config.enable_auto_scaling:
                    self._scale_down_least_used()
                else:
                    raise RuntimeError(f"Maximum agent limit ({self.config.max_agents}) reached")
            
            agent_id = self.lifecycle_manager.spawn_agent(agent_type, config)
            
            # Initialize agent in pool
            if agent_type not in self.agent_pools:
                self.agent_pools[agent_type] = []
            self.agent_pools[agent_type].append(agent_id)
            self.agent_load[agent_id] = 0
            
            if self.monitor:
                self.monitor.record_agent_spawn(agent_id, agent_type)
            
            log.info(f"Spawned agent {agent_id} of type {agent_type}")
            return agent_id
    
    def terminate_agent(self, agent_id: str):
        """Terminate an agent instance"""
        with self.lock:
            agent_info = self.registry.get_agent(agent_id)
            if not agent_info:
                return
            
            # Remove from pools
            agent_type = agent_info['type']
            if agent_type in self.agent_pools:
                self.agent_pools[agent_type] = [
                    a for a in self.agent_pools[agent_type] if a != agent_id
                ]
            
            # Clean up load tracking
            if agent_id in self.agent_load:
                del self.agent_load[agent_id]
            
            # Terminate through lifecycle manager
            self.lifecycle_manager.terminate_agent(agent_id)
            
            if self.monitor:
                self.monitor.record_agent_termination(agent_id)
            
            log.info(f"Terminated agent {agent_id}")
    
    def submit_task(self, task: Dict[str, Any], priority: int = 5) -> str:
        """Submit a task for execution"""
        task_id = str(uuid.uuid4())
        task['id'] = task_id
        task['submitted_at'] = datetime.now()
        task['status'] = 'queued'
        
        self.task_queue.put((priority, task_id, task))
        
        if self.monitor:
            self.monitor.record_task_submission(task_id, task)
        
        # Trigger task processing
        self.thread_pool.submit(self._process_task, task_id, task)
        
        log.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    def _process_task(self, task_id: str, task: Dict[str, Any]):
        """Process a single task"""
        try:
            with self.lock:
                self.active_tasks[task_id] = task
                task['status'] = 'processing'
            
            # Select best agent for task
            agent_id = self._select_agent(task)
            if not agent_id:
                raise RuntimeError("No suitable agent available")
            
            # Update agent load
            with self.lock:
                self.agent_load[agent_id] += 1
            
            # Execute task
            agent = self.registry.get_agent(agent_id)['instance']
            result = self._execute_task_on_agent(agent, task)
            
            # Update task status
            with self.lock:
                task['status'] = 'completed'
                task['result'] = result
                task['completed_at'] = datetime.now()
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
                self.agent_load[agent_id] -= 1
            
            if self.monitor:
                self.monitor.record_task_completion(task_id, result)
            
            log.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            self._handle_task_failure(task_id, task, e)
    
    def _select_agent(self, task: Dict[str, Any]) -> Optional[str]:
        """Select the best agent for a task based on capabilities and load"""
        required_capabilities = task.get('required_capabilities', [])
        
        # Find capable agents
        capable_agents = []
        for agent_id, agent_info in self.registry.active_agents.items():
            agent_capabilities = agent_info.get('capabilities', [])
            if all(cap in agent_capabilities for cap in required_capabilities):
                capable_agents.append(agent_id)
        
        if not capable_agents:
            # Try to spawn a new agent if auto-scaling is enabled
            if self.config.enable_auto_scaling:
                agent_type = task.get('agent_type', 'default')
                if agent_type in self.registry.registered_types:
                    return self.spawn_agent(agent_type)
            return None
        
        # Select based on load balancing policy
        if self.config.load_balancing_policy == "round_robin":
            return self._round_robin_select(capable_agents)
        elif self.config.load_balancing_policy == "least_loaded":
            return self._least_loaded_select(capable_agents)
        elif self.config.load_balancing_policy == "random":
            import random
            return random.choice(capable_agents)
        else:
            return capable_agents[0]
    
    def _round_robin_select(self, agents: List[str]) -> str:
        """Round-robin agent selection"""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        selected = agents[self._rr_index % len(agents)]
        self._rr_index += 1
        return selected
    
    def _least_loaded_select(self, agents: List[str]) -> str:
        """Select the least loaded agent"""
        return min(agents, key=lambda a: self.agent_load.get(a, 0))
    
    def _execute_task_on_agent(self, agent: Any, task: Dict[str, Any]) -> Any:
        """Execute task on the selected agent"""
        # This would integrate with the actual agent's run method
        # For now, returning a placeholder
        task_type = task.get('type', 'default')
        task_data = task.get('data', {})
        
        # Send task through message bus
        message = Message(
            sender="orchestrator",
            recipient=agent.id if hasattr(agent, 'id') else str(agent),
            content=task_data,
            metadata={'task_id': task['id'], 'task_type': task_type}
        )
        
        response = self.message_bus.send_and_wait(message, timeout=self.config.task_timeout)
        return response.content if response else None
    
    def _handle_task_failure(self, task_id: str, task: Dict[str, Any], error: Exception):
        """Handle task execution failure"""
        with self.lock:
            task['status'] = 'failed'
            task['error'] = str(error)
            task['failed_at'] = datetime.now()
            
            # Check retry policy
            retry_count = task.get('retry_count', 0)
            if retry_count < self.config.retry_policy['max_retries']:
                task['retry_count'] = retry_count + 1
                task['status'] = 'retrying'
                
                # Resubmit with backoff
                backoff = min(
                    self.config.retry_policy['backoff_factor'] ** retry_count,
                    self.config.retry_policy['max_backoff']
                )
                threading.Timer(backoff, self._retry_task, args=(task_id, task)).start()
            else:
                self.failed_tasks.append(task)
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
        
        if self.monitor:
            self.monitor.record_task_failure(task_id, error)
        
        log.error(f"Task {task_id} failed: {error}")
    
    def _retry_task(self, task_id: str, task: Dict[str, Any]):
        """Retry a failed task"""
        log.info(f"Retrying task {task_id} (attempt {task.get('retry_count', 1)})")
        self._process_task(task_id, task)
    
    def _scale_down_least_used(self):
        """Scale down by terminating the least used agent"""
        if not self.agent_load:
            return
        
        least_used = min(self.agent_load.items(), key=lambda x: x[1])
        if least_used[1] == 0:  # Only terminate if agent is idle
            self.terminate_agent(least_used[0])
    
    def perform_health_check(self):
        """Perform health check on all agents"""
        unhealthy_agents = []
        
        for agent_id in list(self.registry.active_agents.keys()):
            if not self.lifecycle_manager.health_check(agent_id):
                unhealthy_agents.append(agent_id)
        
        # Handle unhealthy agents
        for agent_id in unhealthy_agents:
            log.warning(f"Agent {agent_id} is unhealthy")
            if self.config.failover_enabled:
                self._handle_agent_failure(agent_id)
    
    def _handle_agent_failure(self, agent_id: str):
        """Handle agent failure with failover"""
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return
        
        agent_type = agent_info['type']
        
        # Terminate failed agent
        self.terminate_agent(agent_id)
        
        # Spawn replacement if needed
        if agent_type in self.agent_pools and len(self.agent_pools[agent_type]) < 2:
            try:
                new_agent_id = self.spawn_agent(agent_type, agent_info.get('config', {}))
                log.info(f"Spawned replacement agent {new_agent_id} for failed agent {agent_id}")
            except Exception as e:
                log.error(f"Failed to spawn replacement agent: {e}")
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint of the current orchestrator state"""
        checkpoint = {
            'orchestrator_id': self.orchestrator_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'active_agents': list(self.registry.active_agents.keys()),
            'active_tasks': list(self.active_tasks.keys()),
            'completed_tasks_count': len(self.completed_tasks),
            'failed_tasks_count': len(self.failed_tasks),
            'agent_load': self.agent_load.copy()
        }
        
        checkpoint_file = Path(f"checkpoint_{self.orchestrator_id}_{int(time.time())}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        log.info(f"Created checkpoint: {checkpoint_file}")
        return str(checkpoint_file)
    
    def restore_from_checkpoint(self, checkpoint_file: str):
        """Restore orchestrator state from checkpoint"""
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        log.info(f"Restoring from checkpoint: {checkpoint_file}")
        
        # Restore configuration
        for key, value in checkpoint['config'].items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Note: Agent restoration would require re-instantiation
        # This is a simplified version
        log.info("Checkpoint restored (partial restoration)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            'orchestrator_id': self.orchestrator_id,
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'active_agents': len(self.registry.active_agents),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'agent_load': self.agent_load.copy(),
            'config': self.config.__dict__
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        log.info(f"Shutting down orchestrator {self.orchestrator_id}")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for active tasks
        timeout = 30
        start = time.time()
        while self.active_tasks and (time.time() - start) < timeout:
            time.sleep(0.5)
        
        # Terminate all agents
        for agent_id in list(self.registry.active_agents.keys()):
            self.terminate_agent(agent_id)
        
        # Shutdown pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Final checkpoint
        if self.config.enable_checkpointing:
            self.create_checkpoint()
        
        log.info("Orchestrator shutdown complete")