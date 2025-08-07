"""
Task dispatcher with intelligent distribution and load balancing.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from queue import PriorityQueue, Queue
import hashlib
import json

log = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class DistributionStrategy(Enum):
    """Task distribution strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    WEIGHTED = "weighted"
    AFFINITY = "affinity"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"


@dataclass
class Task:
    """Task definition"""
    id: str
    type: str
    payload: Any
    priority: int = 5
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: List[str] = field(default_factory=list)
    preferred_agent: Optional[str] = None
    timeout: int = 300  # seconds
    max_retries: int = 3
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class TaskGroup:
    """Group of related tasks"""
    id: str
    tasks: List[Task]
    parallel: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Load balancing for task distribution"""
    
    def __init__(self, strategy: DistributionStrategy = DistributionStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.agent_loads: Dict[str, int] = {}
        self.agent_weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.consistent_hash_ring: Dict[int, str] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def select_agent(self, 
                    available_agents: List[str],
                    task: Task,
                    agent_capabilities: Dict[str, List[str]] = None) -> Optional[str]:
        """Select best agent for task"""
        
        if not available_agents:
            return None
        
        # Filter by capabilities if required
        if task.required_capabilities and agent_capabilities:
            capable_agents = []
            for agent in available_agents:
                caps = agent_capabilities.get(agent, [])
                if all(req in caps for req in task.required_capabilities):
                    capable_agents.append(agent)
            available_agents = capable_agents
        
        if not available_agents:
            return None
        
        # Apply strategy
        if self.strategy == DistributionStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self.strategy == DistributionStrategy.LEAST_LOADED:
            return self._least_loaded_select(available_agents)
        elif self.strategy == DistributionStrategy.RANDOM:
            import random
            return random.choice(available_agents)
        elif self.strategy == DistributionStrategy.WEIGHTED:
            return self._weighted_select(available_agents)
        elif self.strategy == DistributionStrategy.AFFINITY:
            return self._affinity_select(available_agents, task)
        elif self.strategy == DistributionStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(task)
        elif self.strategy == DistributionStrategy.ADAPTIVE:
            return self._adaptive_select(available_agents, task)
        else:
            return available_agents[0]
    
    def _round_robin_select(self, agents: List[str]) -> str:
        """Round-robin selection"""
        selected = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        return selected
    
    def _least_loaded_select(self, agents: List[str]) -> str:
        """Select least loaded agent"""
        loads = [(self.agent_loads.get(a, 0), a) for a in agents]
        loads.sort()
        return loads[0][1]
    
    def _weighted_select(self, agents: List[str]) -> str:
        """Weighted random selection"""
        import random
        weights = [self.agent_weights.get(a, 1.0) for a in agents]
        return random.choices(agents, weights=weights)[0]
    
    def _affinity_select(self, agents: List[str], task: Task) -> str:
        """Select based on task affinity"""
        # Prefer agent that previously handled similar tasks
        if task.metadata.get('affinity_key'):
            key = task.metadata['affinity_key']
            for agent in agents:
                if self._has_affinity(agent, key):
                    return agent
        
        # Fall back to least loaded
        return self._least_loaded_select(agents)
    
    def _consistent_hash_select(self, task: Task) -> str:
        """Consistent hashing for stable assignment"""
        if not self.consistent_hash_ring:
            return None
        
        # Hash the task
        task_hash = int(hashlib.md5(task.id.encode()).hexdigest(), 16)
        
        # Find closest node in ring
        keys = sorted(self.consistent_hash_ring.keys())
        for key in keys:
            if key >= task_hash:
                return self.consistent_hash_ring[key]
        
        # Wrap around
        return self.consistent_hash_ring[keys[0]]
    
    def _adaptive_select(self, agents: List[str], task: Task) -> str:
        """Adaptive selection based on performance"""
        # Score agents based on historical performance
        scores = []
        for agent in agents:
            history = self.performance_history.get(agent, [1.0])
            avg_performance = sum(history) / len(history)
            current_load = self.agent_loads.get(agent, 0)
            
            # Combined score: performance vs load
            score = avg_performance / (current_load + 1)
            scores.append((score, agent))
        
        # Select best scoring agent
        scores.sort(reverse=True)
        return scores[0][1]
    
    def _has_affinity(self, agent: str, key: str) -> bool:
        """Check if agent has affinity for key"""
        # This would check historical task assignments
        # Simplified for now
        return hash(agent + key) % 2 == 0
    
    def update_load(self, agent: str, delta: int):
        """Update agent load"""
        self.agent_loads[agent] = max(0, self.agent_loads.get(agent, 0) + delta)
    
    def update_performance(self, agent: str, performance: float):
        """Update agent performance history"""
        if agent not in self.performance_history:
            self.performance_history[agent] = []
        
        self.performance_history[agent].append(performance)
        
        # Keep last 100 entries
        if len(self.performance_history[agent]) > 100:
            self.performance_history[agent] = self.performance_history[agent][-100:]
    
    def add_to_hash_ring(self, agent: str, replicas: int = 3):
        """Add agent to consistent hash ring"""
        for i in range(replicas):
            key = f"{agent}:{i}"
            hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
            self.consistent_hash_ring[hash_val] = agent
    
    def remove_from_hash_ring(self, agent: str):
        """Remove agent from consistent hash ring"""
        to_remove = []
        for hash_val, agent_id in self.consistent_hash_ring.items():
            if agent_id == agent:
                to_remove.append(hash_val)
        
        for hash_val in to_remove:
            del self.consistent_hash_ring[hash_val]


class TaskDispatcher:
    """
    Intelligent task dispatcher with advanced distribution capabilities.
    
    Features:
    - Multiple distribution strategies
    - Task dependencies and DAG execution
    - Priority-based scheduling
    - Task batching and grouping
    - Circuit breaker pattern
    - Back-pressure handling
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        # Task management
        self.task_queue = PriorityQueue()
        self.pending_tasks: Dict[str, Task] = {}
        self.executing_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Dependencies
        self.task_dependencies: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Load balancing
        self.load_balancer = LoadBalancer(
            DistributionStrategy(config.load_balancing_policy)
        )
        
        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_dispatched': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'avg_execution_time': 0
        }
        
        # Synchronization
        self.lock = threading.RLock()
        
        log.info("Task dispatcher initialized")
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task for dispatch"""
        with self.lock:
            # Validate task
            if not self._validate_task(task):
                return False
            
            # Check dependencies
            if task.dependencies:
                self.task_dependencies[task.id] = task.dependencies
                self._update_dependency_graph(task)
            
            # Add to queue
            task.status = TaskStatus.QUEUED
            self.pending_tasks[task.id] = task
            self.task_queue.put((task.priority, task.created_at, task))
            
            self.stats['tasks_submitted'] += 1
            
            log.debug(f"Task {task.id} submitted with priority {task.priority}")
            return True
    
    def submit_task_group(self, group: TaskGroup) -> bool:
        """Submit a group of tasks"""
        with self.lock:
            # Set up dependencies if sequential
            if not group.parallel and len(group.tasks) > 1:
                for i in range(1, len(group.tasks)):
                    group.tasks[i].dependencies.append(group.tasks[i-1].id)
            
            # Submit all tasks
            for task in group.tasks:
                task.metadata['group_id'] = group.id
                if not self.submit_task(task):
                    return False
            
            log.info(f"Task group {group.id} submitted with {len(group.tasks)} tasks")
            return True
    
    def dispatch_task(self, 
                     available_agents: List[str],
                     agent_capabilities: Dict[str, List[str]] = None) -> Optional[Tuple[Task, str]]:
        """Dispatch next available task to an agent"""
        with self.lock:
            # Get next eligible task
            task = self._get_next_eligible_task()
            if not task:
                return None
            
            # Check preferred agent
            if task.preferred_agent and task.preferred_agent in available_agents:
                selected_agent = task.preferred_agent
            else:
                # Select agent using load balancer
                selected_agent = self.load_balancer.select_agent(
                    available_agents, task, agent_capabilities
                )
            
            if not selected_agent:
                # No suitable agent, requeue task
                self.task_queue.put((task.priority, task.created_at, task))
                return None
            
            # Check circuit breaker
            if not self._check_circuit_breaker(selected_agent):
                log.warning(f"Circuit breaker open for agent {selected_agent}")
                # Try another agent
                remaining_agents = [a for a in available_agents if a != selected_agent]
                if remaining_agents:
                    return self.dispatch_task(remaining_agents, agent_capabilities)
                return None
            
            # Assign task
            task.status = TaskStatus.ASSIGNED
            task.assigned_to = selected_agent
            task.assigned_at = datetime.now()
            
            # Move to executing
            del self.pending_tasks[task.id]
            self.executing_tasks[task.id] = task
            
            # Update load
            self.load_balancer.update_load(selected_agent, 1)
            
            self.stats['tasks_dispatched'] += 1
            
            log.info(f"Task {task.id} dispatched to agent {selected_agent}")
            return task, selected_agent
    
    def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark task as completed"""
        with self.lock:
            if task_id not in self.executing_tasks:
                return False
            
            task = self.executing_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update load
            if task.assigned_to:
                self.load_balancer.update_load(task.assigned_to, -1)
                
                # Update performance
                execution_time = (task.completed_at - task.started_at).total_seconds()
                performance = 1.0 / max(execution_time, 1.0)  # Higher is better
                self.load_balancer.update_performance(task.assigned_to, performance)
            
            # Move to completed
            del self.executing_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            # Update statistics
            self.stats['tasks_completed'] += 1
            self._update_avg_execution_time(task)
            
            # Check dependencies
            self._process_completed_dependencies(task_id)
            
            # Reset circuit breaker
            if task.assigned_to:
                self._reset_circuit_breaker(task.assigned_to)
            
            log.info(f"Task {task_id} completed successfully")
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        with self.lock:
            if task_id not in self.executing_tasks:
                return False
            
            task = self.executing_tasks[task_id]
            task.error = error
            
            # Update load
            if task.assigned_to:
                self.load_balancer.update_load(task.assigned_to, -1)
                
                # Trip circuit breaker
                self._trip_circuit_breaker(task.assigned_to)
            
            # Check retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Requeue with delay
                threading.Timer(
                    2 ** task.retry_count,  # Exponential backoff
                    self._retry_task,
                    args=(task,)
                ).start()
                
                self.stats['tasks_retried'] += 1
                log.info(f"Task {task_id} will be retried (attempt {task.retry_count})")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                # Move to failed
                del self.executing_tasks[task_id]
                self.failed_tasks[task_id] = task
                
                self.stats['tasks_failed'] += 1
                log.error(f"Task {task_id} failed: {error}")
            
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self.lock:
            task = None
            
            # Find task
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                del self.pending_tasks[task_id]
            elif task_id in self.executing_tasks:
                task = self.executing_tasks[task_id]
                del self.executing_tasks[task_id]
                
                # Update load
                if task.assigned_to:
                    self.load_balancer.update_load(task.assigned_to, -1)
            
            if task:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                log.info(f"Task {task_id} cancelled")
                return True
            
            return False
    
    def _validate_task(self, task: Task) -> bool:
        """Validate task before submission"""
        if not task.id or not task.type:
            return False
        
        # Check for circular dependencies
        if task.dependencies:
            if task.id in task.dependencies:
                log.error(f"Task {task.id} has circular dependency")
                return False
        
        return True
    
    def _get_next_eligible_task(self) -> Optional[Task]:
        """Get next task that's ready to execute"""
        while not self.task_queue.empty():
            priority, created_at, task = self.task_queue.get()
            
            # Check if dependencies are satisfied
            if task.dependencies:
                unsatisfied = []
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        unsatisfied.append(dep_id)
                
                if unsatisfied:
                    # Dependencies not satisfied, requeue
                    self.task_queue.put((priority, created_at, task))
                    continue
            
            return task
        
        return None
    
    def _update_dependency_graph(self, task: Task):
        """Update dependency graph"""
        for dep_id in task.dependencies:
            if dep_id not in self.dependency_graph:
                self.dependency_graph[dep_id] = []
            self.dependency_graph[dep_id].append(task.id)
    
    def _process_completed_dependencies(self, task_id: str):
        """Process tasks waiting on completed dependency"""
        if task_id in self.dependency_graph:
            waiting_tasks = self.dependency_graph[task_id]
            
            for waiting_id in waiting_tasks:
                if waiting_id in self.pending_tasks:
                    waiting_task = self.pending_tasks[waiting_id]
                    
                    # Remove satisfied dependency
                    if task_id in waiting_task.dependencies:
                        waiting_task.dependencies.remove(task_id)
                    
                    # If all dependencies satisfied, make eligible
                    if not waiting_task.dependencies:
                        log.debug(f"Task {waiting_id} dependencies satisfied")
            
            # Clean up graph
            del self.dependency_graph[task_id]
    
    def _retry_task(self, task: Task):
        """Retry a failed task"""
        with self.lock:
            task.status = TaskStatus.QUEUED
            self.pending_tasks[task.id] = task
            self.task_queue.put((task.priority - 1, task.created_at, task))  # Higher priority
    
    def _check_circuit_breaker(self, agent_id: str) -> bool:
        """Check if circuit breaker allows request"""
        if agent_id not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[agent_id]
        
        if breaker['state'] == 'open':
            # Check if enough time has passed
            if time.time() - breaker['opened_at'] > breaker['timeout']:
                breaker['state'] = 'half_open'
                breaker['test_requests'] = 0
            else:
                return False
        
        if breaker['state'] == 'half_open':
            breaker['test_requests'] += 1
            if breaker['test_requests'] > 3:
                breaker['state'] = 'closed'
        
        return True
    
    def _trip_circuit_breaker(self, agent_id: str):
        """Trip circuit breaker for agent"""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                'state': 'closed',
                'failures': 0,
                'timeout': 30
            }
        
        breaker = self.circuit_breakers[agent_id]
        breaker['failures'] += 1
        
        if breaker['failures'] >= 3:
            breaker['state'] = 'open'
            breaker['opened_at'] = time.time()
            log.warning(f"Circuit breaker opened for agent {agent_id}")
    
    def _reset_circuit_breaker(self, agent_id: str):
        """Reset circuit breaker for agent"""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id]['failures'] = 0
            self.circuit_breakers[agent_id]['state'] = 'closed'
    
    def _update_avg_execution_time(self, task: Task):
        """Update average execution time statistic"""
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            
            # Running average
            n = self.stats['tasks_completed']
            current_avg = self.stats['avg_execution_time']
            self.stats['avg_execution_time'] = (current_avg * (n - 1) + execution_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dispatcher statistics"""
        with self.lock:
            return {
                **self.stats,
                'pending_tasks': len(self.pending_tasks),
                'executing_tasks': len(self.executing_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'queue_size': self.task_queue.qsize(),
                'agent_loads': self.load_balancer.agent_loads.copy()
            }