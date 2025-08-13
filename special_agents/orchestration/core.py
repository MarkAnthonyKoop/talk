"""
Core Orchestration Engine

Provides the fundamental orchestration capabilities for managing agents,
tasks, and their execution lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Union
from queue import Queue, PriorityQueue
import threading

from agent.agent import Agent

log = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    BLOCKED = auto()
    RETRYING = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    execution_time: float = 0.0
    retry_count: int = 0


@dataclass
class Task:
    """Represents a unit of work to be executed by an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: str = "generic"
    priority: TaskPriority = TaskPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: Set[str] = field(default_factory=set)
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    
    # Callbacks
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    on_progress: Optional[Callable] = None
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        return self.priority.value < other.priority.value
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "priority": self.priority.name,
            "status": self.status.name,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "payload": self.payload
        }


@dataclass
class AgentCapabilities:
    """Defines what an agent can do."""
    agent_id: str
    agent_type: str
    capabilities: Set[str]
    max_concurrent_tasks: int = 1
    supported_task_types: Set[str] = field(default_factory=set)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    availability: bool = True
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle a specific task."""
        if not self.availability:
            return False
        
        # Check task type compatibility
        if task.type and self.supported_task_types:
            if task.type not in self.supported_task_types:
                return False
        
        # Check required capabilities
        return task.required_capabilities.issubset(self.capabilities)


class AgentPool:
    """Manages a pool of agents for task execution."""
    
    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: Dict[str, Agent] = {}
        self.capabilities: Dict[str, AgentCapabilities] = {}
        self.busy_agents: Set[str] = set()
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self._lock = threading.RLock()
        
    def register_agent(self, agent: Agent, capabilities: AgentCapabilities) -> str:
        """Register an agent in the pool."""
        with self._lock:
            agent_id = capabilities.agent_id
            self.agents[agent_id] = agent
            self.capabilities[agent_id] = capabilities
            log.info(f"Registered agent {agent_id} with capabilities {capabilities.capabilities}")
            return agent_id
    
    def unregister_agent(self, agent_id: str):
        """Remove an agent from the pool."""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                del self.capabilities[agent_id]
                self.busy_agents.discard(agent_id)
                log.info(f"Unregistered agent {agent_id}")
    
    def find_suitable_agent(self, task: Task) -> Optional[str]:
        """Find an available agent that can handle the task."""
        with self._lock:
            for agent_id, caps in self.capabilities.items():
                if agent_id not in self.busy_agents and caps.can_handle_task(task):
                    return agent_id
            return None
    
    def assign_task(self, task: Task, agent_id: str) -> bool:
        """Assign a task to an agent."""
        with self._lock:
            if agent_id not in self.agents or agent_id in self.busy_agents:
                return False
            
            self.busy_agents.add(agent_id)
            self.task_assignments[task.id] = agent_id
            task.assigned_agent = agent_id
            return True
    
    def release_agent(self, agent_id: str, task_id: str):
        """Mark an agent as available after task completion."""
        with self._lock:
            self.busy_agents.discard(agent_id)
            self.task_assignments.pop(task_id, None)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        with self._lock:
            return {
                "total_agents": len(self.agents),
                "busy_agents": len(self.busy_agents),
                "available_agents": len(self.agents) - len(self.busy_agents),
                "active_tasks": len(self.task_assignments),
                "agents": {
                    agent_id: {
                        "busy": agent_id in self.busy_agents,
                        "capabilities": list(caps.capabilities)
                    }
                    for agent_id, caps in self.capabilities.items()
                }
            }


class Orchestrator:
    """
    Main orchestration engine that coordinates task execution across agents.
    
    Features:
    - Task decomposition and dependency management
    - Dynamic agent allocation
    - Parallel and sequential execution
    - Result aggregation and synthesis
    - Error handling and retry logic
    - Progress monitoring
    """
    
    def __init__(self,
                 agent_pool: Optional[AgentPool] = None,
                 max_parallel_tasks: int = 4,
                 executor_type: str = "thread",
                 enable_monitoring: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            agent_pool: Pool of agents to use
            max_parallel_tasks: Maximum tasks to run in parallel
            executor_type: "thread" or "process" 
            enable_monitoring: Enable task monitoring
        """
        self.agent_pool = agent_pool or AgentPool()
        self.max_parallel_tasks = max_parallel_tasks
        self.enable_monitoring = enable_monitoring
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: PriorityQueue = PriorityQueue()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Execution management
        if executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_parallel_tasks)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)
        
        self.running_tasks: Dict[str, Any] = {}  # task_id -> Future
        self._shutdown = False
        self._lock = threading.RLock()
        
        # Monitoring
        self.start_time = datetime.now()
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_task_time": 0.0
        }
        
        # Start background scheduler
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        log.info(f"Orchestrator initialized with {max_parallel_tasks} parallel slots")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        with self._lock:
            self.tasks[task.id] = task
            self.metrics["tasks_submitted"] += 1
            
            # Check if task can be executed immediately
            if task.can_execute(self.completed_tasks):
                task.status = TaskStatus.QUEUED
                self.task_queue.put((task.priority.value, task.id))
                log.info(f"Task {task.id} queued for execution")
            else:
                task.status = TaskStatus.BLOCKED
                log.info(f"Task {task.id} blocked on dependencies: {task.dependencies}")
            
            return task.id
    
    def decompose_task(self, parent_task: Task, subtasks: List[Task]) -> List[str]:
        """Decompose a task into subtasks."""
        subtask_ids = []
        
        with self._lock:
            for subtask in subtasks:
                subtask.parent_task_id = parent_task.id
                parent_task.subtasks.append(subtask.id)
                subtask_ids.append(self.submit_task(subtask))
        
        return subtask_ids
    
    def _scheduler_loop(self):
        """Background scheduler that assigns tasks to agents."""
        while not self._shutdown:
            try:
                # Check for ready tasks
                self._check_blocked_tasks()
                
                # Process task queue
                if not self.task_queue.empty() and len(self.running_tasks) < self.max_parallel_tasks:
                    priority, task_id = self.task_queue.get(timeout=0.1)
                    
                    with self._lock:
                        task = self.tasks.get(task_id)
                        if task and task.status == TaskStatus.QUEUED:
                            self._execute_task(task)
                
                # Check running tasks
                self._check_running_tasks()
                
                time.sleep(0.1)
                
            except Exception as e:
                log.error(f"Scheduler error: {e}")
    
    def _check_blocked_tasks(self):
        """Check if any blocked tasks can now execute."""
        with self._lock:
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.BLOCKED:
                    if task.can_execute(self.completed_tasks):
                        task.status = TaskStatus.QUEUED
                        self.task_queue.put((task.priority.value, task.id))
                        log.info(f"Task {task.id} unblocked and queued")
    
    def _execute_task(self, task: Task):
        """Execute a task on an available agent."""
        # Find suitable agent
        agent_id = self.agent_pool.find_suitable_agent(task)
        
        if not agent_id:
            log.warning(f"No suitable agent for task {task.id}, requeuing")
            self.task_queue.put((task.priority.value, task.id))
            return
        
        # Assign and execute
        if self.agent_pool.assign_task(task, agent_id):
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            future = self.executor.submit(self._run_task_on_agent, task, agent_id)
            self.running_tasks[task.id] = future
            
            log.info(f"Task {task.id} started on agent {agent_id}")
    
    def _run_task_on_agent(self, task: Task, agent_id: str) -> TaskResult:
        """Run a task on a specific agent."""
        start_time = time.time()
        result = TaskResult(task_id=task.id, status=TaskStatus.RUNNING)
        
        try:
            agent = self.agent_pool.get_agent(agent_id)
            if not agent:
                raise RuntimeError(f"Agent {agent_id} not found")
            
            # Execute task based on type
            if task.type == "completion":
                output = agent.run(task.payload.get("prompt", ""))
            elif task.type == "analysis":
                # Custom analysis logic
                output = self._run_analysis_task(agent, task)
            elif task.type == "synthesis":
                # Custom synthesis logic
                output = self._run_synthesis_task(agent, task)
            else:
                # Generic execution
                output = agent.run(json.dumps(task.payload))
            
            result.output = output
            result.status = TaskStatus.COMPLETED
            
            # Call completion callback
            if task.on_complete:
                task.on_complete(result)
                
        except Exception as e:
            log.error(f"Task {task.id} failed: {e}")
            result.status = TaskStatus.FAILED
            result.error = str(e)
            
            # Handle retry logic
            if task.max_retries > 0 and result.retry_count < task.max_retries:
                result.retry_count += 1
                time.sleep(task.retry_delay)
                return self._run_task_on_agent(task, agent_id)
            
            # Call error callback
            if task.on_error:
                task.on_error(result)
        
        finally:
            result.execution_time = time.time() - start_time
            result.metrics = {
                "agent_id": agent_id,
                "execution_time": result.execution_time,
                "retry_count": result.retry_count
            }
            
            # Release agent
            self.agent_pool.release_agent(agent_id, task.id)
        
        return result
    
    def _run_analysis_task(self, agent: Agent, task: Task) -> Any:
        """Run an analysis task."""
        context = task.payload.get("context", {})
        target = task.payload.get("target", "")
        
        prompt = f"""Analyze the following:
Target: {target}
Context: {json.dumps(context, indent=2)}

Provide a comprehensive analysis including:
1. Key insights
2. Patterns identified
3. Recommendations
4. Potential issues
"""
        return agent.run(prompt)
    
    def _run_synthesis_task(self, agent: Agent, task: Task) -> Any:
        """Run a synthesis task."""
        inputs = task.payload.get("inputs", [])
        format = task.payload.get("format", "summary")
        
        prompt = f"""Synthesize the following inputs into a {format}:

Inputs:
{json.dumps(inputs, indent=2)}

Create a cohesive {format} that integrates all the input information.
"""
        return agent.run(prompt)
    
    def _check_running_tasks(self):
        """Check status of running tasks."""
        completed = []
        
        with self._lock:
            for task_id, future in list(self.running_tasks.items()):
                if future.done():
                    try:
                        result = future.result(timeout=0.1)
                        task = self.tasks[task_id]
                        task.result = result
                        task.completed_at = datetime.now()
                        task.status = result.status
                        
                        if result.status == TaskStatus.COMPLETED:
                            self.completed_tasks.add(task_id)
                            self.metrics["tasks_completed"] += 1
                            log.info(f"Task {task_id} completed successfully")
                        else:
                            self.failed_tasks.add(task_id)
                            self.metrics["tasks_failed"] += 1
                            log.error(f"Task {task_id} failed: {result.error}")
                        
                        completed.append(task_id)
                        
                    except Exception as e:
                        log.error(f"Error getting task result for {task_id}: {e}")
                        completed.append(task_id)
            
            # Remove completed tasks from running
            for task_id in completed:
                self.running_tasks.pop(task_id, None)
    
    def wait_for_completion(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for specific tasks or all tasks to complete."""
        start_time = time.time()
        target_tasks = set(task_ids) if task_ids else set(self.tasks.keys())
        
        while True:
            with self._lock:
                completed = target_tasks.intersection(self.completed_tasks.union(self.failed_tasks))
                
                if len(completed) == len(target_tasks):
                    return {
                        task_id: self.tasks[task_id].result
                        for task_id in target_tasks
                        if self.tasks[task_id].result
                    }
            
            if timeout and (time.time() - start_time) > timeout:
                log.warning(f"Timeout waiting for tasks: {target_tasks - completed}")
                break
            
            time.sleep(0.1)
        
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        with self._lock:
            return {
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "metrics": self.metrics,
                "tasks": {
                    "total": len(self.tasks),
                    "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                    "queued": sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED),
                    "running": len(self.running_tasks),
                    "completed": len(self.completed_tasks),
                    "failed": len(self.failed_tasks),
                    "blocked": sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED)
                },
                "agent_pool": self.agent_pool.get_pool_status()
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the orchestrator."""
        log.info("Shutting down orchestrator...")
        self._shutdown = True
        
        if wait:
            # Wait for running tasks
            self.executor.shutdown(wait=True)
            
            # Wait for scheduler thread
            self.scheduler_thread.join(timeout=5)
        
        log.info("Orchestrator shutdown complete")


# Export main components
__all__ = ['Orchestrator', 'Task', 'TaskStatus', 'TaskPriority', 'TaskResult', 'AgentPool', 'AgentCapabilities']