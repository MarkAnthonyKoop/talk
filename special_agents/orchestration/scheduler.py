"""
Task Scheduling and Queue Management System

Provides intelligent task scheduling with multiple strategies,
priority management, and resource optimization.
"""

from __future__ import annotations

import heapq
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set, Tuple, Callable

from special_agents.orchestration.core import Task, TaskStatus, TaskPriority

log = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    FIFO = auto()           # First In First Out
    LIFO = auto()           # Last In First Out  
    PRIORITY = auto()       # Priority-based
    ROUND_ROBIN = auto()    # Round-robin across agents
    WEIGHTED = auto()       # Weighted by task cost
    DEADLINE = auto()       # Earliest deadline first
    SHORTEST_JOB = auto()   # Shortest job first
    FAIR_SHARE = auto()     # Fair share across task types
    ADAPTIVE = auto()       # Adaptive based on performance


@dataclass
class ScheduledTask:
    """Wrapper for scheduled tasks."""
    task: Task
    scheduled_time: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    retry_count: int = 0
    last_attempt: Optional[datetime] = None
    
    def __lt__(self, other):
        """Enable heap operations."""
        # Compare by priority first, then scheduled time
        if self.task.priority != other.task.priority:
            return self.task.priority.value < other.task.priority.value
        return self.scheduled_time < other.scheduled_time
    
    def time_until_deadline(self) -> Optional[float]:
        """Get time remaining until deadline in seconds."""
        if self.deadline:
            return (self.deadline - datetime.now()).total_seconds()
        return None
    
    def is_overdue(self) -> bool:
        """Check if task is past its deadline."""
        if self.deadline:
            return datetime.now() > self.deadline
        return False


class TaskScheduler(ABC):
    """Abstract base class for task schedulers."""
    
    @abstractmethod
    def add_task(self, task: ScheduledTask):
        """Add a task to the scheduler."""
        pass
    
    @abstractmethod
    def get_next_task(self) -> Optional[ScheduledTask]:
        """Get the next task to execute."""
        pass
    
    @abstractmethod
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler."""
        pass
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size."""
        pass


class FIFOScheduler(TaskScheduler):
    """First In First Out scheduler."""
    
    def __init__(self):
        self.queue = deque()
        self.task_map = {}
        self._lock = threading.Lock()
    
    def add_task(self, task: ScheduledTask):
        with self._lock:
            self.queue.append(task)
            self.task_map[task.task.id] = task
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        with self._lock:
            if self.queue:
                task = self.queue.popleft()
                del self.task_map[task.task.id]
                return task
            return None
    
    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                self.queue.remove(task)
                del self.task_map[task_id]
                return True
            return False
    
    def get_queue_size(self) -> int:
        return len(self.queue)


class PriorityScheduler(TaskScheduler):
    """Priority-based task scheduler."""
    
    def __init__(self):
        self.heap = []
        self.task_map = {}
        self._lock = threading.Lock()
        self._counter = 0  # Tie-breaker for equal priorities
    
    def add_task(self, task: ScheduledTask):
        with self._lock:
            # Use counter as tie-breaker
            heapq.heappush(self.heap, (
                task.task.priority.value,
                self._counter,
                task
            ))
            self._counter += 1
            self.task_map[task.task.id] = task
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        with self._lock:
            while self.heap:
                _, _, task = heapq.heappop(self.heap)
                if task.task.id in self.task_map:
                    del self.task_map[task.task.id]
                    return task
            return None
    
    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self.task_map:
                del self.task_map[task_id]
                # Mark as removed (lazy deletion)
                return True
            return False
    
    def get_queue_size(self) -> int:
        return len(self.task_map)


class DeadlineScheduler(TaskScheduler):
    """Earliest Deadline First scheduler."""
    
    def __init__(self):
        self.heap = []
        self.task_map = {}
        self._lock = threading.Lock()
    
    def add_task(self, task: ScheduledTask):
        with self._lock:
            # Sort by deadline, with None deadlines last
            deadline_key = task.deadline.timestamp() if task.deadline else float('inf')
            heapq.heappush(self.heap, (deadline_key, task))
            self.task_map[task.task.id] = task
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        with self._lock:
            while self.heap:
                _, task = heapq.heappop(self.heap)
                if task.task.id in self.task_map:
                    del self.task_map[task.task.id]
                    return task
            return None
    
    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self.task_map:
                del self.task_map[task_id]
                return True
            return False
    
    def get_queue_size(self) -> int:
        return len(self.task_map)


class AdaptiveScheduler(TaskScheduler):
    """
    Adaptive scheduler that adjusts strategy based on performance metrics.
    """
    
    def __init__(self):
        self.strategies = {
            SchedulingStrategy.PRIORITY: PriorityScheduler(),
            SchedulingStrategy.DEADLINE: DeadlineScheduler(),
            SchedulingStrategy.FIFO: FIFOScheduler()
        }
        self.current_strategy = SchedulingStrategy.PRIORITY
        self.performance_window = deque(maxlen=100)
        self.strategy_scores = defaultdict(float)
        self._lock = threading.Lock()
    
    def add_task(self, task: ScheduledTask):
        with self._lock:
            self.strategies[self.current_strategy].add_task(task)
    
    def get_next_task(self) -> Optional[ScheduledTask]:
        with self._lock:
            task = self.strategies[self.current_strategy].get_next_task()
            
            # Adapt strategy based on performance
            self._adapt_strategy()
            
            return task
    
    def remove_task(self, task_id: str) -> bool:
        with self._lock:
            return self.strategies[self.current_strategy].remove_task(task_id)
    
    def get_queue_size(self) -> int:
        return self.strategies[self.current_strategy].get_queue_size()
    
    def record_performance(self, task_id: str, success: bool, execution_time: float):
        """Record task performance for adaptation."""
        with self._lock:
            self.performance_window.append({
                "strategy": self.current_strategy,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now()
            })
    
    def _adapt_strategy(self):
        """Adapt scheduling strategy based on recent performance."""
        if len(self.performance_window) < 20:
            return
        
        # Calculate scores for each strategy
        for metric in self.performance_window:
            strategy = metric["strategy"]
            score = 1.0 if metric["success"] else -1.0
            
            # Adjust for execution time
            if metric["execution_time"] < 1.0:
                score += 0.5
            elif metric["execution_time"] > 10.0:
                score -= 0.5
            
            self.strategy_scores[strategy] += score
        
        # Find best performing strategy
        best_strategy = max(self.strategy_scores, key=self.strategy_scores.get)
        
        if best_strategy != self.current_strategy:
            log.info(f"Adapting scheduling strategy from {self.current_strategy} to {best_strategy}")
            
            # Migrate tasks to new strategy
            old_scheduler = self.strategies[self.current_strategy]
            new_scheduler = self.strategies[best_strategy]
            
            # Transfer all tasks
            while old_scheduler.get_queue_size() > 0:
                task = old_scheduler.get_next_task()
                if task:
                    new_scheduler.add_task(task)
            
            self.current_strategy = best_strategy


class TaskQueue:
    """
    Advanced task queue with multiple scheduling strategies and features.
    """
    
    def __init__(self,
                 strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY,
                 max_size: Optional[int] = None,
                 enable_batching: bool = True,
                 batch_size: int = 10):
        """
        Initialize task queue.
        
        Args:
            strategy: Scheduling strategy to use
            max_size: Maximum queue size (None for unlimited)
            enable_batching: Enable task batching
            batch_size: Number of tasks to batch together
        """
        self.strategy = strategy
        self.max_size = max_size
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        
        # Create scheduler based on strategy
        self.scheduler = self._create_scheduler(strategy)
        
        # Task tracking
        self.pending_tasks = {}
        self.task_history = deque(maxlen=1000)
        
        # Batching
        self.task_batches = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "total_rejected": 0,
            "average_wait_time": 0.0,
            "max_queue_size": 0
        }
        
        self._lock = threading.RLock()
        
        log.info(f"Task queue initialized with {strategy} strategy")
    
    def _create_scheduler(self, strategy: SchedulingStrategy) -> TaskScheduler:
        """Create scheduler based on strategy."""
        if strategy == SchedulingStrategy.FIFO:
            return FIFOScheduler()
        elif strategy == SchedulingStrategy.PRIORITY:
            return PriorityScheduler()
        elif strategy == SchedulingStrategy.DEADLINE:
            return DeadlineScheduler()
        elif strategy == SchedulingStrategy.ADAPTIVE:
            return AdaptiveScheduler()
        else:
            # Default to priority scheduler
            return PriorityScheduler()
    
    def enqueue(self, task: Task, deadline: Optional[datetime] = None,
                estimated_duration: float = 0.0) -> bool:
        """
        Add a task to the queue.
        
        Args:
            task: Task to enqueue
            deadline: Optional deadline for the task
            estimated_duration: Estimated task duration in seconds
            
        Returns:
            True if task was enqueued, False if rejected
        """
        with self._lock:
            # Check queue size limit
            if self.max_size and self.scheduler.get_queue_size() >= self.max_size:
                log.warning(f"Queue full, rejecting task {task.id}")
                self.stats["total_rejected"] += 1
                return False
            
            # Create scheduled task
            scheduled_task = ScheduledTask(
                task=task,
                deadline=deadline,
                estimated_duration=estimated_duration
            )
            
            # Add to scheduler
            self.scheduler.add_task(scheduled_task)
            self.pending_tasks[task.id] = scheduled_task
            
            # Update statistics
            self.stats["total_enqueued"] += 1
            self.stats["max_queue_size"] = max(
                self.stats["max_queue_size"],
                self.scheduler.get_queue_size()
            )
            
            # Handle batching
            if self.enable_batching:
                self._add_to_batch(task)
            
            log.debug(f"Enqueued task {task.id} with priority {task.priority}")
            
            return True
    
    def dequeue(self, count: int = 1) -> List[Task]:
        """
        Get next tasks from the queue.
        
        Args:
            count: Number of tasks to dequeue
            
        Returns:
            List of tasks
        """
        with self._lock:
            tasks = []
            
            for _ in range(count):
                scheduled_task = self.scheduler.get_next_task()
                
                if scheduled_task:
                    task = scheduled_task.task
                    
                    # Calculate wait time
                    wait_time = (datetime.now() - scheduled_task.scheduled_time).total_seconds()
                    
                    # Update statistics
                    self.stats["total_dequeued"] += 1
                    self._update_average_wait_time(wait_time)
                    
                    # Record in history
                    self.task_history.append({
                        "task_id": task.id,
                        "dequeued_at": datetime.now(),
                        "wait_time": wait_time
                    })
                    
                    # Remove from pending
                    self.pending_tasks.pop(task.id, None)
                    
                    tasks.append(task)
                else:
                    break
            
            return tasks
    
    def dequeue_batch(self, task_type: Optional[str] = None) -> List[Task]:
        """
        Dequeue a batch of similar tasks.
        
        Args:
            task_type: Optional task type to filter by
            
        Returns:
            List of batched tasks
        """
        if not self.enable_batching:
            return self.dequeue(self.batch_size)
        
        with self._lock:
            if task_type and task_type in self.task_batches:
                batch = self.task_batches[task_type][:self.batch_size]
                self.task_batches[task_type] = self.task_batches[task_type][self.batch_size:]
                
                # Remove from scheduler
                for task in batch:
                    self.scheduler.remove_task(task.id)
                    self.pending_tasks.pop(task.id, None)
                
                return batch
            else:
                return self.dequeue(self.batch_size)
    
    def peek(self, count: int = 1) -> List[Task]:
        """
        Peek at next tasks without removing them.
        
        Args:
            count: Number of tasks to peek at
            
        Returns:
            List of tasks
        """
        with self._lock:
            # This is a simplified peek - actual implementation would
            # depend on the specific scheduler
            tasks = []
            temp_removed = []
            
            for _ in range(count):
                scheduled_task = self.scheduler.get_next_task()
                if scheduled_task:
                    tasks.append(scheduled_task.task)
                    temp_removed.append(scheduled_task)
                else:
                    break
            
            # Re-add tasks
            for scheduled_task in temp_removed:
                self.scheduler.add_task(scheduled_task)
            
            return tasks
    
    def remove(self, task_id: str) -> bool:
        """
        Remove a specific task from the queue.
        
        Args:
            task_id: Task ID to remove
            
        Returns:
            True if task was removed, False otherwise
        """
        with self._lock:
            if task_id in self.pending_tasks:
                success = self.scheduler.remove_task(task_id)
                if success:
                    del self.pending_tasks[task_id]
                return success
            return False
    
    def requeue(self, task: Task, increase_priority: bool = True) -> bool:
        """
        Requeue a task (e.g., after failure).
        
        Args:
            task: Task to requeue
            increase_priority: Whether to increase task priority
            
        Returns:
            True if task was requeued
        """
        with self._lock:
            # Increase priority if requested
            if increase_priority and task.priority.value > 0:
                task.priority = TaskPriority(task.priority.value - 1)
            
            return self.enqueue(task)
    
    def _add_to_batch(self, task: Task):
        """Add task to appropriate batch."""
        self.task_batches[task.type].append(task)
    
    def _update_average_wait_time(self, wait_time: float):
        """Update average wait time statistic."""
        alpha = 0.1  # Exponential moving average factor
        self.stats["average_wait_time"] = (
            alpha * wait_time +
            (1 - alpha) * self.stats["average_wait_time"]
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status and statistics."""
        with self._lock:
            return {
                "strategy": self.strategy.name,
                "current_size": self.scheduler.get_queue_size(),
                "max_size": self.max_size,
                "statistics": dict(self.stats),
                "pending_tasks": len(self.pending_tasks),
                "batch_sizes": {
                    task_type: len(batch)
                    for task_type, batch in self.task_batches.items()
                }
            }
    
    def clear(self):
        """Clear all tasks from the queue."""
        with self._lock:
            # Clear scheduler
            while self.scheduler.get_next_task():
                pass
            
            # Clear tracking
            self.pending_tasks.clear()
            self.task_batches.clear()
            
            log.info("Task queue cleared")


# Export main components
__all__ = [
    'TaskQueue',
    'TaskScheduler',
    'SchedulingStrategy',
    'ScheduledTask',
    'PriorityScheduler',
    'FIFOScheduler',
    'DeadlineScheduler',
    'AdaptiveScheduler'
]