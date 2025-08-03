#!/usr/bin/env python3
"""
Dynamic Agent Spawning and Role Assignment System

This module provides dynamic scaling and role management for multi-agent collaboration.
It can automatically spawn new agents based on workload, assign optimal roles based on
capabilities and requirements, and distribute tasks efficiently across the agent pool.

Features:
- Dynamic agent spawning and termination based on demand
- Intelligent role assignment using capability matching
- Load balancing with sophisticated task distribution
- Integration with AgentMessageBus and collaboration systems
- Automatic scaling policies and thresholds
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
import uuid
import importlib
import inspect

from .agent_message_bus import AgentMessageBus, MessageType
from .collaborative_decision_making import VotingSystem, DecisionType, VoteType

log = logging.getLogger(__name__)

class AgentState(Enum):
    """States an agent can be in."""
    SPAWNING = "spawning"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentTemplate:
    """Template for spawning agents."""
    agent_type: str
    class_name: str
    module_path: str
    capabilities: List[str]
    resource_requirements: Dict[str, Any]
    initialization_params: Dict[str, Any]
    max_concurrent_tasks: int = 5
    
@dataclass
class Task:
    """A task to be executed by an agent."""
    task_id: str
    task_type: str
    priority: TaskPriority
    requirements: List[str]  # Required capabilities
    content: Dict[str, Any]
    created_at: float = None
    deadline: Optional[float] = None
    assigned_agent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

@dataclass
class AgentInstance:
    """Information about a spawned agent instance."""
    agent_id: str
    agent_type: str
    template: AgentTemplate
    state: AgentState
    spawned_at: float
    last_activity: float
    current_tasks: List[str]
    completed_tasks: int
    failed_tasks: int
    capabilities: List[str]
    performance_score: float = 1.0
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}

class AgentSpawner:
    """
    Dynamically spawns and manages agent instances based on workload and requirements.
    
    Monitors system load and automatically creates or terminates agents to maintain
    optimal performance and resource utilization.
    """
    
    def __init__(self, message_bus: AgentMessageBus, voting_system: VotingSystem = None):
        """
        Initialize the agent spawner.
        
        Args:
            message_bus: AgentMessageBus for communication
            voting_system: Optional voting system for scaling decisions
        """
        self.message_bus = message_bus
        self.voting_system = voting_system
        
        # Agent management
        self.agent_templates: Dict[str, AgentTemplate] = {}
        self.active_agents: Dict[str, AgentInstance] = {}
        self.agent_objects: Dict[str, Any] = {}  # Actual agent instances
        
        # Scaling configuration
        self.min_agents_per_type: Dict[str, int] = {}
        self.max_agents_per_type: Dict[str, int] = {}
        self.scaling_thresholds = {
            "cpu_high": 80.0,
            "cpu_low": 20.0,
            "queue_high": 10,
            "queue_low": 2,
            "response_time_high": 5.0
        }
        
        # Task queue and load balancing
        self.task_queue = queue.PriorityQueue()
        self.task_history: List[Task] = []
        
        # Performance monitoring
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "agent_utilization": 0.0,
            "scaling_events": 0
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._load_balancing_task: Optional[asyncio.Task] = None
        self._running = False
        
        log.info("AgentSpawner initialized")
    
    async def start(self):
        """Start the agent spawning system."""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._load_balancing_task = asyncio.create_task(self._load_balancing_loop())
        
        # Subscribe to system events
        await self.message_bus.subscribe("agent.*", self._handle_agent_event, "spawner")
        await self.message_bus.subscribe("task.*", self._handle_task_event, "spawner")
        
        log.info("AgentSpawner started")
    
    async def stop(self):
        """Stop the agent spawning system."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._load_balancing_task:
            self._load_balancing_task.cancel()
        
        # Terminate all agents
        for agent_id in list(self.active_agents.keys()):
            await self.terminate_agent(agent_id)
        
        log.info("AgentSpawner stopped")
    
    def register_agent_template(self, template: AgentTemplate):
        """
        Register a template for spawning agents.
        
        Args:
            template: AgentTemplate defining how to create agents of this type
        """
        self.agent_templates[template.agent_type] = template
        
        # Set default scaling limits
        if template.agent_type not in self.min_agents_per_type:
            self.min_agents_per_type[template.agent_type] = 1
        if template.agent_type not in self.max_agents_per_type:
            self.max_agents_per_type[template.agent_type] = 10
        
        log.info(f"Registered agent template: {template.agent_type}")
    
    def set_scaling_limits(self, agent_type: str, min_agents: int, max_agents: int):
        """Set scaling limits for an agent type."""
        self.min_agents_per_type[agent_type] = min_agents
        self.max_agents_per_type[agent_type] = max_agents
        log.info(f"Set scaling limits for {agent_type}: {min_agents}-{max_agents}")
    
    async def spawn_agent(self, agent_type: str, reason: str = "manual") -> Optional[str]:
        """
        Spawn a new agent instance.
        
        Args:
            agent_type: Type of agent to spawn
            reason: Reason for spawning (for logging/monitoring)
            
        Returns:
            Agent ID if successful, None if failed
        """
        if agent_type not in self.agent_templates:
            log.error(f"No template found for agent type: {agent_type}")
            return None
        
        template = self.agent_templates[agent_type]
        
        # Check scaling limits
        current_count = len([a for a in self.active_agents.values() 
                           if a.agent_type == agent_type and a.state != AgentState.TERMINATED])
        
        if current_count >= self.max_agents_per_type.get(agent_type, 10):
            log.warning(f"Cannot spawn {agent_type}: at max limit ({current_count})")
            return None
        
        agent_id = f"{agent_type}_{int(time.time())}_{len(self.active_agents)}"
        
        try:
            # Create agent instance
            agent_instance = AgentInstance(
                agent_id=agent_id,
                agent_type=agent_type,
                template=template,
                state=AgentState.SPAWNING,
                spawned_at=time.time(),
                last_activity=time.time(),
                current_tasks=[],
                completed_tasks=0,
                failed_tasks=0,
                capabilities=template.capabilities.copy()
            )
            
            # Dynamically import and instantiate agent
            agent_object = await self._create_agent_object(template, agent_id)
            if not agent_object:
                log.error(f"Failed to create agent object for {agent_id}")
                return None
            
            # Register with systems
            self.active_agents[agent_id] = agent_instance
            self.agent_objects[agent_id] = agent_object
            
            # Register with message bus and voting
            await self.message_bus.register_agent(
                agent_id, agent_type, template.capabilities
            )
            
            if self.voting_system:
                self.voting_system.register_agent(
                    agent_id, agent_type, template.capabilities
                )
            
            # Update state
            agent_instance.state = AgentState.ACTIVE
            
            # Notify other agents
            await self.message_bus.publish(
                "agent.spawned",
                MessageType.AGENT_JOIN,
                {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "reason": reason,
                    "capabilities": template.capabilities
                },
                "spawner"
            )
            
            self.performance_metrics["scaling_events"] += 1
            log.info(f"Successfully spawned agent: {agent_id} (reason: {reason})")
            return agent_id
            
        except Exception as e:
            log.error(f"Error spawning agent {agent_id}: {e}")
            # Cleanup on failure
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            if agent_id in self.agent_objects:
                del self.agent_objects[agent_id]
            return None
    
    async def _create_agent_object(self, template: AgentTemplate, agent_id: str) -> Optional[Any]:
        """Dynamically create an agent object from template."""
        try:
            # Import the module
            module = importlib.import_module(template.module_path)
            
            # Get the class
            agent_class = getattr(module, template.class_name)
            
            # Prepare initialization parameters
            init_params = template.initialization_params.copy()
            init_params['name'] = agent_id
            init_params['capabilities'] = template.capabilities
            
            # Create instance
            if inspect.iscoroutinefunction(agent_class.__init__):
                agent_object = await agent_class(**init_params)
            else:
                agent_object = agent_class(**init_params)
            
            return agent_object
            
        except Exception as e:
            log.error(f"Failed to create agent object: {e}")
            return None
    
    async def terminate_agent(self, agent_id: str, reason: str = "manual") -> bool:
        """
        Terminate an agent instance.
        
        Args:
            agent_id: ID of agent to terminate
            reason: Reason for termination
            
        Returns:
            True if successful
        """
        if agent_id not in self.active_agents:
            log.warning(f"Cannot terminate unknown agent: {agent_id}")
            return False
        
        agent_instance = self.active_agents[agent_id]
        agent_instance.state = AgentState.TERMINATING
        
        try:
            # Reassign any current tasks
            if agent_instance.current_tasks:
                for task_id in agent_instance.current_tasks:
                    await self._reassign_task(task_id, agent_id)
            
            # Unregister from systems
            await self.message_bus.unregister_agent(agent_id)
            
            if self.voting_system:
                self.voting_system.unregister_agent(agent_id)
            
            # Cleanup agent object
            if agent_id in self.agent_objects:
                agent_obj = self.agent_objects[agent_id]
                if hasattr(agent_obj, 'cleanup'):
                    await agent_obj.cleanup()
                del self.agent_objects[agent_id]
            
            # Update state and remove
            agent_instance.state = AgentState.TERMINATED
            del self.active_agents[agent_id]
            
            # Notify other agents
            await self.message_bus.publish(
                "agent.terminated",
                MessageType.AGENT_LEAVE,
                {
                    "agent_id": agent_id,
                    "reason": reason,
                    "completed_tasks": agent_instance.completed_tasks,
                    "failed_tasks": agent_instance.failed_tasks
                },
                "spawner"
            )
            
            self.performance_metrics["scaling_events"] += 1
            log.info(f"Terminated agent: {agent_id} (reason: {reason})")
            return True
            
        except Exception as e:
            log.error(f"Error terminating agent {agent_id}: {e}")
            return False
    
    async def submit_task(self, task: Task) -> bool:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            True if task was queued successfully
        """
        try:
            # Add to queue with priority
            priority = -task.priority.value  # Negative for high priority first
            self.task_queue.put((priority, time.time(), task))
            
            self.task_history.append(task)
            
            log.info(f"Task submitted: {task.task_id} (priority: {task.priority.name})")
            
            # Trigger load balancing
            await self.message_bus.publish(
                "task.submitted",
                MessageType.TASK_ASSIGNMENT,
                {"task_id": task.task_id, "priority": task.priority.name},
                "spawner"
            )
            
            return True
            
        except Exception as e:
            log.error(f"Error submitting task {task.task_id}: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Background monitoring and scaling decisions."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                # Check if scaling is needed
                scale_decisions = await self._analyze_scaling_needs()
                
                for decision in scale_decisions:
                    if decision["action"] == "scale_up":
                        await self.spawn_agent(decision["agent_type"], decision["reason"])
                    elif decision["action"] == "scale_down":
                        await self._scale_down_agent_type(decision["agent_type"], decision["reason"])
                
                # Update performance metrics
                await self._update_performance_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
    
    async def _load_balancing_loop(self):
        """Background task assignment and load balancing."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Process pending tasks
                await self._process_task_queue()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in load balancing loop: {e}")
    
    async def _analyze_scaling_needs(self) -> List[Dict[str, Any]]:
        """Analyze current load and determine scaling needs."""
        decisions = []
        
        # Calculate current metrics
        queue_size = self.task_queue.qsize()
        total_agents = len(self.active_agents)
        
        # Check each agent type
        for agent_type, template in self.agent_templates.items():
            type_agents = [a for a in self.active_agents.values() 
                          if a.agent_type == agent_type and a.state == AgentState.ACTIVE]
            
            current_count = len(type_agents)
            min_count = self.min_agents_per_type.get(agent_type, 1)
            max_count = self.max_agents_per_type.get(agent_type, 10)
            
            # Calculate utilization
            busy_agents = sum(1 for a in type_agents if len(a.current_tasks) > 0)
            utilization = (busy_agents / current_count) if current_count > 0 else 0
            
            # Scale up conditions
            if (current_count < max_count and 
                (queue_size > self.scaling_thresholds["queue_high"] or
                 utilization > 0.8)):
                decisions.append({
                    "action": "scale_up",
                    "agent_type": agent_type,
                    "reason": f"high_load (queue: {queue_size}, util: {utilization:.1%})"
                })
            
            # Scale down conditions
            elif (current_count > min_count and 
                  queue_size < self.scaling_thresholds["queue_low"] and
                  utilization < 0.3):
                decisions.append({
                    "action": "scale_down", 
                    "agent_type": agent_type,
                    "reason": f"low_load (queue: {queue_size}, util: {utilization:.1%})"
                })
        
        return decisions
    
    async def _scale_down_agent_type(self, agent_type: str, reason: str):
        """Scale down agents of a specific type."""
        type_agents = [a for a in self.active_agents.values() 
                      if (a.agent_type == agent_type and 
                          a.state == AgentState.ACTIVE and
                          len(a.current_tasks) == 0)]  # Only terminate idle agents
        
        if type_agents:
            # Terminate the oldest idle agent
            oldest_agent = min(type_agents, key=lambda a: a.last_activity)
            await self.terminate_agent(oldest_agent.agent_id, reason)
    
    async def _process_task_queue(self):
        """Process pending tasks and assign to agents."""
        while not self.task_queue.empty():
            try:
                # Get next task
                priority, timestamp, task = self.task_queue.get_nowait()
                
                # Find best agent for the task
                best_agent = await self._find_best_agent_for_task(task)
                
                if best_agent:
                    await self._assign_task_to_agent(task, best_agent)
                else:
                    # No suitable agent available, put back in queue
                    self.task_queue.put((priority, timestamp, task))
                    break  # Stop processing to avoid infinite loop
                    
            except queue.Empty:
                break
            except Exception as e:
                log.error(f"Error processing task queue: {e}")
    
    async def _find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best agent to handle a task."""
        suitable_agents = []
        
        for agent_id, agent_instance in self.active_agents.items():
            # Check if agent is available
            if (agent_instance.state != AgentState.ACTIVE or
                len(agent_instance.current_tasks) >= agent_instance.template.max_concurrent_tasks):
                continue
            
            # Check capability match
            if not all(req in agent_instance.capabilities for req in task.requirements):
                continue
            
            # Calculate suitability score
            capability_score = len([cap for cap in agent_instance.capabilities 
                                  if cap in task.requirements]) / len(task.requirements)
            load_score = 1.0 - (len(agent_instance.current_tasks) / 
                               agent_instance.template.max_concurrent_tasks)
            performance_score = agent_instance.performance_score
            
            total_score = (capability_score * 0.5 + 
                          load_score * 0.3 + 
                          performance_score * 0.2)
            
            suitable_agents.append((agent_id, total_score))
        
        if suitable_agents:
            # Return agent with highest score
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    async def _assign_task_to_agent(self, task: Task, agent_id: str):
        """Assign a task to a specific agent."""
        agent_instance = self.active_agents[agent_id]
        agent_instance.current_tasks.append(task.task_id)
        agent_instance.last_activity = time.time()
        task.assigned_agent = agent_id
        
        # Send task to agent
        await self.message_bus.publish(
            f"agent.{agent_id}.task",
            MessageType.TASK_ASSIGNMENT,
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "priority": task.priority.name,
                "content": task.content,
                "deadline": task.deadline
            },
            "spawner"
        )
        
        log.info(f"Assigned task {task.task_id} to agent {agent_id}")
    
    async def _reassign_task(self, task_id: str, failed_agent_id: str):
        """Reassign a task from a failed agent."""
        # Find the task in history
        task = next((t for t in self.task_history if t.task_id == task_id), None)
        if task:
            task.assigned_agent = None
            await self.submit_task(task)
            log.info(f"Reassigned task {task_id} from failed agent {failed_agent_id}")
    
    async def _handle_agent_event(self, topic: str, message: dict):
        """Handle agent-related events."""
        # Update agent states based on events
        pass
    
    async def _handle_task_event(self, topic: str, message: dict):
        """Handle task-related events."""
        # Update task completion statistics
        if "completed" in topic:
            self.performance_metrics["tasks_completed"] += 1
        elif "failed" in topic:
            self.performance_metrics["tasks_failed"] += 1
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        total_tasks = sum(len(a.current_tasks) for a in self.active_agents.values())
        total_capacity = sum(a.template.max_concurrent_tasks for a in self.active_agents.values())
        
        self.performance_metrics["agent_utilization"] = (
            total_tasks / total_capacity if total_capacity > 0 else 0
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current spawner status."""
        agent_counts = {}
        for agent_type in self.agent_templates:
            count = len([a for a in self.active_agents.values() 
                        if a.agent_type == agent_type and a.state == AgentState.ACTIVE])
            agent_counts[agent_type] = count
        
        return {
            "total_agents": len(self.active_agents),
            "agent_counts_by_type": agent_counts,
            "pending_tasks": self.task_queue.qsize(),
            "completed_tasks": self.performance_metrics["tasks_completed"],
            "failed_tasks": self.performance_metrics["tasks_failed"],
            "agent_utilization": self.performance_metrics["agent_utilization"],
            "scaling_events": self.performance_metrics["scaling_events"],
            "templates_registered": len(self.agent_templates)
        }

# Example usage
async def main():
    """Example usage of the dynamic agent spawning system."""
    from .agent_message_bus import AgentMessageBus
    from .collaborative_decision_making import VotingSystem
    
    # Initialize systems
    message_bus = AgentMessageBus()
    voting_system = VotingSystem()
    spawner = AgentSpawner(message_bus, voting_system)
    
    await message_bus.start()
    await spawner.start()
    
    # Register agent templates
    coder_template = AgentTemplate(
        agent_type="coder",
        class_name="CodeAgent",
        module_path="special_agents.code_agent",
        capabilities=["python", "javascript", "code_generation"],
        resource_requirements={"memory": "512MB", "cpu": "1 core"},
        initialization_params={"timeout": 30},
        max_concurrent_tasks=3
    )
    
    spawner.register_agent_template(coder_template)
    spawner.set_scaling_limits("coder", 1, 5)
    
    # Spawn initial agents
    agent_id = await spawner.spawn_agent("coder", "initial_spawn")
    print(f"Spawned agent: {agent_id}")
    
    # Submit tasks
    task = Task(
        task_id="test_task_1",
        task_type="code_generation",
        priority=TaskPriority.HIGH,
        requirements=["python"],
        content={"description": "Create a hello world function"}
    )
    
    await spawner.submit_task(task)
    
    # Monitor for a bit
    await asyncio.sleep(5)
    
    # Get status
    status = spawner.get_status()
    print(f"Spawner status: {status}")
    
    await spawner.stop()
    await message_bus.stop()

if __name__ == "__main__":
    asyncio.run(main())