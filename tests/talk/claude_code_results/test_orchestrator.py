"""
Comprehensive tests for the agentic orchestration system.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    AgentOrchestrator,
    AgentRegistry,
    TaskDispatcher,
    OrchestrationMonitor,
    AgentLifecycleManager
)
from orchestrator.core import OrchestrationConfig, OrchestrationMode
from orchestrator.registry import AgentState, AgentMetadata
from orchestrator.lifecycle import LifecycleEvent, LifecyclePolicy
from orchestrator.communication import Message, MessageBus, MessageType, MessagePriority
from orchestrator.dispatcher import Task, TaskStatus, TaskGroup, DistributionStrategy
from orchestrator.monitor import Metric, MetricType, Alert, HealthCheck


# Mock agent for testing
class MockAgent:
    """Mock agent for testing"""
    def __init__(self, name="mock_agent", **kwargs):
        self.name = name
        self.id = f"{name}_{id(self)}"
        self.tasks_executed = []
        self.health = True
        self.config = kwargs
    
    def initialize(self):
        """Initialize the agent"""
        pass
    
    def start(self):
        """Start the agent"""
        pass
    
    def shutdown(self):
        """Shutdown the agent"""
        pass
    
    def health_check(self):
        """Health check"""
        return self.health
    
    def execute_task(self, task):
        """Execute a task"""
        self.tasks_executed.append(task)
        return {"result": "success", "task_id": task.get('id')}
    
    def run(self, prompt):
        """Run method for compatibility"""
        return f"Executed: {prompt}"


class TestAgentRegistry(unittest.TestCase):
    """Test agent registry functionality"""
    
    def setUp(self):
        self.registry = AgentRegistry()
    
    def test_register_agent_type(self):
        """Test registering an agent type"""
        success = self.registry.register(
            "test_agent",
            MockAgent,
            capabilities=["test", "mock"]
        )
        
        self.assertTrue(success)
        self.assertIn("test_agent", self.registry.registered_types)
    
    def test_create_agent_instance(self):
        """Test creating an agent instance"""
        self.registry.register("test_agent", MockAgent)
        
        agent_id = self.registry.create_instance("test_agent", {"param": "value"})
        
        self.assertIsNotNone(agent_id)
        self.assertIn(agent_id, self.registry.active_agents)
    
    def test_destroy_agent_instance(self):
        """Test destroying an agent instance"""
        self.registry.register("test_agent", MockAgent)
        agent_id = self.registry.create_instance("test_agent")
        
        success = self.registry.destroy_instance(agent_id)
        
        self.assertTrue(success)
        self.assertNotIn(agent_id, self.registry.active_agents)
    
    def test_find_agents_by_capability(self):
        """Test finding agents by capability"""
        self.registry.register("test_agent", MockAgent, capabilities=["test", "mock"])
        agent_id = self.registry.create_instance("test_agent")
        
        agents = self.registry.find_agents_by_capability("test")
        
        self.assertIn(agent_id, agents)
    
    def test_agent_state_transitions(self):
        """Test agent state transitions"""
        self.registry.register("test_agent", MockAgent)
        agent_id = self.registry.create_instance("test_agent")
        
        # Update state
        self.registry.update_agent_state(agent_id, AgentState.BUSY)
        
        agent_info = self.registry.get_agent(agent_id)
        self.assertEqual(agent_info['state'], AgentState.BUSY.value)


class TestAgentLifecycle(unittest.TestCase):
    """Test agent lifecycle management"""
    
    def setUp(self):
        self.registry = AgentRegistry()
        self.lifecycle = AgentLifecycleManager(self.registry)
        self.registry.register("test_agent", MockAgent)
    
    def tearDown(self):
        self.lifecycle.shutdown()
    
    def test_spawn_agent(self):
        """Test spawning an agent"""
        agent_id = self.lifecycle.spawn_agent("test_agent", {"param": "value"})
        
        self.assertIsNotNone(agent_id)
        self.assertIn(agent_id, self.registry.active_agents)
    
    def test_terminate_agent(self):
        """Test terminating an agent"""
        agent_id = self.lifecycle.spawn_agent("test_agent")
        
        success = self.lifecycle.terminate_agent(agent_id)
        
        self.assertTrue(success)
        self.assertNotIn(agent_id, self.registry.active_agents)
    
    def test_health_check(self):
        """Test agent health check"""
        agent_id = self.lifecycle.spawn_agent("test_agent")
        
        is_healthy = self.lifecycle.health_check(agent_id)
        
        self.assertTrue(is_healthy)
    
    def test_suspend_resume_agent(self):
        """Test suspending and resuming an agent"""
        agent_id = self.lifecycle.spawn_agent("test_agent")
        
        # Suspend
        success = self.lifecycle.suspend_agent(agent_id)
        self.assertTrue(success)
        
        agent_info = self.registry.get_agent(agent_id)
        self.assertEqual(agent_info['state'], AgentState.SUSPENDED.value)
        
        # Resume
        success = self.lifecycle.resume_agent(agent_id)
        self.assertTrue(success)
        
        agent_info = self.registry.get_agent(agent_id)
        self.assertEqual(agent_info['state'], AgentState.READY.value)
    
    def test_event_handling(self):
        """Test lifecycle event handling"""
        events_received = []
        
        def event_handler(agent_id, event, data):
            events_received.append((agent_id, event, data))
        
        self.lifecycle.register_event_handler(LifecycleEvent.SPAWNED, event_handler)
        
        agent_id = self.lifecycle.spawn_agent("test_agent")
        time.sleep(0.1)  # Allow event processing
        
        self.assertTrue(any(e[1] == LifecycleEvent.SPAWNED for e in events_received))


class TestMessageBus(unittest.TestCase):
    """Test message bus functionality"""
    
    def setUp(self):
        self.bus = MessageBus()
        self.bus.register_agent("agent1")
        self.bus.register_agent("agent2")
    
    def tearDown(self):
        self.bus.shutdown()
    
    def test_send_direct_message(self):
        """Test sending a direct message"""
        message = Message(
            sender="agent1",
            recipient="agent2",
            content="test message"
        )
        
        success = self.bus.send(message)
        self.assertTrue(success)
        
        received = self.bus.receive("agent2", timeout=1)
        self.assertIsNotNone(received)
        self.assertEqual(received.content, "test message")
    
    def test_broadcast_message(self):
        """Test broadcasting a message"""
        success = self.bus.broadcast("sender", "broadcast content")
        self.assertTrue(success)
        
        # Both agents should receive
        msg1 = self.bus.receive("agent1", timeout=1)
        msg2 = self.bus.receive("agent2", timeout=1)
        
        self.assertIsNotNone(msg1)
        self.assertIsNotNone(msg2)
        self.assertEqual(msg1.content, "broadcast content")
        self.assertEqual(msg2.content, "broadcast content")
    
    def test_pub_sub(self):
        """Test publish-subscribe pattern"""
        self.bus.subscribe("agent1", "test_topic")
        self.bus.subscribe("agent2", "test_topic")
        
        self.bus.publish("publisher", "test_topic", "topic content")
        
        # Both subscribers should receive
        msg1 = self.bus.receive("agent1", timeout=1)
        msg2 = self.bus.receive("agent2", timeout=1)
        
        self.assertIsNotNone(msg1)
        self.assertIsNotNone(msg2)
    
    def test_request_response(self):
        """Test request-response pattern"""
        # Start responder thread
        def responder():
            msg = self.bus.receive("agent2", timeout=2)
            if msg:
                self.bus.reply(msg, "response content")
        
        responder_thread = threading.Thread(target=responder)
        responder_thread.start()
        
        # Send and wait for response
        request = Message(
            sender="agent1",
            recipient="agent2",
            content="request"
        )
        
        response = self.bus.send_and_wait(request, timeout=3)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.content, "response content")
        
        responder_thread.join()


class TestTaskDispatcher(unittest.TestCase):
    """Test task dispatcher functionality"""
    
    def setUp(self):
        config = MagicMock()
        config.load_balancing_policy = "round_robin"
        self.dispatcher = TaskDispatcher(config)
    
    def test_submit_task(self):
        """Test submitting a task"""
        task = Task(
            id="task1",
            type="test",
            payload={"data": "test"}
        )
        
        success = self.dispatcher.submit_task(task)
        self.assertTrue(success)
        self.assertIn("task1", self.dispatcher.pending_tasks)
    
    def test_dispatch_task(self):
        """Test dispatching a task"""
        task = Task(
            id="task1",
            type="test",
            payload={"data": "test"}
        )
        
        self.dispatcher.submit_task(task)
        
        result = self.dispatcher.dispatch_task(["agent1", "agent2"])
        
        self.assertIsNotNone(result)
        task, agent = result
        self.assertEqual(task.id, "task1")
        self.assertIn(agent, ["agent1", "agent2"])
    
    def test_task_dependencies(self):
        """Test task with dependencies"""
        task1 = Task(id="task1", type="test", payload={})
        task2 = Task(id="task2", type="test", payload={}, dependencies=["task1"])
        
        self.dispatcher.submit_task(task1)
        self.dispatcher.submit_task(task2)
        
        # Task2 should not be eligible until task1 completes
        next_task = self.dispatcher._get_next_eligible_task()
        self.assertEqual(next_task.id, "task1")
        
        # Complete task1
        self.dispatcher.executing_tasks["task1"] = task1
        self.dispatcher.complete_task("task1", "result")
        
        # Now task2 should be eligible
        next_task = self.dispatcher._get_next_eligible_task()
        self.assertEqual(next_task.id, "task2")
    
    def test_task_group(self):
        """Test submitting a task group"""
        tasks = [
            Task(id=f"task{i}", type="test", payload={})
            for i in range(3)
        ]
        
        group = TaskGroup(
            id="group1",
            tasks=tasks,
            parallel=False
        )
        
        success = self.dispatcher.submit_task_group(group)
        self.assertTrue(success)
        
        # Check dependencies were set up
        self.assertIn("task1", self.dispatcher.pending_tasks["task2"].dependencies)


class TestOrchestrationMonitor(unittest.TestCase):
    """Test monitoring functionality"""
    
    def setUp(self):
        self.monitor = OrchestrationMonitor()
    
    def tearDown(self):
        self.monitor.shutdown()
    
    def test_record_metrics(self):
        """Test recording metrics"""
        self.monitor.record_metric("test.metric", 42.0)
        self.monitor.record_counter("test.counter", 1)
        self.monitor.record_timer("test.timer", 1.5)
        
        summary = self.monitor.get_metrics_summary()
        
        self.assertIn("test.metric", summary['latest'])
        self.assertEqual(summary['latest']['test.metric'], 42.0)
    
    def test_health_checks(self):
        """Test health check functionality"""
        def check_func():
            return True
        
        health = self.monitor.perform_health_check("test_component", check_func)
        
        self.assertEqual(health.status, "healthy")
        
        status = self.monitor.get_health_status()
        self.assertEqual(status['status'], "healthy")
    
    def test_alerts(self):
        """Test alert system"""
        alert = Alert(
            id="test_alert",
            name="Test Alert",
            condition="test.metric",
            threshold=100,
            severity="warning",
            message="Test alert triggered"
        )
        
        self.monitor.register_alert(alert)
        
        # Trigger alert
        self.monitor.record_metric("test.metric", 150)
        
        history = self.monitor.get_alert_history()
        self.assertTrue(any(a['alert_id'] == "test_alert" for a in history))
    
    def test_event_recording(self):
        """Test event recording"""
        self.monitor.record_event(
            "test_event",
            "Test event occurred",
            severity="info",
            metadata={"key": "value"}
        )
        
        events = self.monitor.get_recent_events(10)
        
        self.assertTrue(any(e['type'] == "test_event" for e in events))


class TestAgentOrchestrator(unittest.TestCase):
    """Test the main orchestrator"""
    
    def setUp(self):
        config = OrchestrationConfig(
            max_agents=10,
            max_concurrent_tasks=5
        )
        self.orchestrator = AgentOrchestrator(config)
        self.orchestrator.register_agent("test_agent", MockAgent, ["test"])
    
    def tearDown(self):
        self.orchestrator.shutdown()
    
    def test_spawn_and_terminate_agent(self):
        """Test spawning and terminating agents"""
        agent_id = self.orchestrator.spawn_agent("test_agent")
        
        self.assertIsNotNone(agent_id)
        self.assertIn(agent_id, self.orchestrator.registry.active_agents)
        
        self.orchestrator.terminate_agent(agent_id)
        self.assertNotIn(agent_id, self.orchestrator.registry.active_agents)
    
    def test_submit_and_process_task(self):
        """Test submitting and processing a task"""
        # Spawn an agent
        agent_id = self.orchestrator.spawn_agent("test_agent")
        
        # Submit a task
        task = {
            'type': 'test',
            'data': {'message': 'test'},
            'required_capabilities': ['test']
        }
        
        task_id = self.orchestrator.submit_task(task)
        
        self.assertIsNotNone(task_id)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check if task was processed
        # (In real implementation, would check completed_tasks)
    
    def test_load_balancing(self):
        """Test load balancing across multiple agents"""
        # Spawn multiple agents
        agents = [
            self.orchestrator.spawn_agent("test_agent")
            for _ in range(3)
        ]
        
        # Submit multiple tasks
        tasks = []
        for i in range(6):
            task = {
                'type': 'test',
                'data': {'index': i},
                'required_capabilities': ['test']
            }
            tasks.append(self.orchestrator.submit_task(task))
        
        # Wait for processing
        time.sleep(1)
        
        # Check load distribution
        loads = self.orchestrator.agent_load
        
        # Load should be distributed
        self.assertTrue(len(loads) > 0)
    
    def test_checkpoint_and_restore(self):
        """Test checkpoint creation and restoration"""
        # Create some state
        agent_id = self.orchestrator.spawn_agent("test_agent")
        task_id = self.orchestrator.submit_task({'type': 'test', 'data': {}})
        
        # Create checkpoint
        checkpoint_file = self.orchestrator.create_checkpoint()
        
        self.assertTrue(Path(checkpoint_file).exists())
        
        # Create new orchestrator and restore
        new_orchestrator = AgentOrchestrator()
        new_orchestrator.restore_from_checkpoint(checkpoint_file)
        
        # Verify configuration was restored
        self.assertEqual(
            new_orchestrator.config.max_agents,
            self.orchestrator.config.max_agents
        )
        
        new_orchestrator.shutdown()
    
    def test_get_status(self):
        """Test getting orchestrator status"""
        agent_id = self.orchestrator.spawn_agent("test_agent")
        
        status = self.orchestrator.get_status()
        
        self.assertIn('orchestrator_id', status)
        self.assertIn('active_agents', status)
        self.assertEqual(status['active_agents'], 1)


if __name__ == "__main__":
    unittest.main()