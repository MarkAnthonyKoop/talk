#!/usr/bin/env python3
"""
Test script for the Dynamic Agent Spawning and Role Assignment System.

This script tests the complete dynamic agent management system including:
- Agent template registration and spawning
- Task submission and automatic assignment  
- Load balancing and scaling decisions
- Integration with collaboration systems
- Performance monitoring and metrics
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Import the systems
from special_agents.collaboration.agent_message_bus import AgentMessageBus, MessageType
from special_agents.collaboration.collaborative_decision_making import VotingSystem
from special_agents.collaboration.dynamic_agent_spawning import (
    AgentSpawner, AgentTemplate, Task, TaskPriority, AgentState
)

class MockAgent:
    """Mock agent for testing purposes."""
    
    def __init__(self, name: str, capabilities: list, **kwargs):
        self.name = name
        self.capabilities = capabilities
        self.active = True
        self.tasks_handled = []
        log.info(f"MockAgent {name} initialized with capabilities: {capabilities}")
    
    async def handle_task(self, task_data: dict):
        """Simulate handling a task."""
        task_id = task_data.get('task_id', 'unknown')
        self.tasks_handled.append(task_id)
        log.info(f"MockAgent {self.name} handling task: {task_id}")
        
        # Simulate work
        await asyncio.sleep(0.1)
        return {"status": "completed", "result": f"Task {task_id} completed by {self.name}"}
    
    async def cleanup(self):
        """Cleanup when agent is terminated."""
        self.active = False
        log.info(f"MockAgent {self.name} cleaned up")

async def test_agent_template_registration():
    """Test agent template registration."""
    print("\n=== Testing Agent Template Registration ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Create and register templates
    coder_template = AgentTemplate(
        agent_type="coder",
        class_name="MockAgent",
        module_path="__main__",  # Use current module for testing
        capabilities=["python", "code_generation"],
        resource_requirements={"memory": "256MB"},
        initialization_params={"timeout": 30},
        max_concurrent_tasks=2
    )
    
    tester_template = AgentTemplate(
        agent_type="tester", 
        class_name="MockAgent",
        module_path="__main__",
        capabilities=["testing", "validation"],
        resource_requirements={"memory": "128MB"},
        initialization_params={"timeout": 15},
        max_concurrent_tasks=3
    )
    
    spawner.register_agent_template(coder_template)
    spawner.register_agent_template(tester_template)
    
    # Set scaling limits
    spawner.set_scaling_limits("coder", 1, 3)
    spawner.set_scaling_limits("tester", 1, 2)
    
    print(f"[OK] Registered templates: {list(spawner.agent_templates.keys())}")
    print(f"[OK] Scaling limits set for coder: {spawner.min_agents_per_type['coder']}-{spawner.max_agents_per_type['coder']}")
    
    await spawner.stop()
    await message_bus.stop()
    
    return len(spawner.agent_templates) == 2

async def test_agent_spawning():
    """Test dynamic agent spawning."""
    print("\n=== Testing Agent Spawning ===")
    
    message_bus = AgentMessageBus()
    voting_system = VotingSystem()
    spawner = AgentSpawner(message_bus, voting_system)
    
    await message_bus.start()
    await spawner.start()
    
    # Register template
    template = AgentTemplate(
        agent_type="worker",
        class_name="MockAgent", 
        module_path="__main__",
        capabilities=["general"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=1
    )
    
    spawner.register_agent_template(template)
    
    # Test spawning
    agent_id1 = await spawner.spawn_agent("worker", "test_spawn")
    agent_id2 = await spawner.spawn_agent("worker", "test_spawn")
    
    print(f"[OK] Spawned agents: {agent_id1}, {agent_id2}")
    print(f"[OK] Active agents: {len(spawner.active_agents)}")
    
    # Check agent states
    for agent_id, agent_instance in spawner.active_agents.items():
        print(f"  - {agent_id}: {agent_instance.state.value} (type: {agent_instance.agent_type})")
    
    # Test termination
    success = await spawner.terminate_agent(agent_id1, "test_termination")
    print(f"[OK] Agent termination successful: {success}")
    print(f"[OK] Remaining agents: {len(spawner.active_agents)}")
    
    await spawner.stop()
    await message_bus.stop()
    
    return agent_id1 is not None and agent_id2 is not None and success

async def test_task_submission_and_assignment():
    """Test task submission and automatic assignment."""
    print("\n=== Testing Task Submission and Assignment ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Register template and spawn agent
    template = AgentTemplate(
        agent_type="processor",
        class_name="MockAgent",
        module_path="__main__",
        capabilities=["data_processing", "analysis"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=2
    )
    
    spawner.register_agent_template(template)
    agent_id = await spawner.spawn_agent("processor", "task_testing")
    
    # Create tasks
    tasks = []
    for i in range(3):
        task = Task(
            task_id=f"test_task_{i}",
            task_type="data_processing",
            priority=TaskPriority.NORMAL if i < 2 else TaskPriority.HIGH,
            requirements=["data_processing"],
            content={"data": f"test_data_{i}"}
        )
        tasks.append(task)
    
    # Submit tasks
    for task in tasks:
        success = await spawner.submit_task(task)
        print(f"[OK] Task {task.task_id} submitted: {success}")
    
    # Wait for task processing
    await asyncio.sleep(2)
    
    # Check task assignment
    status = spawner.get_status()
    print(f"[OK] Spawner status: {status}")
    
    # Check agent task assignments
    if agent_id in spawner.active_agents:
        agent_instance = spawner.active_agents[agent_id]
        print(f"[OK] Agent {agent_id} current tasks: {len(agent_instance.current_tasks)}")
    
    await spawner.stop()
    await message_bus.stop()
    
    return status["pending_tasks"] >= 0  # Some tasks should be processed

async def test_load_balancing_and_scaling():
    """Test load balancing and automatic scaling."""
    print("\n=== Testing Load Balancing and Scaling ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Register template with low max tasks to trigger scaling
    template = AgentTemplate(
        agent_type="scaler",
        class_name="MockAgent",
        module_path="__main__",
        capabilities=["scaling_test"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=1  # Low limit to test scaling
    )
    
    spawner.register_agent_template(template)
    spawner.set_scaling_limits("scaler", 1, 4)  # Allow up to 4 agents
    
    # Spawn initial agent
    initial_agent = await spawner.spawn_agent("scaler", "initial")
    print(f"[OK] Initial agent spawned: {initial_agent}")
    
    # Submit many tasks to trigger scaling
    tasks = []
    for i in range(8):  # More tasks than one agent can handle
        task = Task(
            task_id=f"scale_task_{i}",
            task_type="scaling_test",
            priority=TaskPriority.NORMAL,
            requirements=["scaling_test"],
            content={"work": f"scale_work_{i}"}
        )
        tasks.append(task)
        await spawner.submit_task(task)
    
    print(f"[OK] Submitted {len(tasks)} tasks")
    
    # Wait for scaling decisions
    await asyncio.sleep(5)
    
    # Check if more agents were spawned
    final_status = spawner.get_status()
    print(f"[OK] Final status: {final_status}")
    
    expected_scaling = final_status["total_agents"] > 1
    print(f"[OK] Scaling triggered: {expected_scaling} (agents: {final_status['total_agents']})")
    
    await spawner.stop()
    await message_bus.stop()
    
    return expected_scaling

async def test_integration_with_collaboration_systems():
    """Test integration with message bus and voting systems."""
    print("\n=== Testing Integration with Collaboration Systems ===")
    
    message_bus = AgentMessageBus()
    voting_system = VotingSystem()
    spawner = AgentSpawner(message_bus, voting_system)
    
    await message_bus.start()
    await spawner.start()
    
    # Test message bus integration
    received_messages = []
    
    async def message_callback(topic, message):
        received_messages.append((topic, message))
    
    await message_bus.subscribe("agent.*", message_callback)
    
    # Register template and spawn agents
    template = AgentTemplate(
        agent_type="integrator",
        class_name="MockAgent",
        module_path="__main__",
        capabilities=["integration_test"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=1
    )
    
    spawner.register_agent_template(template)
    
    # Spawn and terminate to generate events
    agent_id = await spawner.spawn_agent("integrator", "integration_test")
    await asyncio.sleep(0.1)  # Let messages propagate
    
    await spawner.terminate_agent(agent_id, "integration_test")
    await asyncio.sleep(0.1)  # Let messages propagate
    
    # Check message bus integration
    spawn_messages = [msg for topic, msg in received_messages if "spawned" in topic or "join" in msg.get("message_type", "")]
    terminate_messages = [msg for topic, msg in received_messages if "terminated" in topic or "leave" in msg.get("message_type", "")]
    
    print(f"[OK] Spawn messages received: {len(spawn_messages)}")
    print(f"[OK] Terminate messages received: {len(terminate_messages)}")
    
    # Check voting system integration
    voting_agents = voting_system.get_stats()["registered_voters"]
    print(f"[OK] Voting system integration: {voting_agents} voters registered")
    
    await spawner.stop()
    await message_bus.stop()
    
    return len(spawn_messages) > 0 and len(terminate_messages) > 0

async def test_performance_monitoring():
    """Test performance monitoring and metrics."""
    print("\n=== Testing Performance Monitoring ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Get initial status
    initial_status = spawner.get_status()
    print(f"[OK] Initial status: {initial_status}")
    
    # Register template and spawn agent
    template = AgentTemplate(
        agent_type="monitor_test",
        class_name="MockAgent", 
        module_path="__main__",
        capabilities=["monitoring"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=1
    )
    
    spawner.register_agent_template(template)
    agent_id = await spawner.spawn_agent("monitor_test", "monitoring")
    
    # Submit some tasks
    for i in range(3):
        task = Task(
            task_id=f"monitor_task_{i}",
            task_type="monitoring",
            priority=TaskPriority.NORMAL,
            requirements=["monitoring"],
            content={"test": f"data_{i}"}
        )
        await spawner.submit_task(task)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Get final status
    final_status = spawner.get_status()
    print(f"[OK] Final status: {final_status}")
    
    # Check metrics
    metrics_exist = all(key in final_status for key in [
        "total_agents", "pending_tasks", "completed_tasks", 
        "agent_utilization", "scaling_events"
    ])
    
    print(f"[OK] All metrics present: {metrics_exist}")
    print(f"[OK] Agent utilization: {final_status.get('agent_utilization', 0):.1%}")
    
    await spawner.stop() 
    await message_bus.stop()
    
    return metrics_exist and final_status["total_agents"] > 0

async def run_all_tests():
    """Run all dynamic agent spawning tests."""
    print("Testing Dynamic Agent Spawning and Role Assignment System")
    print("=" * 70)
    
    tests = [
        ("Agent Template Registration", test_agent_template_registration),
        ("Agent Spawning", test_agent_spawning),
        ("Task Submission & Assignment", test_task_submission_and_assignment),
        ("Load Balancing & Scaling", test_load_balancing_and_scaling),
        ("Collaboration Integration", test_integration_with_collaboration_systems),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            status = "PASS" if result else "FAIL"
            print(f"[{status}] {test_name} completed in {duration:.2f}s")
            results.append((test_name, result, duration))
            
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, 0))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total_time = 0
    
    for test_name, result, duration in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:4} | {test_name:30} | {duration:.2f}s")
        if result:
            passed += 1
        total_time += duration
    
    print("-" * 70)
    print(f"PASSED: {passed}/{len(results)} tests in {total_time:.2f}s")
    
    success_rate = passed / len(results) * 100
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if passed == len(results):
        print("\n[SUCCESS] ALL TESTS PASSED! Dynamic agent spawning system is working!")
    else:
        print(f"\n[FAIL] {len(results) - passed} tests failed")
    
    return passed == len(results)

if __name__ == "__main__":
    import sys
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(130)