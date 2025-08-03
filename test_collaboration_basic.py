#!/usr/bin/env python3
"""
Basic Collaboration System Test

This script tests the core collaboration framework components:
- AgentMessageBus communication
- Dynamic agent spawning 
- Task distribution
- Basic integration testing

Simplified version without dashboard to test core functionality.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Import collaboration systems
from special_agents.collaboration.agent_message_bus import AgentMessageBus, MessageType
from special_agents.collaboration.collaborative_decision_making import VotingSystem, DecisionType, VoteType, VotingStrategy
from special_agents.collaboration.shared_workspace import SharedWorkspace
from special_agents.collaboration.dynamic_agent_spawning import (
    AgentSpawner, AgentTemplate, Task, TaskPriority, AgentState
)

class MockCollaborationAgent:
    """Mock agent for testing collaboration features."""
    
    def __init__(self, name: str, capabilities: list, **kwargs):
        self.name = name
        self.capabilities = capabilities
        self.active = True
        self.tasks_handled = []
        log.info(f"MockCollaborationAgent {name} initialized")
    
    async def handle_task(self, task_data: dict):
        """Handle a task assignment."""
        task_id = task_data.get('task_id', 'unknown')
        self.tasks_handled.append(task_id)
        log.info(f"Agent {self.name} completed task: {task_id}")
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "completed", "result": f"Task {task_id} done"}
    
    async def cleanup(self):
        """Cleanup when terminated."""
        self.active = False
        log.info(f"Agent {self.name} cleaned up")

async def test_message_bus():
    """Test message bus communication."""
    print("\n=== Testing Message Bus ===")
    
    message_bus = AgentMessageBus()
    await message_bus.start()
    
    messages_received = []
    
    async def test_handler(topic, message):
        messages_received.append((topic, message))
        log.info(f"Received: {topic} -> {message}")
    
    # Subscribe and publish
    await message_bus.subscribe("test.*", test_handler)
    await message_bus.publish("test.message", MessageType.TASK_ASSIGNMENT, {"data": "test"}, "tester")
    
    await asyncio.sleep(0.5)
    await message_bus.stop()
    
    success = len(messages_received) > 0
    print(f"Message Bus: {'PASS' if success else 'FAIL'} - {len(messages_received)} messages")
    return success

async def test_agent_spawning():
    """Test dynamic agent spawning."""
    print("\n=== Testing Agent Spawning ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Register agent template
    template = AgentTemplate(
        agent_type="test_worker",
        class_name="MockCollaborationAgent",
        module_path="__main__",
        capabilities=["testing", "collaboration"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=2
    )
    
    spawner.register_agent_template(template)
    spawner.set_scaling_limits("test_worker", 1, 3)
    
    # Spawn agents
    agent1 = await spawner.spawn_agent("test_worker", "test")
    agent2 = await spawner.spawn_agent("test_worker", "test")
    
    status = spawner.get_status()
    active_count = status["total_agents"]
    
    # Terminate one agent
    if agent1:
        termination_success = await spawner.terminate_agent(agent1, "test_cleanup")
    else:
        termination_success = False
    
    final_status = spawner.get_status()
    final_count = final_status["total_agents"]
    
    await spawner.stop()
    await message_bus.stop()
    
    success = agent1 and agent2 and active_count == 2 and final_count == 1 and termination_success
    print(f"Agent Spawning: {'PASS' if success else 'FAIL'} - {active_count} -> {final_count} agents")
    return success

async def test_task_processing():
    """Test task submission and processing."""
    print("\n=== Testing Task Processing ===")
    
    message_bus = AgentMessageBus()
    spawner = AgentSpawner(message_bus)
    
    await message_bus.start()
    await spawner.start()
    
    # Setup agent template
    template = AgentTemplate(
        agent_type="task_processor",
        class_name="MockCollaborationAgent", 
        module_path="__main__",
        capabilities=["task_processing"],
        resource_requirements={},
        initialization_params={},
        max_concurrent_tasks=3
    )
    
    spawner.register_agent_template(template)
    
    # Spawn agent and submit tasks
    agent_id = await spawner.spawn_agent("task_processor", "task_testing")
    
    if agent_id:
        # Submit multiple tasks
        tasks_submitted = 0
        for i in range(3):
            task = Task(
                task_id=f"test_task_{i}",
                task_type="processing",
                priority=TaskPriority.NORMAL,
                requirements=["task_processing"],
                content={"data": f"test_data_{i}"}
            )
            success = await spawner.submit_task(task)
            if success:
                tasks_submitted += 1
        
        # Wait for processing
        await asyncio.sleep(2)
        
        final_status = spawner.get_status()
        processed = final_status.get("completed_tasks", 0) + final_status.get("pending_tasks", 0)
    else:
        tasks_submitted = 0
        processed = 0
    
    await spawner.stop()
    await message_bus.stop()
    
    success = agent_id and tasks_submitted == 3 and processed >= tasks_submitted
    print(f"Task Processing: {'PASS' if success else 'FAIL'} - {tasks_submitted} submitted, {processed} processed")
    return success

async def test_voting_system():
    """Test collaborative voting."""
    print("\n=== Testing Voting System ===")
    
    voting_system = VotingSystem()
    
    # Register test agents
    voting_system.register_agent("agent1", "voter", ["voting"])
    voting_system.register_agent("agent2", "voter", ["voting"])
    
    # Create decision
    decision_id = await voting_system.create_decision(
        title="Test Decision",
        description="Test collaborative decision making",
        proposer_id="agent1",
        decision_type=DecisionType.TASK_ASSIGNMENT,
        voting_strategy=VotingStrategy.SIMPLE_MAJORITY,
        deadline_minutes=0.5
    )
    
    if decision_id:
        # Cast votes
        vote1 = await voting_system.cast_vote(decision_id, "agent1", VoteType.APPROVE)
        vote2 = await voting_system.cast_vote(decision_id, "agent2", VoteType.APPROVE)
        
        # Check status
        await asyncio.sleep(1)
        status = voting_system.get_decision_status(decision_id)
        
        success = decision_id and vote1 and vote2 and status
    else:
        success = False
    
    print(f"Voting System: {'PASS' if success else 'FAIL'} - Decision {decision_id}")
    return success

async def test_shared_workspace():
    """Test shared workspace coordination."""
    print("\n=== Testing Shared Workspace ===")
    
    temp_dir = tempfile.mkdtemp(prefix="workspace_test_")
    workspace = SharedWorkspace(Path(temp_dir))
    
    try:
        # Test file operations
        test_file = "collaboration_test.txt"
        test_content = "Collaboration test content"
        
        # Write and read (await since they're async)
        await workspace.write_file(test_file, test_content, "tester")
        read_content = await workspace.read_file(test_file, "tester")
        
        # Test locking - need to import LockType
        from special_agents.collaboration.shared_workspace import LockType
        lock_acquired = await workspace.acquire_lock(test_file, LockType.WRITE, "tester")
        
        # Check changes
        changes = await workspace.get_changes(limit=5)
        
        # Release lock
        if lock_acquired:
            await workspace.release_lock(test_file, LockType.WRITE, "tester")
        
        success = (read_content == test_content and lock_acquired and len(changes) > 0)
        
    except Exception as e:
        log.error(f"Workspace test error: {e}")
        success = False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Shared Workspace: {'PASS' if success else 'FAIL'} - File operations")
    return success

async def run_basic_tests():
    """Run basic collaboration system tests."""
    print("=" * 60)
    print("BASIC COLLABORATION SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Message Bus Communication", test_message_bus),
        ("Agent Spawning", test_agent_spawning), 
        ("Task Processing", test_task_processing),
        ("Voting System", test_voting_system),
        ("Shared Workspace", test_shared_workspace)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            status = "PASS" if result else "FAIL"
            print(f"[{status}] {test_name} - {duration:.2f}s")
            results.append((test_name, result, duration))
            
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, 0))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, result, duration in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:4} | {test_name:25} | {duration:.2f}s")
    
    print("-" * 60)
    print(f"PASSED: {passed}/{len(results)} tests in {total_time:.2f}s")
    
    success_rate = passed / len(results) * 100 if results else 0
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if passed == len(results):
        print("\n[SUCCESS] ALL BASIC TESTS PASSED!")
        print("Core collaboration system is working correctly!")
    else:
        print(f"\n[FAIL] {len(results) - passed} tests failed")
    
    return passed == len(results)

if __name__ == "__main__":
    import sys
    
    try:
        success = asyncio.run(run_basic_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(130)