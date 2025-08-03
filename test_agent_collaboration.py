#!/usr/bin/env python3
"""
Test script for the real-time agent collaboration system.

This script demonstrates the complete agent collaboration system including:
- Agent message bus for real-time communication
- Shared workspace coordination with file locking
- Collaborative decision making with voting
- Multi-agent workflow coordination

Features tested:
- Multiple agents working simultaneously
- Message publishing and subscription
- File operations with conflict prevention
- Voting on collaborative decisions
- Real-time status monitoring
"""

import asyncio
import logging
import time
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Import collaboration components
from special_agents.collaboration.agent_message_bus import (
    AgentMessageBus, MessageType, Message
)
from special_agents.collaboration.shared_workspace import (
    SharedWorkspace, LockType
)
from special_agents.collaboration.collaborative_decision_making import (
    VotingSystem, DecisionType, VoteType, VotingStrategy
)

class CollaborativeAgent:
    """A simulated agent that participates in collaboration."""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: list):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.message_bus = None
        self.workspace = None
        self.voting_system = None
        self.active = False
        
        # Agent state
        self.current_task = None
        self.messages_received = []
        self.files_worked_on = []
        self.votes_cast = []
    
    async def connect_to_systems(self, message_bus: AgentMessageBus, 
                                workspace: SharedWorkspace, 
                                voting_system: VotingSystem):
        """Connect agent to collaboration systems."""
        self.message_bus = message_bus
        self.workspace = workspace
        self.voting_system = voting_system
        
        # Register with systems
        await self.message_bus.register_agent(
            self.agent_id, self.agent_type, self.capabilities
        )
        
        self.voting_system.register_agent(
            self.agent_id, self.agent_type, self.capabilities, weight=1.0
        )
        
        # Subscribe to relevant message topics
        await self.message_bus.subscribe("task.*", self._handle_task_message, self.agent_id)
        await self.message_bus.subscribe("decision.*", self._handle_decision_message, self.agent_id)
        await self.message_bus.subscribe("workspace.*", self._handle_workspace_message, self.agent_id)
        
        self.active = True
        log.info(f"Agent {self.agent_id} connected to collaboration systems")
    
    async def _handle_task_message(self, topic: str, message: dict):
        """Handle task-related messages."""
        self.messages_received.append(message)
        
        if message.get('content', {}).get('assigned_to') == self.agent_id:
            task = message['content'].get('task_description', '')
            self.current_task = task
            log.info(f"Agent {self.agent_id} received task assignment: {task}")
            
            # Acknowledge task assignment
            await self.message_bus.publish(
                "task.acknowledged",
                MessageType.STATUS_UPDATE,
                {
                    "agent_id": self.agent_id,
                    "task": task,
                    "status": "accepted"
                },
                self.agent_id
            )
    
    async def _handle_decision_message(self, topic: str, message: dict):
        """Handle decision-related messages."""
        self.messages_received.append(message)
        
        if message['message_type'] == MessageType.DECISION_REQUEST.value:
            decision_id = message['content'].get('decision_id')
            if decision_id:
                # Simulate decision making logic
                vote = await self._make_decision(decision_id, message['content'])
                if vote:
                    self.votes_cast.append(vote)
    
    async def _handle_workspace_message(self, topic: str, message: dict):
        """Handle workspace-related messages."""
        self.messages_received.append(message)
        
        if message['message_type'] == MessageType.CODE_CHANGE.value:
            file_path = message['content'].get('file_path')
            if file_path:
                log.info(f"Agent {self.agent_id} notified of change to {file_path}")
    
    async def _make_decision(self, decision_id: str, decision_content: dict) -> bool:
        """Make a decision on a voting matter."""
        # Simulate decision logic based on agent type and capabilities
        if self.agent_type == "coder":
            # Coders generally approve code-related decisions
            if decision_content.get('decision_type') == DecisionType.CODE_MERGE.value:
                vote_type = VoteType.APPROVE
            else:
                vote_type = VoteType.ABSTAIN
        elif self.agent_type == "tester":
            # Testers are more cautious
            vote_type = VoteType.REJECT if "untested" in str(decision_content) else VoteType.APPROVE
        else:
            # Default to approval
            vote_type = VoteType.APPROVE
        
        reasoning = f"Decision made by {self.agent_type} agent based on capabilities"
        
        success = await self.voting_system.cast_vote(
            decision_id, self.agent_id, vote_type, reasoning
        )
        
        if success:
            log.info(f"Agent {self.agent_id} voted {vote_type.value} on decision {decision_id}")
        
        return success
    
    async def work_on_file(self, file_path: str, content: str) -> bool:
        """Simulate working on a file."""
        if not self.workspace:
            return False
        
        try:
            # Try to write to the file
            success = await self.workspace.write_file(file_path, content, self.agent_id)
            
            if success:
                self.files_worked_on.append(file_path)
                
                # Notify other agents of the change
                await self.message_bus.publish(
                    "workspace.file_changed",
                    MessageType.CODE_CHANGE,
                    {
                        "file_path": file_path,
                        "agent_id": self.agent_id,
                        "change_type": "modify"
                    },
                    self.agent_id
                )
                
                log.info(f"Agent {self.agent_id} successfully modified {file_path}")
            else:
                log.warning(f"Agent {self.agent_id} failed to modify {file_path} (lock conflict?)")
            
            return success
            
        except Exception as e:
            log.error(f"Agent {self.agent_id} error working on {file_path}: {e}")
            return False
    
    async def send_heartbeat(self):
        """Send heartbeat to message bus."""
        if self.message_bus and self.active:
            await self.message_bus.heartbeat(
                self.agent_id, 
                "busy" if self.current_task else "idle",
                {"current_task": self.current_task}
            )
    
    async def disconnect(self):
        """Disconnect from collaboration systems."""
        self.active = False
        if self.message_bus:
            await self.message_bus.unregister_agent(self.agent_id)
        if self.voting_system:
            self.voting_system.unregister_agent(self.agent_id)
        
        log.info(f"Agent {self.agent_id} disconnected")

async def test_message_bus():
    """Test the agent message bus functionality."""
    print("\n=== Testing Agent Message Bus ===")
    
    bus = AgentMessageBus()
    await bus.start()
    
    # Create test agents
    agent1 = CollaborativeAgent("agent1", "coder", ["python", "javascript"])
    agent2 = CollaborativeAgent("agent2", "tester", ["pytest", "selenium"])
    
    # Register agents
    agent1.message_bus = bus
    await agent1.message_bus.register_agent("agent1", "coder", ["python", "javascript"])
    agent2.message_bus = bus
    await agent2.message_bus.register_agent("agent2", "tester", ["pytest", "selenium"])
    
    # Test message publishing and subscription
    received_messages = []
    
    async def test_callback(topic, message):
        received_messages.append((topic, message))
    
    await bus.subscribe("test.*", test_callback)
    
    # Publish test messages
    await bus.publish("test.message1", MessageType.STATUS_UPDATE, 
                     {"content": "Hello from agent1"}, "agent1")
    await bus.publish("test.message2", MessageType.TASK_ASSIGNMENT,
                     {"content": "Task for agent2"}, "agent1")
    
    # Wait for message processing
    await asyncio.sleep(0.1)
    
    # Check results
    print(f"[OK] Messages received: {len(received_messages)}")
    for topic, msg in received_messages:
        print(f"  - {topic}: {msg['content']}")
    
    # Test agent discovery
    agents = bus.get_online_agents()
    print(f"[OK] Online agents: {list(agents.keys())}")
    
    # Test stats
    stats = bus.get_stats()
    print(f"[OK] Bus stats: {stats['total_messages']} messages, {stats['registered_agents']} agents")
    
    await bus.stop()
    return len(received_messages) == 2

async def test_shared_workspace():
    """Test the shared workspace functionality."""
    print("\n=== Testing Shared Workspace ===")
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = SharedWorkspace(temp_dir, enable_git=False)
        
        # Test file operations
        success1 = await workspace.create_file("test1.py", "print('Hello from agent1')", "agent1")
        success2 = await workspace.create_file("test2.py", "print('Hello from agent2')", "agent2")
        
        print(f"[OK] File creation: agent1={success1}, agent2={success2}")
        
        # Test concurrent access
        file_path = "shared.py"
        
        # Both agents try to work on the same file
        task1 = workspace.write_file(file_path, "# Modified by agent1", "agent1")
        task2 = workspace.write_file(file_path, "# Modified by agent2", "agent2")
        
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        successful_writes = sum(1 for r in results if r is True)
        
        print(f"[OK] Concurrent writes: {successful_writes}/2 succeeded (expected: 1 due to locking)")
        
        # Test file listing
        files = workspace.list_files()
        print(f"[OK] Files in workspace: {[f['path'] for f in files]}")
        
        # Test change history
        changes = workspace.get_changes()
        print(f"[OK] Change history: {len(changes)} changes")
        
        return successful_writes == 1  # Only one should succeed due to locking

async def test_collaborative_decision_making():
    """Test the collaborative decision making system."""
    print("\n=== Testing Collaborative Decision Making ===")
    
    voting_system = VotingSystem(default_deadline_minutes=0.1)  # Short deadline for testing
    
    # Register agents with different roles and weights
    voting_system.register_agent("agent1", "lead", ["architecture", "python"], 2.0)
    voting_system.register_agent("agent2", "senior", ["testing", "qa"], 1.5)
    voting_system.register_agent("agent3", "regular", ["python"], 1.0)
    
    # Create a decision
    decision_id = await voting_system.create_decision(
        title="Implement new authentication system",
        description="Should we implement OAuth2 authentication?",
        decision_type=DecisionType.ARCHITECTURE_CHANGE,
        proposer_id="agent1",
        voting_strategy=VotingStrategy.WEIGHTED,
        deadline_minutes=0.05,  # 3 seconds
        quorum_required=2
    )
    
    print(f"[OK] Decision created: {decision_id}")
    
    # Cast votes
    await voting_system.cast_vote(decision_id, "agent1", VoteType.APPROVE, "Good security improvement")
    await voting_system.cast_vote(decision_id, "agent2", VoteType.APPROVE, "Supports modern standards")
    await voting_system.cast_vote(decision_id, "agent3", VoteType.REJECT, "Too complex for our needs")
    
    # Wait for deadline
    await asyncio.sleep(0.1)
    
    # Check result
    result = await voting_system.get_result(decision_id)
    print(f"[OK] Decision result: {result['status']} - {result['reasoning']}")
    
    # Test stats
    stats = voting_system.get_stats()
    print(f"[OK] Voting stats: {stats['total_decisions']} decisions, {stats['approval_rate']:.1%} approval rate")
    
    return result['status'] == "approved"

async def test_full_collaboration_scenario():
    """Test a complete collaboration scenario with all systems."""
    print("\n=== Testing Full Collaboration Scenario ===")
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize systems
        message_bus = AgentMessageBus()
        workspace = SharedWorkspace(temp_dir, enable_git=False)
        voting_system = VotingSystem(default_deadline_minutes=0.1)
        
        await message_bus.start()
        
        # Create agents
        agents = [
            CollaborativeAgent("coder1", "coder", ["python", "architecture"]),
            CollaborativeAgent("coder2", "coder", ["python", "testing"]),
            CollaborativeAgent("tester1", "tester", ["pytest", "qa"]),
        ]
        
        # Connect agents to systems
        for agent in agents:
            await agent.connect_to_systems(message_bus, workspace, voting_system)
        
        # Scenario: Collaborative code development
        print("[SCENARIO] Starting collaborative development session...")
        
        # 1. Agent1 creates initial file
        await agents[0].work_on_file("calculator.py", """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
""")
        
        # 2. Propose adding multiplication feature
        decision_id = await voting_system.create_decision(
            title="Add multiplication to calculator",
            description="Should we add a multiply method to the Calculator class?",
            decision_type=DecisionType.CODE_MERGE,
            proposer_id="coder1",
            voting_strategy=VotingStrategy.SIMPLE_MAJORITY,
            deadline_minutes=0.05
        )
        
        # 3. Agents vote on the decision
        await asyncio.sleep(0.01)  # Brief delay for decision creation
        await agents[0]._make_decision(decision_id, {
            "decision_type": DecisionType.CODE_MERGE.value,
            "description": "Add multiplication method"
        })
        await agents[1]._make_decision(decision_id, {
            "decision_type": DecisionType.CODE_MERGE.value,
            "description": "Add multiplication method"
        })
        await agents[2]._make_decision(decision_id, {
            "decision_type": DecisionType.CODE_MERGE.value,
            "description": "Add multiplication method"
        })
        
        # 4. Wait for decision and check result
        await asyncio.sleep(0.1)
        result = await voting_system.get_result(decision_id)
        print(f"[SCENARIO] Vote result: {result['status']}")
        
        # 5. If approved, implement the feature
        if result['status'] == "approved":
            await agents[1].work_on_file("calculator.py", """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
""")
            print("[SCENARIO] Feature implemented")
        
        # 6. Tester creates test file
        await agents[2].work_on_file("test_calculator.py", """
import pytest
from calculator import Calculator

def test_calculator():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.subtract(5, 2) == 3
    if hasattr(calc, 'multiply'):
        assert calc.multiply(3, 4) == 12
""")
        
        # 7. Send heartbeats
        for agent in agents:
            await agent.send_heartbeat()
        
        # 8. Check final state
        files = workspace.list_files()
        changes = workspace.get_changes()
        stats = message_bus.get_stats()
        
        print(f"[SCENARIO] Final state:")
        print(f"  - Files created: {len(files)}")
        print(f"  - Changes made: {len(changes)}")
        print(f"  - Messages exchanged: {stats['total_messages']}")
        print(f"  - Agents active: {stats['online_agents']}")
        
        # Cleanup
        for agent in agents:
            await agent.disconnect()
        
        await message_bus.stop()
        
        return len(files) >= 2 and result['status'] == "approved"

async def run_all_tests():
    """Run all collaboration system tests."""
    print("Testing Real-Time Agent Collaboration System")
    print("=" * 60)
    
    tests = [
        ("Message Bus", test_message_bus),
        ("Shared Workspace", test_shared_workspace),
        ("Decision Making", test_collaborative_decision_making),
        ("Full Collaboration", test_full_collaboration_scenario)
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
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total_time = 0
    
    for test_name, result, duration in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:4} | {test_name:20} | {duration:.2f}s")
        if result:
            passed += 1
        total_time += duration
    
    print("-" * 60)
    print(f"PASSED: {passed}/{len(results)} tests in {total_time:.2f}s")
    
    success_rate = passed / len(results) * 100
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if passed == len(results):
        print("\n[SUCCESS] ALL TESTS PASSED! Real-time agent collaboration system is working!")
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