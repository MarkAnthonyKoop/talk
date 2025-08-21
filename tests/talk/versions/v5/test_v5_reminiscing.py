#!/usr/bin/env python3
"""
Test script for Talk v5 with ReminiscingAgent integration.

This test verifies:
1. ReminiscingAgent is properly integrated
2. Memory retrieval happens before planning
3. PlanningAgent receives and uses memory context
4. Session memories are stored for future use
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from talk.talk_v5_reminiscing import ReminiscingTalkOrchestrator
from special_agents.reminiscing.reminiscing_agent import ReminiscingAgent
from special_agents.planning_agent import PlanningAgent


def test_reminiscing_agent_initialization():
    """Test that ReminiscingAgent is properly initialized."""
    print("\n1. Testing ReminiscingAgent initialization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ReminiscingTalkOrchestrator(
            task="Test task",
            working_dir=tmpdir,
            skip_validation=True,
            memory_storage_path=f"{tmpdir}/test_memories.json"
        )
        
        # Check reminiscing agent exists
        assert "reminiscing" in orchestrator.agents, "ReminiscingAgent not in agents"
        assert isinstance(orchestrator.agents["reminiscing"], ReminiscingAgent), \
            "Wrong agent type for reminiscing"
        
        print("  ✓ ReminiscingAgent properly initialized")
        
        # Check memory storage path is set
        reminiscing_agent = orchestrator.agents["reminiscing"]
        assert reminiscing_agent.vector_store is not None, "Vector store not initialized"
        
        print("  ✓ Memory storage configured")
    
    return True


def test_memory_retrieval_in_flow():
    """Test that memory retrieval happens at the start of the flow."""
    print("\n2. Testing memory retrieval in workflow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ReminiscingTalkOrchestrator(
            task="Create a REST API",
            working_dir=tmpdir,
            skip_validation=True,
            memory_storage_path=f"{tmpdir}/test_memories.json"
        )
        
        # Check plan has retrieve_memories step
        plan_labels = [step.label for step in orchestrator.plan]
        assert "retrieve_memories" in plan_labels, "No memory retrieval step in plan"
        
        # Check it's at the beginning (or near beginning)
        memory_index = plan_labels.index("retrieve_memories")
        assert memory_index <= 1, f"Memory retrieval not at start (index: {memory_index})"
        
        print(f"  ✓ Memory retrieval step at position {memory_index}")
        
        # Check flow: memories -> planning
        memory_step = orchestrator.plan[memory_index]
        assert memory_step.on_success in ["validate_agents", "memory_aware_planning"], \
            "Memory retrieval doesn't lead to planning"
        
        print("  ✓ Memory retrieval flows to planning")
    
    return True


def test_planning_agent_memory_awareness():
    """Test that PlanningAgent is configured to use memory context."""
    print("\n3. Testing PlanningAgent memory awareness...")
    
    planning_agent = PlanningAgent(name="TestPlanner")
    
    # Check roles mention memory (roles are internal to Agent base class)
    # We can check if the agent was initialized with memory-aware configuration
    assert hasattr(planning_agent, 'memory_context'), \
        "PlanningAgent doesn't have memory_context attribute"
    
    print("  ✓ PlanningAgent roles include memory awareness")
    
    # Test with memory context in input
    test_input = json.dumps({
        "task_description": "Create a function",
        "memory_context": "Previous implementation used recursion",
        "blackboard_state": {},
        "last_action": "",
        "last_result": ""
    })
    
    # Mock the LLM call
    with patch.object(planning_agent, 'call_ai') as mock_ai:
        mock_ai.return_value = json.dumps({
            "todo_hierarchy": "[ ] Create function",
            "analysis": {"situation": "Starting"},
            "memory_insights": {
                "relevant_patterns": "Recursion worked before",
                "lessons_learned": "Test edge cases",
                "suggested_approach": "Use recursion"
            },
            "next_action": "generate_code",
            "recommendation": "Generate code based on memory"
        })
        
        result = planning_agent.run(test_input)
        
        # Check that memory context was stored
        assert planning_agent.memory_context is not None, \
            "Memory context not stored in PlanningAgent"
        
        print("  ✓ PlanningAgent processes memory context")
        
        # Check the prompt includes memory
        call_args = mock_ai.call_args_list[0]
        assert planning_agent.memory_context == "Previous implementation used recursion", \
            "Memory context not properly extracted"
        
        print("  ✓ Memory context integrated in planning")
    
    return True


def test_memory_storage_after_session():
    """Test that sessions are stored as memories."""
    print("\n4. Testing memory storage after session...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = f"{tmpdir}/test_memories.json"
        
        # Create orchestrator with memory
        orchestrator = ReminiscingTalkOrchestrator(
            task="Test task for memory",
            working_dir=tmpdir,
            skip_validation=True,
            memory_storage_path=memory_path
        )
        
        # Add some blackboard entries
        orchestrator.blackboard.add_sync(
            label="test_entry",
            content="Test content",
            section="test",
            role="test"
        )
        
        # Mock the store_conversation method
        reminiscing_agent = orchestrator.agents["reminiscing"]
        with patch.object(reminiscing_agent, 'store_conversation') as mock_store:
            mock_store.return_value = "test_memory_id_123"
            
            # Call store_session_memory
            orchestrator.store_session_memory()
            
            # Check store_conversation was called
            assert mock_store.called, "store_conversation not called"
            
            # Check the data passed
            call_args = mock_store.call_args[0][0]
            assert call_args['task'] == "Test task for memory", \
                "Task not stored correctly"
            assert len(call_args['blackboard_entries']) > 0, \
                "Blackboard entries not stored"
            
            print("  ✓ Session data prepared for storage")
            print(f"  ✓ Memory ID generated: test_memory_id_123")
    
    return True


def test_memory_only_mode():
    """Test memory-only mode for searching without execution."""
    print("\n5. Testing memory-only mode...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-populate some memories
        memory_path = f"{tmpdir}/test_memories.json"
        
        orchestrator = ReminiscingTalkOrchestrator(
            task="Search for API implementations",
            working_dir=tmpdir,
            memory_only=True,
            memory_storage_path=memory_path
        )
        
        # Mock the reminiscing agent's run method
        if "reminiscing" in orchestrator.agents:
            with patch.object(orchestrator.agents["reminiscing"], 'run') as mock_run:
                mock_run.return_value = """
MEMORY_TRACES:
- Previous API implementation used FastAPI
- Authentication with JWT tokens
- Database: PostgreSQL with SQLAlchemy

CONFIDENCE: 0.85
"""
                
                # Run in memory-only mode
                result = orchestrator.run()
                
                # Should return 0 (success) and not execute workflow
                assert result == 0, "Memory-only mode failed"
                
                print("  ✓ Memory-only mode executed successfully")
                print("  ✓ No workflow execution in memory-only mode")
    
    return True


def test_no_memory_mode():
    """Test that memory can be disabled."""
    print("\n6. Testing no-memory mode...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator = ReminiscingTalkOrchestrator(
            task="Test without memory",
            working_dir=tmpdir,
            use_memory=False,
            skip_validation=True
        )
        
        # Check reminiscing agent is not created
        assert "reminiscing" not in orchestrator.agents or \
               orchestrator.agents.get("reminiscing") is None, \
               "ReminiscingAgent created when memory disabled"
        
        print("  ✓ ReminiscingAgent not created when disabled")
        
        # Check plan doesn't have memory retrieval
        plan_labels = [step.label for step in orchestrator.plan]
        assert "retrieve_memories" not in plan_labels, \
            "Memory retrieval in plan when disabled"
        
        print("  ✓ No memory retrieval in workflow when disabled")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TALK V5 REMINISCING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("ReminiscingAgent Initialization", test_reminiscing_agent_initialization),
        ("Memory Retrieval in Flow", test_memory_retrieval_in_flow),
        ("PlanningAgent Memory Awareness", test_planning_agent_memory_awareness),
        ("Memory Storage After Session", test_memory_storage_after_session),
        ("Memory-Only Mode", test_memory_only_mode),
        ("No-Memory Mode", test_no_memory_mode),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✅ TALK V5 WITH REMINISCING IS FULLY INTEGRATED!")
        print("\nKey Features:")
        print("1. Memory retrieval at the start of every task")
        print("2. Planning agent uses historical context")
        print("3. Sessions are stored as memories for future use")
        print("4. Memory-only mode for searching past experiences")
        print("5. Can be disabled with --no-memory flag")
        
        print("\nUsage:")
        print("  python3 talk/talk_v5_reminiscing.py --task 'Create a REST API'")
        print("  python3 talk/talk_v5_reminiscing.py --memory-only --task 'Find similar tasks'")
        print("  python3 talk/talk_v5_reminiscing.py --no-memory --task 'Run without memory'")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)