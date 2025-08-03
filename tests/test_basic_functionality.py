#!/usr/bin/env python3
"""
Test suite for the Talk Agent Framework.

This file contains comprehensive tests for the core functionality of the Talk Agent Framework,
including agent creation, message handling, backend loading, runtime execution, special agents,
blackboard memory, and configuration.

All tests use the stub backend to avoid requiring external API keys.

Run with:
    pytest tests/test_basic_functionality.py -v
"""

import os
import pytest
import tempfile
import asyncio
from typing import Dict, List, Optional

# Agent components
from agent.agent import Agent
from agent.messages import Message, Role, MessageList
from agent.settings import Settings
from agent.llm_backends import get_backend, LLMBackend, StubBackend, LLMBackendError

# Runtime components
from runtime.blackboard import Blackboard, BlackboardEntry
from runtime.step import Step
from runtime.plan_runner import PlanRunner

# Special agents
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from special_agents.branching_agent import BranchingAgent
from special_agents.shell_agent import ShellAgent


# ===== Fixtures =====

@pytest.fixture
def stub_agent():
    """Fixture for a basic agent with stub backend."""
    return Agent(overrides={"provider": {"type": "stub"}})


@pytest.fixture
def message_list():
    """Fixture for a basic message list with some sample messages."""
    return MessageList(
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
    )


@pytest.fixture
def blackboard():
    """Fixture for a basic blackboard."""
    return Blackboard()


@pytest.fixture
def temp_dir():
    """Fixture for a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# ===== Basic Agent Tests =====

def test_agent_creation():
    """Test that agents can be created with different configurations."""
    # Default agent
    agent1 = Agent()
    assert agent1.id is not None
    assert agent1.settings is not None
    
    # Agent with stub backend
    agent2 = Agent(overrides={"provider": {"type": "stub"}})
    assert agent2.settings.provider.type == "stub"
    
    # Agent with name and ID
    agent3 = Agent(name="test_agent", id="test123")
    assert agent3.name == "test_agent"
    assert agent3.id == "test123"


def test_agent_conversation():
    """Test basic conversation with an agent."""
    agent = Agent(overrides={"provider": {"type": "stub"}})
    
    # Initial conversation should be empty
    assert len(agent.conversation.messages) == 0
    
    # Run a message
    response = agent.run("Hello")
    assert response is not None
    assert isinstance(response, str)
    
    # Conversation should now have two messages (user and assistant)
    assert len(agent.conversation.messages) == 2
    assert agent.conversation.messages[0].role == Role.USER
    assert agent.conversation.messages[0].content == "Hello"
    assert agent.conversation.messages[1].role == Role.ASSISTANT
    
    # Run another message
    response2 = agent.run("How are you?")
    assert response2 is not None
    
    # Conversation should now have four messages
    assert len(agent.conversation.messages) == 4


def test_agent_with_initial_messages():
    """Test creating an agent with initial messages."""
    initial_messages = MessageList(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
    )
    
    agent = Agent(
        overrides={"provider": {"type": "stub"}},
        initial_messages=initial_messages
    )
    
    # Check that the initial messages were set
    assert len(agent.conversation.messages) == 3
    assert agent.conversation.messages[0].role == Role.SYSTEM
    assert agent.conversation.messages[1].role == Role.USER
    assert agent.conversation.messages[2].role == Role.ASSISTANT
    
    # Run a new message
    response = agent.run("How are you?")
    assert response is not None
    
    # Conversation should now have five messages
    assert len(agent.conversation.messages) == 5


# ===== Message Tests =====

def test_message_creation():
    """Test creating messages with different roles and content."""
    # Basic message
    msg1 = Message(role="user", content="Hello")
    assert msg1.role == Role.USER
    assert msg1.content == "Hello"
    
    # System message
    msg2 = Message(role="system", content="You are a helpful assistant.")
    assert msg2.role == Role.SYSTEM
    assert msg2.content == "You are a helpful assistant."
    
    # Assistant message
    msg3 = Message(role="assistant", content="Hi there!")
    assert msg3.role == Role.ASSISTANT
    assert msg3.content == "Hi there!"
    
    # Empty content
    msg4 = Message(role="user", content="")
    assert msg4.role == Role.USER
    assert msg4.content == ""


def test_message_list_operations(message_list):
    """Test operations on message lists."""
    # Check initial state
    assert len(message_list.messages) == 3
    
    # Add a message
    message_list.add(Message(role="assistant", content="I'm doing well!"))
    assert len(message_list.messages) == 4
    assert message_list.messages[-1].content == "I'm doing well!"
    
    # Clear messages
    message_list.clear()
    assert len(message_list.messages) == 0
    
    # Add multiple messages
    message_list.add(Message(role="user", content="Hello again"))
    message_list.add(Message(role="assistant", content="Welcome back!"))
    assert len(message_list.messages) == 2


def test_message_validation():
    """Test message validation."""
    # Valid messages should not raise exceptions
    Message(role="user", content="Hello")
    Message(role="assistant", content="Hi")
    Message(role="system", content="You are an AI")
    
    # Invalid role should raise ValueError
    with pytest.raises(ValueError):
        Message(role="invalid_role", content="Hello")
    
    # None content is allowed
    Message(role="user", content=None)


# ===== Backend Tests =====

def test_backend_loading():
    """Test loading different backends."""
    # Stub backend should always work
    backend = get_backend({"type": "stub"})
    assert isinstance(backend, StubBackend)
    
    # Invalid backend should raise an error and suggest using stub
    with pytest.raises(LLMBackendError):
        get_backend({"type": "nonexistent_backend"})


def test_backend_fallback():
    """Test backend fallback mechanism."""
    # Create an agent with a backend that would fail
    # It should fall back to the stub backend
    agent = Agent()
    
    # Force a backend error
    agent.settings.provider.type = "invalid_backend"
    
    # This should trigger fallback to stub
    agent._setup_backend()
    
    # Check that we're using the stub backend
    assert isinstance(agent.backend, StubBackend)
    
    # Run a message to confirm it works
    response = agent.run("Hello after fallback")
    assert response is not None


def test_backend_completion():
    """Test that backends can complete messages."""
    backend = StubBackend({})
    
    # Create some test messages
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="How are you?"),
    ]
    
    # Complete the messages
    response = backend.complete(messages)
    
    # Check the response
    assert response is not None
    assert response.role == Role.ASSISTANT
    assert isinstance(response.content, str)
    assert len(response.content) > 0


# ===== Runtime Tests =====

def test_blackboard_operations(blackboard):
    """Test blackboard operations."""
    # Initially empty
    assert len(blackboard.entries) == 0
    
    # Add an entry
    blackboard.set("test", "Hello world")
    assert "test" in blackboard.entries
    assert blackboard.get("test") == "Hello world"
    
    # Update an entry
    blackboard.set("test", "Updated value")
    assert blackboard.get("test") == "Updated value"
    
    # Add an entry with metadata
    blackboard.set("meta_test", "With metadata", meta={"source": "test", "timestamp": 123})
    entry = blackboard.entries["meta_test"]
    assert entry.value == "With metadata"
    assert entry.meta["source"] == "test"
    assert entry.meta["timestamp"] == 123
    
    # Get a non-existent entry
    assert blackboard.get("nonexistent") is None
    assert blackboard.get("nonexistent", default="default") == "default"


def test_step_creation():
    """Test creating workflow steps."""
    # Basic step
    step1 = Step(name="test", agent="test_agent")
    assert step1.name == "test"
    assert step1.agent == "test_agent"
    assert step1.prompt is None
    
    # Step with prompt
    step2 = Step(name="prompt_test", agent="test_agent", prompt="Hello {name}")
    assert step2.name == "prompt_test"
    assert step2.agent == "test_agent"
    assert step2.prompt == "Hello {name}"
    
    # Step with dependencies
    step3 = Step(name="dep_test", agent="test_agent", depends_on=["step1", "step2"])
    assert step3.name == "dep_test"
    assert step3.depends_on == ["step1", "step2"]


@pytest.mark.asyncio
async def test_plan_runner_execution():
    """Test executing a workflow with PlanRunner."""
    # Create a blackboard
    blackboard = Blackboard()
    
    # Create agents
    agents = {
        "agent1": Agent(name="agent1", overrides={"provider": {"type": "stub"}}),
        "agent2": Agent(name="agent2", overrides={"provider": {"type": "stub"}}),
    }
    
    # Create steps
    steps = [
        Step(name="step1", agent="agent1", prompt="Hello"),
        Step(name="step2", agent="agent2", prompt="Process this: {step1.output}"),
    ]
    
    # Create and run the workflow
    runner = PlanRunner(steps, agents, blackboard)
    result = await runner.run("Start the workflow")
    
    # Check the result
    assert result is not None
    assert isinstance(result, str)
    
    # Check the blackboard
    assert "step1" in blackboard.entries
    assert "step2" in blackboard.entries
    assert blackboard.get("step1") is not None
    assert blackboard.get("step2") is not None


@pytest.mark.asyncio
async def test_plan_runner_with_dependencies():
    """Test PlanRunner with step dependencies."""
    # Create a blackboard
    blackboard = Blackboard()
    
    # Create agents
    agents = {
        "agent1": Agent(name="agent1", overrides={"provider": {"type": "stub"}}),
        "agent2": Agent(name="agent2", overrides={"provider": {"type": "stub"}}),
        "agent3": Agent(name="agent3", overrides={"provider": {"type": "stub"}}),
    }
    
    # Create steps with dependencies
    steps = [
        Step(name="step1", agent="agent1"),
        Step(name="step2", agent="agent2"),
        Step(name="step3", agent="agent3", depends_on=["step1", "step2"],
             prompt="Combine: {step1.output} and {step2.output}"),
    ]
    
    # Create and run the workflow
    runner = PlanRunner(steps, agents, blackboard)
    result = await runner.run("Execute with dependencies")
    
    # Check the result
    assert result is not None
    
    # Check the blackboard - all steps should have executed
    assert "step1" in blackboard.entries
    assert "step2" in blackboard.entries
    assert "step3" in blackboard.entries


# ===== Special Agents Tests =====

def test_code_agent():
    """Test the CodeAgent for generating diffs."""
    agent = CodeAgent(overrides={"provider": {"type": "stub"}})
    
    # Run a code generation request
    result = agent.run("""
    Generate a Python function that calculates the Fibonacci sequence.
    """)
    
    # In stub mode, we'll get a mock response, but we can check it's working
    assert result is not None
    assert isinstance(result, str)


def test_file_agent(temp_dir):
    """Test the FileAgent for applying diffs."""
    # Create a test file
    test_file = os.path.join(temp_dir, "test.py")
    with open(test_file, "w") as f:
        f.write("def hello():\n    print('Hello world')\n")
    
    # Create a file agent
    agent = FileAgent(base_dir=temp_dir)
    
    # Create a simple diff
    diff = """--- test.py
+++ test.py
@@ -1,2 +1,5 @@
 def hello():
-    print('Hello world')
+    print('Hello, world!')
+
+def goodbye():
+    print('Goodbye, world!')
"""
    
    # Apply the diff
    result = agent.run(diff)
    
    # Check the result
    assert "Applied" in result or "would apply" in result
    
    # In stub mode, no actual changes are made, so we don't check the file content


def test_branching_agent():
    """Test the BranchingAgent for control flow."""
    agent = BranchingAgent(overrides={"provider": {"type": "stub"}})
    
    # Run a branching request
    result = agent.run("""
    If the user asks about Python, go to step 'python_info'.
    If the user asks about JavaScript, go to step 'js_info'.
    Otherwise, go to step 'general_info'.
    
    User query: How do I use dictionaries in Python?
    """)
    
    # Check the result - should indicate a branch
    assert result is not None
    assert isinstance(result, str)


# ===== Settings and Configuration Tests =====

def test_settings_creation():
    """Test creating settings with different configurations."""
    # Default settings
    settings1 = Settings()
    assert settings1.provider is not None
    assert settings1.conversation is not None
    
    # Settings with overrides
    settings2 = Settings.model_validate({
        "provider": {
            "type": "stub",
            "model_name": "test-model",
            "temperature": 0.5,
        },
        "conversation": {
            "log_path": "./test_logs",
            "log_format": "jsonl",
        }
    })
    
    assert settings2.provider.type == "stub"
    assert settings2.provider.model_name == "test-model"
    assert settings2.provider.temperature == 0.5
    assert settings2.conversation.log_path == "./test_logs"
    assert settings2.conversation.log_format == "jsonl"


def test_settings_from_dict():
    """Test creating settings from a dictionary."""
    config = {
        "provider": {
            "type": "stub",
            "model_name": "dict-model",
            "temperature": 0.7,
        },
        "conversation": {
            "log_enabled": False,
        }
    }
    
    settings = Settings.model_validate(config)
    assert settings.provider.type == "stub"
    assert settings.provider.model_name == "dict-model"
    assert settings.provider.temperature == 0.7
    assert settings.conversation.log_enabled is False


def test_settings_with_env_vars(monkeypatch):
    """Test settings with environment variables."""
    # Set environment variables
    monkeypatch.setenv("TALK_PROVIDER_TYPE", "stub")
    monkeypatch.setenv("TALK_MODEL_NAME", "env-model")
    monkeypatch.setenv("TALK_TEMPERATURE", "0.8")
    
    # Create settings - should pick up env vars
    settings = Settings()
    
    # Check that environment variables were applied
    assert settings.provider.type == "stub"
    assert settings.provider.model_name == "env-model"
    assert settings.provider.temperature == 0.8


def test_agent_with_settings():
    """Test creating an agent with custom settings."""
    settings = Settings.model_validate({
        "provider": {
            "type": "stub",
            "model_name": "custom-model",
            "temperature": 0.3,
        }
    })
    
    agent = Agent(settings=settings)
    assert agent.settings.provider.type == "stub"
    assert agent.settings.provider.model_name == "custom-model"
    assert agent.settings.provider.temperature == 0.3


# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_complete_workflow():
    """Integration test for a complete workflow with multiple components."""
    # Create a blackboard
    blackboard = Blackboard()
    
    # Create agents
    agents = {
        "researcher": Agent(name="researcher", overrides={"provider": {"type": "stub"}}),
        "writer": Agent(name="writer", overrides={"provider": {"type": "stub"}}),
        "reviewer": Agent(name="reviewer", overrides={"provider": {"type": "stub"}}),
        "code_gen": CodeAgent(name="code_gen", overrides={"provider": {"type": "stub"}}),
    }
    
    # Create steps
    steps = [
        Step(name="research", agent="researcher", 
             prompt="Research the topic: Python generators"),
        
        Step(name="write", agent="writer", depends_on=["research"],
             prompt="Write an article based on this research: {research.output}"),
        
        Step(name="review", agent="reviewer", depends_on=["write"],
             prompt="Review this article: {write.output}"),
        
        Step(name="generate_code", agent="code_gen", depends_on=["research"],
             prompt="Generate example code for: {research.output}"),
        
        Step(name="final", agent="writer", depends_on=["review", "generate_code"],
             prompt="Revise the article based on this feedback: {review.output}\n\nInclude this code: {generate_code.output}"),
    ]
    
    # Create and run the workflow
    runner = PlanRunner(steps, agents, blackboard)
    result = await runner.run("Create an article about Python generators")
    
    # Check the result
    assert result is not None
    assert isinstance(result, str)
    
    # Check the blackboard - all steps should have executed
    assert "research" in blackboard.entries
    assert "write" in blackboard.entries
    assert "review" in blackboard.entries
    assert "generate_code" in blackboard.entries
    assert "final" in blackboard.entries


def test_conversation_persistence(temp_dir):
    """Test that conversations can be persisted and loaded."""
    # Create a log path in the temp directory
    log_path = os.path.join(temp_dir, "conversation.jsonl")
    
    # Create an agent with logging enabled
    agent = Agent(overrides={
        "provider": {"type": "stub"},
        "conversation": {
            "log_enabled": True,
            "log_path": log_path,
            "log_format": "jsonl",
        }
    })
    
    # Run some messages
    agent.run("Hello")
    agent.run("How are you?")
    
    # Check that the log file was created
    assert os.path.exists(log_path)
    
    # Create a new agent that should load the conversation
    agent2 = Agent(overrides={
        "provider": {"type": "stub"},
        "conversation": {
            "log_enabled": True,
            "log_path": log_path,
            "log_format": "jsonl",
        }
    })
    
    # The new agent should have the conversation loaded
    assert len(agent2.conversation.messages) >= 4  # At least 4 messages (2 user, 2 assistant)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

