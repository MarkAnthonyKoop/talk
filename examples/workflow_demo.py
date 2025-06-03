#!/usr/bin/env python3
"""
Talk Agent Framework - Workflow Demonstration
=============================================

This example demonstrates the key capabilities of the Talk Agent Framework:
1. Simple agent conversations
2. Multi-step workflows with PlanRunner
3. Special agents (CodeAgent, FileAgent)
4. Blackboard shared memory
5. Different backend configurations

Run this demo with:
    python examples/workflow_demo.py

No API keys required - uses stub backend by default.
"""

import asyncio
import os
import tempfile
from typing import Dict

# Core components
from agent.agent import Agent
from agent.messages import Message, Role

# Runtime components
from runtime.blackboard import Blackboard
from runtime.step import Step
from runtime.plan_runner import PlanRunner

# Special agents
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent


def print_section(title):
    """Helper to print formatted section headers"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}\n")


def demo_simple_conversation():
    """Demonstrates a simple conversation with a single agent"""
    print_section("1. Simple Agent Conversation")
    
    # Create an agent with the stub backend
    agent = Agent(overrides={"provider": {"type": "stub"}})
    
    # Run a simple conversation
    print("User: Hello, agent!")
    response = agent.run("Hello, agent!")
    print(f"Agent: {response}")
    
    # Add another turn to the conversation
    print("\nUser: What can you help me with?")
    response = agent.run("What can you help me with?")
    print(f"Agent: {response}")
    
    # Show conversation history
    print("\nConversation History:")
    for msg in agent.conversation.messages:
        print(f"[{msg.role.value}] {msg.content}")


async def demo_multi_step_workflow():
    """Demonstrates a multi-step workflow with PlanRunner"""
    print_section("2. Multi-Step Workflow")
    
    # Create a shared blackboard
    blackboard = Blackboard()
    
    # Create agents with different configurations but same stub backend
    agents = {
        "greeter": Agent(name="greeter", overrides={"provider": {"type": "stub"}}),
        "analyzer": Agent(name="analyzer", overrides={"provider": {"type": "stub"}}),
        "summarizer": Agent(name="summarizer", overrides={"provider": {"type": "stub"}}),
    }
    
    # Define workflow steps
    steps = [
        Step(label="welcome", agent_key="greeter", message="Greet the user warmly"),
        Step(label="analyze", agent_key="analyzer", message="Analyze the following text: {input}"),
        Step(label="summarize", agent_key="summarizer", 
             message="Summarize the analysis: {analyze.output}"),
    ]
    
    # Create and run the workflow
    runner = PlanRunner(steps, agents, blackboard)
    result = await runner.run("This is sample text to analyze")
    
    # Print the workflow results
    print("Final result:", result)
    
    # Show blackboard contents
    print("\nBlackboard Contents:")
    for key, entry in blackboard.entries.items():
        print(f"- {key}: {entry.value}")


async def demo_special_agents():
    """Demonstrates the special agents for code generation and file operations"""
    print_section("3. Special Agents (CodeAgent, FileAgent)")
    
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple Python file to modify
        initial_file = os.path.join(temp_dir, "example.py")
        with open(initial_file, "w") as f:
            f.write("def hello():\n    print('Hello world')\n")
        
        print(f"Created initial file at: {initial_file}")
        print("Initial content:")
        with open(initial_file, "r") as f:
            print(f.read())
        
        # Create special agents
        code_agent = CodeAgent(overrides={"provider": {"type": "stub"}})
        file_agent = FileAgent(base_dir=temp_dir)
        
        # Generate a diff to modify the file
        print("\nGenerating code diff...")
        diff = code_agent.run(f"""
        Modify the file {initial_file} to:
        1. Add a greeting parameter
        2. Add a main function that calls hello()
        """)
        print("Generated diff:")
        print(diff)
        
        # Apply the diff to the file
        print("\nApplying diff...")
        result = file_agent.run(diff)
        print(f"File agent result: {result}")
        
        # Show the modified file
        print("\nModified file content:")
        try:
            with open(initial_file, "r") as f:
                print(f.read())
        except FileNotFoundError:
            print("Note: In stub mode, no actual file changes are made")


async def demo_blackboard_memory():
    """Demonstrates using the Blackboard for shared memory between agents"""
    print_section("4. Blackboard Shared Memory")
    
    # Create a shared blackboard
    blackboard = Blackboard()
    
    # Create agents
    agents = {
        "researcher": Agent(name="researcher", overrides={"provider": {"type": "stub"}}),
        "writer": Agent(name="writer", overrides={"provider": {"type": "stub"}}),
        "reviewer": Agent(name="reviewer", overrides={"provider": {"type": "stub"}}),
    }
    
    # Define workflow steps with shared memory
    steps = [
        # First agent stores information in the blackboard
        Step(label="research", agent_key="researcher", 
             message="Research facts about AI and store them in structured format"),
        
        # Second agent uses the research and adds to the blackboard
        Step(label="draft", agent_key="writer", 
             message="Write an article based on these facts: {research.output}"),
        
        # Third agent reviews and provides feedback
        Step(label="review", agent_key="reviewer", 
             message="Review this article: {draft.output}"),
        
        # Writer makes revisions based on feedback
        Step(label="revise", agent_key="writer", 
             message="Revise the article based on this feedback: {review.output}"),
    ]
    
    # Run the workflow
    runner = PlanRunner(steps, agents, blackboard)
    result = await runner.run("Write an article about artificial intelligence")
    
    # Print the final result
    print("Final article:", result)
    
    # Show the evolution of the content through the blackboard
    print("\nContent Evolution in Blackboard:")
    for step_name in ["research", "draft", "review", "revise"]:
        if step_name in blackboard.entries:
            print(f"\n--- {step_name.upper()} ---")
            print(blackboard.entries[step_name].value)


def demo_backend_configurations():
    """Demonstrates different backend configurations"""
    print_section("5. Different Backend Configurations")
    
    # Create agents with different backend configurations
    configurations = [
        {"name": "Default Stub", "overrides": {"provider": {"type": "stub"}}},
        {"name": "OpenAI-like", "overrides": {
            "provider": {"type": "stub"},
            "model_name": "gpt-4-like",
            "temperature": 0.7,
            "max_tokens": 2000
        }},
        {"name": "Anthropic-like", "overrides": {
            "provider": {"type": "stub"},
            "model_name": "claude-3-opus-like",
            "temperature": 0.5,
            "max_tokens": 4000
        }},
        {"name": "Shell Backend", "overrides": {"provider": {"type": "shell"}}},
    ]
    
    # Try each configuration
    for config in configurations:
        print(f"\n--- {config['name']} ---")
        agent = Agent(overrides=config["overrides"])
        response = agent.run("Tell me about the Talk Agent Framework")
        print(f"Response: {response}")
        print(f"Backend type: {agent.settings.provider.type}")
        print(f"Model: {agent.settings.provider.model_name if hasattr(agent.settings.provider, 'model_name') else 'N/A'}")


async def main():
    """Run all demonstrations"""
    demo_simple_conversation()
    await demo_multi_step_workflow()
    await demo_special_agents()
    await demo_blackboard_memory()
    demo_backend_configurations()


if __name__ == "__main__":
    # Ensure the examples directory exists
    os.makedirs("examples", exist_ok=True)
    
    # Run the async demo
    asyncio.run(main())
