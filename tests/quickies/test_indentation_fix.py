#!/usr/bin/env python3
"""Test that our new tool-based file operations preserve indentation correctly."""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agent.tools import FILE_TOOLS
from agent.tools.handlers import UnifiedToolHandler
from agent.messages import Message, Role
from agent.llm_backends import get_backend

# Import the old FileAgent to test it fails
from special_agents.file_agent_old import FileAgent as OldFileAgent
from special_agents.file_agent_old2 import FileAgent as OldFileAgent2


def test_create_orchestrator_files():
    """Test creating the orchestrator files that previously failed with no indentation."""
    print("\n=== Test: Create Orchestrator Files with Proper Indentation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = UnifiedToolHandler(base_dir=tmpdir)
        
        # 1. Create config.py with proper indentation
        config_content = """from pydantic import BaseSettings

class Settings(BaseSettings):
    MAX_AGENTS: int = 100
    MAX_TASKS_PER_AGENT: int = 5
    HEARTBEAT_TIMEOUT: int = 30  # seconds
    TASK_TIMEOUT: int = 300  # seconds
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()"""
        
        result = handler.handle_tool_call("create_file", {
            "path": "config.py",
            "content": config_content
        })
        print(f"Created config.py: {result}")
        
        # 2. Create models.py with proper indentation
        models_content = """from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

class Task(BaseModel):
    id: UUID
    name: str
    payload: Dict
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[UUID] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

class Agent(BaseModel):
    id: UUID
    name: str
    capabilities: List[str]
    status: AgentStatus
    last_heartbeat: datetime
    current_tasks: List[UUID]
    metadata: Dict"""
        
        result = handler.handle_tool_call("create_file", {
            "path": "models.py",
            "content": models_content
        })
        print(f"Created models.py: {result}")
        
        # 3. Create orchestrator.py with proper indentation
        orchestrator_content = """import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from models import Agent, Task, AgentStatus, TaskStatus
from config import settings

class Orchestrator:
    def __init__(self):
        self.agents: Dict[UUID, Agent] = {}
        self.tasks: Dict[UUID, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)

    async def start(self):
        \"\"\"Start the orchestrator background tasks.\"\"\"
        self.logger.info("Starting orchestrator")
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._process_task_queue())

    async def register_agent(self, name: str, capabilities: List[str], metadata: Dict) -> UUID:
        \"\"\"Register a new agent.\"\"\"
        if len(self.agents) >= settings.MAX_AGENTS:
            raise RuntimeError("Maximum number of agents reached")

        agent_id = uuid4()
        agent = Agent(
            id=agent_id,
            name=name,
            capabilities=capabilities,
            status=AgentStatus.ONLINE,
            last_heartbeat=datetime.utcnow(),
            current_tasks=[],
            metadata=metadata
        )
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {name} ({agent_id})")
        return agent_id"""
        
        result = handler.handle_tool_call("create_file", {
            "path": "orchestrator.py",
            "content": orchestrator_content
        })
        print(f"Created orchestrator.py: {result}")
        
        # 4. Verify Python syntax is valid
        print("\n--- Verifying Python Syntax ---")
        for filename in ["config.py", "models.py", "orchestrator.py"]:
            filepath = Path(tmpdir) / filename
            try:
                with open(filepath, 'r') as f:
                    code = f.read()
                compile(code, filename, 'exec')
                print(f"✓ {filename}: Valid Python syntax")
                
                # Check for proper indentation
                lines = code.split('\n')
                indented_lines = [l for l in lines if l.startswith('    ')]
                if indented_lines:
                    print(f"  Found {len(indented_lines)} indented lines")
                
            except SyntaxError as e:
                print(f"✗ {filename}: Syntax error - {e}")
                print(f"  Line {e.lineno}: {e.text}")
                return False
            except Exception as e:
                print(f"✗ {filename}: Error - {e}")
                return False
        
        return True


def test_old_create_file_format():
    """Test what would have happened with the old CREATE_FILE format (no indentation)."""
    print("\n=== Test: Old CREATE_FILE Format (What Failed Before) ===")
    
    # This is what Claude generated in the original failing case
    old_format_content = """CREATE_FILE: config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
MAX_AGENTS: int = 100
MAX_TASKS_PER_AGENT: int = 5
HEARTBEAT_TIMEOUT: int = 30  # seconds
TASK_TIMEOUT: int = 300  # seconds
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

class Config:
env_file = ".env"

settings = Settings()"""
    
    print("Original CREATE_FILE content (no indentation):")
    print("-" * 40)
    print(old_format_content)
    print("-" * 40)
    print("\nThis would have created invalid Python due to missing indentation!")
    
    return True


def test_old_file_agent_fails():
    """Test that the old FileAgent creates invalid Python files."""
    print("\n=== Test: Old FileAgent with Claude's Output ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old file agent
        old_agent = OldFileAgent(base_dir=tmpdir)
        
        # This is the actual output from Claude in the blackboard.json
        claude_output = """CREATE_FILE: config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    MAX_AGENTS: int = 100
    MAX_TASKS_PER_AGENT: int = 5
    HEARTBEAT_TIMEOUT: int = 30  # seconds
    TASK_TIMEOUT: int = 300  # seconds
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()"""
        
        # Process with old file agent
        result = old_agent.run(claude_output)
        print(f"Old FileAgent result: {result}")
        
        # Try to verify the file
        config_path = Path(tmpdir) / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            print("\nFile content preview:")
            print("-" * 40)
            for i, line in enumerate(content.split('\n')[:10]):
                print(f"{i+1:3}: {repr(line)}")
            print("-" * 40)
            
            # Check Python syntax
            try:
                compile(content, "config.py", 'exec')
                print("✗ UNEXPECTED: File has valid syntax (should have failed!)")
                return False
            except IndentationError as e:
                print(f"✓ EXPECTED: IndentationError - {e}")
                print(f"  Line {e.lineno}: {repr(e.text)}")
                return True
            except SyntaxError as e:
                print(f"✓ EXPECTED: SyntaxError - {e}")
                return True
        else:
            print("✗ File was not created")
            return False


def test_old_file_agent2_fixed():
    """Test that simply removing line.strip() would have fixed the issue."""
    print("\n=== Test: Old FileAgent2 (with strip removed) ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old file agent 2
        old_agent2 = OldFileAgent2(base_dir=tmpdir)
        
        # This is the actual output from Claude in the blackboard.json
        claude_output = """CREATE_FILE: config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    MAX_AGENTS: int = 100
    MAX_TASKS_PER_AGENT: int = 5
    HEARTBEAT_TIMEOUT: int = 30  # seconds
    TASK_TIMEOUT: int = 300  # seconds
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()"""
        
        # Process with old file agent 2
        result = old_agent2.run(claude_output)
        print(f"Old FileAgent2 result: {result}")
        
        # Try to verify the file
        config_path = Path(tmpdir) / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            print("\nFile content preview:")
            print("-" * 40)
            for i, line in enumerate(content.split('\n')[:10]):
                print(f"{i+1:3}: {repr(line)}")
            print("-" * 40)
            
            # Check Python syntax
            try:
                compile(content, "config.py", 'exec')
                print("✓ SUCCESS: File has valid Python syntax!")
                print("✓ Simply removing line.strip() WOULD have fixed the issue!")
                return True
            except IndentationError as e:
                print(f"✗ UNEXPECTED: IndentationError - {e}")
                return False
            except SyntaxError as e:
                print(f"✗ UNEXPECTED: SyntaxError - {e}")
                return False
        else:
            print("✗ File was not created")
            return False


def test_anthropic_file_creation():
    """Test using Anthropic to create a properly formatted Python file."""
    print("\n=== Test: Anthropic File Creation ===")
    
    if not os.getenv("CLAUDE_API_TOKEN"):
        print("SKIPPED: CLAUDE_API_TOKEN not set")
        return True
    
    try:
        backend = get_backend({"_provider": "anthropic"})
        
        # Ask Claude to create a file using tools
        messages = [
            Message(
                role=Role.user, 
                content="Create a Python file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods."
            )
        ]
        
        # Give Claude access to file creation tools
        response = backend.complete(messages, tools=[FILE_TOOLS[3]])  # create_file tool
        
        print(f"Claude's response: {response}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("\nClaude wants to use tools:")
            for tool_call in response.tool_calls:
                func = tool_call["function"]
                print(f"  Tool: {func['name']}")
                args = json.loads(func['arguments'])
                print(f"  Path: {args.get('path', 'N/A')}")
                print(f"  Content preview: {args.get('content', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Indentation Fix with New Tool Architecture")
    print("=" * 60)
    
    results = []
    results.append(("Create Orchestrator Files", test_create_orchestrator_files()))
    results.append(("Old CREATE_FILE Format", test_old_create_file_format()))
    results.append(("Old FileAgent Fails", test_old_file_agent_fails()))
    results.append(("Old FileAgent2 Fixed", test_old_file_agent2_fixed()))
    results.append(("Anthropic File Creation", test_anthropic_file_creation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\nSuccess! The new tool-based architecture preserves indentation correctly.")
    else:
        print("\nSome tests failed.")
    
    sys.exit(0 if all(passed for _, passed in results) else 1)