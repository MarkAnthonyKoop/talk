# Communication Architecture

## Table of Contents
1. [Overview](#overview)
2. [LLM Communication Management](#llm-communication-management)
3. [Class Responsibilities](#class-responsibilities)
4. [Blackboard System](#blackboard-system)
5. [Agent-to-Agent Communication](#agent-to-agent-communication)
6. [File Creation and Storage](#file-creation-and-storage)
7. [Non-Blackboard Communication](#non-blackboard-communication)
8. [Data Flow Diagrams](#data-flow-diagrams)
9. [Advanced Topics](#advanced-topics)

## Overview

The Talk system implements a sophisticated multi-agent architecture where autonomous agents communicate with each other through a centralized blackboard system. This document provides a detailed technical explanation of how LLM communications are managed, stored, and used throughout the system.

The core communication architecture follows a blackboard pattern with provenance tracking:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   CodeAgent   │     │   FileAgent   │     │   TestAgent   │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│                      Blackboard                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │                 BlackboardEntry                  │    │
│  │  - id: UUID                                      │    │
│  │  - section: string                               │    │
│  │  - role: string                                  │    │
│  │  - author: string (agent ID)                     │    │
│  │  - label: string (step label)                    │    │
│  │  - content: Any                                  │    │
│  │  - meta: Dict[str, Any]                          │    │
│  │  - ts: timestamp                                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     PlanRunner                           │
│  Executes steps and coordinates agent communication      │
└─────────────────────────────────────────────────────────┘
```

## LLM Communication Management

### Core Communication Flow

1. **Initialization**: Each agent is initialized with a provider configuration (Google/OpenAI)
2. **Backend Setup**: The `_setup_backend()` method in `Agent` class configures the appropriate LLM backend
3. **Message Exchange**: Agents communicate with LLMs through the `call_ai()` method
4. **Response Handling**: Responses are captured, processed, and stored in the conversation history
5. **Persistence**: All communications are logged to disk via `ConversationLog` and the blackboard

### LLM Backend Management

The system uses a pluggable backend architecture to support multiple LLM providers:

```python
# From agent/agent.py
def _setup_backend(self):
    try:
        # Handle both Pydantic v1 and v2 model serialization
        try:
            provider_dict = self.cfg.provider.model_dump(mode="python")
        except AttributeError:
            # Fallback for Pydantic v1
            provider_dict = self.cfg.provider.dict()
        
        self.backend: LLMBackend = get_backend(provider_dict)
        
        if self.cfg.debug.verbose:
            log.info("[agent] Using backend=%s model=%s",
                     self.backend.__class__.__name__,
                     getattr(self.backend, "model_name", "n/a"))
    except Exception as exc:  # Various exception handlers for graceful fallback
        log.exception("Backend error: %s. Falling back to stub backend.", exc)
        self.cfg.debug.mock_mode = True
        self.backend = get_backend({"type": "stub"})
```

### Provider Switching

Agents can dynamically switch between different LLM providers using the `switch_provider` method:

```python
# From agent/agent.py
def switch_provider(self, **provider_kw: Any) -> bool:
    """
    Dynamically switch to a different LLM provider / model.
    """
    try:
        # Normalize common aliases
        if "type" in provider_kw and "provider" not in provider_kw:
            provider_kw["provider"] = provider_kw["type"]

        # Mutate the active Settings instance
        for key, value in provider_kw.items():
            # Update configuration
            if hasattr(self.cfg.llm, key):
                setattr(self.cfg.llm, key, value)
                continue
            if hasattr(self.cfg.provider, key):
                setattr(self.cfg.provider, key, value)

        # Re-establish backend with new configuration
        self._setup_backend()
        return True
    except Exception as exc:
        log.exception("Could not switch provider: %s", exc)
        return False
```

## Class Responsibilities

### Agent Class

The `Agent` class (`agent/agent.py`) is the foundation for all LLM communication:

- **Responsibilities**:
  - Manages LLM backend configuration and initialization
  - Handles conversation history via `ConversationLog`
  - Provides the core `run()` and `call_ai()` methods for LLM interaction
  - Implements provider switching functionality
  - Maintains agent identity with UUID-based IDs

- **Key Methods**:
  - `run(user_text)`: Main entry point for agent execution
  - `call_ai()`: Makes the actual LLM API call
  - `switch_provider(**kwargs)`: Changes LLM provider dynamically
  - `_append(role, content)`: Adds messages to conversation history
  - `_pydantic_msgs()`: Converts history to Message objects

### Specialized Agents

Specialized agents inherit from the base `Agent` class and add domain-specific functionality:

#### CodeAgent (`special_agents/code_agent.py`)
- **Responsibilities**:
  - Generates code changes as unified diffs
  - Extracts code from markdown responses
  - Validates diff format
  - Handles error recovery

#### FileAgent (`special_agents/file_agent.py`)
- **Responsibilities**:
  - Applies diffs to create/modify files
  - Creates backups of modified files
  - Lists files in the working directory
  - Reads file content
  - Ensures file operations stay within the working directory

#### TestAgent (`special_agents/test_agent.py`)
- **Responsibilities**:
  - Runs tests with various frameworks (pytest, unittest)
  - Parses test results into structured format
  - Handles test timeouts
  - Provides detailed test failure analysis

### Blackboard Class

The `Blackboard` class (`plan_runner/blackboard.py`) is the central communication hub:

- **Responsibilities**:
  - Stores all agent communications with provenance
  - Provides async-safe CRUD operations
  - Supports querying by section, role, author, or label
  - Offers both async and sync interfaces

### PlanRunner Class

The `PlanRunner` class (`plan_runner/plan_runner.py`) orchestrates the execution flow:

- **Responsibilities**:
  - Executes steps in the correct order
  - Routes messages between agents via the blackboard
  - Handles parallel execution
  - Manages control flow based on step success/failure

### TalkOrchestrator Class

The `TalkOrchestrator` class (`talk/talk.py`) is the top-level coordinator:

- **Responsibilities**:
  - Creates and initializes all agents
  - Sets up the execution plan
  - Manages interactive/non-interactive modes
  - Handles timeouts and interrupts
  - Persists blackboard state to disk

## Blackboard System

### BlackboardEntry Structure

The `BlackboardEntry` class defines the structure of all communications:

```python
@dataclass(slots=True)
class BlackboardEntry:
    """
    A single record on the blackboard.

    section : logical namespace (e.g. "tasks", "artifacts")
    role    : usually "user", "assistant", or domain-specific tag
    author  : agent id that wrote the entry (provenance)
    label   : plan-step label that produced the entry
    meta    : arbitrary small dict for extra flags / scores
    """

    id: str = field(default_factory=_uuid)
    section: str = "default"
    role: str = "assistant"
    author: str = "system"
    label: str = "unlabelled"
    content: Any = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
```

### Blackboard Operations

The `Blackboard` class provides both async and sync methods for CRUD operations:

```python
# Async methods
async def add(self, entry: BlackboardEntry) -> None: ...
async def update(self, entry_id: str, **patch) -> None: ...
async def get(self, entry_id: str) -> Optional[BlackboardEntry]: ...
async def query(self, *, section=None, role=None, author=None, label=None) -> List[BlackboardEntry]: ...

# Sync methods
def add_sync(self, label: str, content: Any, **kwargs) -> None: ...
def entries(self) -> List[BlackboardEntry]: ...
def query_sync(self, *, section=None, role=None, author=None, label=None) -> List[BlackboardEntry]: ...
```

### Blackboard Persistence

The blackboard state is persisted to disk as JSON in the `TalkOrchestrator.run()` method:

```python
# From talk/talk.py
try:
    blackboard_file = self.working_dir / "blackboard.json"
    with open(blackboard_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": self.task,
                "working_dir": str(self.working_dir),
                "entries": [
                    {
                        "id": str(entry.id),
                        "section": entry.section,
                        "label": entry.label,
                        "author": entry.author,
                        "content": entry.content,
                        "timestamp": (
                            entry.ts
                            if isinstance(entry.ts, str)
                            else datetime.fromtimestamp(entry.ts).isoformat()
                            if isinstance(entry.ts, (int, float))
                            else str(entry.ts)
                        ),
                    }
                    for entry in self.blackboard.entries()
                ],
            },
            f,
            indent=2,
        )
    print(f"Blackboard state saved to: {blackboard_file}")
except Exception as e:
    log.warning("Failed to save blackboard state: %s", e)
```

## Agent-to-Agent Communication

### Communication Flow

Agents communicate with each other exclusively through the blackboard. The typical flow is:

1. `PlanRunner` executes a step by calling an agent's `run()` method
2. The agent processes the input and returns a result
3. `PlanRunner` adds the result to the blackboard with the step's label
4. The next agent receives this result as input to its `run()` method

```
┌────────────┐          ┌────────────┐          ┌────────────┐
│ CodeAgent  │          │ FileAgent  │          │ TestAgent  │
└─────┬──────┘          └─────┬──────┘          └─────┬──────┘
      │                       │                       │
      │ 1. Generate code      │                       │
      │ as unified diff       │                       │
      ▼                       │                       │
┌─────────────────────────────┴───────────────────────┴─────┐
│                      Blackboard                            │
│                                                            │
│  Entry: {                                                  │
│    "id": "67961e3f147a495c9b202f8a14540187",              │
│    "label": "generate_code",                               │
│    "author": "CodeAgent-abc123",                           │
│    "content": "--- a/hello.py\n+++ b/hello.py\n@@ ..."     │
│  }                                                         │
└─────────────────────────────┬───────────────────────┬─────┘
                              │                       │
                              │ 2. Apply diff         │
                              │ to create file        │
                              ▼                       │
                        ┌────────────┐               │
                        │ FileAgent  │               │
                        └─────┬──────┘               │
                              │                       │
                              │ 3. Report             │
                              │ file created          │
                              ▼                       │
┌─────────────────────────────────────────────────────┴─────┐
│                      Blackboard                            │
│                                                            │
│  Entry: {                                                  │
│    "id": "7ae8715f19d744ab8c2325928c7ee407",              │
│    "label": "apply_changes",                               │
│    "author": "FileAgent-def456",                           │
│    "content": "PATCH_APPLIED: hello.py\n..."              │
│  }                                                         │
└─────────────────────────────────────────────────┬─────────┘
                                                  │
                                                  │ 4. Run tests
                                                  │ on created file
                                                  ▼
                                            ┌────────────┐
                                            │ TestAgent  │
                                            └─────┬──────┘
                                                  │
                                                  │ 5. Report
                                                  │ test results
                                                  ▼
┌────────────────────────────────────────────────────────────┐
│                      Blackboard                             │
│                                                             │
│  Entry: {                                                   │
│    "id": "630f4e37478c409d8ec11e56068ce1d5",               │
│    "label": "run_tests",                                    │
│    "author": "TestAgent-ghi789",                            │
│    "content": "TEST_RESULTS: FAILURE\nExit Code: 4\n..."    │
│  }                                                          │
└────────────────────────────────────────────────────────────┘
```

### Implementation in PlanRunner

The `PlanRunner._run_single()` method is where agent communication happens:

```python
def _run_single(self, step: Step, prompt: str) -> str:
    log.debug("→ %s", step.label)
    agent = self.agents[step.agent_key]
    result = agent.run(prompt)
    self.bb.add(step.label, result)
    return result
```

## File Creation and Storage

### File Creation Process

File creation happens in the `FileAgent._apply_diff()` method, which:

1. Takes a unified diff as input
2. Creates a temporary file for the diff
3. Extracts affected file paths
4. Creates backups of affected files
5. Applies the patch using the system's `patch` command
6. Returns a status message

```python
# From special_agents/file_agent.py
def _apply_diff(self, diff_text: str) -> str:
    """
    Apply a unified diff using the system's patch command.
    """
    if not diff_text.strip():
        return "No changes to apply (empty diff)"
    
    # Create a temporary file for the diff
    with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as temp_diff:
        temp_diff.write(diff_text)
        diff_path = temp_diff.name
    
    try:
        # Extract affected file paths from the diff
        affected_files = self._extract_file_paths(diff_text)
        
        # Create backup of affected files
        backup_paths = self._backup_files(affected_files)
        
        # Apply the patch
        # Using -p1 to strip the first path component (a/ and b/ prefixes)
        # Using --forward to apply only if the patch can be applied in forward direction
        result = subprocess.run(
            ["patch", "-p1", "--forward", "-i", diff_path],
            cwd=str(self.base_dir),
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        # Clean up the temporary diff file
        os.unlink(diff_path)
        
        # Process the result
        if result.returncode == 0:
            # Success
            return f"PATCH_APPLIED: {', '.join(affected_files)}\n{result.stdout}"
        else:
            # Failed to apply patch
            self._restore_backups(backup_paths)
            return f"PATCH_FAILED: {result.stderr}\n{result.stdout}"
    except Exception as e:
        # Handle exceptions
        return f"ERROR: {str(e)}"
```

### File Storage Locations

Files are created and stored in the following locations:

1. **Working Directory**: Set during `TalkOrchestrator` initialization
   ```python
   # From talk/talk.py
   self.working_dir = self._create_versioned_dir(base_dir)
   ```

2. **Versioned Directories**: Created as `talk1`, `talk2`, etc.
   ```python
   # From talk/talk.py
   def _create_versioned_dir(self, base_dir: Optional[str] = None) -> Path:
       """
       Create a versioned directory for this Talk session.
       """
       # Use the provided base directory or the current directory
       base = Path(base_dir or ".").expanduser().resolve()
       
       # Find the next available version number
       version = 1
       while True:
           versioned_dir = base / f"talk{version}"
           if not versioned_dir.exists():
               break
           version += 1
       
       # Create the directory
       versioned_dir.mkdir(parents=True, exist_ok=True)
       log.info("Working directory: %s", versioned_dir)
       
       return versioned_dir
   ```

3. **Backup Directory**: Created as `.talk_backups` within the working directory
   ```python
   # From special_agents/file_agent.py
   def _backup_files(self, file_paths: List[str]) -> Dict[str, str]:
       """Create backups of files that will be modified."""
       backup_dir = self.base_dir / ".talk_backups"
       backup_dir.mkdir(exist_ok=True)
       
       backup_paths = {}
       for file_path in file_paths:
           file_obj = self.base_dir / file_path
           if file_obj.exists():
               timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
               backup_name = f"{file_path}_{timestamp}.bak"
               backup_path = backup_dir / backup_name
               shutil.copy2(file_obj, backup_path)
               backup_paths[file_path] = str(backup_path)
               log.info(f"Backed up {file_path} to {backup_path}")
       
       return backup_paths
   ```

### File Safety Mechanisms

The `FileAgent` implements several safety mechanisms:

1. **Path Validation**: Ensures all file operations stay within the base directory
   ```python
   def _is_safe_path(self, path: str) -> bool:
       """Check if a path is within the base directory."""
       target_path = (self.base_dir / path).resolve()
       return str(target_path).startswith(str(self.base_dir.resolve()))
   ```

2. **Automatic Backups**: Creates backups before modifying files
3. **Rollback on Failure**: Restores backups if patch application fails

## Non-Blackboard Communication

While the blackboard is the primary communication mechanism, there are a few instances of direct communication:

### Agent-to-LLM Communication

The `Agent.call_ai()` method communicates directly with the LLM backend:

```python
def call_ai(self) -> str:
    if self.cfg.debug.mock_mode:
        return "[mock] This would be the assistant reply."

    try:
        assistant_msg = self.backend.complete(self._pydantic_msgs())
        return assistant_msg.content or ""
    except LLMBackendError as exc:
        log.error("Backend error: %s", exc)
        return f"[backend error] {exc}"
```

### Conversation Logging

The `ConversationLog` class handles direct file I/O for persistence:

```python
# From agent/storage.py
def append(self, msg: Message):
    """Append a message and flush to disk if needed."""
    self._messages.append(msg)
    self._since_flush += 1
    if self._since_flush >= self.flush_every:
        self._flush_to_disk()
        self._since_flush = 0

def _flush_to_disk(self):
    """Write recent messages to disk."""
    with open(self.path, "a", encoding="utf-8") as f:
        for msg in self._messages[-self._since_flush:]:
            f.write(json.dumps(msg.model_dump()) + "\n")
```

### File System Interaction

The `FileAgent` interacts directly with the file system:

```python
# From special_agents/file_agent.py
def read_file(self, file_path: str) -> str:
    """Read the content of a file."""
    if not self._is_safe_path(file_path):
        return f"ERROR: Path {file_path} is outside the base directory"
    
    try:
        file_obj = self.base_dir / file_path
        if not file_obj.exists():
            return f"ERROR: File {file_path} does not exist"
        
        with open(file_obj, 'r') as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"ERROR: {str(e)}"
```

## Data Flow Diagrams

### Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TalkOrchestrator                          │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Create Plan │  │ Init Agents │  │ Setup Working Dir   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────┬───────────┘  │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     PlanRunner                          │ │
│  │                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │ Execute Step│  │ Route Msgs  │  │ Handle Failures │ │ │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │ │
│  └─────────┼─────────────────┼─────────────────┼───────────┘ │
└─────────────────────────────────────────────────────────────┘
              │                 │                 │
              ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                        Blackboard                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    BlackboardEntry                      │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────┬─────────────────┬────────────────────┬───────────┘
           │                 │                    │
           ▼                 ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  CodeAgent   │    │  FileAgent   │    │    TestAgent     │
│              │    │              │    │                  │
│ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────────┐ │
│ │Generate  │ │    │ │Apply Diff│ │    │ │Run Tests     │ │
│ │Diff      │ │    │ │          │ │    │ │              │ │
│ └────┬─────┘ │    │ └────┬─────┘ │    │ └──────┬───────┘ │
│      │       │    │      │       │    │        │         │
│      ▼       │    │      ▼       │    │        ▼         │
│ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────────┐ │
│ │LLM API   │ │    │ │File I/O  │ │    │ │Test Runner   │ │
│ │Call      │ │    │ │          │ │    │ │              │ │
│ └──────────┘ │    │ └──────────┘ │    │ └──────────────┘ │
└──────────────┘    └──────────────┘    └──────────────────┘
```

### Message Flow for File Creation

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  User Input   │     │  Blackboard   │     │   File System │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │ 1. Task Description │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │                     │ 2. Store Task       │
        │                     ├────────────────────►│
        │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│   CodeAgent   │     │   Blackboard  │     │   File System │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │ 3. Generate Diff    │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │                     │ 4. Store Diff       │
        │                     ├────────────────────►│
        │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│   FileAgent   │     │   Blackboard  │     │   File System │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │ 5. Read Diff        │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │ 6. Create Temp File │                     │
        ├─────────────────────────────────────────►│
        │                     │                     │
        │ 7. Create Backups   │                     │
        ├─────────────────────────────────────────►│
        │                     │                     │
        │ 8. Apply Patch      │                     │
        ├─────────────────────────────────────────►│
        │                     │                     │
        │ 9. Report Result    │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │                     │ 10. Store Result    │
        │                     ├────────────────────►│
        │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│   TestAgent   │     │   Blackboard  │     │   File System │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │ 11. Read Results    │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │ 12. Run Tests       │                     │
        ├─────────────────────────────────────────►│
        │                     │                     │
        │ 13. Report Results  │                     │
        ├────────────────────►│                     │
        │                     │                     │
        │                     │ 14. Store Results   │
        │                     ├────────────────────►│
        │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│TalkOrchestrator│    │   Blackboard  │     │   File System │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │ 15. Save Blackboard │                     │
        ├─────────────────────────────────────────►│
        │                     │                     │
```

## Advanced Topics

### Provenance Tracking

The system implements thorough provenance tracking through:

1. **UUID-based Agent Identity**: Each agent has a unique ID
   ```python
   # From agent/agent.py
   self.id = id or f"{self.name}-{uuid.uuid4().hex[:8]}"
   ```

2. **BlackboardEntry Metadata**: Each entry tracks its author and timestamp
   ```python
   # From plan_runner/blackboard.py
   @dataclass(slots=True)
   class BlackboardEntry:
       id: str = field(default_factory=_uuid)
       author: str = "system"
       ts: float = field(default_factory=time.time)
   ```

3. **Step Labeling**: Each plan step has a unique label for tracing
   ```python
   # From plan_runner/step.py
   @dataclass
   class Step:
       label: str
       agent_key: str
       message: str
   ```

### Error Handling and Recovery

The system implements robust error handling:

1. **Agent Backend Fallback**: Falls back to stub backend if API keys are missing
   ```python
   # From agent/agent.py
   except LLMBackendError as exc:
       log.exception("Backend initialization error: %s. Falling back to stub backend.", exc)
       self.cfg.debug.mock_mode = True
       self.backend = get_backend({"type": "stub"})
   ```

2. **File Operation Safety**: Validates paths and creates backups
   ```python
   # From special_agents/file_agent.py
   def _is_safe_path(self, path: str) -> bool:
       """Check if a path is within the base directory."""
       target_path = (self.base_dir / path).resolve()
       return str(target_path).startswith(str(self.base_dir.resolve()))
   ```

3. **Timeout Handling**: Implements timeouts for long-running operations
   ```python
   # From talk/talk.py
   def _timeout_handler(self, signum, frame):
       """Handle timeout by logging and exiting gracefully."""
       elapsed_minutes = (time.time() - self.start_time) / 60
       log.error(f"Execution timed out after {elapsed_minutes:.1f} minutes")
       print(f"\n[WARNING] Execution timed out after {elapsed_minutes:.1f} minutes")
   ```

### Blackboard Persistence

The blackboard state is persisted to disk as JSON in the `TalkOrchestrator.run()` method:

```python
# From talk/talk.py
blackboard_file = self.working_dir / "blackboard.json"
with open(blackboard_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "task": self.task,
            "working_dir": str(self.working_dir),
            "entries": [
                {
                    "id": str(entry.id),
                    "section": entry.section,
                    "label": entry.label,
                    "author": entry.author,
                    "content": entry.content,
                    "timestamp": (
                        entry.ts
                        if isinstance(entry.ts, str)
                        else datetime.fromtimestamp(entry.ts).isoformat()
                        if isinstance(entry.ts, (int, float))
                        else str(entry.ts)
                    ),
                }
                for entry in self.blackboard.entries()
            ],
        },
        f,
        indent=2,
    )
```

This persistence enables:
- Post-run analysis of agent interactions
- Debugging of complex workflows
- Audit trails for all file modifications
- Resumption of interrupted workflows
