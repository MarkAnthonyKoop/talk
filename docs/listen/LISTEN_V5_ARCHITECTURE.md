# Listen v5 Architecture: System Automation & Command Interface

## Overview

Listen v5 extends v4's conversational AI with **actionable intelligence** - the ability to understand intent, plan actions, and execute system commands safely. The architecture leverages MCP (Model Context Protocol) for secure command execution while maintaining Listen's conversational nature.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Listen v5 Core                          │
├─────────────────────────────────────────────────────────────────┤
│                    v4 Capabilities                             │
│  • Speaker ID & Diarization    • Context Detection            │
│  • Voice Enrollment            • Response Generation           │
│  • Conversation Management     • TTS Integration               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEW v5 Components                           │
├─────────────────────────────────────────────────────────────────┤
│                Intent Detection & Classification                │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ ActionIntentAgent│  │ SafetyValidator │  │ CommandPlanner │  │
│  │                  │  │                 │  │                │  │
│  │ • Conversation   │  │ • Whitelist     │  │ • Task         │  │
│  │ • Action Request │  │ • Confirmation  │  │   Breakdown    │  │
│  │ • Mixed Mode     │  │ • Audit Trail   │  │ • Dependency   │  │
│  └──────────────────┘  └─────────────────┘  │   Resolution   │  │
│                                             └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   System Action Layer                          │
├─────────────────────────────────────────────────────────────────┤
│   ┌──────────────────────────────────────────────────────────┐   │
│   │              MCP Integration Manager                     │   │
│   │  • Shell command routing                                │   │
│   │  • Server health monitoring                             │   │
│   │  • Protocol validation                                  │   │
│   └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│   │FileSystemMCP│  │ProcessMCP   │  │WebActionMCP │             │
│   │             │  │             │  │             │             │
│   │• List/Read  │  │• ps/top     │  │• curl/wget  │             │
│   │• Write/Move │  │• start/stop │  │• API calls  │             │
│   │• Organize   │  │• monitoring │  │• downloads  │             │
│   └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Specialized Domain Agents                         │
├─────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │GitAgent     │  │DevToolsAgent│  │SystemMaintenanceAgent  │ │
│   │             │  │             │  │                         │ │
│   │• status     │  │• npm/pip    │  │• disk cleanup           │ │
│   │• commit/push│  │• docker     │  │• log rotation           │ │
│   │• branch     │  │• testing    │  │• service health         │ │
│   └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Layer                             │
├─────────────────────────────────────────────────────────────────┤
│        MCP Shell Servers (@hdresearch/mcp-shell)               │
│                         │                                       │
│        ┌────────────────┼────────────────┐                     │
│        ▼                ▼                ▼                     │
│   ┌─────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │Security │    │   System    │    │  Audit &    │            │
│   │Sandbox  │    │  Commands   │    │  Logging    │            │
│   │         │    │             │    │             │            │
│   │• Whitelist   │• bash/zsh   │    │• Command log│            │
│   │• Timeout     │• PowerShell │    │• Results    │            │
│   │• Resource    │• File Ops   │    │• Errors     │            │
│   │  Limits      │• Network    │    │• Timing     │            │
│   └─────────────┘└─────────────┘    └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Intent Detection & Classification

#### ActionIntentAgent
```python
class ActionIntentAgent(Agent):
    """Detects when user wants system actions vs conversation."""
    
    def classify_intent(self, text: str, context: List[Dict]) -> IntentClassification:
        """
        Returns:
        - conversation: Regular chat response needed
        - action_request: User wants system action
        - mixed_mode: Both conversation and action needed
        """
```

**Detection Patterns:**
- **Action Keywords**: "run", "execute", "install", "check", "find", "organize"  
- **Command Indicators**: File paths, package names, system references
- **Task Verbs**: "backup", "download", "restart", "update", "clean"
- **Context Clues**: Previous commands, current directory mentions

#### SafetyValidator  
```python
class SafetyValidator:
    """Validates commands for security before execution."""
    
    DANGER_COMMANDS = ["rm -rf", "sudo rm", "format", "dd if=", ":(){ :|:& };:"]
    REQUIRES_CONFIRMATION = ["rm", "mv", "chmod", "chown", "sudo"]
    SAFE_COMMANDS = ["ls", "ps", "df", "top", "cat", "grep", "find"]
    
    def validate_command(self, command: str, context: Dict) -> ValidationResult:
        """Multi-layer security validation."""
```

#### CommandPlanner
```python
class CommandPlanner(Agent):  
    """Breaks down complex requests into safe, executable steps."""
    
    def plan_execution(self, request: str, context: Dict) -> ExecutionPlan:
        """
        Returns structured plan with:
        - Sequential steps
        - Dependencies 
        - Safety checkpoints
        - Rollback points
        """
```

### 2. MCP Integration Layer

#### MCP Integration Manager
```python
class MCPIntegrationManager:
    """Manages communication with MCP shell servers."""
    
    def __init__(self):
        self.servers = {
            "shell": MCPShellServer("@hdresearch/mcp-shell"),
            "filesystem": MCPFileSystemServer(), 
            "process": MCPProcessServer(),
            "web": MCPWebActionServer()
        }
    
    async def execute_command(self, server: str, command: Dict) -> MCPResponse:
        """Route command to appropriate MCP server with monitoring."""
```

### 3. Specialized Domain Agents

#### GitAgent
```python
class GitAgent(Agent):
    """Handles Git repository operations safely."""
    
    async def run(self, prompt: str) -> str:
        # Parse git-specific commands
        # Validate repository state
        # Execute via MCP with appropriate permissions
```

#### DevToolsAgent  
```python
class DevToolsAgent(Agent):
    """Manages development tools and package managers."""
    
    SUPPORTED_TOOLS = {
        "python": ["pip", "poetry", "conda"],
        "node": ["npm", "yarn", "pnpm"], 
        "docker": ["build", "run", "ps", "logs"],
        "testing": ["pytest", "jest", "go test"]
    }
```

#### SystemMaintenanceAgent
```python
class SystemMaintenanceAgent(Agent):
    """Handles routine system maintenance tasks."""
    
    def cleanup_logs(self) -> str:
        """Safely clean old log files with size/date limits."""
        
    def check_disk_space(self) -> str:  
        """Monitor disk usage and suggest cleanup actions."""
```

## Security Architecture

### Multi-Layer Security Model

```
┌─────────────────────────────────────────────┐
│           User Request                      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│    Layer 1: Intent Validation              │
│    • Action vs conversation detection      │
│    • Context appropriateness check         │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│    Layer 2: Command Planning               │  
│    • Task breakdown & dependency check     │
│    • Risk assessment per step              │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│    Layer 3: Safety Validation              │
│    • Whitelist/blacklist checking          │
│    • Destructive operation detection       │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│    Layer 4: User Confirmation              │
│    • High-risk operations require approval │
│    • Clear explanation of consequences     │
└─────────────────┬───────────────────────────┘
                  │  
┌─────────────────▼───────────────────────────┐
│    Layer 5: MCP Server Security            │
│    • Server-side validation & sandboxing   │
│    • Resource limits & timeout controls    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐  
│    Layer 6: System Execution               │
│    • Audit logging of all actions          │
│    • Result validation & error handling    │
└─────────────────────────────────────────────┘
```

### Permission Levels

```python
class PermissionLevel(Enum):
    READ_ONLY = 1      # ls, cat, ps, df - safe read operations
    SAFE_WRITE = 2     # mkdir, touch, echo - non-destructive writes  
    STANDARD = 3       # cp, mv (non-system), chmod (user files)
    ELEVATED = 4       # sudo operations, system file modifications
    DANGEROUS = 5      # rm -rf, format, dd - requires confirmation
```

## Integration with Listen v4

### Enhanced Components

#### Updated ContextRelevanceAgent
```python
# v4: Only detected conversation triggers
should_respond = detect_wake_words() or detect_questions()

# v5: Also detects action intents  
should_respond = detect_conversation() or detect_action_intent()
action_mode = classify_intent_type()  # conversation | action | mixed
```

#### Enhanced ResponseGenerator
```python
# v4: Only generated conversational responses
response = generate_conversation_response()

# v5: Routes to appropriate handler
if intent_type == "conversation":
    response = generate_conversation_response()
elif intent_type == "action": 
    response = await execute_system_action()
elif intent_type == "mixed":
    response = await handle_mixed_mode()
```

## Example Usage Scenarios

### 1. File Organization
```
User: "Hey Listen, organize my Downloads folder"

Flow:
1. ActionIntentAgent → action_request
2. CommandPlanner → breaks into steps:
   - List Downloads contents
   - Categorize by file type  
   - Create organized folders
   - Move files safely
3. SafetyValidator → validates each step
4. FileSystemMCP → executes via MCP shell server
5. Response → "Organized 47 files into Documents, Images, and Archives"
```

### 2. Development Workflow
```
User: "Check if my server is running and restart it if needed"

Flow:
1. ActionIntentAgent → action_request  
2. CommandPlanner → multi-step plan:
   - Check process status
   - If not running, restart service
   - Verify restart success
3. ProcessMCP → ps aux | grep server
4. SystemMaintenanceAgent → sudo systemctl restart myserver
5. Confirmation → "Server was down, restarted successfully"
```

### 3. Mixed Mode Conversation
```
User: "What time is it? Also backup my code folder"

Flow:
1. ActionIntentAgent → mixed_mode
2. Parallel execution:
   - ResponseGenerator → "It's 3:47 PM"  
   - GitAgent → backs up code folder
3. Combined response → "It's 3:47 PM. I've also backed up your code folder to ~/backups/code_20250822.tar.gz"
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] ActionIntentAgent implementation
- [ ] SafetyValidator with basic whitelist/blacklist
- [ ] MCP Integration Manager setup
- [ ] Basic file operations via MCP

### Phase 2: Command Planning (Week 3-4) 
- [ ] CommandPlanner with task breakdown
- [ ] User confirmation workflows
- [ ] Audit logging system
- [ ] Error recovery mechanisms

### Phase 3: Domain Agents (Week 5-6)
- [ ] GitAgent for repository operations  
- [ ] DevToolsAgent for package management
- [ ] SystemMaintenanceAgent for cleanup tasks
- [ ] Web interaction capabilities

### Phase 4: Advanced Features (Week 7-8)
- [ ] Mixed mode conversation handling
- [ ] Learning from user patterns
- [ ] Performance optimization
- [ ] Comprehensive testing

## Configuration

### MCP Server Setup
```json
{
  "mcp_servers": {
    "shell": {
      "command": "npx",
      "args": ["@hdresearch/mcp-shell"],
      "timeout": 30000,
      "security": {
        "whitelist": ["ls", "ps", "df", "cat", "grep"],
        "blacklist": ["rm -rf", "sudo rm", "format"],
        "require_confirmation": ["rm", "mv", "chmod"]
      }
    }
  }
}
```

### Safety Configuration  
```json
{
  "safety": {
    "max_command_length": 1000,
    "execution_timeout": 60,
    "require_confirmation_for": [
      "destructive_operations",
      "system_modifications", 
      "network_operations"
    ],
    "audit_log_path": "~/.listen/audit.log"
  }
}
```

## Success Metrics

1. **Safety**: Zero accidental destructive operations
2. **Usability**: 90%+ user requests handled correctly  
3. **Performance**: <5 second response time for simple commands
4. **Reliability**: 99%+ uptime for MCP server connections
5. **User Satisfaction**: Clear, helpful responses for all scenarios

This architecture provides a robust foundation for Listen v5's system automation capabilities while maintaining the conversational intelligence that makes Listen unique.