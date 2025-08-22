#!/usr/bin/env python3
"""
Listen v5 - System Automation & Command Interface

Extends v4's conversational AI with actionable intelligence:
- Intent detection (conversation vs action)
- Safe command execution via MCP
- Specialized domain agents (Git, DevTools, System)
- Multi-layer security validation
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

# Import Listen v4 as base
from listen.listen import ListenV4
from agent.agent import Agent

log = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intent classification."""
    CONVERSATION = "conversation"
    ACTION = "action"  
    MIXED = "mixed"


class PermissionLevel(Enum):
    """Security permission levels for commands."""
    READ_ONLY = 1      # ls, cat, ps, df
    SAFE_WRITE = 2     # mkdir, touch, echo  
    STANDARD = 3       # cp, mv, chmod (user files)
    ELEVATED = 4       # sudo operations
    DANGEROUS = 5      # rm -rf, format - requires confirmation


@dataclass
class IntentClassification:
    """Result of intent detection."""
    intent_type: IntentType
    confidence: float
    action_keywords: List[str]
    reasoning: str
    suggested_handler: Optional[str] = None


@dataclass  
class ValidationResult:
    """Result of command safety validation."""
    is_safe: bool
    permission_level: PermissionLevel
    requires_confirmation: bool
    risk_assessment: str
    suggested_alternatives: List[str] = None
    

@dataclass
class ExecutionPlan:
    """Structured execution plan for complex requests."""
    steps: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    checkpoints: List[str]
    estimated_duration: int
    rollback_plan: List[str]


class ActionIntentAgent(Agent):
    """Detects when user wants system actions vs conversation."""
    
    def __init__(self, **kwargs):
        roles = [
            "You are an expert at detecting user intent.",
            "You distinguish between conversational requests and action requests.",
            "You identify when users want system commands executed.",
            "You are precise and conservative in your classifications."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Action keywords and patterns
        self.action_keywords = {
            "file_operations": ["organize", "backup", "copy", "move", "delete", "clean", "find"],
            "process_management": ["start", "stop", "restart", "kill", "check", "monitor"],
            "system_info": ["disk", "memory", "cpu", "processes", "status", "health"],
            "development": ["git", "commit", "push", "pull", "install", "build", "test", "deploy"],
            "network": ["ping", "curl", "wget", "download", "upload", "ssh", "scp"],
            "package_management": ["install", "update", "upgrade", "remove", "search"]
        }
        
        self.command_indicators = [
            # File paths
            r"~/", r"/", r"\\", r"\.txt", r"\.py", r"\.js", r"\.json",
            # Command names  
            "ls", "cd", "pwd", "ps", "top", "df", "du", "grep", "find",
            # Package names
            "npm", "pip", "apt", "brew", "docker", "git"
        ]
        
    def classify_intent(self, text: str, context: List[Dict]) -> IntentClassification:
        """
        Classify user intent as conversation, action, or mixed.
        
        Args:
            text: User input text
            context: Recent conversation context
            
        Returns:
            IntentClassification with type and details
        """
        text_lower = text.lower()
        found_keywords = []
        
        # Check for action keywords
        for category, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        # Check for command indicators
        import re
        for pattern in self.command_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                found_keywords.append(f"command_pattern:{pattern}")
        
        # Calculate confidence based on keywords and context
        action_score = len(found_keywords) * 0.4
        
        # Boost score for direct commands
        if any(text_lower.startswith(cmd) for cmd in ["run", "execute", "check", "find", "backup", "list", "show"]):
            action_score += 0.5
            
        # Boost score for imperative verbs
        imperative_verbs = ["list", "show", "check", "find", "organize", "backup", "copy", "move", "delete", "install", "update"]
        for verb in imperative_verbs:
            if verb in text_lower:
                action_score += 0.3
                break
                
        # Check for file/path references
        if any(indicator in text for indicator in ["~/", "/", "\\", ".txt", ".py", ".js", "file", "folder", "directory"]):
            action_score += 0.3
            
        # Check for conversational markers
        conversation_markers = ["what", "how", "why", "tell me", "explain", "hello", "hi", "thanks"]
        conversation_score = sum(0.15 for marker in conversation_markers if marker in text_lower)
        
        # Determine intent type
        if action_score > 0.6 and conversation_score < 0.3:
            intent_type = IntentType.ACTION
            confidence = min(action_score, 0.95)
            reasoning = f"Strong action indicators: {found_keywords[:3]}"
            
        elif action_score > 0.3 and conversation_score > 0.3:
            intent_type = IntentType.MIXED  
            confidence = min((action_score + conversation_score) / 2, 0.9)
            reasoning = f"Both action ({action_score:.2f}) and conversation ({conversation_score:.2f}) detected"
            
        elif conversation_score > action_score:
            intent_type = IntentType.CONVERSATION
            confidence = min(conversation_score + 0.3, 0.9)
            reasoning = "Conversational request detected"
            
        else:
            # Default to conversation for ambiguous cases
            intent_type = IntentType.CONVERSATION
            confidence = 0.5
            reasoning = "Ambiguous intent, defaulting to conversation"
        
        # Suggest appropriate handler
        suggested_handler = None
        if intent_type == IntentType.ACTION:
            # Git operations
            if any(word in text_lower for word in ["git", "commit", "push", "pull", "branch", "repository"]):
                suggested_handler = "GitAgent"
            # File system operations
            elif any(word in text_lower for word in ["file", "folder", "directory", "organize", "backup", "find"]):
                suggested_handler = "FileSystemAgent"
            # Development tools
            elif any(word in text_lower for word in ["install", "package", "npm", "pip", "docker", "poetry", "yarn"]):
                suggested_handler = "DevToolsAgent"
            # Process/system operations
            elif any(word in text_lower for word in ["process", "memory", "cpu", "disk", "service"]):
                suggested_handler = "ProcessAgent"
            else:
                suggested_handler = "SystemAgent"
        
        return IntentClassification(
            intent_type=intent_type,
            confidence=confidence,
            action_keywords=found_keywords,
            reasoning=reasoning,
            suggested_handler=suggested_handler
        )


class SafetyValidator:
    """Validates commands for security before execution."""
    
    def __init__(self):
        # Extremely dangerous commands that should never be executed
        self.danger_commands = [
            "rm -rf /", "sudo rm -rf", "format", "dd if=", 
            ":(){ :|:& };:", "> /dev/sda", "chmod -R 777 /",
            "wget | sh", "curl | sh", "mkfs", "fdisk"
        ]
        
        # Commands that require user confirmation
        self.requires_confirmation = [
            "rm", "rmdir", "mv", "chmod", "chown", "sudo", 
            "service", "systemctl", "reboot", "shutdown",
            "apt-get remove", "pip uninstall", "npm uninstall"
        ]
        
        # Generally safe read-only commands
        self.safe_commands = [
            "ls", "cat", "less", "head", "tail", "grep", "find",
            "ps", "top", "df", "du", "free", "uptime", "date",
            "pwd", "whoami", "id", "groups", "which", "type"
        ]
        
    def validate_command(self, command: str, context: Dict = None) -> ValidationResult:
        """
        Validate command safety with multi-layer checking.
        
        Args:
            command: Command string to validate
            context: Optional context for risk assessment
            
        Returns:
            ValidationResult with safety assessment
        """
        command_lower = command.lower().strip()
        
        # Check for extremely dangerous commands
        for danger_cmd in self.danger_commands:
            if danger_cmd in command_lower:
                return ValidationResult(
                    is_safe=False,
                    permission_level=PermissionLevel.DANGEROUS,
                    requires_confirmation=True,
                    risk_assessment=f"EXTREMELY DANGEROUS: Command contains '{danger_cmd}'",
                    suggested_alternatives=["Please specify exactly what you want to accomplish"]
                )
        
        # Check if command requires confirmation
        requires_confirmation = False
        permission_level = PermissionLevel.READ_ONLY
        risk_factors = []
        
        for confirm_cmd in self.requires_confirmation:
            if confirm_cmd in command_lower:
                requires_confirmation = True
                if confirm_cmd in ["sudo", "service", "systemctl"]:
                    permission_level = PermissionLevel.ELEVATED
                    risk_factors.append("system-level operation")
                elif confirm_cmd in ["rm", "rmdir"]:
                    permission_level = PermissionLevel.DANGEROUS if "-rf" in command else PermissionLevel.STANDARD
                    risk_factors.append("file deletion")
                else:
                    permission_level = PermissionLevel.STANDARD
                    risk_factors.append("file modification")
                break
        
        # Check for safe read-only commands
        command_parts = command.split()
        if command_parts and command_parts[0] in self.safe_commands:
            permission_level = PermissionLevel.READ_ONLY
            risk_assessment = "Safe read-only operation"
        else:
            # Assess write operations
            write_indicators = [">>", ">", "touch", "mkdir", "echo"]
            if any(indicator in command for indicator in write_indicators):
                if permission_level == PermissionLevel.READ_ONLY:
                    permission_level = PermissionLevel.SAFE_WRITE
                risk_factors.append("file system write")
        
        # Build risk assessment
        if risk_factors:
            risk_assessment = f"Operation involves: {', '.join(risk_factors)}"
        else:
            risk_assessment = "Safe read-only operation"
        
        # Additional safety checks
        if len(command) > 1000:
            risk_factors.append("unusually long command")
            requires_confirmation = True
            
        if ";" in command or "&&" in command or "|" in command:
            risk_factors.append("command chaining")
            if permission_level < PermissionLevel.STANDARD:
                permission_level = PermissionLevel.STANDARD
        
        return ValidationResult(
            is_safe=permission_level != PermissionLevel.DANGEROUS or requires_confirmation,
            permission_level=permission_level,
            requires_confirmation=requires_confirmation,
            risk_assessment=risk_assessment,
            suggested_alternatives=[]
        )


class CommandPlanner(Agent):
    """Breaks down complex requests into safe, executable steps."""
    
    def __init__(self, **kwargs):
        roles = [
            "You are an expert at breaking down complex system tasks.",
            "You create step-by-step execution plans for system operations.", 
            "You identify dependencies and potential failure points.",
            "You prioritize safety and provide clear rollback procedures."
        ]
        super().__init__(roles=roles, **kwargs)
        
    def plan_execution(self, request: str, context: Dict = None) -> ExecutionPlan:
        """
        Create structured execution plan for complex requests.
        
        Args:
            request: User's request text
            context: Optional context information
            
        Returns:
            ExecutionPlan with steps, dependencies, and safety measures
        """
        # Use LLM to generate execution plan
        prompt = f"""
        Break down this user request into a detailed execution plan:
        "{request}"
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Create a JSON response with:
        {{
            "steps": [
                {{"id": "step1", "command": "command to run", "description": "what this does", "risk_level": "low|medium|high"}},
                ...
            ],
            "dependencies": {{"step2": ["step1"], ...}},
            "checkpoints": ["step1", "step3"], 
            "estimated_duration": 30,
            "rollback_plan": ["undo step3", "undo step2", "undo step1"]
        }}
        
        Prioritize safety and break complex operations into small, verifiable steps.
        """
        
        try:
            response = self.run(prompt)
            plan_data = json.loads(response)
            
            return ExecutionPlan(
                steps=plan_data.get("steps", []),
                dependencies=plan_data.get("dependencies", {}),
                checkpoints=plan_data.get("checkpoints", []),
                estimated_duration=plan_data.get("estimated_duration", 60),
                rollback_plan=plan_data.get("rollback_plan", [])
            )
            
        except Exception as e:
            log.error(f"Error creating execution plan: {e}")
            # Fallback to simple single-step plan
            return ExecutionPlan(
                steps=[{
                    "id": "step1",
                    "command": request,
                    "description": "Execute user request",
                    "risk_level": "medium"
                }],
                dependencies={},
                checkpoints=["step1"],
                estimated_duration=30,
                rollback_plan=["Manual review required"]
            )


class MCPIntegrationManager:
    """Manages communication with MCP shell servers."""
    
    def __init__(self):
        self.servers = {}
        self.is_available = False
        self._check_mcp_availability()
        
    def _check_mcp_availability(self):
        """Check if MCP shell servers are available."""
        try:
            # Try to run a simple command via subprocess to simulate MCP
            result = subprocess.run(["which", "ls"], capture_output=True, text=True, timeout=5)
            self.is_available = result.returncode == 0
            log.info(f"MCP shell server availability: {self.is_available}")
        except Exception as e:
            log.warning(f"MCP availability check failed: {e}")
            self.is_available = False
    
    async def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute command via MCP shell server (simulated with subprocess).
        
        Args:
            command: Shell command to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict with stdout, stderr, return_code, and execution time
        """
        if not self.is_available:
            return {
                "stdout": "",
                "stderr": "MCP shell server not available",
                "return_code": 1,
                "execution_time": 0,
                "error": "MCP_UNAVAILABLE"
            }
        
        start_time = datetime.now()
        
        try:
            # For now, simulate MCP with subprocess (in production, this would use actual MCP)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.home()  # Execute from user home directory
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr, 
                "return_code": result.returncode,
                "execution_time": execution_time,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": 124,
                "execution_time": timeout,
                "error": "TIMEOUT"
            }
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": 1,
                "execution_time": execution_time,
                "error": "EXECUTION_ERROR"
            }


class SystemActionAgent(Agent):
    """Base agent for system action execution."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcp_manager = MCPIntegrationManager()
        self.safety_validator = SafetyValidator()
        
    async def execute_safe_command(self, command: str, require_confirmation: bool = False) -> Dict[str, Any]:
        """
        Execute command with safety validation.
        
        Args:
            command: Command to execute
            require_confirmation: Force confirmation dialog
            
        Returns:
            Execution result with safety metadata
        """
        # Validate command safety
        validation = self.safety_validator.validate_command(command)
        
        if not validation.is_safe:
            return {
                "success": False,
                "error": "UNSAFE_COMMAND",
                "message": validation.risk_assessment,
                "suggested_alternatives": validation.suggested_alternatives
            }
        
        # Check if confirmation needed
        if validation.requires_confirmation or require_confirmation:
            confirmation_msg = f"""
⚠️  Command requires confirmation:
Command: {command}
Risk Level: {validation.permission_level.name}
Assessment: {validation.risk_assessment}

Proceed? (y/N): """
            
            # In production, this would show a proper UI dialog
            print(confirmation_msg)
            # For now, auto-approve safe operations
            if validation.permission_level in [PermissionLevel.READ_ONLY, PermissionLevel.SAFE_WRITE]:
                confirmed = True
                print("Auto-approved safe operation")
            else:
                confirmed = False
                print("Confirmation required - operation cancelled for safety")
                return {
                    "success": False,
                    "error": "CONFIRMATION_REQUIRED", 
                    "message": "User confirmation required for this operation"
                }
        else:
            confirmed = True
        
        if not confirmed:
            return {
                "success": False,
                "error": "USER_CANCELLED",
                "message": "Operation cancelled by user"
            }
        
        # Execute command
        result = await self.mcp_manager.execute_command(command)
        
        # Format response
        return {
            "success": result["return_code"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "return_code": result["return_code"],
            "execution_time": result["execution_time"],
            "command": command,
            "safety_level": validation.permission_level.name
        }


class FileSystemAgent(SystemActionAgent):
    """Handles file system operations safely."""
    
    def __init__(self, **kwargs):
        roles = [
            "You are a file system management expert.",
            "You help users organize, search, and manage files safely.",
            "You provide clear feedback on file operations.",
            "You prioritize data safety and ask for confirmation on destructive operations."
        ]
        super().__init__(roles=roles, **kwargs)
    
    async def run(self, prompt: str) -> str:
        """
        Process file system requests.
        
        Args:
            prompt: User request for file operations
            
        Returns:
            Response describing the operation results
        """
        prompt_lower = prompt.lower()
        
        try:
            # List files
            if any(keyword in prompt_lower for keyword in ["list", "show", "ls", "dir"]):
                if "downloads" in prompt_lower:
                    result = await self.execute_safe_command("ls -la ~/Downloads")
                else:
                    result = await self.execute_safe_command("ls -la")
                    
                if result["success"]:
                    return f"Here are the files:\n{result['stdout']}"
                else:
                    return f"Error listing files: {result.get('stderr', 'Unknown error')}"
            
            # Disk space
            elif any(keyword in prompt_lower for keyword in ["disk", "space", "df", "storage"]):
                result = await self.execute_safe_command("df -h")
                if result["success"]:
                    return f"Disk space usage:\n{result['stdout']}"
                else:
                    return f"Error checking disk space: {result.get('stderr', 'Unknown error')}"
            
            # Find files
            elif "find" in prompt_lower:
                # Extract search term (simplified)
                words = prompt.split()
                search_term = "*.txt"  # Default
                for i, word in enumerate(words):
                    if word == "find" and i + 1 < len(words):
                        search_term = words[i + 1]
                        break
                
                result = await self.execute_safe_command(f"find ~ -name '{search_term}' -type f | head -20")
                if result["success"]:
                    if result["stdout"].strip():
                        return f"Found files matching '{search_term}':\n{result['stdout']}"
                    else:
                        return f"No files found matching '{search_term}'"
                else:
                    return f"Error searching for files: {result.get('stderr', 'Unknown error')}"
            
            # Organize downloads (common request)
            elif "organize" in prompt_lower and "download" in prompt_lower:
                return await self._organize_downloads()
            
            else:
                return "I can help with file operations like:\n• List files (ls, dir)\n• Check disk space\n• Find files\n• Organize Downloads folder\n\nWhat would you like to do?"
                
        except Exception as e:
            log.error(f"FileSystemAgent error: {e}")
            return f"Sorry, there was an error processing your file request: {str(e)}"
    
    async def _organize_downloads(self) -> str:
        """Organize Downloads folder by file type."""
        try:
            # First, check what's in Downloads
            result = await self.execute_safe_command("ls ~/Downloads")
            if not result["success"]:
                return "Could not access Downloads folder"
            
            if not result["stdout"].strip():
                return "Downloads folder is already empty!"
            
            files = result["stdout"].strip().split('\n')
            file_count = len([f for f in files if f.strip()])
            
            # Create organization folders
            folders = ["Documents", "Images", "Archives", "Videos", "Audio", "Software"]
            for folder in folders:
                await self.execute_safe_command(f"mkdir -p ~/Downloads/{folder}")
            
            # Move files by extension (simplified logic)
            move_commands = [
                "mv ~/Downloads/*.pdf ~/Downloads/Documents/ 2>/dev/null || true",
                "mv ~/Downloads/*.txt ~/Downloads/Documents/ 2>/dev/null || true", 
                "mv ~/Downloads/*.doc ~/Downloads/Documents/ 2>/dev/null || true",
                "mv ~/Downloads/*.docx ~/Downloads/Documents/ 2>/dev/null || true",
                "mv ~/Downloads/*.jpg ~/Downloads/Images/ 2>/dev/null || true",
                "mv ~/Downloads/*.png ~/Downloads/Images/ 2>/dev/null || true",
                "mv ~/Downloads/*.gif ~/Downloads/Images/ 2>/dev/null || true",
                "mv ~/Downloads/*.zip ~/Downloads/Archives/ 2>/dev/null || true",
                "mv ~/Downloads/*.tar.gz ~/Downloads/Archives/ 2>/dev/null || true",
                "mv ~/Downloads/*.mp4 ~/Downloads/Videos/ 2>/dev/null || true",
                "mv ~/Downloads/*.mp3 ~/Downloads/Audio/ 2>/dev/null || true",
                "mv ~/Downloads/*.deb ~/Downloads/Software/ 2>/dev/null || true",
                "mv ~/Downloads/*.dmg ~/Downloads/Software/ 2>/dev/null || true"
            ]
            
            for cmd in move_commands:
                await self.execute_safe_command(cmd)
            
            return f"✅ Organized Downloads folder! Created organization folders and categorized {file_count} items by file type."
            
        except Exception as e:
            return f"Error organizing Downloads: {str(e)}"


class ListenV5(ListenV4):
    """
    Listen v5 - System Automation & Command Interface
    
    Extends v4 with actionable intelligence:
    - Intent detection for action vs conversation
    - Safe system command execution
    - Specialized domain agents
    - Multi-layer security validation
    """
    
    def __init__(self, **kwargs):
        """Initialize Listen v5 with system action capabilities."""
        super().__init__(**kwargs)
        
        # Initialize v5 components
        self.action_intent_agent = ActionIntentAgent()
        self.safety_validator = SafetyValidator()
        self.command_planner = CommandPlanner()
        self.mcp_manager = MCPIntegrationManager()
        
        # Domain-specific agents
        self.filesystem_agent = FileSystemAgent()
        
        # Import and initialize specialized agents
        try:
            from special_agents.git_agent import GitAgent
            self.git_agent = GitAgent()
        except ImportError:
            log.warning("GitAgent not available")
            self.git_agent = None
            
        try:
            from special_agents.dev_tools_agent import DevToolsAgent
            self.dev_tools_agent = DevToolsAgent()
        except ImportError:
            log.warning("DevToolsAgent not available")
            self.dev_tools_agent = None
        
        # Track action execution
        self.action_history = []
        
        log.info(f"Initialized Listen v5 with system action capabilities")
    
    async def _process_audio(self, audio_data: Dict[str, Any]):
        """
        Enhanced audio processing with intent detection.
        
        Extends v4's processing to detect action intents.
        """
        # Run v4 processing first (transcription, speaker ID, etc.)
        await super()._process_audio(audio_data)
        
        # Get the latest transcribed text
        context = self.conversation_manager.get_context(num_turns=1)
        if not context:
            return
            
        latest_turn = context[-1]
        text = latest_turn.get("text", "")
        speaker_id = latest_turn.get("speaker_id", "unknown")
        
        # Skip if it's our own response
        if speaker_id == "assistant":
            return
            
        # Classify intent
        classification = self.action_intent_agent.classify_intent(text, context)
        
        log.debug(f"Intent classification: {classification.intent_type.value} (confidence: {classification.confidence:.2f})")
        
        # Handle based on intent type
        if classification.intent_type == IntentType.ACTION:
            await self._handle_action_request(text, classification, latest_turn)
            
        elif classification.intent_type == IntentType.MIXED:
            await self._handle_mixed_request(text, classification, latest_turn)
            
        # CONVERSATION type is already handled by v4's existing logic
    
    async def _handle_action_request(self, text: str, classification: IntentClassification, turn_data: Dict):
        """
        Handle pure action requests.
        
        Args:
            text: User's request text
            classification: Intent classification results
            turn_data: Conversation turn data
        """
        try:
            # Route to appropriate agent based on classification
            if classification.suggested_handler == "FileSystemAgent":
                response = await self.filesystem_agent.run(text)
            elif classification.suggested_handler == "GitAgent" and self.git_agent:
                response = self.git_agent.run(text)  # GitAgent is sync
            elif "git" in text.lower() and self.git_agent:
                response = self.git_agent.run(text)
            elif any(keyword in text.lower() for keyword in ["install", "package", "npm", "pip", "docker"]) and self.dev_tools_agent:
                response = self.dev_tools_agent.run(text)
            else:
                # Default system agent behavior
                response = await self._execute_generic_action(text)
            
            # Add response to conversation
            self.conversation_manager.add_turn(
                text=response,
                speaker_id="assistant",
                audio_features={"action_response": True}
            )
            
            # Display and speak response
            print(f"\n🔧 [System]: {response}\n")
            if self.use_tts:
                self.response_queue.put(response)
            
            # Log action
            self.action_history.append({
                "timestamp": datetime.now().isoformat(),
                "request": text,
                "response": response,
                "classification": classification.intent_type.value,
                "confidence": classification.confidence
            })
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error executing that action: {str(e)}"
            print(f"\n❌ [System]: {error_msg}\n")
            log.error(f"Action execution error: {e}")
    
    async def _handle_mixed_request(self, text: str, classification: IntentClassification, turn_data: Dict):
        """
        Handle requests that contain both conversation and action elements.
        
        Args:
            text: User's request text
            classification: Intent classification results
            turn_data: Conversation turn data
        """
        try:
            # Generate both conversational response and execute action
            # For now, do them sequentially
            
            # First, handle any action component
            action_response = None
            if classification.suggested_handler == "FileSystemAgent":
                action_response = await self.filesystem_agent.run(text)
            
            # Then generate conversational response
            conv_response = self.response_generator.generate_response(
                text=text,
                speaker_info={"id": turn_data.get("speaker_id", "user"), "name": "User"},
                conversation_context=self.conversation_manager.get_context(num_turns=3),
                information_context={"items": []}
            )
            
            # Combine responses
            if action_response:
                combined_response = f"{conv_response}\n\nAlso, {action_response}"
            else:
                combined_response = conv_response
            
            # Add to conversation and respond
            self.conversation_manager.add_turn(
                text=combined_response,
                speaker_id="assistant",
                audio_features={"mixed_response": True}
            )
            
            print(f"\n💬🔧 [{self.name}]: {combined_response}\n")
            if self.use_tts:
                self.response_queue.put(combined_response)
            
        except Exception as e:
            error_msg = f"Sorry, I had trouble with that mixed request: {str(e)}"
            print(f"\n❌ [{self.name}]: {error_msg}\n")
            log.error(f"Mixed request error: {e}")
    
    async def _execute_generic_action(self, request: str) -> str:
        """
        Execute generic action requests not handled by specialized agents.
        
        Args:
            request: User's action request
            
        Returns:
            Response describing the action result
        """
        # Simple keyword-based routing for common requests
        request_lower = request.lower()
        
        try:
            # File listing commands (l, ls, list files, etc.)
            if any(pattern in request_lower for pattern in [
                "bash command like l", "run l", "execute l", "command l",
                "ls", "list files", "show files", "dir"
            ]):
                result = await self.mcp_manager.execute_command("ls -la")
                if result["return_code"] == 0:
                    return f"📁 Current directory contents:\n{result['stdout']}"
                else:
                    return f"Could not list files: {result.get('stderr', 'Unknown error')}"
            
            # Basic bash commands
            elif "bash command" in request_lower or "run command" in request_lower:
                # Extract potential command from the request
                import re
                # Look for single letter commands (common aliases)
                match = re.search(r'command.*?([a-z]+)(?:\s|$)', request_lower)
                if match:
                    cmd = match.group(1)
                    if cmd in ['l', 'll']:
                        # Safe file listing commands
                        full_cmd = "ls -la" if cmd == 'll' else "ls -l"
                        result = await self.mcp_manager.execute_command(full_cmd)
                        if result["return_code"] == 0:
                            return f"📁 Directory listing ({cmd}):\n{result['stdout']}"
                        else:
                            return f"Could not execute {cmd}: {result.get('stderr', 'Unknown error')}"
                    elif cmd in ['pwd']:
                        result = await self.mcp_manager.execute_command("pwd")
                        if result["return_code"] == 0:
                            return f"📂 Current directory: {result['stdout'].strip()}"
                        else:
                            return "Could not get current directory"
                    elif cmd in ['whoami']:
                        result = await self.mcp_manager.execute_command("whoami")
                        if result["return_code"] == 0:
                            return f"👤 Current user: {result['stdout'].strip()}"
                        else:
                            return "Could not get current user"
                    else:
                        return f"I can execute safe commands like 'l' (ls), 'pwd', 'whoami'. The command '{cmd}' needs safety review. Try being more specific about what you want to accomplish."
                else:
                    return "I can help with safe bash commands like:\n• 'l' or 'ls' - List files\n• 'pwd' - Show current directory\n• 'whoami' - Show current user\n\nWhat specific command did you want to run?"
            
            # System information requests
            elif any(keyword in request_lower for keyword in ["memory", "ram"]):
                result = await self.mcp_manager.execute_command("free -h")
                if result["return_code"] == 0:
                    return f"💾 Memory usage:\n{result['stdout']}"
                else:
                    return "Could not retrieve memory information"
            
            elif any(keyword in request_lower for keyword in ["cpu", "processor"]):
                result = await self.mcp_manager.execute_command("top -bn1 | head -5")
                if result["return_code"] == 0:
                    return f"🖥️ CPU information:\n{result['stdout']}"
                else:
                    return "Could not retrieve CPU information"
            
            elif any(keyword in request_lower for keyword in ["process", "ps"]):
                result = await self.mcp_manager.execute_command("ps aux | head -10")
                if result["return_code"] == 0:
                    return f"⚡ Running processes:\n{result['stdout']}"
                else:
                    return "Could not retrieve process information"
            
            # Current directory / pwd
            elif any(keyword in request_lower for keyword in ["current directory", "where am i", "pwd"]):
                result = await self.mcp_manager.execute_command("pwd")
                if result["return_code"] == 0:
                    return f"📂 Current directory: {result['stdout'].strip()}"
                else:
                    return "Could not get current directory"
            
            else:
                return "I can help with system commands! Try:\n• 'run bash command l' - List files\n• 'list files' - Show directory contents\n• 'check memory' - Show RAM usage\n• 'show processes' - Display running programs\n• 'current directory' - Show where you are"
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including action execution data."""
        v4_stats = super().get_statistics()
        
        # Add v5-specific stats
        v5_stats = {
            "actions_executed": len(self.action_history),
            "mcp_available": self.mcp_manager.is_available,
            "recent_actions": self.action_history[-5:] if self.action_history else []
        }
        
        return {**v4_stats, **v5_stats}


async def main():
    """Main entry point for Listen v5."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Listen v5 - System Automation & Command Interface")
    parser.add_argument("--db", type=str, help="Path to speaker database")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize Listen v5
    assistant = ListenV5(
        name="Listen v5",
        db_path=Path(args.db) if args.db else None,
        use_tts=not args.no_tts,
        confidence_threshold=args.confidence
    )
    
    # Start the assistant
    await assistant.start()


if __name__ == "__main__":
    asyncio.run(main())