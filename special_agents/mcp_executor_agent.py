#!/usr/bin/env python3
"""
MCP Executor Agent

Executes system commands via Model Context Protocol.
Part of Listen v7's agentic architecture.
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from agent.agent import Agent

log = logging.getLogger(__name__)


class MCPExecutorAgent(Agent):
    """
    Specialized agent for command execution via MCP.
    
    Handles:
    - Filesystem operations
    - Shell commands
    - Git operations
    - Development tool commands
    
    With multi-tier fallback:
    - Anthropic MCP (when available)
    - Standalone MCP servers (npx)
    - Direct subprocess execution
    """
    
    def __init__(self, anthropic_key: Optional[str] = None, **kwargs):
        roles = [
            "You execute system commands safely and reliably",
            "You manage Model Context Protocol communications",
            "You handle filesystem, shell, and git operations",
            "You ensure command security and validation",
            "You provide detailed execution results and error handling"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.mcp_available = self._check_mcp_availability()
        self.command_history = []
        self.safety_mode = True
        
        log.info(f"MCPExecutorAgent initialized (MCP: {self.mcp_available})")
    
    def _check_mcp_availability(self) -> bool:
        """Check if MCP services are available."""
        # Check for Anthropic SDK
        try:
            import anthropic
            if self.anthropic_key:
                return True
        except ImportError:
            pass
        
        # Check for npx (standalone MCP servers)
        try:
            result = subprocess.run(
                ["npx", "--version"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                return True
        except:
            pass
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of execution services."""
        return {
            "status": "healthy",
            "mcp_available": self.mcp_available,
            "anthropic_configured": bool(self.anthropic_key),
            "safety_mode": self.safety_mode,
            "commands_executed": len(self.command_history)
        }
    
    async def execute(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a command via MCP or fallback.
        
        This is the main action called by PlanRunner.
        """
        # Validate command safety
        if self.safety_mode:
            is_safe, reason = self._validate_command_safety(command, parameters)
            if not is_safe:
                return {
                    "success": False,
                    "error": f"Command blocked: {reason}",
                    "command": command
                }
        
        # Try MCP execution first
        if self.mcp_available:
            try:
                result = await self._execute_via_mcp(command, parameters)
                if result["success"]:
                    self._record_command(command, result, "mcp")
                    return result
            except Exception as e:
                log.warning(f"MCP execution failed: {e}, falling back")
        
        # Fallback to direct execution
        result = await self._execute_direct(command, parameters)
        self._record_command(command, result, "direct")
        return result
    
    async def execute_simple(self, command: str) -> Dict[str, Any]:
        """
        Execute simple commands quickly.
        
        Optimized path for common operations.
        """
        # Map common phrases to commands
        command_map = {
            "list files": "ls",
            "list my files": "ls -la",
            "show files": "ls",
            "current directory": "pwd",
            "what directory": "pwd",
            "disk usage": "df -h",
            "memory usage": "free -h",
            "running processes": "ps aux | head -20"
        }
        
        actual_command = command_map.get(command.lower(), command)
        
        # Direct execution for speed
        return await self._execute_direct(actual_command, {})
    
    def _validate_command_safety(self, command: str, parameters: Dict[str, Any]) -> tuple:
        """Validate command safety before execution."""
        # Dangerous commands
        dangerous_commands = [
            "rm -rf",
            "format",
            "dd",
            "mkfs",
            "> /dev/",
            "fork bomb",
            ":(){ :|:& };:",
        ]
        
        command_lower = command.lower()
        
        # Check for dangerous patterns
        for dangerous in dangerous_commands:
            if dangerous in command_lower:
                return False, f"Dangerous command pattern: {dangerous}"
        
        # Check for system paths
        system_paths = ["/etc", "/sys", "/proc", "/boot", "/dev"]
        for path in system_paths:
            if path in command and "ls" not in command_lower:
                return False, f"Access to system path restricted: {path}"
        
        # Check sudo
        if "sudo" in command_lower:
            return False, "Sudo commands require explicit permission"
        
        return True, "Command validated"
    
    async def _execute_via_mcp(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute via MCP (Anthropic or standalone)."""
        # Try to import and use real MCP
        try:
            from listen.versions.listen_v6_mcp_integration import RealMCPIntegrationManager
            
            mcp_manager = RealMCPIntegrationManager(self.anthropic_key)
            if mcp_manager.is_available:
                await mcp_manager.initialize()
                result = await mcp_manager.execute_shell_command(command)
                
                return {
                    "success": result.success,
                    "stdout": result.output or "",
                    "stderr": result.error or "",
                    "command": command,
                    "service": "mcp"
                }
        except:
            pass
        
        # If MCP not available, will fall through to direct execution
        raise RuntimeError("MCP not available")
    
    async def _execute_direct(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Direct command execution via subprocess."""
        try:
            # Handle different command types
            if parameters and parameters.get("operation") == "list":
                # Filesystem list operation
                path = parameters.get("path", ".")
                result = subprocess.run(
                    ["ls", "-la", path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                # General command execution
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=os.environ.copy()
                )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": command,
                "service": "subprocess"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 30 seconds",
                "command": command,
                "service": "subprocess"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "service": "subprocess"
            }
    
    def _record_command(self, command: str, result: Dict[str, Any], service: str):
        """Record command execution for history."""
        self.command_history.append({
            "command": command,
            "success": result.get("success", False),
            "service": service,
            "timestamp": os.popen("date").read().strip()
        })
        
        # Keep only last 100 commands
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]
    
    async def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command execution history."""
        return self.command_history[-limit:]
    
    async def receive_message(self, source_agent: str, message: Any) -> Any:
        """
        Receive inter-agent messages.
        
        This implements Talk's agent communication protocol.
        """
        if source_agent == "security_validator":
            # Security agent can update safety settings
            if isinstance(message, dict) and "safety_mode" in message:
                self.safety_mode = message["safety_mode"]
                return {"acknowledged": True, "safety_mode": self.safety_mode}
        
        elif source_agent == "orchestrator":
            # Orchestrator can request command execution
            if isinstance(message, dict) and "execute" in message:
                command = message["execute"]
                params = message.get("parameters", {})
                return await self.execute(command, params)
        
        return {"error": f"Unknown message from {source_agent}"}
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Agent interface for LLM-based command generation.
        
        Can generate commands from natural language.
        """
        if "execute" in prompt.lower():
            # Extract command from context
            command = context.get("command", "") if context else ""
            if command:
                result = await self.execute(command)
                if result["success"]:
                    return f"Executed: {command}\nOutput: {result.get('stdout', '')[:200]}"
                else:
                    return f"Failed: {command}\nError: {result.get('error', result.get('stderr', ''))}"
        
        elif "history" in prompt.lower():
            history = await self.get_command_history(5)
            return f"Recent commands: {[h['command'] for h in history]}"
        
        return f"MCPExecutorAgent: {prompt}"
    
    async def cleanup(self):
        """Clean up execution resources."""
        # Save command history if needed
        if self.command_history:
            log.info(f"MCPExecutorAgent executed {len(self.command_history)} commands")
        
        log.info("MCPExecutorAgent cleanup complete")