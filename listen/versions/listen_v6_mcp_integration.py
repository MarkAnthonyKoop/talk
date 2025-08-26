#!/usr/bin/env python3
"""
Listen v6 - Real MCP (Model Context Protocol) Integration

This module provides actual MCP integration for Claude Code and other MCP servers,
replacing the placeholder implementation with real protocol support.
"""

import asyncio
import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

log = logging.getLogger(__name__)

# Check for Anthropic SDK with MCP support
try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types.beta import (
        BetaRequestMCPServerToolConfigurationParam,
        BetaRequestMCPServerURLDefinitionParam,
        BetaMCPToolUseBlockParam,
        BetaRequestMCPToolResultBlockParam
    )
    ANTHROPIC_MCP_AVAILABLE = True
except ImportError:
    ANTHROPIC_MCP_AVAILABLE = False
    log.warning("⚠️  Anthropic SDK with MCP support not available")

# Check for standalone MCP client
try:
    import mcp_client
    STANDALONE_MCP_AVAILABLE = True
except ImportError:
    STANDALONE_MCP_AVAILABLE = False
    

class MCPServerType(Enum):
    """Types of MCP servers."""
    FILESYSTEM = "filesystem"
    SHELL = "shell"
    GIT = "git"
    WEB = "web"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    type: MCPServerType
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    capabilities: List[str] = None
    auto_start: bool = True


@dataclass
class MCPToolResult:
    """Result from an MCP tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class MCPServerManager:
    """Manages MCP server lifecycle and connections."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.initialize_default_servers()
    
    def initialize_default_servers(self):
        """Initialize default MCP server configurations."""
        
        # Filesystem MCP server
        self.servers["filesystem"] = MCPServerConfig(
            name="filesystem",
            type=MCPServerType.FILESYSTEM,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/home/tony"],
            capabilities=["read", "write", "list", "search"],
            auto_start=True
        )
        
        # Shell/Bash MCP server  
        self.servers["shell"] = MCPServerConfig(
            name="shell",
            type=MCPServerType.SHELL,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-shell"],
            env={"SHELL_SAFE_MODE": "true"},
            capabilities=["execute", "environment", "process"],
            auto_start=True
        )
        
        # Git MCP server
        self.servers["git"] = MCPServerConfig(
            name="git",
            type=MCPServerType.GIT,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-git"],
            capabilities=["status", "diff", "commit", "branch", "log"],
            auto_start=False
        )
        
        # Web browser MCP server
        self.servers["web"] = MCPServerConfig(
            name="web",
            type=MCPServerType.WEB,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-puppeteer"],
            capabilities=["browse", "screenshot", "scrape"],
            auto_start=False
        )
        
        log.info(f"✅ Initialized {len(self.servers)} MCP server configurations")
    
    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server."""
        if server_name not in self.servers:
            log.error(f"❌ Unknown server: {server_name}")
            return False
        
        if server_name in self.processes:
            log.info(f"ℹ️  Server {server_name} already running")
            return True
        
        config = self.servers[server_name]
        
        try:
            # Build command
            cmd = [config.command] + (config.args or [])
            
            # Set up environment
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
            
            # Start server process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Wait briefly to check if it started successfully
            await asyncio.sleep(0.5)
            
            if process.poll() is None:
                self.processes[server_name] = process
                log.info(f"✅ Started MCP server: {server_name}")
                return True
            else:
                stderr = process.stderr.read() if process.stderr else ""
                log.error(f"❌ Failed to start {server_name}: {stderr}")
                return False
                
        except Exception as e:
            log.error(f"❌ Error starting {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop an MCP server."""
        if server_name not in self.processes:
            log.info(f"ℹ️  Server {server_name} not running")
            return True
        
        try:
            process = self.processes[server_name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process(process)),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Force kill if not stopped
                process.kill()
                await asyncio.sleep(0.5)
            
            del self.processes[server_name]
            log.info(f"✅ Stopped MCP server: {server_name}")
            return True
            
        except Exception as e:
            log.error(f"❌ Error stopping {server_name}: {e}")
            return False
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for a process to exit."""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def start_all_auto_start(self):
        """Start all servers configured for auto-start."""
        for name, config in self.servers.items():
            if config.auto_start:
                await self.start_server(name)
    
    async def stop_all(self):
        """Stop all running servers."""
        for name in list(self.processes.keys()):
            await self.stop_server(name)


class RealMCPIntegrationManager:
    """Real MCP integration using actual protocol implementation."""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.server_manager = MCPServerManager()
        self.is_available = False
        
        if ANTHROPIC_MCP_AVAILABLE and self.api_key:
            try:
                self.client = AsyncAnthropic(api_key=self.api_key)
                self.is_available = True
                log.info("✅ Real MCP integration initialized with Anthropic SDK")
            except Exception as e:
                log.error(f"❌ Failed to initialize Anthropic client: {e}")
                self.is_available = False
        else:
            log.warning("⚠️  MCP integration not available - missing Anthropic SDK or API key")
    
    async def initialize(self):
        """Initialize MCP servers and connections."""
        if not self.is_available:
            log.warning("⚠️  MCP not available, skipping initialization")
            return False
        
        # Start default MCP servers
        await self.server_manager.start_all_auto_start()
        
        return True
    
    async def execute_filesystem_operation(
        self, 
        operation: str, 
        path: str, 
        content: Optional[str] = None
    ) -> MCPToolResult:
        """Execute filesystem operations via MCP."""
        
        if not self.is_available:
            # Fallback to direct filesystem operations
            return await self._fallback_filesystem_operation(operation, path, content)
        
        try:
            # Ensure filesystem server is running
            if "filesystem" not in self.server_manager.processes:
                await self.server_manager.start_server("filesystem")
            
            # Create MCP tool use request
            tool_use = BetaMCPToolUseBlockParam(
                type="mcp_tool_use",
                mcp_server_name="filesystem",
                tool_name=f"filesystem_{operation}",
                input={
                    "path": path,
                    "content": content
                } if content else {"path": path}
            )
            
            # Execute via Anthropic client
            response = await self.client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Execute filesystem {operation} on {path}"
                }],
                tools=[tool_use],
                mcp_servers=[
                    BetaRequestMCPServerToolConfigurationParam(
                        name="filesystem",
                        url=BetaRequestMCPServerURLDefinitionParam(
                            type="stdio",
                            command="npx",
                            args=["-y", "@modelcontextprotocol/server-filesystem", "/home/tony"]
                        )
                    )
                ]
            )
            
            # Extract result
            if response.content and len(response.content) > 0:
                result = response.content[0]
                return MCPToolResult(
                    success=True,
                    output=result.text if hasattr(result, 'text') else str(result),
                    metadata={"service": "mcp_filesystem"}
                )
            else:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error="No response from MCP server"
                )
                
        except Exception as e:
            log.error(f"❌ MCP filesystem operation failed: {e}")
            return await self._fallback_filesystem_operation(operation, path, content)
    
    async def execute_shell_command(
        self, 
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> MCPToolResult:
        """Execute shell commands via MCP."""
        
        if not self.is_available:
            # Fallback to subprocess
            return await self._fallback_shell_command(command, working_dir, env)
        
        try:
            # Ensure shell server is running
            if "shell" not in self.server_manager.processes:
                await self.server_manager.start_server("shell")
            
            # Create MCP tool use request
            tool_use = BetaMCPToolUseBlockParam(
                type="mcp_tool_use",
                mcp_server_name="shell",
                tool_name="shell_execute",
                input={
                    "command": command,
                    "cwd": working_dir,
                    "env": env or {}
                }
            )
            
            # Execute via Anthropic client
            response = await self.client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Execute shell command: {command}"
                }],
                tools=[tool_use],
                mcp_servers=[
                    BetaRequestMCPServerToolConfigurationParam(
                        name="shell",
                        url=BetaRequestMCPServerURLDefinitionParam(
                            type="stdio",
                            command="npx",
                            args=["-y", "@modelcontextprotocol/server-shell"]
                        )
                    )
                ]
            )
            
            # Extract result
            if response.content and len(response.content) > 0:
                result = response.content[0]
                return MCPToolResult(
                    success=True,
                    output=result.text if hasattr(result, 'text') else str(result),
                    metadata={"service": "mcp_shell", "command": command}
                )
            else:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error="No response from MCP server"
                )
                
        except Exception as e:
            log.error(f"❌ MCP shell command failed: {e}")
            return await self._fallback_shell_command(command, working_dir, env)
    
    async def execute_git_operation(
        self,
        operation: str,
        repo_path: str,
        args: Optional[Dict[str, Any]] = None
    ) -> MCPToolResult:
        """Execute Git operations via MCP."""
        
        if not self.is_available:
            # Fallback to git commands
            return await self._fallback_git_operation(operation, repo_path, args)
        
        try:
            # Ensure git server is running
            if "git" not in self.server_manager.processes:
                await self.server_manager.start_server("git")
            
            # Create MCP tool use request
            tool_use = BetaMCPToolUseBlockParam(
                type="mcp_tool_use", 
                mcp_server_name="git",
                tool_name=f"git_{operation}",
                input={
                    "repo": repo_path,
                    **(args or {})
                }
            )
            
            # Execute via Anthropic client
            response = await self.client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Execute git {operation} in {repo_path}"
                }],
                tools=[tool_use],
                mcp_servers=[
                    BetaRequestMCPServerToolConfigurationParam(
                        name="git",
                        url=BetaRequestMCPServerURLDefinitionParam(
                            type="stdio",
                            command="npx",
                            args=["-y", "@modelcontextprotocol/server-git"]
                        )
                    )
                ]
            )
            
            # Extract result
            if response.content and len(response.content) > 0:
                result = response.content[0]
                return MCPToolResult(
                    success=True,
                    output=result.text if hasattr(result, 'text') else str(result),
                    metadata={"service": "mcp_git", "operation": operation}
                )
            else:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error="No response from MCP server"
                )
                
        except Exception as e:
            log.error(f"❌ MCP git operation failed: {e}")
            return await self._fallback_git_operation(operation, repo_path, args)
    
    async def _fallback_filesystem_operation(
        self, 
        operation: str, 
        path: str, 
        content: Optional[str] = None
    ) -> MCPToolResult:
        """Fallback filesystem operations using Python."""
        try:
            path_obj = Path(path)
            
            if operation == "read":
                output = path_obj.read_text()
            elif operation == "write":
                path_obj.write_text(content or "")
                output = f"Wrote to {path}"
            elif operation == "list":
                output = [str(p) for p in path_obj.iterdir()]
            elif operation == "exists":
                output = path_obj.exists()
            elif operation == "delete":
                if path_obj.is_file():
                    path_obj.unlink()
                else:
                    path_obj.rmdir()
                output = f"Deleted {path}"
            else:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}"
                )
            
            return MCPToolResult(
                success=True,
                output=output,
                metadata={"service": "fallback_filesystem"}
            )
            
        except Exception as e:
            return MCPToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _fallback_shell_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> MCPToolResult:
        """Fallback shell execution using subprocess."""
        try:
            # Set up environment
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                env=cmd_env,
                timeout=30
            )
            
            return MCPToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                metadata={
                    "service": "fallback_shell",
                    "return_code": result.returncode
                }
            )
            
        except subprocess.TimeoutExpired:
            return MCPToolResult(
                success=False,
                output=None,
                error="Command timed out after 30 seconds"
            )
        except Exception as e:
            return MCPToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _fallback_git_operation(
        self,
        operation: str,
        repo_path: str,
        args: Optional[Dict[str, Any]] = None
    ) -> MCPToolResult:
        """Fallback Git operations using git commands."""
        try:
            git_commands = {
                "status": "git status",
                "diff": "git diff",
                "log": "git log --oneline -10",
                "branch": "git branch -a",
                "commit": f"git commit -m '{args.get('message', 'Commit via MCP')}'" if args else "git commit"
            }
            
            command = git_commands.get(operation)
            if not command:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown git operation: {operation}"
                )
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=10
            )
            
            return MCPToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                metadata={
                    "service": "fallback_git",
                    "operation": operation
                }
            )
            
        except Exception as e:
            return MCPToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available MCP tools across servers."""
        tools = {}
        
        # Filesystem tools
        tools["filesystem"] = [
            "read", "write", "list", "exists", "delete", 
            "mkdir", "copy", "move", "search"
        ]
        
        # Shell tools
        tools["shell"] = [
            "execute", "environment", "process_list",
            "kill_process", "working_directory"
        ]
        
        # Git tools
        tools["git"] = [
            "status", "diff", "log", "branch", "commit",
            "push", "pull", "checkout", "merge"
        ]
        
        # Web tools (if available)
        if "web" in self.server_manager.servers:
            tools["web"] = [
                "browse", "screenshot", "scrape",
                "click", "type", "wait"
            ]
        
        return tools
    
    async def cleanup(self):
        """Clean up MCP resources."""
        await self.server_manager.stop_all()
        log.info("✅ MCP resources cleaned up")


class SimplifiedMCPClient:
    """Simplified MCP client for direct server communication."""
    
    def __init__(self):
        self.servers = {}
        self.stdio_connections = {}
    
    async def connect_stdio_server(self, name: str, command: List[str]) -> bool:
        """Connect to an MCP server via stdio."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.stdio_connections[name] = process
            
            # Send initialization with proper capabilities
            init_request = json.dumps({
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "Listen v6",
                        "version": "1.0"
                    },
                    "capabilities": {}
                },
                "id": 1
            }) + "\n"
            
            process.stdin.write(init_request.encode())
            await process.stdin.drain()
            
            # Read response
            response_line = await process.stdout.readline()
            response = json.loads(response_line.decode())
            
            if "result" in response:
                log.info(f"✅ Connected to MCP server: {name}")
                return True
            else:
                log.error(f"❌ Failed to initialize MCP server {name}: {response}")
                return False
                
        except Exception as e:
            log.error(f"❌ Error connecting to MCP server {name}: {e}")
            return False
    
    async def call_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """Call a tool on an MCP server."""
        
        if server_name not in self.stdio_connections:
            return MCPToolResult(
                success=False,
                output=None,
                error=f"Server {server_name} not connected"
            )
        
        try:
            process = self.stdio_connections[server_name]
            
            # Create tool call request
            request = json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                },
                "id": 2
            }) + "\n"
            
            # Send request
            process.stdin.write(request.encode())
            await process.stdin.drain()
            
            # Read response
            response_line = await process.stdout.readline()
            response = json.loads(response_line.decode())
            
            if "result" in response:
                return MCPToolResult(
                    success=True,
                    output=response["result"],
                    metadata={"server": server_name, "tool": tool_name}
                )
            else:
                return MCPToolResult(
                    success=False,
                    output=None,
                    error=response.get("error", {}).get("message", "Unknown error")
                )
                
        except Exception as e:
            return MCPToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def disconnect_all(self):
        """Disconnect all MCP servers."""
        for name, process in self.stdio_connections.items():
            try:
                process.terminate()
                await process.wait()
                log.info(f"✅ Disconnected MCP server: {name}")
            except Exception as e:
                log.error(f"❌ Error disconnecting {name}: {e}")
        
        self.stdio_connections.clear()


async def test_mcp_integration():
    """Test the real MCP integration."""
    print("🧪 Testing Real MCP Integration")
    print("=" * 50)
    
    # Test with Anthropic SDK
    if ANTHROPIC_MCP_AVAILABLE:
        print("\n1. Testing Anthropic MCP Integration...")
        
        manager = RealMCPIntegrationManager()
        if manager.is_available:
            await manager.initialize()
            
            # Test filesystem operation
            result = await manager.execute_filesystem_operation("list", "/home/tony/talk")
            print(f"   Filesystem list: success={result.success}")
            
            # Test shell command
            result = await manager.execute_shell_command("echo 'Hello from MCP'")
            print(f"   Shell command: success={result.success}")
            if result.success:
                print(f"   Output: {result.output[:50]}...")
            
            # Test git operation
            result = await manager.execute_git_operation("status", "/home/tony/talk")
            print(f"   Git status: success={result.success}")
            
            # List available tools
            tools = await manager.list_available_tools()
            print(f"   Available tool categories: {list(tools.keys())}")
            
            await manager.cleanup()
        else:
            print("   ⚠️  Anthropic MCP not available - need API key")
    else:
        print("   ⚠️  Anthropic SDK not installed")
    
    # Test simplified client
    print("\n2. Testing Simplified MCP Client...")
    
    client = SimplifiedMCPClient()
    
    # Try to connect to filesystem server
    connected = await client.connect_stdio_server(
        "filesystem",
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/tony"]
    )
    
    if connected:
        # Test tool call
        result = await client.call_tool(
            "filesystem",
            "read_file",
            {"path": "/home/tony/talk/README.md"}
        )
        print(f"   Tool call result: success={result.success}")
        
        await client.disconnect_all()
    else:
        print("   ⚠️  Could not connect to MCP server (npx may be required)")
    
    print("\n✅ MCP integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())