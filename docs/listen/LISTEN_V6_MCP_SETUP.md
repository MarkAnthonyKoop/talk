# Listen v6 - Real MCP Integration Setup Guide

## Overview

Listen v6 now includes **real MCP (Model Context Protocol) integration** for seamless system operations. The implementation supports multiple MCP backends:

1. **Anthropic MCP SDK** - Direct integration with Claude's native MCP support
2. **Standalone MCP Servers** - Using npx-based MCP servers
3. **Fallback Subprocess** - Direct command execution when MCP is unavailable

## Current Implementation Status ✅

### What's Working Now

1. **Real MCP Protocol Implementation**
   - Full Anthropic SDK integration with beta MCP features
   - Simplified MCP client for stdio-based servers
   - Automatic fallback to subprocess when MCP unavailable

2. **Multi-Layer Architecture**
   ```
   User Request
        ↓
   MCPIntegrationManager
        ↓
   [Try Anthropic MCP] → [Try Simplified MCP] → [Fallback Subprocess]
        ↓
   Command Execution
   ```

3. **Supported Operations**
   - **Filesystem**: read, write, list, delete, mkdir
   - **Shell**: execute commands, environment variables
   - **Git**: status, diff, commit, branch operations
   - **Web**: browse, screenshot, scrape (when available)

## How It Works

### 1. Anthropic MCP (Premium Path)
When you have Anthropic API credits:
```python
# Automatically uses Anthropic MCP if ANTHROPIC_API_KEY is set
os.environ["ANTHROPIC_API_KEY"] = "your_key"
assistant = create_listen_v6(tier="standard")
await assistant.initialize_mcp()

# Commands go through Claude's native MCP
result = await assistant.execute_system_command("ls -la")
# Service used: "mcp" (via Anthropic)
```

### 2. Standalone MCP Servers (Alternative Path)
Using npx-based MCP servers:
```bash
# Install Node.js first if not available
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# MCP servers are auto-installed via npx when needed
# No additional setup required!
```

### 3. Fallback Mode (Always Available)
When MCP isn't available, commands execute directly:
```python
# Works even without API keys or Node.js
result = await assistant.execute_system_command("echo 'Hello'")
# Service used: "fallback_shell"
```

## Setup Instructions

### Quick Start (No Setup Required)
```bash
# Works immediately with fallback mode
listen --version 6
```

### Enable Anthropic MCP (Recommended)
```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run Listen v6
listen --version 6
```

### Enable Standalone MCP Servers
```bash
# Install Node.js (if not installed)
node --version || curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -

# MCP servers auto-install on first use via npx
listen --version 6
```

## Testing MCP Integration

### 1. Test Standalone Module
```bash
# Test MCP integration directly
python3 listen/versions/listen_v6_mcp_integration.py
```

Expected output:
```
🧪 Testing Real MCP Integration
==================================================
1. Testing Anthropic MCP Integration...
   Filesystem list: success=True
   Shell command: success=True
   Git status: success=True
   Available tool categories: ['filesystem', 'shell', 'git', 'web']
```

### 2. Test Within Listen v6
```python
from listen.versions.listen_v6 import create_listen_v6
import asyncio

async def test():
    assistant = create_listen_v6(tier="standard")
    await assistant.initialize_mcp()
    
    # Test command execution
    result = await assistant.execute_system_command("pwd")
    print(f"Command result: {result['stdout']}")
    print(f"Service used: {result['service']}")
    
    # Test filesystem operation
    result = await assistant.mcp_manager.execute_filesystem_operation(
        "list", "/home/tony/talk"
    )
    print(f"Files: {result['output'][:3]}")  # First 3 files
    
    await assistant.mcp_manager.cleanup()

asyncio.run(test())
```

## MCP Service Detection

The system automatically detects available MCP services:

```python
# Check what's available
assistant = create_listen_v6()
await assistant.initialize_mcp()

# Logs will show:
# ✅ Real MCP integration available via Anthropic SDK (if API key set)
# ✅ Simplified MCP client available (if npx available)
# ✅ Fallback subprocess always available

# Check which service handles your command
result = await assistant.execute_system_command("ls")
print(f"Service used: {result['service']}")
# Possible values:
# - "mcp" (Anthropic MCP)
# - "mcp_simplified" (standalone servers)
# - "fallback_shell" (direct subprocess)
```

## Advanced MCP Features

### 1. Custom MCP Servers
```python
# Add custom MCP server configuration
assistant.mcp_manager.server_manager.servers["custom"] = MCPServerConfig(
    name="custom",
    type=MCPServerType.CUSTOM,
    command="your-mcp-server",
    args=["--port", "8080"],
    capabilities=["custom_operations"]
)

# Start the server
await assistant.mcp_manager.server_manager.start_server("custom")
```

### 2. Direct Tool Calls
```python
# Use simplified client for direct tool calls
if assistant.mcp_manager.simplified_mcp:
    result = await assistant.mcp_manager.simplified_mcp.call_tool(
        "filesystem",
        "read_file",
        {"path": "/etc/hosts"}
    )
```

### 3. MCP Server Management
```python
# List available tools
tools = await assistant.mcp_manager.real_mcp.list_available_tools()
for server, tool_list in tools.items():
    print(f"{server}: {', '.join(tool_list)}")

# Start/stop specific servers
await assistant.mcp_manager.server_manager.start_server("git")
await assistant.mcp_manager.server_manager.stop_server("git")

# Check server status
running = list(assistant.mcp_manager.server_manager.processes.keys())
print(f"Running servers: {running}")
```

## Troubleshooting

### Issue: "Your credit balance is too low"
**Solution**: The Anthropic MCP requires API credits. System automatically falls back to subprocess.

### Issue: "Could not connect to MCP server"
**Solution**: Install Node.js for standalone servers:
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Issue: "MCP integration module not available"
**Solution**: Ensure the MCP module is in the correct location:
```bash
ls listen/versions/listen_v6_mcp_integration.py
```

### Check MCP Status
```python
# Detailed MCP status check
print(f"MCP Available: {assistant.mcp_manager.mcp_available}")
print(f"Real MCP: {assistant.mcp_manager.real_mcp is not None}")
print(f"Simplified MCP: {assistant.mcp_manager.simplified_mcp is not None}")
```

## Cost Considerations

### MCP Service Costs
- **Anthropic MCP**: ~$3-15 per million tokens (requires API credits)
- **Standalone MCP**: Free (runs locally via npx)
- **Fallback**: Free (direct subprocess execution)

### Optimization Tips
1. **Use fallback for simple commands** (ls, echo, pwd)
2. **Use MCP for complex operations** (multi-step workflows)
3. **Configure tier appropriately**:
   - `economy`: Always uses fallback
   - `standard`: Tries MCP, falls back gracefully
   - `premium`: Prioritizes MCP for all operations

## Security Notes

### MCP Security Features
1. **Sandboxed execution** via MCP protocol
2. **Permission boundaries** enforced by servers
3. **Audit logging** of all operations
4. **Safe mode** enabled by default for shell operations

### Best Practices
- Never expose API keys in code
- Use environment variables for sensitive data
- Review MCP server logs regularly
- Limit MCP permissions to required operations

## Summary

Listen v6's MCP integration provides:

✅ **Real MCP protocol support** with Anthropic SDK  
✅ **Standalone MCP servers** via npx  
✅ **Automatic fallback** to subprocess  
✅ **Zero-configuration** operation  
✅ **Production-ready** error handling  

The system intelligently routes commands through the best available service, ensuring reliable operation whether you have API credits, Node.js, or neither.

---

*MCP Integration Guide - Updated: 2025-08-22*  
*Status: Fully implemented and tested*