# agent/tools/handlers.py
"""Handlers for tool operations."""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any

class ToolError(Exception):
    """Error raised by tool handlers."""
    pass

class FileOperationHandler:
    """Handles file operations from tool calls."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()
    
    def read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            file_path = self._resolve_path(path)
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ToolError(f"Failed to read file: {e}")
    
    def write_file(self, path: str, content: str) -> str:
        """Write content to file."""
        try:
            file_path = self._resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            raise ToolError(f"Failed to write file: {e}")
    
    def create_file(self, path: str, content: str) -> str:
        """Create a new file."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            raise ToolError(f"File already exists: {path}")
        return self.write_file(path, content)
    
    def edit_file(self, path: str, search: str, replace: str) -> str:
        """Edit file using search and replace."""
        try:
            content = self.read_file(path)
            if search not in content:
                raise ToolError(f"Search text not found in {path}")
            
            new_content = content.replace(search, replace, 1)
            self.write_file(path, new_content)
            return f"Successfully replaced text in {path}"
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to edit file: {e}")
    
    def list_files(self, directory: str = ".") -> str:
        """List files in directory."""
        try:
            dir_path = self._resolve_path(directory)
            if not dir_path.is_dir():
                raise ToolError(f"Not a directory: {directory}")
            
            files = []
            for item in sorted(dir_path.iterdir()):
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    files.append(f"{item.name}/")
            
            return "\n".join(files) if files else "Empty directory"
        except Exception as e:
            raise ToolError(f"Failed to list files: {e}")


class ShellOperationHandler:
    """Handles shell operations from tool calls."""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    def run_shell_command(self, command: str, working_dir: str = ".") -> str:
        """Execute shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=working_dir
            )
            
            output = result.stdout or ""
            error = result.stderr or ""
            
            if result.returncode != 0:
                return f"Command failed (exit {result.returncode}):\n{error}\n{output}"
            
            return output if output else "Command completed successfully"
            
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {self.timeout} seconds")
        except Exception as e:
            raise ToolError(f"Failed to execute command: {e}")


class UnifiedToolHandler:
    """Unified handler for all tool operations."""
    
    def __init__(self, base_dir: str = ".", shell_timeout: int = 60):
        self.file_handler = FileOperationHandler(base_dir)
        self.shell_handler = ShellOperationHandler(shell_timeout)
    
    def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Route tool call to appropriate handler."""
        # File operations
        if tool_name == "read_file":
            return self.file_handler.read_file(parameters["path"])
        elif tool_name == "write_file":
            return self.file_handler.write_file(parameters["path"], parameters["content"])
        elif tool_name == "create_file":
            return self.file_handler.create_file(parameters["path"], parameters["content"])
        elif tool_name == "edit_file":
            return self.file_handler.edit_file(
                parameters["path"], 
                parameters["search"], 
                parameters["replace"]
            )
        elif tool_name == "list_files":
            return self.file_handler.list_files(parameters.get("directory", "."))
        
        # Shell operations
        elif tool_name == "run_shell_command":
            return self.shell_handler.run_shell_command(
                parameters["command"],
                parameters.get("working_dir", ".")
            )
        
        else:
            raise ToolError(f"Unknown tool: {tool_name}")