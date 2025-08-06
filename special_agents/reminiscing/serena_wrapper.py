#!/usr/bin/env python3
"""
Serena MCP Server Wrapper

This wrapper provides easy access to Serena's semantic code analysis capabilities
from the Talk framework. It handles the UV environment and provides a clean interface
to Serena's tools.
"""

import subprocess
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

class SerenaWrapper:
    """
    Wrapper for Serena MCP server that provides semantic code analysis.
    
    This class bridges between Talk's pip-based environment and Serena's UV-based
    environment, allowing you to leverage Serena's powerful LSP-based semantic
    search without disrupting your existing setup.
    """
    
    def __init__(self, serena_path: Optional[str] = None):
        """
        Initialize the Serena wrapper.
        
        Args:
            serena_path: Path to Serena installation (defaults to relative path)
        """
        if serena_path is None:
            # Default to the serena directory we installed
            self.serena_path = Path(__file__).parent / "serena"
        else:
            self.serena_path = Path(serena_path)
        
        if not self.serena_path.exists():
            raise FileNotFoundError(f"Serena installation not found at {self.serena_path}")
        
        # Check if UV is available
        self.uv_path = self._find_uv()
        if not self.uv_path:
            raise RuntimeError("UV not found. Please install UV first.")
    
    def _find_uv(self) -> Optional[str]:
        """Find the UV executable."""
        # Check common paths
        uv_paths = [
            os.path.expanduser("~/.local/bin/uv"),
            "/usr/local/bin/uv",
            "uv"  # In PATH
        ]
        
        for path in uv_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return None
    
    def _run_serena_command(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """
        Run a Serena command and return the result.
        
        Args:
            args: Command arguments to pass to Serena
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with stdout, stderr, and return_code
        """
        cmd = [
            self.uv_path, "run", "python", "scripts/mcp_server.py"
        ] + args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.serena_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PATH": f"{os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '')}"}
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1,
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False
            }
    
    def get_serena_info(self) -> Dict[str, Any]:
        """Get information about the Serena installation."""
        result = self._run_serena_command(["--help"], timeout=10)
        
        return {
            "serena_path": str(self.serena_path),
            "uv_path": self.uv_path,
            "available": result["success"],
            "version_info": result.get("stdout", "") if result["success"] else None,
            "error": result.get("stderr", "") if not result["success"] else None
        }
    
    def start_mcp_server(self, 
                        project_path: Optional[str] = None,
                        port: int = 9121,
                        context: str = "ide-assistant",
                        mode: List[str] = None) -> subprocess.Popen:
        """
        Start Serena MCP server in the background.
        
        Args:
            project_path: Path to project to analyze
            port: Port for SSE server
            context: Serena context to use
            mode: List of modes to activate
            
        Returns:
            Popen object for the running server
        """
        if mode is None:
            mode = ["interactive", "editing"]
        
        args = [
            "--transport", "sse",
            "--port", str(port),
            "--context", context
        ]
        
        # Add modes
        for m in mode:
            args.extend(["--mode", m])
        
        # Add project if specified
        if project_path:
            args.extend(["--project", project_path])
        
        cmd = [
            self.uv_path, "run", "python", "scripts/mcp_server.py"
        ] + args
        
        # Start the server
        process = subprocess.Popen(
            cmd,
            cwd=self.serena_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PATH": f"{os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '')}"}
        )
        
        return process
    
    def semantic_search_codebase(self, 
                                project_path: str,
                                query: str,
                                language: Optional[str] = None,
                                max_results: int = 10) -> Dict[str, Any]:
        """
        Perform semantic search on a codebase using Serena's LSP capabilities.
        
        This is a conceptual method - in practice, you'd need to interact with
        Serena through its MCP protocol or adapt its internal APIs.
        
        Args:
            project_path: Path to the codebase
            query: Search query
            language: Programming language filter
            max_results: Maximum results to return
            
        Returns:
            Search results with semantic context
        """
        # Note: This is a placeholder. In practice, you would:
        # 1. Start an MCP server
        # 2. Send MCP requests for semantic search
        # 3. Parse and return the results
        
        return {
            "query": query,
            "project_path": project_path,
            "results": [],
            "message": "Semantic search requires MCP protocol implementation",
            "recommendation": "Use start_mcp_server() and interact via MCP client"
        }
    
    def test_installation(self) -> Dict[str, Any]:
        """Test that Serena is properly installed and working."""
        info = self.get_serena_info()
        
        if not info["available"]:
            return {
                "success": False,
                "message": "Serena is not available",
                "details": info
            }
        
        # Try to get tool list
        result = self._run_serena_command(["--help"], timeout=10)
        
        return {
            "success": result["success"],
            "message": "Serena is working" if result["success"] else "Serena test failed",
            "serena_path": str(self.serena_path),
            "uv_available": bool(self.uv_path),
            "details": result
        }

def main():
    """Test the Serena wrapper."""
    try:
        wrapper = SerenaWrapper()
        test_result = wrapper.test_installation()
        
        print("Serena Wrapper Test Results:")
        print("=" * 40)
        print(f"Success: {test_result['success']}")
        print(f"Message: {test_result['message']}")
        print(f"Serena Path: {test_result['serena_path']}")
        print(f"UV Available: {test_result['uv_available']}")
        
        if test_result["success"]:
            print("\n✅ Serena is ready to use!")
            print("\nNext steps:")
            print("1. Use wrapper.start_mcp_server() to start semantic search server")
            print("2. Connect Claude Code or other MCP client to the server")
            print("3. Enjoy semantic code search and editing capabilities!")
        else:
            print(f"\n❌ Serena test failed: {test_result['details'].get('stderr', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error testing Serena wrapper: {e}")

if __name__ == "__main__":
    main()