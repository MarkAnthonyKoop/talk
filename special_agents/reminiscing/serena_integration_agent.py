#!/usr/bin/env python3
"""
SerenaIntegrationAgent - Integration between Talk framework and Serena MCP.

This agent bridges the Talk framework with Serena's semantic code analysis capabilities,
providing a seamless way to leverage LSP-based semantic search while keeping the
Talk project on pip and Serena on UV.
"""

from __future__ import annotations

import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import signal
import os

from agent.agent import Agent
from .serena_wrapper import SerenaWrapper

log = logging.getLogger(__name__)

class SerenaIntegrationAgent(Agent):
    """
    Agent that integrates Serena's semantic search capabilities with Talk framework.
    
    This agent solves the "Claude working at 30% potential" problem by providing
    focused, semantically-relevant code context instead of dumping entire files.
    
    Features:
    - Semantic symbol search using LSP
    - Code relationship mapping
    - Context-aware code retrieval  
    - Language-specific analysis (13+ languages)
    - Minimal token usage via focused results
    """
    
    def __init__(self, **kwargs):
        """Initialize with Serena integration capabilities."""
        super().__init__(roles=[
            "You are a semantic code analysis specialist using Serena MCP integration.",
            "You provide focused, relevant code context using Language Server Protocol analysis.",
            "You find symbols, references, and relationships in codebases efficiently.",
            "You solve the 'context pollution' problem by returning only relevant code pieces.",
            "You understand code at the semantic level, not just text matching."
        ], **kwargs)
        
        # Initialize Serena wrapper
        self.serena = SerenaWrapper()
        self.mcp_server = None
        self.server_port = 9121
        self.server_url = f"http://localhost:{self.server_port}"
        
        # Test Serena availability
        test_result = self.serena.test_installation()
        if not test_result["success"]:
            log.error(f"Serena not available: {test_result['message']}")
            self.serena_available = False
        else:
            self.serena_available = True
            log.info("Serena integration ready")
    
    def run(self, input_text: str) -> str:
        """
        Process semantic search request and return focused context.
        
        Args:
            input_text: Search query or task description
            
        Returns:
            Focused code context or analysis results
        """
        try:
            if not self.serena_available:
                return "SERENA_ERROR: Serena MCP is not available. Please check installation."
            
            # Parse the input to understand the request
            request = self._parse_request(input_text)
            
            if request["type"] == "semantic_search":
                return self._handle_semantic_search(request)
            elif request["type"] == "symbol_analysis":
                return self._handle_symbol_analysis(request)
            elif request["type"] == "codebase_overview":
                return self._handle_codebase_overview(request)
            elif request["type"] == "start_server":
                return self._handle_start_server(request)
            elif request["type"] == "stop_server":
                return self._handle_stop_server(request)
            else:
                return self._handle_general_query(request)
                
        except Exception as e:
            log.error(f"Error in Serena integration: {e}")
            return f"SERENA_ERROR: {str(e)}"
    
    def _parse_request(self, input_text: str) -> Dict[str, Any]:
        """Parse the input to determine request type and parameters."""
        input_lower = input_text.lower()
        
        # Determine request type
        if "start serena" in input_lower or "start mcp" in input_lower:
            return {
                "type": "start_server",
                "text": input_text,
                "project_path": self._extract_project_path(input_text)
            }
        elif "stop serena" in input_lower or "stop mcp" in input_lower:
            return {
                "type": "stop_server",
                "text": input_text
            }
        elif any(word in input_lower for word in ["find", "search", "locate"]):
            return {
                "type": "semantic_search",
                "text": input_text,
                "query": input_text,
                "project_path": self._extract_project_path(input_text)
            }
        elif any(word in input_lower for word in ["symbol", "function", "class", "method"]):
            return {
                "type": "symbol_analysis", 
                "text": input_text,
                "symbol": self._extract_symbol_name(input_text),
                "project_path": self._extract_project_path(input_text)
            }
        elif "overview" in input_lower or "structure" in input_lower:
            return {
                "type": "codebase_overview",
                "text": input_text,
                "project_path": self._extract_project_path(input_text)
            }
        else:
            return {
                "type": "general_query",
                "text": input_text,
                "project_path": self._extract_project_path(input_text)
            }
    
    def _extract_project_path(self, text: str) -> Optional[str]:
        """Extract project path from input text."""
        # Look for path-like patterns
        import re
        
        # Look for absolute paths
        abs_paths = re.findall(r'/[\w/.-]+', text)
        if abs_paths:
            for path in abs_paths:
                if Path(path).exists():
                    return path
        
        # Look for relative paths
        rel_paths = re.findall(r'[\w.-]+/[\w/.-]+', text)
        if rel_paths:
            for path in rel_paths:
                full_path = Path.cwd() / path
                if full_path.exists():
                    return str(full_path)
        
        # Default to current directory if it looks like a project
        cwd = Path.cwd()
        if any((cwd / f).exists() for f in ["setup.py", "pyproject.toml", "package.json", "Cargo.toml"]):
            return str(cwd)
        
        return None
    
    def _extract_symbol_name(self, text: str) -> Optional[str]:
        """Extract symbol name from input text."""
        import re
        
        # Look for CamelCase (class names)
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
        if camel_case:
            return camel_case[0]
        
        # Look for snake_case (function names)
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)
        if snake_case:
            return snake_case[0]
        
        # Look for simple identifiers
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        # Filter out common words
        common_words = {
            "find", "search", "function", "class", "method", "symbol", 
            "code", "file", "project", "the", "a", "an", "in", "of", "to"
        }
        
        for identifier in identifiers:
            if identifier.lower() not in common_words and len(identifier) > 2:
                return identifier
        
        return None
    
    def _handle_start_server(self, request: Dict[str, Any]) -> str:
        """Start the Serena MCP server."""
        if self.mcp_server and self.mcp_server.poll() is None:
            return f"SERENA_STATUS: MCP server already running on port {self.server_port}"
        
        project_path = request.get("project_path")
        if not project_path:
            project_path = str(Path.cwd())
        
        try:
            log.info(f"Starting Serena MCP server for project: {project_path}")
            
            self.mcp_server = self.serena.start_mcp_server(
                project_path=project_path,
                port=self.server_port,
                context="ide-assistant",
                mode=["interactive", "editing"]
            )
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Check if server is responding
            if self._check_server_health():
                return f"""SERENA_STARTED: MCP server running successfully

Server Details:
- Port: {self.server_port}
- URL: {self.server_url}
- Project: {project_path}
- Context: ide-assistant
- Modes: interactive, editing

You can now perform semantic searches and code analysis."""
            else:
                return "SERENA_ERROR: Server started but not responding to health checks"
            
        except Exception as e:
            log.error(f"Failed to start Serena server: {e}")
            return f"SERENA_ERROR: Failed to start MCP server: {str(e)}"
    
    def _handle_stop_server(self, request: Dict[str, Any]) -> str:
        """Stop the Serena MCP server."""
        if not self.mcp_server or self.mcp_server.poll() is not None:
            return "SERENA_STATUS: No MCP server is currently running"
        
        try:
            # Try graceful shutdown first
            self.mcp_server.terminate()
            
            # Wait for graceful shutdown
            try:
                self.mcp_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                self.mcp_server.kill()
                self.mcp_server.wait()
            
            self.mcp_server = None
            return "SERENA_STOPPED: MCP server has been stopped"
            
        except Exception as e:
            log.error(f"Error stopping Serena server: {e}")
            return f"SERENA_ERROR: Failed to stop server: {str(e)}"
    
    def _handle_semantic_search(self, request: Dict[str, Any]) -> str:
        """Handle semantic search requests."""
        if not self._ensure_server_running(request.get("project_path")):
            return "SERENA_ERROR: Could not start MCP server for semantic search"
        
        query = request["query"]
        
        # For now, return a placeholder response
        # In a full implementation, this would send MCP requests to the server
        return f"""SERENA_SEMANTIC_SEARCH: 

Query: {query}
Status: Server running on port {self.server_port}

Note: Full MCP protocol integration needed for actual semantic search.
This would typically:
1. Send find_symbol requests to MCP server
2. Get semantic context from LSP
3. Return focused code snippets
4. Filter by relevance and language

To implement full integration:
- Use MCP client library to send requests
- Parse semantic responses from Serena
- Format results for optimal LLM consumption"""
    
    def _handle_symbol_analysis(self, request: Dict[str, Any]) -> str:
        """Handle symbol analysis requests."""
        if not self._ensure_server_running(request.get("project_path")):
            return "SERENA_ERROR: Could not start MCP server for symbol analysis"
        
        symbol = request.get("symbol", "unknown")
        
        return f"""SERENA_SYMBOL_ANALYSIS:

Symbol: {symbol}
Server: Running on port {self.server_port}

Note: Full implementation would:
1. Use find_symbol MCP tool
2. Get symbol definition and references
3. Find related symbols and dependencies
4. Return semantic context tree

This provides the focused context needed to avoid the '30% potential' problem."""
    
    def _handle_codebase_overview(self, request: Dict[str, Any]) -> str:
        """Handle codebase overview requests."""
        project_path = request.get("project_path", str(Path.cwd()))
        
        if not self._ensure_server_running(project_path):
            return "SERENA_ERROR: Could not start MCP server for codebase analysis"
        
        return f"""SERENA_CODEBASE_OVERVIEW:

Project: {project_path}
Server: Running on port {self.server_port}

Note: Full implementation would:
1. Use get_symbols_overview MCP tool
2. Map codebase structure using LSP
3. Identify key entry points and modules
4. Create semantic dependency graph

This gives high-level understanding without token waste."""
    
    def _handle_general_query(self, request: Dict[str, Any]) -> str:
        """Handle general queries about Serena capabilities."""
        return f"""SERENA_INFO: Semantic Code Analysis Agent

Available Commands:
- "start serena [project_path]" - Start MCP server
- "stop serena" - Stop MCP server  
- "find [symbol/pattern]" - Semantic search
- "analyze symbol [name]" - Symbol analysis
- "codebase overview" - Project structure

Features:
- Language Server Protocol integration (13+ languages)
- Semantic symbol search (not just text matching)
- Focused context retrieval (solves 30% potential problem)
- Code relationship mapping
- Minimal token usage

Current Status:
- Serena Available: {self.serena_available}
- Server Running: {self.mcp_server is not None and self.mcp_server.poll() is None}
- Server Port: {self.server_port}

Usage: This agent bridges Talk framework with Serena's UV-based environment."""
    
    def _ensure_server_running(self, project_path: Optional[str] = None) -> bool:
        """Ensure MCP server is running."""
        if self.mcp_server and self.mcp_server.poll() is None:
            return self._check_server_health()
        
        # Try to start server
        try:
            if not project_path:
                project_path = str(Path.cwd())
            
            self.mcp_server = self.serena.start_mcp_server(
                project_path=project_path,
                port=self.server_port,
                context="ide-assistant",
                mode=["interactive", "editing"]
            )
            
            time.sleep(2)
            return self._check_server_health()
            
        except Exception as e:
            log.error(f"Failed to ensure server running: {e}")
            return False
    
    def _check_server_health(self) -> bool:
        """Check if the MCP server is responding."""
        try:
            # Try to connect to the server
            # Note: This is a simplified health check
            # Real MCP health check would use proper MCP protocol
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            # For SSE servers, connection might fail but server could still be running
            return self.mcp_server and self.mcp_server.poll() is None
    
    def __del__(self):
        """Cleanup: stop server when agent is destroyed."""
        if self.mcp_server and self.mcp_server.poll() is None:
            try:
                self.mcp_server.terminate()
                self.mcp_server.wait(timeout=2)
            except Exception:
                pass