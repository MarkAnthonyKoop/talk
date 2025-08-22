#!/usr/bin/env python3
"""
DevToolsAgent - Manages development tools and package managers safely.

This agent provides safe package management operations and development tool assistance
with validation and user-friendly responses.
"""

import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from agent.agent import Agent

log = logging.getLogger(__name__)


class DevToolsAgent(Agent):
    """
    Agent that manages development tools and package managers safely.
    
    This agent:
    1. Provides safe package management operations
    2. Checks tool availability and versions
    3. Handles common development workflows
    4. Gives clear feedback on operations
    """
    
    def __init__(self, **kwargs):
        """Initialize the DevToolsAgent."""
        roles = [
            "You are a development tools expert assistant.",
            "You help users with package managers and development tools safely.",
            "You check tool availability and provide installation guidance.",
            "You prioritize project safety and warn about destructive operations.",
            "You provide clear, actionable development advice."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Supported tools and their commands
        self.supported_tools = {
            "python": {
                "managers": ["pip", "poetry", "conda", "pipenv"],
                "info_commands": {
                    "pip": ["pip --version", "pip list"],
                    "poetry": ["poetry --version", "poetry show"],
                    "conda": ["conda --version", "conda list"],
                    "pipenv": ["pipenv --version", "pipenv graph"]
                },
                "safe_commands": {
                    "pip": ["pip list", "pip show", "pip check"],
                    "poetry": ["poetry show", "poetry check", "poetry env info"],
                    "conda": ["conda list", "conda info", "conda env list"],
                    "pipenv": ["pipenv graph", "pipenv check"]
                }
            },
            "node": {
                "managers": ["npm", "yarn", "pnpm"],
                "info_commands": {
                    "npm": ["npm --version", "npm list"],
                    "yarn": ["yarn --version", "yarn list"],
                    "pnpm": ["pnpm --version", "pnpm list"]
                },
                "safe_commands": {
                    "npm": ["npm list", "npm outdated", "npm audit"],
                    "yarn": ["yarn list", "yarn outdated", "yarn audit"],
                    "pnpm": ["pnpm list", "pnpm outdated", "pnpm audit"]
                }
            },
            "docker": {
                "commands": ["docker", "docker-compose"],
                "safe_commands": [
                    "docker --version", "docker images", "docker ps", 
                    "docker stats", "docker system df"
                ]
            },
            "testing": {
                "tools": ["pytest", "jest", "go test", "cargo test"],
                "safe_commands": [
                    "pytest --version", "jest --version"
                ]
            }
        }
        
        log.info("DevToolsAgent initialized")
    
    def _run_command(self, command: str, cwd: Path = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Run a development command safely with error handling.
        
        Args:
            command: Command to run
            cwd: Working directory (defaults to current)
            timeout: Command timeout in seconds
            
        Returns:
            Dict with stdout, stderr, return_code, and success status
        """
        try:
            result = subprocess.run(
                command.split(),
                cwd=cwd or Path.cwd(),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": 124,
                "success": False,
                "command": command
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Command not found: {command.split()[0]}",
                "return_code": 127,
                "success": False,
                "command": command
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": 1,
                "success": False,
                "command": command
            }
    
    def _check_tool_availability(self, tool: str) -> Dict[str, Any]:
        """Check if a development tool is available and get version."""
        tool_path = shutil.which(tool)
        if not tool_path:
            return {"available": False, "path": None, "version": None}
        
        # Try to get version
        version_commands = [
            f"{tool} --version",
            f"{tool} -v", 
            f"{tool} version"
        ]
        
        version = None
        for cmd in version_commands:
            result = self._run_command(cmd, timeout=10)
            if result["success"]:
                version = result["stdout"].strip().split('\n')[0]
                break
        
        return {
            "available": True,
            "path": tool_path,
            "version": version
        }
    
    def _detect_project_type(self, path: Path = None) -> Dict[str, Any]:
        """Detect project type based on files in directory."""
        search_path = path or Path.cwd()
        
        project_info = {
            "type": "unknown",
            "files": [],
            "package_managers": []
        }
        
        # Check for common project files
        project_files = {
            "python": ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile", "environment.yml"],
            "node": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
            "rust": ["Cargo.toml"],
            "go": ["go.mod"],
            "java": ["pom.xml", "build.gradle"],
            "docker": ["Dockerfile", "docker-compose.yml"]
        }
        
        found_files = []
        for file_type, files in project_files.items():
            for file_name in files:
                file_path = search_path / file_name
                if file_path.exists():
                    found_files.append((file_type, file_name))
        
        if found_files:
            # Determine primary project type
            type_counts = {}
            for file_type, file_name in found_files:
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
                project_info["files"].append(file_name)
            
            # Primary type is the one with most files
            project_info["type"] = max(type_counts.keys(), key=lambda k: type_counts[k])
            
            # Detect package managers
            if "requirements.txt" in project_info["files"]:
                project_info["package_managers"].append("pip")
            if "pyproject.toml" in project_info["files"]:
                project_info["package_managers"].append("poetry")
            if "Pipfile" in project_info["files"]:
                project_info["package_managers"].append("pipenv")
            if "package.json" in project_info["files"]:
                project_info["package_managers"].append("npm")
            if "yarn.lock" in project_info["files"]:
                project_info["package_managers"].append("yarn")
            if "pnpm-lock.yaml" in project_info["files"]:
                project_info["package_managers"].append("pnpm")
        
        return project_info
    
    def run(self, prompt: str) -> str:
        """
        Process development tool requests.
        
        Args:
            prompt: User request related to development tools
            
        Returns:
            Response describing development tool operation results
        """
        prompt_lower = prompt.lower()
        
        try:
            # Check project information
            if any(word in prompt_lower for word in ["project", "detect", "what type"]):
                return self._handle_project_detection()
            
            # Tool availability checks
            elif any(word in prompt_lower for word in ["version", "available", "installed"]):
                return self._handle_tool_check(prompt)
            
            # Package management
            elif any(word in prompt_lower for word in ["install", "add", "package"]):
                return self._handle_package_info(prompt)
            
            # List packages/dependencies
            elif any(word in prompt_lower for word in ["list", "show", "dependencies"]):
                return self._handle_list_packages(prompt)
            
            # Docker operations
            elif "docker" in prompt_lower:
                return self._handle_docker(prompt)
            
            # Testing
            elif any(word in prompt_lower for word in ["test", "pytest", "jest"]):
                return self._handle_testing(prompt)
            
            # General help
            else:
                return self._provide_help()
                
        except Exception as e:
            log.error(f"DevToolsAgent error: {e}")
            return f"❌ Development tools operation failed: {str(e)}"
    
    def _handle_project_detection(self) -> str:
        """Handle project type detection."""
        project_info = self._detect_project_type()
        
        if project_info["type"] == "unknown":
            return "📁 No recognizable project files found in current directory.\n\nLooking for files like package.json, requirements.txt, pyproject.toml, etc."
        
        response_parts = [
            f"📁 **Project Type**: {project_info['type'].title()}",
            f"📄 **Files found**: {', '.join(project_info['files'])}"
        ]
        
        if project_info["package_managers"]:
            response_parts.append(f"📦 **Package managers**: {', '.join(project_info['package_managers'])}")
        
        return "\n".join(response_parts)
    
    def _handle_tool_check(self, prompt: str) -> str:
        """Handle tool availability and version checks."""
        # Extract tool names from prompt
        tools_to_check = []
        for tool_category in self.supported_tools.values():
            if "managers" in tool_category:
                tools_to_check.extend(tool_category["managers"])
            if "commands" in tool_category:
                tools_to_check.extend(tool_category["commands"])
        
        # Check for specific tools mentioned in prompt
        mentioned_tools = [tool for tool in tools_to_check if tool in prompt.lower()]
        
        if not mentioned_tools:
            # Check common tools
            mentioned_tools = ["python", "pip", "node", "npm", "git", "docker"]
        
        results = []
        for tool in mentioned_tools[:6]:  # Limit to 6 tools
            availability = self._check_tool_availability(tool)
            if availability["available"]:
                version_info = f" ({availability['version']})" if availability["version"] else ""
                results.append(f"✅ {tool}{version_info}")
            else:
                results.append(f"❌ {tool} - Not installed")
        
        if results:
            return "🔧 **Tool Availability**:\n" + "\n".join(results)
        else:
            return "❓ No specific tools mentioned. Try asking about Python, Node.js, Docker, etc."
    
    def _handle_package_info(self, prompt: str) -> str:
        """Handle package installation information."""
        project_info = self._detect_project_type()
        
        if project_info["type"] == "unknown":
            return "❓ Unable to determine project type. Please specify what type of packages you want to install (Python, Node.js, etc.)"
        
        response_parts = [f"📦 **Package Installation for {project_info['type'].title()} Project**:"]
        
        if project_info["type"] == "python":
            if "poetry" in project_info["package_managers"]:
                response_parts.append("• poetry add <package-name>")
            elif "pipenv" in project_info["package_managers"]:
                response_parts.append("• pipenv install <package-name>")
            else:
                response_parts.append("• pip install <package-name>")
        
        elif project_info["type"] == "node":
            if "yarn" in project_info["package_managers"]:
                response_parts.append("• yarn add <package-name>")
            elif "pnpm" in project_info["package_managers"]:
                response_parts.append("• pnpm add <package-name>")
            else:
                response_parts.append("• npm install <package-name>")
        
        response_parts.append("\n💡 **Safety Note**: Package installation requires manual commands to ensure you install exactly what you need.")
        
        return "\n".join(response_parts)
    
    def _handle_list_packages(self, prompt: str) -> str:
        """Handle listing installed packages."""
        project_info = self._detect_project_type()
        
        if project_info["type"] == "unknown":
            return "❓ No project files detected. Try 'pip list' or 'npm list' manually to see installed packages."
        
        responses = []
        
        # Try to list packages based on project type
        if project_info["type"] == "python":
            for manager in ["poetry", "pip"]:
                if manager in project_info["package_managers"] or manager == "pip":
                    if manager == "poetry":
                        result = self._run_command("poetry show")
                    else:
                        result = self._run_command("pip list")
                    
                    if result["success"]:
                        lines = result["stdout"].strip().split('\n')
                        package_count = len([l for l in lines if l.strip() and not l.startswith('-')])
                        responses.append(f"📦 **{manager}**: {package_count} packages installed")
                        if len(lines) <= 20:
                            responses.append(result["stdout"])
                        else:
                            responses.append(f"Use '{manager} list' to see all packages")
                        break
        
        elif project_info["type"] == "node":
            for manager in ["yarn", "npm"]:
                if manager in project_info["package_managers"] or manager == "npm":
                    result = self._run_command(f"{manager} list --depth=0")
                    if result["success"]:
                        responses.append(f"📦 **{manager} packages**:")
                        responses.append(result["stdout"][:1000] + "..." if len(result["stdout"]) > 1000 else result["stdout"])
                        break
        
        if responses:
            return "\n".join(responses)
        else:
            return f"❓ Could not list packages for {project_info['type']} project. Try manual commands like 'pip list' or 'npm list'."
    
    def _handle_docker(self, prompt: str) -> str:
        """Handle Docker operations."""
        docker_available = self._check_tool_availability("docker")
        
        if not docker_available["available"]:
            return "❌ Docker is not installed. Install Docker Desktop or Docker Engine to use Docker commands."
        
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["images", "list"]):
            result = self._run_command("docker images")
            if result["success"]:
                return f"🐳 **Docker Images**:\n{result['stdout']}"
            else:
                return f"❌ Could not list Docker images: {result['stderr']}"
        
        elif any(word in prompt_lower for word in ["ps", "containers", "running"]):
            result = self._run_command("docker ps")
            if result["success"]:
                return f"🐳 **Running Containers**:\n{result['stdout']}"
            else:
                return f"❌ Could not list containers: {result['stderr']}"
        
        elif "stats" in prompt_lower:
            result = self._run_command("docker stats --no-stream")
            if result["success"]:
                return f"🐳 **Container Stats**:\n{result['stdout']}"
            else:
                return f"❌ Could not get container stats: {result['stderr']}"
        
        else:
            return """🐳 **Docker Operations**:
• docker images - List Docker images
• docker ps - List running containers
• docker stats - Show container resource usage

💡 **Safety Note**: Container operations (build, run, stop) require manual Docker commands for safety."""
    
    def _handle_testing(self, prompt: str) -> str:
        """Handle testing framework operations."""
        project_info = self._detect_project_type()
        
        response_parts = ["🧪 **Testing Information**:"]
        
        if project_info["type"] == "python":
            pytest_available = self._check_tool_availability("pytest")
            if pytest_available["available"]:
                response_parts.append(f"✅ pytest available ({pytest_available['version']})")
                response_parts.append("• pytest - Run all tests")
                response_parts.append("• pytest -v - Verbose output")
                response_parts.append("• pytest --tb=short - Short traceback")
            else:
                response_parts.append("❌ pytest not installed")
                response_parts.append("• Install: pip install pytest")
        
        elif project_info["type"] == "node":
            # Check if jest is in package.json
            package_json = Path.cwd() / "package.json"
            if package_json.exists():
                try:
                    with open(package_json) as f:
                        data = json.load(f)
                    
                    scripts = data.get("scripts", {})
                    if "test" in scripts:
                        response_parts.append(f"✅ Test script found: {scripts['test']}")
                        response_parts.append("• npm test - Run tests")
                    
                    dependencies = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                    if "jest" in dependencies:
                        response_parts.append("✅ Jest testing framework detected")
                    
                except Exception:
                    pass
        
        response_parts.append("\n💡 **Safety Note**: Test execution requires manual commands to ensure proper test environment.")
        
        return "\n".join(response_parts)
    
    def _provide_help(self) -> str:
        """Provide general development tools help."""
        return """🔧 **Development Tools I can help with**:

📊 **Information**:
  • Project type detection
  • Tool availability and versions
  • Package/dependency listing

📦 **Package Management**:
  • Python: pip, poetry, pipenv, conda
  • Node.js: npm, yarn, pnpm
  • Installation guidance

🐳 **Docker**:
  • List images and containers
  • Container stats and info

🧪 **Testing**:
  • Test framework detection
  • Test execution guidance

💡 **Safety Note**: All installation and modification operations require manual commands to ensure project safety.

What development tools would you like to know about?"""
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Manage development tools, package managers, and project information"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest development tool operations are needed."""
        return [
            "install", "package", "npm", "pip", "poetry", "yarn", "pnpm",
            "docker", "test", "pytest", "jest", "dependencies", "version",
            "available", "project", "build", "dev", "development"
        ]