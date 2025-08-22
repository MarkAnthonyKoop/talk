#!/usr/bin/env python3
"""
GitAgent - Handles Git repository operations safely.

This agent provides safe Git operations with validation and user-friendly responses.
It checks repository state before operations and provides clear feedback.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from agent.agent import Agent

log = logging.getLogger(__name__)


class GitAgent(Agent):
    """
    Agent that handles Git repository operations safely.
    
    This agent:
    1. Validates Git repository state before operations
    2. Provides user-friendly Git command interface
    3. Handles common Git workflows safely
    4. Gives clear feedback on Git operations
    """
    
    def __init__(self, **kwargs):
        """Initialize the GitAgent."""
        roles = [
            "You are a Git expert assistant.",
            "You help users with Git repository operations safely.",
            "You explain Git concepts clearly and provide helpful guidance.",
            "You always check repository state before making changes.",
            "You prioritize data safety and warn about destructive operations."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Common Git operations
        self.safe_operations = {
            "status": "git status --porcelain",
            "log": "git log --oneline -10",
            "branch": "git branch -a",
            "remote": "git remote -v",
            "diff": "git diff --stat",
            "stash_list": "git stash list"
        }
        
        # Operations that modify repository (require validation)
        self.modify_operations = [
            "add", "commit", "push", "pull", "merge", "rebase",
            "checkout", "branch", "stash", "reset", "clean"
        ]
        
        log.info("GitAgent initialized")
    
    def _run_git_command(self, command: str, cwd: Path = None) -> Dict[str, Any]:
        """
        Run a Git command safely with error handling.
        
        Args:
            command: Git command to run
            cwd: Working directory (defaults to current)
            
        Returns:
            Dict with stdout, stderr, return_code, and success status
        """
        try:
            result = subprocess.run(
                command.split(),
                cwd=cwd or Path.cwd(),
                capture_output=True,
                text=True,
                timeout=30
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
                "stderr": "Git command timed out",
                "return_code": 124,
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
    
    def _is_git_repository(self, path: Path = None) -> bool:
        """Check if current directory is a Git repository."""
        result = self._run_git_command("git rev-parse --git-dir", path)
        return result["success"]
    
    def _get_repository_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status."""
        if not self._is_git_repository():
            return {"error": "Not a Git repository"}
        
        status = {}
        
        # Basic status
        result = self._run_git_command("git status --porcelain")
        if result["success"]:
            lines = result["stdout"].strip().split('\n') if result["stdout"].strip() else []
            status["modified_files"] = len([l for l in lines if l.startswith(' M')])
            status["staged_files"] = len([l for l in lines if l.startswith('M ')])
            status["untracked_files"] = len([l for l in lines if l.startswith('??')])
            status["clean"] = len(lines) == 0
        
        # Current branch
        result = self._run_git_command("git branch --show-current")
        if result["success"]:
            status["current_branch"] = result["stdout"].strip()
        
        # Remote status
        result = self._run_git_command("git status -sb")
        if result["success"]:
            first_line = result["stdout"].split('\n')[0] if result["stdout"] else ""
            if "ahead" in first_line:
                status["ahead"] = True
            if "behind" in first_line:
                status["behind"] = True
        
        return status
    
    def run(self, prompt: str) -> str:
        """
        Process Git-related requests.
        
        Args:
            prompt: User request related to Git
            
        Returns:
            Response describing Git operation results
        """
        prompt_lower = prompt.lower()
        
        try:
            # Check if we're in a Git repository
            if not self._is_git_repository():
                if any(word in prompt_lower for word in ["init", "clone"]):
                    return self._handle_repo_creation(prompt)
                else:
                    return "❌ Not in a Git repository. Use 'git init' to create one or 'git clone <url>' to clone an existing repository."
            
            # Git status
            if "status" in prompt_lower:
                return self._handle_status()
            
            # Git log
            elif any(word in prompt_lower for word in ["log", "history", "commits"]):
                return self._handle_log()
            
            # Git branch operations
            elif "branch" in prompt_lower:
                return self._handle_branch(prompt)
            
            # Git add/stage
            elif any(word in prompt_lower for word in ["add", "stage"]):
                return self._handle_add(prompt)
            
            # Git commit
            elif "commit" in prompt_lower:
                return self._handle_commit(prompt)
            
            # Git push
            elif "push" in prompt_lower:
                return self._handle_push()
            
            # Git pull
            elif "pull" in prompt_lower:
                return self._handle_pull()
            
            # Git diff
            elif "diff" in prompt_lower:
                return self._handle_diff()
            
            # Git stash
            elif "stash" in prompt_lower:
                return self._handle_stash(prompt)
            
            # General help
            else:
                return self._provide_help()
                
        except Exception as e:
            log.error(f"GitAgent error: {e}")
            return f"❌ Git operation failed: {str(e)}"
    
    def _handle_status(self) -> str:
        """Handle git status requests."""
        repo_status = self._get_repository_status()
        
        if "error" in repo_status:
            return repo_status["error"]
        
        status_parts = []
        status_parts.append(f"📁 Current branch: {repo_status.get('current_branch', 'unknown')}")
        
        if repo_status.get("clean", False):
            status_parts.append("✅ Working tree clean")
        else:
            if repo_status.get("staged_files", 0) > 0:
                status_parts.append(f"📝 {repo_status['staged_files']} staged files")
            if repo_status.get("modified_files", 0) > 0:
                status_parts.append(f"✏️  {repo_status['modified_files']} modified files")
            if repo_status.get("untracked_files", 0) > 0:
                status_parts.append(f"❓ {repo_status['untracked_files']} untracked files")
        
        if repo_status.get("ahead", False):
            status_parts.append("⬆️  Local commits ahead of remote")
        if repo_status.get("behind", False):
            status_parts.append("⬇️  Local branch behind remote")
        
        return "\n".join(status_parts)
    
    def _handle_log(self) -> str:
        """Handle git log requests."""
        result = self._run_git_command("git log --oneline -10")
        
        if not result["success"]:
            return f"❌ Could not retrieve Git log: {result['stderr']}"
        
        if not result["stdout"].strip():
            return "📋 No commits found"
        
        return f"📋 Recent commits:\n{result['stdout']}"
    
    def _handle_branch(self, prompt: str) -> str:
        """Handle branch-related operations."""
        if "list" in prompt.lower() or prompt.lower().strip() == "branch":
            result = self._run_git_command("git branch -a")
            if result["success"]:
                return f"🌿 Branches:\n{result['stdout']}"
            else:
                return f"❌ Could not list branches: {result['stderr']}"
        
        # For now, just list branches - creating/switching requires more safety checks
        return "🌿 Branch operations: Currently only listing is supported. Use 'git status' to see current branch."
    
    def _handle_add(self, prompt: str) -> str:
        """Handle git add operations."""
        # Get current status first
        result = self._run_git_command("git status --porcelain")
        if not result["success"]:
            return f"❌ Could not check repository status: {result['stderr']}"
        
        if not result["stdout"].strip():
            return "✅ No files to add - working tree is clean"
        
        # For safety, only add specific files or ask for confirmation
        if "all" in prompt.lower() or "." in prompt:
            # Show what would be added
            files = result["stdout"].strip().split('\n')
            untracked = [line[3:] for line in files if line.startswith('??')]
            modified = [line[3:] for line in files if line.startswith(' M')]
            
            file_list = []
            if untracked:
                file_list.append(f"Untracked: {', '.join(untracked[:5])}")
            if modified:
                file_list.append(f"Modified: {', '.join(modified[:5])}")
            
            return f"📝 Files that would be staged:\n{chr(10).join(file_list)}\n\nUse 'git commit' to commit staged changes."
        
        return "📝 To stage files, be specific about which files to add, or use 'git add all' to stage all changes."
    
    def _handle_commit(self, prompt: str) -> str:
        """Handle git commit operations."""
        # Check if there are staged files
        result = self._run_git_command("git diff --cached --name-only")
        if not result["success"]:
            return f"❌ Could not check staged files: {result['stderr']}"
        
        if not result["stdout"].strip():
            return "❌ No staged files to commit. Use 'git add' to stage files first."
        
        staged_files = result["stdout"].strip().split('\n')
        
        # For now, just show what would be committed
        return f"📝 Staged files ready to commit:\n{chr(10).join(f'  • {file}' for file in staged_files)}\n\n💡 Commit operations require manual Git commands for safety."
    
    def _handle_push(self) -> str:
        """Handle git push operations."""
        # Check repository status
        repo_status = self._get_repository_status()
        
        if not repo_status.get("ahead", False):
            return "✅ No local commits to push - already up to date with remote"
        
        return "⬆️  Local commits ready to push to remote.\n💡 Push operations require manual Git commands for safety."
    
    def _handle_pull(self) -> str:
        """Handle git pull operations."""
        repo_status = self._get_repository_status()
        
        if not repo_status.get("behind", False):
            return "✅ Already up to date with remote"
        
        return "⬇️  Remote has new commits available.\n💡 Pull operations require manual Git commands for safety."
    
    def _handle_diff(self) -> str:
        """Handle git diff operations."""
        result = self._run_git_command("git diff --stat")
        
        if not result["success"]:
            return f"❌ Could not get diff: {result['stderr']}"
        
        if not result["stdout"].strip():
            return "📊 No changes in working directory"
        
        return f"📊 Changes in working directory:\n{result['stdout']}"
    
    def _handle_stash(self, prompt: str) -> str:
        """Handle git stash operations."""
        if "list" in prompt.lower():
            result = self._run_git_command("git stash list")
            if result["success"]:
                if result["stdout"].strip():
                    return f"📦 Stashes:\n{result['stdout']}"
                else:
                    return "📦 No stashes found"
            else:
                return f"❌ Could not list stashes: {result['stderr']}"
        
        return "📦 Stash operations: Use 'git stash list' to see saved stashes.\n💡 Creating/applying stashes requires manual Git commands for safety."
    
    def _handle_repo_creation(self, prompt: str) -> str:
        """Handle repository initialization or cloning."""
        if "init" in prompt.lower():
            return "🏁 To initialize a Git repository, run: git init\n💡 Repository creation requires manual Git commands for safety."
        elif "clone" in prompt.lower():
            return "📥 To clone a repository, run: git clone <repository-url>\n💡 Repository cloning requires manual Git commands for safety."
        else:
            return "🏁 Repository operations:\n• git init - Initialize new repository\n• git clone <url> - Clone existing repository"
    
    def _provide_help(self) -> str:
        """Provide general Git help."""
        return """🔧 Git operations I can help with:

📊 **Information**:
  • status - Show repository status
  • log - Show recent commits  
  • branch - List branches
  • diff - Show changes

🔍 **Safe Operations**:
  • Check what files would be staged
  • Show commit-ready files
  • Display remote sync status

💡 **Safety Note**: Destructive operations (commit, push, pull, merge) require manual Git commands for your safety.

What would you like to know about your Git repository?"""
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Handle Git repository operations and provide repository status"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest Git operations are needed."""
        return [
            "git", "repository", "repo", "commit", "push", "pull", "branch",
            "status", "log", "diff", "stash", "merge", "checkout", "clone"
        ]