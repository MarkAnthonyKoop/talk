#!/usr/bin/env python3
"""
SharedWorkspace - Coordinated file system access for collaborative agents.

This module provides a shared workspace that allows multiple agents to work
on the same codebase simultaneously while preventing conflicts and maintaining
consistency. It integrates with version control and provides locking mechanisms.

Features:
- File-level locking for conflict prevention
- Version control integration (Git)
- Change tracking and rollback capabilities
- Atomic operations for file modifications
- Real-time change notifications
- Backup and restore functionality
"""

import os
import shutil
import tempfile
import threading
import time
import hashlib
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import subprocess
import logging

log = logging.getLogger(__name__)

class LockType(Enum):
    """Types of locks that can be acquired on files."""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"

@dataclass
class FileLock:
    """Information about a file lock."""
    file_path: str
    lock_type: LockType
    agent_id: str
    acquired_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

@dataclass
class FileChange:
    """Information about a file change."""
    file_path: str
    change_type: str  # "create", "modify", "delete", "rename"
    agent_id: str
    timestamp: float
    old_content_hash: Optional[str] = None
    new_content_hash: Optional[str] = None
    backup_path: Optional[str] = None
    metadata: Dict[str, Any] = None

class SharedWorkspace:
    """
    Shared workspace for collaborative agent development.
    
    Provides coordinated access to files with locking, version control,
    and change tracking to enable safe multi-agent collaboration.
    """
    
    def __init__(self, workspace_path: str, enable_git: bool = True, 
                 backup_dir: str = None, lock_timeout: float = 300.0):
        """
        Initialize the shared workspace.
        
        Args:
            workspace_path: Path to the workspace directory
            enable_git: Whether to enable Git integration
            backup_dir: Directory for backups (defaults to workspace/.backups)
            lock_timeout: Default lock timeout in seconds
        """
        self.workspace_path = Path(workspace_path).resolve()
        self.enable_git = enable_git
        self.lock_timeout = lock_timeout
        
        # Create workspace directory if it doesn't exist
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Setup backup directory
        if backup_dir:
            self.backup_dir = Path(backup_dir).resolve()
        else:
            self.backup_dir = self.workspace_path / ".backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Lock management
        self.locks: Dict[str, FileLock] = {}  # file_path -> lock
        self.lock = threading.RLock()  # Thread-safe lock management
        self.file_locks: Dict[str, threading.Lock] = {}  # Per-file locks
        
        # Change tracking
        self.changes: List[FileChange] = []
        self.file_hashes: Dict[str, str] = {}  # file_path -> content_hash
        
        # Initialize Git if enabled
        if self.enable_git:
            self._init_git()
        
        # Build initial file hash index
        self._build_file_index()
        
        log.info(f"SharedWorkspace initialized at {self.workspace_path}")
    
    def _init_git(self):
        """Initialize Git repository if not already present."""
        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=self.workspace_path,
                    check=True,
                    capture_output=True
                )
                log.info("Git repository initialized")
            except subprocess.CalledProcessError as e:
                log.warning(f"Failed to initialize Git: {e}")
                self.enable_git = False
    
    def _build_file_index(self):
        """Build index of all files and their content hashes."""
        for file_path in self.workspace_path.rglob("*"):
            if file_path.is_file() and not self._is_ignored_file(file_path):
                rel_path = str(file_path.relative_to(self.workspace_path))
                self.file_hashes[rel_path] = self._get_file_hash(file_path)
    
    def _is_ignored_file(self, file_path: Path) -> bool:
        """Check if file should be ignored (e.g., .git, .backups, temp files)."""
        parts = file_path.parts
        ignored_dirs = {".git", ".backups", "__pycache__", ".pytest_cache", "node_modules"}
        ignored_extensions = {".pyc", ".pyo", ".swp", ".tmp"}
        
        # Check for ignored directories
        if any(part in ignored_dirs for part in parts):
            return True
        
        # Check for ignored extensions
        if file_path.suffix in ignored_extensions:
            return True
        
        return False
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA-256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            log.warning(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def _get_file_lock(self, filename: str) -> threading.Lock:
        """
        Gets a lock for a specific file.
        """
        with self.lock:
            if filename not in self.file_locks:
                self.file_locks[filename] = threading.Lock()
            return self.file_locks[filename]
    
    async def acquire_lock(self, file_path: str, lock_type: LockType, 
                          agent_id: str, timeout: float = None) -> bool:
        """
        Acquire a lock on a file.
        
        Args:
            file_path: Relative path to file from workspace root
            lock_type: Type of lock to acquire
            agent_id: ID of agent requesting lock
            timeout: Lock timeout in seconds (uses default if None)
            
        Returns:
            True if lock acquired successfully
        """
        if timeout is None:
            timeout = self.lock_timeout
        
        abs_path = self.workspace_path / file_path
        file_lock = self._get_file_lock(file_path)
        
        with file_lock:
            # Check if file is already locked
            if file_path in self.locks:
                existing_lock = self.locks[file_path]
                
                # Clean up expired locks
                if existing_lock.is_expired():
                    del self.locks[file_path]
                    log.debug(f"Removed expired lock on {file_path}")
                else:
                    # Check lock compatibility
                    if not self._locks_compatible(existing_lock.lock_type, lock_type, 
                                                 existing_lock.agent_id, agent_id):
                        log.debug(f"Lock conflict on {file_path}: {existing_lock.lock_type} vs {lock_type}")
                        return False
            
            # Acquire the lock
            expires_at = time.time() + timeout if timeout > 0 else None
            lock = FileLock(
                file_path=file_path,
                lock_type=lock_type,
                agent_id=agent_id,
                acquired_at=time.time(),
                expires_at=expires_at
            )
            
            self.locks[file_path] = lock
            log.debug(f"Lock acquired: {file_path} ({lock_type}) by {agent_id}")
            return True
    
    def _locks_compatible(self, existing_type: LockType, requested_type: LockType,
                         existing_agent: str, requesting_agent: str) -> bool:
        """Check if two locks are compatible."""
        # Same agent can upgrade/downgrade locks
        if existing_agent == requesting_agent:
            return True
        
        # Multiple read locks are compatible
        if existing_type == LockType.READ and requested_type == LockType.READ:
            return True
        
        # Exclusive locks are never compatible with others
        if existing_type == LockType.EXCLUSIVE or requested_type == LockType.EXCLUSIVE:
            return False
        
        # Write locks are not compatible with other locks
        if existing_type == LockType.WRITE or requested_type == LockType.WRITE:
            return False
        
        return True
    
    async def release_lock(self, file_path: str, agent_id: str) -> bool:
        """
        Release a lock on a file.
        
        Args:
            file_path: Relative path to file
            agent_id: ID of agent releasing lock
            
        Returns:
            True if lock released successfully
        """
        file_lock = self._get_file_lock(file_path)
        
        with file_lock:
            if file_path not in self.locks:
                return False
            
            existing_lock = self.locks[file_path]
            if existing_lock.agent_id != agent_id:
                log.warning(f"Agent {agent_id} tried to release lock owned by {existing_lock.agent_id}")
                return False
            
            del self.locks[file_path]
            log.debug(f"Lock released: {file_path} by {agent_id}")
            return True
    
    async def read_file(self, file_path: str, agent_id: str) -> Optional[str]:
        """
        Read a file with automatic read lock acquisition.
        
        Args:
            file_path: Relative path to file
            agent_id: ID of agent reading file
            
        Returns:
            File content as string, or None if failed
        """
        # Acquire read lock
        if not await self.acquire_lock(file_path, LockType.READ, agent_id):
            log.warning(f"Could not acquire read lock for {file_path}")
            return None
        
        try:
            abs_path = self.workspace_path / file_path
            if not abs_path.exists():
                return None
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            log.debug(f"File read: {file_path} by {agent_id}")
            return content
            
        except Exception as e:
            log.error(f"Error reading file {file_path}: {e}")
            return None
        finally:
            await self.release_lock(file_path, agent_id)
    
    async def write_file(self, file_path: str, content: str, agent_id: str,
                        create_backup: bool = True) -> bool:
        """
        Write content to a file with automatic write lock acquisition.
        
        Args:
            file_path: Relative path to file
            content: Content to write
            agent_id: ID of agent writing file
            create_backup: Whether to create backup before writing
            
        Returns:
            True if write successful
        """
        # Acquire write lock
        if not await self.acquire_lock(file_path, LockType.WRITE, agent_id):
            log.warning(f"Could not acquire write lock for {file_path}")
            return False
        
        try:
            abs_path = self.workspace_path / file_path
            
            # Create backup if file exists and backup requested
            backup_path = None
            old_hash = None
            if abs_path.exists() and create_backup:
                backup_path = await self._create_backup(file_path, agent_id)
                old_hash = self.file_hashes.get(file_path)
            
            # Ensure parent directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update file hash
            new_hash = self._get_file_hash(abs_path)
            self.file_hashes[file_path] = new_hash
            
            # Record change
            change_type = "create" if old_hash is None else "modify"
            change = FileChange(
                file_path=file_path,
                change_type=change_type,
                agent_id=agent_id,
                timestamp=time.time(),
                old_content_hash=old_hash,
                new_content_hash=new_hash,
                backup_path=backup_path
            )
            self.changes.append(change)
            
            # Git commit if enabled
            if self.enable_git:
                await self._git_commit(file_path, agent_id, change_type)
            
            log.info(f"File {change_type}: {file_path} by {agent_id}")
            return True
            
        except Exception as e:
            log.error(f"Error writing file {file_path}: {e}")
            return False
        finally:
            await self.release_lock(file_path, agent_id)
    
    async def create_file(self, file_path: str, content: str, agent_id: str) -> bool:
        """Create a new file."""
        abs_path = self.workspace_path / file_path
        if abs_path.exists():
            log.warning(f"File already exists: {file_path}")
            return False
        
        return await self.write_file(file_path, content, agent_id, create_backup=False)
    
    async def delete_file(self, file_path: str, agent_id: str, 
                         create_backup: bool = True) -> bool:
        """
        Delete a file with automatic exclusive lock acquisition.
        
        Args:
            file_path: Relative path to file
            agent_id: ID of agent deleting file
            create_backup: Whether to create backup before deletion
            
        Returns:
            True if deletion successful
        """
        # Acquire exclusive lock
        if not await self.acquire_lock(file_path, LockType.EXCLUSIVE, agent_id):
            log.warning(f"Could not acquire exclusive lock for {file_path}")
            return False
        
        try:
            abs_path = self.workspace_path / file_path
            if not abs_path.exists():
                return False
            
            # Create backup if requested
            backup_path = None
            if create_backup:
                backup_path = await self._create_backup(file_path, agent_id)
            
            old_hash = self.file_hashes.get(file_path)
            
            # Delete file
            abs_path.unlink()
            
            # Remove from tracking
            if file_path in self.file_hashes:
                del self.file_hashes[file_path]
            
            # Record change
            change = FileChange(
                file_path=file_path,
                change_type="delete",
                agent_id=agent_id,
                timestamp=time.time(),
                old_content_hash=old_hash,
                backup_path=backup_path
            )
            self.changes.append(change)
            
            # Git commit if enabled
            if self.enable_git:
                await self._git_commit(file_path, agent_id, "delete")
            
            log.info(f"File deleted: {file_path} by {agent_id}")
            return True
            
        except Exception as e:
            log.error(f"Error deleting file {file_path}: {e}")
            return False
        finally:
            await self.release_lock(file_path, agent_id)
    
    async def _create_backup(self, file_path: str, agent_id: str) -> str:
        """Create a backup of a file."""
        abs_path = self.workspace_path / file_path
        if not abs_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.replace('/', '_')}_{timestamp}_{agent_id}.backup"
        backup_path = self.backup_dir / backup_name
        
        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file to backup
        shutil.copy2(abs_path, backup_path)
        
        log.debug(f"Backup created: {backup_path}")
        return str(backup_path.relative_to(self.workspace_path))
    
    async def _git_commit(self, file_path: str, agent_id: str, change_type: str):
        """Commit changes to Git."""
        if not self.enable_git:
            return
        
        try:
            # Stage the file
            if change_type == "delete":
                subprocess.run(
                    ["git", "rm", file_path],
                    cwd=self.workspace_path,
                    check=True,
                    capture_output=True
                )
            else:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=self.workspace_path,
                    check=True,
                    capture_output=True
                )
            
            # Commit with message
            commit_msg = f"{change_type.title()} {file_path} by {agent_id}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace_path,
                check=True,
                capture_output=True
            )
            
            log.debug(f"Git commit: {commit_msg}")
            
        except subprocess.CalledProcessError as e:
            log.warning(f"Git commit failed: {e}")
    
    def list_files(self, pattern: str = "*", include_locked: bool = True) -> List[Dict[str, Any]]:
        """
        List files in the workspace.
        
        Args:
            pattern: Glob pattern to filter files
            include_locked: Whether to include lock information
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for file_path in self.workspace_path.glob(pattern):
            if file_path.is_file() and not self._is_ignored_file(file_path):
                rel_path = str(file_path.relative_to(self.workspace_path))
                
                file_info = {
                    "path": rel_path,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "hash": self.file_hashes.get(rel_path, "")
                }
                
                if include_locked and rel_path in self.locks:
                    lock = self.locks[rel_path]
                    file_info["lock"] = {
                        "type": lock.lock_type.value,
                        "agent_id": lock.agent_id,
                        "acquired_at": lock.acquired_at,
                        "expires_at": lock.expires_at
                    }
                
                files.append(file_info)
        
        return sorted(files, key=lambda f: f["path"])
    
    def get_changes(self, agent_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get change history.
        
        Args:
            agent_id: Optional filter by agent ID
            limit: Maximum number of changes to return
            
        Returns:
            List of change dictionaries
        """
        changes = self.changes
        
        if agent_id:
            changes = [c for c in changes if c.agent_id == agent_id]
        
        # Sort by timestamp (newest first) and limit
        changes = sorted(changes, key=lambda c: c.timestamp, reverse=True)[:limit]
        
        return [asdict(change) for change in changes]
    
    def get_active_locks(self) -> Dict[str, Dict[str, Any]]:
        """Get all active locks."""
        active_locks = {}
        
        for file_path, lock in self.locks.items():
            if not lock.is_expired():
                active_locks[file_path] = {
                    "type": lock.lock_type.value,
                    "agent_id": lock.agent_id,
                    "acquired_at": lock.acquired_at,
                    "expires_at": lock.expires_at
                }
        
        return active_locks
    
    def cleanup_expired_locks(self) -> int:
        """Remove expired locks and return count of removed locks."""
        expired_count = 0
        expired_files = []
        
        for file_path, lock in self.locks.items():
            if lock.is_expired():
                expired_files.append(file_path)
        
        for file_path in expired_files:
            del self.locks[file_path]
            expired_count += 1
            log.debug(f"Removed expired lock on {file_path}")
        
        return expired_count
    
    async def rollback_change(self, change_index: int, agent_id: str) -> bool:
        """
        Rollback a specific change.
        
        Args:
            change_index: Index in the changes list
            agent_id: Agent requesting rollback
            
        Returns:
            True if rollback successful
        """
        if change_index >= len(self.changes):
            return False
        
        change = self.changes[change_index]
        
        # Only allow agent to rollback their own changes
        if change.agent_id != agent_id:
            log.warning(f"Agent {agent_id} tried to rollback change by {change.agent_id}")
            return False
        
        if not change.backup_path:
            log.warning(f"No backup available for change: {change.file_path}")
            return False
        
        try:
            # Restore from backup
            backup_abs_path = self.workspace_path / change.backup_path
            file_abs_path = self.workspace_path / change.file_path
            
            if backup_abs_path.exists():
                shutil.copy2(backup_abs_path, file_abs_path)
                
                # Update file hash
                self.file_hashes[change.file_path] = self._get_file_hash(file_abs_path)
                
                log.info(f"Rollback successful: {change.file_path} by {agent_id}")
                return True
        
        except Exception as e:
            log.error(f"Rollback failed: {e}")
        
        return False

# Example usage
async def main():
    """Example usage of SharedWorkspace."""
    workspace = SharedWorkspace("/tmp/test_workspace")
    
    # Agent 1 creates a file
    success = await workspace.create_file("test.py", "print('Hello World')", "agent1")
    print(f"File created: {success}")
    
    # Agent 2 reads the file
    content = await workspace.read_file("test.py", "agent2") 
    print(f"File content: {content}")
    
    # Agent 1 modifies the file
    success = await workspace.write_file("test.py", "print('Hello Agent World!')", "agent1")
    print(f"File modified: {success}")
    
    # List files
    files = workspace.list_files()
    print(f"Files: {files}")
    
    # Get changes
    changes = workspace.get_changes()
    print(f"Changes: {len(changes)}")

if __name__ == "__main__":
    asyncio.run(main())