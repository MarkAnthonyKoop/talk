#!/usr/bin/env python3
# special_agents/file_agent.py

"""
FileAgent - Specialized agent for applying unified diffs to the filesystem.

This agent takes a unified diff string as input and applies it to the filesystem
using the system's 'patch' command. It handles file creation, modification, and
error cases, and includes backup functionality for safety.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from agent.agent import Agent

log = logging.getLogger(__name__)

class FileAgent(Agent):
    """
    Specialized agent for filesystem operations.
    
    This agent applies unified diffs to the filesystem using the system's
    'patch' command. It doesn't use LLMs for its core functionality but
    inherits from Agent for interface consistency.
    """
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """
        Initialize the FileAgent.
        
        Args:
            base_dir: The base directory for file operations (defaults to current directory)
            **kwargs: Additional arguments passed to the parent Agent class
        """
        # Initialize with empty roles since this agent doesn't use LLM prompting
        super().__init__(roles=[], **kwargs)
        
        # Set the base directory for file operations
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        log.info(f"FileAgent initialized with base directory: {self.base_dir}")
        
        # Create backup directory if it doesn't exist
        self.backup_dir = self.base_dir / ".talk_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def run(self, input_text: str) -> str:
        """
        Apply a unified diff to the filesystem.
        
        Args:
            input_text: A unified diff string to apply
            
        Returns:
            A status message indicating success or failure
        """
        # Record the operation in the conversation log for provenance
        self._append("user", f"Request to apply diff:\n{input_text}")
        
        try:
            # Apply the diff and get the result
            result = self._apply_diff(input_text)
            
            # Record the result
            self._append("assistant", result)
            return result
            
        except Exception as e:
            error_msg = f"ERROR: Failed to apply diff: {str(e)}"
            log.error(error_msg)
            self._append("assistant", error_msg)
            return error_msg
    
    def _apply_diff(self, operations_text: str) -> str:
        """
        Apply structured file operations from CodeAgent.
        
        Args:
            operations_text: Structured file operations (CREATE_FILE, MODIFY_FILE, etc.)
            
        Returns:
            A status message indicating success or failure
        """
        if not operations_text.strip():
            return "No changes to apply (empty operations)"
        
        # Check if this is the new structured format or old diff format
        if "CREATE_FILE:" in operations_text or "MODIFY_FILE:" in operations_text:
            return self._apply_structured_operations(operations_text)
        else:
            # Fallback to old diff format
            return self._apply_unified_diff(operations_text)
    
    def _apply_structured_operations(self, operations_text: str) -> str:
        """
        Apply structured file operations.
        
        Args:
            operations_text: Text containing CREATE_FILE and MODIFY_FILE operations
            
        Returns:
            Status message
        """
        try:
            operations = self._parse_operations(operations_text)
            results = []
            
            for operation in operations:
                if operation['type'] == 'CREATE_FILE':
                    result = self._create_file(operation['filename'], operation['content'])
                    results.append(f"Created {operation['filename']}: {result}")
                elif operation['type'] == 'MODIFY_FILE':
                    result = self._modify_file(operation['filename'], operation['content'])
                    results.append(f"Modified {operation['filename']}: {result}")
                else:
                    results.append(f"Unknown operation: {operation['type']}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error applying operations: {str(e)}"
    
    def _parse_operations(self, operations_text: str) -> List[Dict]:
        """Parse the structured operations text."""
        operations = []
        current_operation = None
        current_content = ""
        
        for line in operations_text.split('\n'):
            line = line.strip()
            
            if line.startswith('CREATE_FILE:') or line.startswith('MODIFY_FILE:'):
                # Save previous operation
                if current_operation:
                    operations.append({
                        'type': current_operation['type'],
                        'filename': current_operation['filename'],
                        'content': current_content.strip()
                    })
                
                # Start new operation
                parts = line.split(':', 1)
                current_operation = {
                    'type': parts[0].strip(),
                    'filename': parts[1].strip()
                }
                current_content = ""
            else:
                current_content += line + "\n"
        
        # Save final operation
        if current_operation:
            operations.append({
                'type': current_operation['type'],
                'filename': current_operation['filename'],
                'content': current_content.strip()
            })
        
        return operations
    
    def _create_file(self, filename: str, content: str) -> str:
        """Create a new file with the given content."""
        try:
            file_path = self.base_dir / filename
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            log.info(f"Created file: {file_path}")
            return "SUCCESS"
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _modify_file(self, filename: str, diff_content: str) -> str:
        """Modify an existing file using a unified diff."""
        try:
            # For now, treat modify as a diff operation
            return self._apply_unified_diff(diff_content)
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _apply_unified_diff(self, diff_text: str) -> str:
        """
        Apply a unified diff using the system's patch command.
        
        Args:
            diff_text: A unified diff string
            
        Returns:
            A status message indicating success or failure
        """
        if not diff_text.strip():
            return "No changes to apply (empty diff)"
        
        # Create a temporary file for the diff
        with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as temp_diff:
            temp_diff.write(diff_text)
            diff_path = temp_diff.name
        
        try:
            # Extract affected file paths from the diff
            affected_files = self._extract_file_paths(diff_text)
            
            # Create backup of affected files
            backup_paths = self._backup_files(affected_files)
            
            # Apply the patch
            # Using -p1 to strip the first path component (a/ and b/ prefixes)
            # Using --forward to apply only if the patch can be applied in forward direction
            result = subprocess.run(
                ["patch", "-p1", "--forward", "-i", diff_path],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            # Clean up the temporary diff file
            os.unlink(diff_path)
            
            # Process the result
            if result.returncode == 0:
                # Success
                return f"PATCH_APPLIED: {', '.join(affected_files)}\n{result.stdout}"
            else:
                # Failed to apply patch
                self._restore_backups(backup_paths)
                return f"PATCH_FAILED: {result.stderr}\n{result.stdout}"
                
        except Exception as e:
            # Clean up and restore on error
            if os.path.exists(diff_path):
                os.unlink(diff_path)
            
            log.error(f"Error applying diff: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def _extract_file_paths(self, diff_text: str) -> List[str]:
        """
        Extract the file paths from a unified diff.
        
        Args:
            diff_text: A unified diff string
            
        Returns:
            A list of file paths affected by the diff
        """
        files = set()
        
        # Look for lines like "+++ b/path/to/file.py"
        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                # Extract the file path, removing "b/" prefix
                file_path = line[6:].strip()
                files.add(file_path)
        
        return list(files)
    
    def _backup_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Create backups of the specified files.
        
        Args:
            file_paths: List of file paths to backup
            
        Returns:
            A dictionary mapping original paths to backup paths
        """
        backup_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in file_paths:
            full_path = self.base_dir / file_path
            
            # Skip if file doesn't exist (might be a new file)
            if not full_path.exists():
                continue
            
            # Create a backup path with timestamp
            backup_name = f"{file_path.replace('/', '_')}_{timestamp}.bak"
            backup_path = self.backup_dir / backup_name
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy the file to the backup location
            shutil.copy2(full_path, backup_path)
            backup_paths[file_path] = str(backup_path)
            log.info(f"Backed up {file_path} to {backup_path}")
        
        return backup_paths
    
    def _restore_backups(self, backup_paths: Dict[str, str]) -> None:
        """
        Restore files from backups.
        
        Args:
            backup_paths: Dictionary mapping original paths to backup paths
        """
        for original_path, backup_path in backup_paths.items():
            full_original_path = self.base_dir / original_path
            
            # Restore the file
            shutil.copy2(backup_path, full_original_path)
            log.info(f"Restored {original_path} from {backup_path}")
    
    def read_file(self, file_path: str) -> str:
        """
        Read the content of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            The content of the file as a string
        """
        full_path = self.base_dir / file_path
        
        if not full_path.exists():
            return f"ERROR: File not found: {file_path}"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            log.error(f"Error reading file {file_path}: {str(e)}")
            return f"ERROR: Failed to read file: {str(e)}"
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file exists, False otherwise
        """
        full_path = self.base_dir / file_path
        return full_path.exists()
    
    def list_files(self, directory: str = "") -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Relative path to the directory to list
            
        Returns:
            A list of file paths in the directory
        """
        dir_path = self.base_dir / directory
        
        if not dir_path.exists() or not dir_path.is_dir():
            log.error(f"Directory not found or not a directory: {directory}")
            return []
        
        try:
            # Get all files recursively
            files = []
            for root, _, filenames in os.walk(dir_path):
                rel_root = os.path.relpath(root, self.base_dir)
                for filename in filenames:
                    # Skip backup files and hidden files
                    if filename.startswith('.') or filename.endswith('.bak'):
                        continue
                    
                    rel_path = os.path.join(rel_root, filename)
                    # Normalize path separators
                    rel_path = rel_path.replace('\\', '/')
                    files.append(rel_path)
            
            return sorted(files)
        except Exception as e:
            log.error(f"Error listing files in {directory}: {str(e)}")
            return []
