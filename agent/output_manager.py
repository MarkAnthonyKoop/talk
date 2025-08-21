#!/usr/bin/env python3
"""
Output Manager - Centralized output directory management for Talk framework.

This module provides utilities for creating timestamped output directories
and ensuring all logs, test outputs, and runtime artifacts are stored
in a consistent, organized structure.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.settings import Settings


class OutputManager:
    """Manages output directory structure with timestamps and consistent naming."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize with settings instance."""
        self.settings = settings or Settings()
        # Try to get output_root, fallback to .talk directory
        try:
            self.output_root = self.settings.paths.output_root
        except AttributeError:
            self.output_root = Path.cwd() / ".talk"
        
    def create_session_dir(self, session_type: str = "session", custom_name: Optional[str] = None) -> Path:
        """
        Create a timestamped session directory.
        
        Args:
            session_type: Type of session (e.g., 'talk', 'test', 'demo')
            custom_name: Optional custom name to append
            
        Returns:
            Path to the created session directory
            
        Example output: output/2025-08-02_15-30-45_talk_fibonacci/
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if custom_name:
            dir_name = f"{timestamp}_{session_type}_{custom_name}"
        else:
            dir_name = f"{timestamp}_{session_type}"
            
        session_dir = self.output_root / dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        (session_dir / "logs").mkdir(exist_ok=True)
        (session_dir / "workspace").mkdir(exist_ok=True)
        (session_dir / "artifacts").mkdir(exist_ok=True)
        
        # Create session info file
        self._create_session_info(session_dir, session_type, custom_name)
        
        return session_dir
    
    def create_test_dir(self, test_name: str, test_id: Optional[str] = None) -> Path:
        """
        Create a test output directory with consistent naming.
        
        Args:
            test_name: Name of the test
            test_id: Optional test ID (will generate one if not provided)
            
        Returns:
            Path to the created test directory
        """
        if test_id is None:
            import uuid
            test_id = uuid.uuid4().hex[:8]
            
        return self.create_session_dir("test", f"{test_name}_{test_id}")
    
    def get_logs_dir(self, session_dir: Optional[Path] = None) -> Path:
        """
        Get the logs directory for a session or the global logs directory.
        
        Args:
            session_dir: Optional session directory. If provided, returns session/logs/
                        If None, returns the global logs directory
        """
        if session_dir:
            return session_dir / "logs"
        else:
            return self.settings.paths.logs_dir
    
    def cleanup_old_sessions(self, days_to_keep: int = 30) -> None:
        """
        Clean up old session directories based on age.
        
        Args:
            days_to_keep: Number of days of sessions to keep
        """
        if not self.output_root.exists():
            return
            
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for session_dir in self.output_root.iterdir():
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff_time:
                import shutil
                shutil.rmtree(session_dir)
                print(f"Cleaned up old session: {session_dir.name}")
    
    def _create_session_info(self, session_dir: Path, session_type: str, custom_name: Optional[str]) -> None:
        """Create a session info file with metadata."""
        import json
        
        session_info = {
            "session_type": session_type,
            "custom_name": custom_name,
            "created_at": datetime.now().isoformat(),
            "session_dir": str(session_dir),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "working_directory": str(Path.cwd()),
        }
        
        with open(session_dir / "session_info.json", "w") as f:
            json.dump(session_info, f, indent=2)


# Global convenience functions
def create_session_dir(session_type: str = "session", custom_name: Optional[str] = None) -> Path:
    """Create a timestamped session directory using default settings."""
    return OutputManager().create_session_dir(session_type, custom_name)


def create_test_dir(test_name: str, test_id: Optional[str] = None) -> Path:
    """Create a test output directory using default settings."""
    return OutputManager().create_test_dir(test_name, test_id)


def get_output_root() -> Path:
    """Get the root output directory path."""
    try:
        return Settings().paths.output_root
    except AttributeError:
        return Path.cwd() / ".talk"