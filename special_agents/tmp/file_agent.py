# special_agents/file_agent.py

import subprocess
import tempfile
import os
import uuid
from typing import Dict, Optional, Any

from agent.agent import Agent

class FileAgent(Agent):
    """
    Applies a unified diff to the working directory.
    Uses `patch` if available; otherwise applies lines manually.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None,
        id: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, id=id, overrides=overrides)

    def run(self, prompt: str) -> str:
        """
        Run method expected by PlanRunner.
        
        Parameters
        ----------
        prompt : str
            The unified diff text to apply
            
        Returns
        -------
        str
            Status message indicating whether the patch was applied successfully
        """
        return self.act(prompt, None)
        
    def act(self, diff_text: str, blackboard=None) -> str:
        """
        Apply a unified diff to the working directory.
        
        Parameters
        ----------
        diff_text : str
            Unified diff text to apply
        blackboard : Optional
            Blackboard instance (unused)
            
        Returns
        -------
        str
            Status message indicating whether the patch was applied successfully
        """
        # 1. Write diff to a temp file
        tmp_path = tempfile.mkstemp(prefix="talk_diff_", suffix=".patch")[1]
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(diff_text)

        # 2. Run `patch -p0 < diff`
        try:
            subprocess.run(
                ["patch", "-p0", "-i", tmp_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            status = "PATCH_APPLIED"
        except subprocess.CalledProcessError as exc:
            status = f"PATCH_FAILED: {exc.stderr}"

        # 3. Cleanup
        os.remove(tmp_path)
        return status

