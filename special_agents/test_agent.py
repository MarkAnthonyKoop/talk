# special_agents/test_agent.py

import subprocess
import shlex
from typing import Dict, Optional, Any

from agent.agent import Agent

class TestAgent(Agent):
    """Runs pytest (or another command) and returns output + exit code."""

    def __init__(
        self, 
        cmd: str = "pytest -q", 
        name: Optional[str] = None,
        id: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, id=id, overrides=overrides)
        self.cmd = cmd

    def run(self, prompt: str) -> str:
        """
        Run method expected by PlanRunner.
        
        Parameters
        ----------
        prompt : str
            The input prompt (ignored for TestAgent)
            
        Returns
        -------
        str
            The output of the test command with exit code
        """
        return self.act(prompt, None)
        
    def act(self, task: str, blackboard=None) -> str:  # noqa: D401
        """
        Execute the configured test command.
        
        Parameters
        ----------
        task : str
            Input task (ignored by TestAgent)
        blackboard : Optional
            Blackboard instance (unused)
            
        Returns
        -------
        str
            Formatted string with exit code and command output
        """
        # `task` is ignored; we just run the configured test command.
        proc = subprocess.run(
            shlex.split(self.cmd),
            capture_output=True,
            text=True,
        )
        return f"EXIT {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
