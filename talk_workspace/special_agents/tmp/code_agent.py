# special_agents/code_agent.py

from typing import Dict, Optional, Any

from agent.agent import Agent 

class CodeAgent(Agent):
    """
    Generates a unified diff for a given task.
    The prompt instructs the LLM to emit only a valid UNIX unified diff.
    """

    DIFF_INSTRUCTIONS = (
        "Return a valid unified diff **only**. Do not wrap it in markdown. "
        "Start each file section with --- / +++ headers."
    )

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
            The input task description for generating a diff
            
        Returns
        -------
        str
            A unified diff for the requested task
        """
        return self.act(prompt, None)
        
    def act(self, task: str, blackboard=None) -> str:  # noqa: D401
        """
        Generate a unified diff for the given task.
        
        Parameters
        ----------
        task : str
            Description of the code changes to make
        blackboard : Optional
            Blackboard instance (unused)
            
        Returns
        -------
        str
            Unified diff format text
        """
        prompt = (
            "You are an expert Python developer.\n"
            f"{self.DIFF_INSTRUCTIONS}\n"
            f"Task:\n{task}"
        )
        return super().run(prompt)   # CoreAgent.run -> LLM -> string

