"""
Specialised Agent that uses the ShellBackend.
"""

from agent.agent import Agent
from agent.llm_backends import get_backend


class ShellAgent(Agent):
    def __init__(self, **kwargs):
        cfg = {"provider": {"type": "shell", "model_name": "gpt-4o-mini"}}
        cfg.update(kwargs.pop("cfg_overrides", {}))
        super().__init__(cfg_overrides=cfg, **kwargs)

