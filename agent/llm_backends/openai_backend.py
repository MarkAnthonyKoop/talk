from __future__ import annotations

import importlib
import logging
import os
from typing import Dict, List

from agent.messages import Message
from . import LLMBackend, LLMBackendError

log = logging.getLogger(__name__)

class OpenaiBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key_var = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_var) or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError(f"environment variable '{api_key_var}' not set")
        
        try:
            openai = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMBackendError("pip install openai>=1.0.0") from exc
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = cfg.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)
    
    def complete(self, messages: List[Message]) -> Message:
        """Send messages to OpenAI and return the response."""
        # Convert to OpenAI format
        conv = [
            {"role": m.role.value, "content": m.content or ""}
            for m in messages
            if m.content
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conv,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # Extract the content from the response
            text = response.choices[0].message.content.strip()
            return Message(role="assistant", content=text)  # type: ignore[arg-type]
            
        except Exception as exc:
            raise LLMBackendError(f"OpenAI API error: {exc}") from exc
