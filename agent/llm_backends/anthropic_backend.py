from __future__ import annotations

import importlib
import json
import os
from typing import Dict, List, Optional

from agent.messages import Message
from . import LLMBackend, LLMBackendError

class AnthropicBackend(LLMBackend):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key = os.getenv("CLAUDE_API_TOKEN") or cfg.get("api_key")
        if not api_key:
            raise LLMBackendError("CLAUDE_API_TOKEN not set")

        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError as exc:
            raise LLMBackendError("pip install anthropic") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = cfg.get("model_name", "claude-3-5-sonnet-20241022")
        self.max_tokens = cfg.get("max_tokens", 1500)
        self.temperature = cfg.get("temperature", 0.7)
    
    @property
    def supports_tools(self) -> bool:
        """Anthropic supports native tool calling."""
        return True
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI format tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        return anthropic_tools

    def complete(self, messages: List[Message], tools: Optional[List[Dict]] = None) -> Message:
        conv = []
        for m in messages:
            if m.role.value == "tool":
                # Convert tool responses to Anthropic format
                conv.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": getattr(m, "tool_call_id", ""),
                        "content": m.content or ""
                    }]
                })
            elif m.content:
                conv.append({"role": m.role.value, "content": m.content})

        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": conv,
            }
            
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
            
            resp = self.client.messages.create(**kwargs)
            
            # Handle tool calls
            if hasattr(resp, 'stop_reason') and resp.stop_reason == 'tool_use':
                # Extract tool calls from response
                tool_calls = []
                for content in resp.content:
                    if content.type == 'tool_use':
                        tool_calls.append({
                            "id": content.id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": json.dumps(content.input)
                            }
                        })
                
                # Return message with tool calls
                # Convert to JSON string for the Json field
                from agent.messages import Role
                return Message(
                    role=Role.assistant, 
                    content=None,
                    tool_calls=json.dumps(tool_calls)  # Json field expects string
                )
            else:
                # Regular text response
                text = resp.content[0].text.strip()
                return Message(role="assistant", content=text)  # type: ignore[arg-type]

        except Exception as exc:
            raise LLMBackendError(exc) from exc

