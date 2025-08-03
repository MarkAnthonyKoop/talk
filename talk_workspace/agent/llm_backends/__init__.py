# agent/llm_backends/__init__.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from agent.messages import Message

class LLMBackendError(Exception):
    """Exception raised by LLM backends."""
    pass

class LLMBackend(ABC):
    """Abstract base class for all LLM backends."""
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
    
    @abstractmethod
    def complete(self, messages: List[Message]) -> Message:
        """Complete a conversation with the LLM."""
        pass

class StubBackend(LLMBackend):
    """Stub backend for testing that returns mock responses."""
    
    def complete(self, messages: List[Message]) -> Message:
        """Return a mock response for testing."""
        from agent.messages import Role
        return Message(role=Role.ASSISTANT, content="[mock] This would be the assistant reply.")

def get_backend(provider_dict: Dict) -> LLMBackend:
    import logging
    log = logging.getLogger("llm")
    log.info(f"Selected LLM backend config: {provider_dict}")
    """Factory function to get the appropriate backend from a provider dict."""
    
    # Handle stub backend for testing
    if provider_dict.get("type") == "stub":
        return StubBackend(provider_dict)
    
    # Extract provider name from the dict structure
    # The provider_dict contains the full provider settings 
    provider_name = None
    provider_cfg = None
    
    # Find which provider is configured
    for key, value in provider_dict.items():
        if isinstance(value, dict) and value:  # Non-empty dict indicates active provider
            provider_name = key
            provider_cfg = value
            break
    
    if not provider_name:
        # Default to google if no provider is explicitly configured
        provider_name = "google"
        provider_cfg = provider_dict.get("google", {})
    
    provider_map = {
        "openai": "openai_backend.OpenaiBackend",
        "anthropic": "anthropic_backend.AnthropicBackend", 
        "google": "gemini_backend.GeminiBackend",
        "gemini": "gemini_backend.GeminiBackend",
        "fireworks": "fireworks_backend.FireworksBackend",
        "openrouter": "openrouter_backend.OpenRouterBackend",
        "perplexity": "perplexity_backend.PerplexityBackend",
        "openai_compatible": "openai_compatible.OpenAICompatibleBackend",
        "shell": "shell_backend.ShellBackend",
    }
    
    if provider_name not in provider_map:
        raise LLMBackendError(f"Unknown provider: {provider_name}")
    
    module_path, class_name = provider_map[provider_name].rsplit(".", 1)
    
    try:
        import importlib
        module = importlib.import_module(f".{module_path}", package="agent.llm_backends")
        backend_class = getattr(module, class_name)
        return backend_class(provider_cfg)
    except Exception as exc:
        raise LLMBackendError(f"Failed to load backend {provider_name}: {exc}") from exc

__all__ = ["LLMBackend", "LLMBackendError", "get_backend"]
