"""
LLM Backends package.

This package provides a pluggable interface for different LLM providers.
Each backend implements the same interface but connects to a different API.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from agent.messages import Message

log = logging.getLogger(__name__)

class LLMBackendError(Exception):
    """Raised when an LLM backend encounters an error."""
    pass

class LLMBackend(ABC):
    """
    Abstract base class for all LLM backends.
    
    Each concrete implementation must:
    1. Accept a config dict in __init__
    2. Implement complete() to return a Message
    """
    
    def __init__(self, cfg: Dict):
        """Initialize with provider-specific config."""
        self.cfg = cfg
    
    @abstractmethod
    def complete(self, messages: List[Message]) -> Message:
        """
        Send messages to the LLM and return the response.
        
        Parameters
        ----------
        messages : List[Message]
            The conversation history as Message objects.
            
        Returns
        -------
        Message
            The LLM's response as a Message object.
            
        Raises
        ------
        LLMBackendError
            If the LLM API call fails.
        """
        pass

class StubBackend(LLMBackend):
    """Mock backend that returns canned responses (for testing)."""
    
    def complete(self, messages: List[Message]) -> Message:
        """Return a fixed response."""
        return Message(role="assistant", content="[stub] This is a mock response from the stub backend.")

def get_backend(cfg: Dict) -> LLMBackend:
    """
    Factory function to create an LLM backend based on config.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary with at least a 'type' key.
        
    Returns
    -------
    LLMBackend
        An instance of the appropriate LLM backend.
        
    Raises
    ------
    LLMBackendError
        If the backend type is unknown or cannot be instantiated.
    """
    backend_type = cfg.get("type", "stub")
    
    # Special case for stub backend
    if backend_type == "stub":
        return StubBackend(cfg)
    
    # Map backend types to module names
    backend_modules = {
        "openai": "openai_backend",
        "anthropic": "anthropic_backend",
        "gemini": "gemini_backend",
        "openrouter": "openrouter_backend",
        "perplexity": "perplexity_backend",
        "fireworks": "fireworks_backend",
        "shell": "shell_backend",
    }
    
    if backend_type not in backend_modules:
        raise LLMBackendError(f"Unknown backend type: {backend_type}")
    
    module_name = backend_modules[backend_type]
    
    try:
        # Import the module dynamically
        module = importlib.import_module(f"agent.llm_backends.{module_name}")
        
        # Get the class (assuming it follows the naming convention)
        class_name = "".join(word.capitalize() for word in backend_type.split("_")) + "Backend"
        backend_class = getattr(module, class_name)
        
        # Create and return an instance
        return backend_class(cfg)
    except ImportError as exc:
        raise LLMBackendError(f"Failed to import backend module '{module_name}': {exc}") from exc
    except AttributeError as exc:
        raise LLMBackendError(f"Backend class '{class_name}' not found in module '{module_name}': {exc}") from exc
    except Exception as exc:
        raise LLMBackendError(f"Failed to instantiate backend '{backend_type}': {exc}") from exc

__all__ = ["LLMBackend", "LLMBackendError", "get_backend"]
