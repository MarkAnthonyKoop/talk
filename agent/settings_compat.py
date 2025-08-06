#!/usr/bin/env python3
"""
Compatibility layer for Pydantic v1/v2 differences.
This provides a BaseSettings that works with both versions.
"""

import os
from typing import Any, Dict

# Try to detect which Pydantic version we have
try:
    # Pydantic v2 with pydantic-settings
    from pydantic_settings import BaseSettings as PydanticBaseSettings
    from pydantic import Field
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1
        from pydantic import BaseSettings as PydanticBaseSettings, Field
        PYDANTIC_V2 = False
    except ImportError:
        # Create a minimal BaseSettings that doesn't require pydantic at all
        class PydanticBaseSettings:
            """Minimal BaseSettings replacement when pydantic is not available."""
            def __init__(self, **kwargs):
                # Set attributes from kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
                
                # Load from environment variables
                cls_name = self.__class__.__name__
                for key in dir(self.__class__):
                    if not key.startswith('_'):
                        attr = getattr(self.__class__, key)
                        if not callable(attr):
                            # Check for env var
                            env_key = f"TALK_{key.upper()}"
                            if env_key in os.environ:
                                setattr(self, key, os.environ[env_key])
                            elif not hasattr(self, key):
                                # Set default
                                setattr(self, key, attr)
        
        Field = lambda **kwargs: kwargs.get('default', None)
        PYDANTIC_V2 = False


class BaseSettings(PydanticBaseSettings):
    """
    Compatibility wrapper for BaseSettings that works with both Pydantic v1 and v2.
    """
    
    def __init__(self, **kwargs):
        if PYDANTIC_V2:
            super().__init__(**kwargs)
        else:
            # For Pydantic v1 or no pydantic
            try:
                super().__init__(**kwargs)
            except:
                # Fallback for when pydantic is completely broken
                for key, value in kwargs.items():
                    setattr(self, key, value)
    
    @classmethod
    def model_config(cls, **kwargs):
        """Compatibility method for model_config."""
        if PYDANTIC_V2:
            from pydantic_settings import SettingsConfigDict
            return SettingsConfigDict(**kwargs)
        else:
            # For v1, we handle this differently
            return {}


def SettingsConfigDict(**kwargs):
    """Compatibility function for SettingsConfigDict."""
    if PYDANTIC_V2:
        from pydantic_settings import SettingsConfigDict as RealSettingsConfigDict
        return RealSettingsConfigDict(**kwargs)
    else:
        # For v1 or no pydantic, return empty dict
        return {}


def model_validator(**kwargs):
    """Compatibility decorator for model_validator."""
    def decorator(func):
        if PYDANTIC_V2:
            from pydantic import model_validator as real_model_validator
            return real_model_validator(**kwargs)(func)
        else:
            # For v1 or no pydantic, just return the function unchanged
            return func
    return decorator