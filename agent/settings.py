#
# Changes from Original & Architectural Theory:
# This file manages application configuration using Pydantic settings, which allows for
# loading from environment variables. The original file provided a solid foundation.
#
# The key architectural change implemented here is the addition of a global override
# for the model name via the `TALK_FORCE_MODEL` environment variable. This supports
# the broader testing strategy of "configuration-driven testing," where we can alter
# the application's behavior for tests without changing the code itself.
#
# In the `resolve` classmethod, logic was added to check for this environment
# variable. If it's set, it overrides any other model configuration. This makes it
# trivial to write a test that asserts a specific agent is configured to use a
# specific model, without needing to manually patch the agent's instance variables.
#

"""
Manages application settings using Pydantic's BaseSettings.

This module defines the structure of the application's configuration,
including settings for the LLM provider and debugging options. It allows for
loading settings from environment variables and supports overriding them for
different contexts, such as testing.
"""

from __future__ import annotations
import os
from copy import deepcopy
from pathlib import Path
from typing import Mapping
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = dict
from pydantic import Field, model_validator

class LLMSettings(BaseSettings):
    """Settings related to the LLM provider."""
    model_config = SettingsConfigDict(env_prefix="TALK_LLM_", extra="ignore")
    provider: str = "anthropic"

class GoogleSettings(BaseSettings):
    """Settings specific to the Google LLM provider."""
    model_config = SettingsConfigDict(env_prefix="TALK_GOOGLE_", extra="ignore")
    model_name: str = "gemini-2.0-flash"
    project: str = ""
    location: str = ""

class AnthropicSettings(BaseSettings):
    """Settings specific to the Anthropic LLM provider."""
    model_config = SettingsConfigDict(env_prefix="TALK_ANTHROPIC_", extra="ignore")
    model_name: str = "claude-3-5-sonnet-20241022"
    api_key: str = ""
    max_tokens: int = 4000
    temperature: float = 0.7

class ProviderSettings(BaseSettings):
    """Container for all provider-specific settings."""
    model_config = SettingsConfigDict(extra="ignore")
    google: GoogleSettings = GoogleSettings()
    anthropic: AnthropicSettings = AnthropicSettings()

class PathSettings(BaseSettings):
    """Settings for file system paths."""
    model_config = SettingsConfigDict(env_prefix="PATH_", extra="ignore")
    base_dir: Path = Path(os.getenv("SANDBOX_DIR", "~/sandbox")).expanduser()
    
    # Centralized output directory structure
    output_root: Path = Field(
        default_factory=lambda: Path.cwd() / ".talk"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".talk" / "logs"
    )

    @model_validator(mode="after")
    def _ensure_dirs(self):
        # create the directories that are needed
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self

class LoggingSettings(BaseSettings):
    """Settings for logging behavior.""" 
    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")
    flush_every: int = Field(default=10, env="LOG_FLUSH_EVERY")

class DebugSettings(BaseSettings):
    """Settings for debugging purposes."""
    model_config = SettingsConfigDict(env_prefix="DEBUG_", extra="ignore")
    verbose: bool = Field(default=False, env="DEBUG_VERBOSE")
    mock_mode: bool = Field(default=False, env="DEBUG_MOCK_MODE")

class Settings(BaseSettings):
    """The main settings object for the entire application."""
    model_config = SettingsConfigDict(extra="ignore")
    llm: LLMSettings = LLMSettings()
    provider: ProviderSettings = ProviderSettings()
    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    debug: DebugSettings = DebugSettings()

    @classmethod
    def resolve(cls, overrides: dict | Settings | None = None) -> Settings:
        """
        Creates a Settings object, merging defaults with overrides.
        
        This method is central to the configuration system. It starts with base
        defaults, loads any values from environment variables, and then merges
        in any explicit `overrides` provided. Finally, it applies any
        global environment variable overrides like `TALK_FORCE_MODEL`.

        Args:
            overrides: A dictionary or Settings object with values to override.

        Returns:
            A fully resolved Settings instance.
        """
        # Start with the base configuration derived from defaults and env vars.
        base = cls().model_dump(mode="python")

        # Merge in any explicit override dictionary or Settings object.
        if isinstance(overrides, Settings):
            overrides = overrides.model_dump(mode="python")
        if isinstance(overrides, Mapping):
            Settings._deep_merge(base, overrides)
        
        # --- ARCHITECTURAL CHANGE: Global Model Override via Environment Variable ---
        # This allows tests or users to force a specific model globally, which is
        # simpler and more reliable than patching agent instances.
        if fm_env := os.getenv("TALK_FORCE_MODEL"):
            # Auto-detect provider based on model name
            if "claude" in fm_env.lower():
                # Claude models use Anthropic provider
                base["llm"]["provider"] = "anthropic"
                if "anthropic" not in base["provider"]:
                    base["provider"]["anthropic"] = {}
                base["provider"]["anthropic"]["model_name"] = fm_env
            elif "gemini" in fm_env.lower() or "flash" in fm_env.lower():
                # Gemini models use Google provider
                base["llm"]["provider"] = "google"
                if "google" not in base["provider"]:
                    base["provider"]["google"] = {}
                base["provider"]["google"]["model_name"] = fm_env
            elif "gpt" in fm_env.lower():
                # GPT models use OpenAI provider
                base["llm"]["provider"] = "openai"
                if "openai" not in base["provider"]:
                    base["provider"]["openai"] = {}
                base["provider"]["openai"]["model_name"] = fm_env
            else:
                # Default to google for unknown models
                if "google" not in base["provider"]:
                    base["provider"]["google"] = {}
                base["provider"]["google"]["model_name"] = fm_env
            
        return cls(**base)

    @staticmethod
    def _deep_merge(dst: dict, src: Mapping) -> dict:
        """Recursively merges a source dictionary into a destination dictionary."""
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, Mapping):
                Settings._deep_merge(dst[k], v)
            else:
                dst[k] = deepcopy(v)
        return dst

    def get_provider_settings(self):
        """Gets the settings for the currently configured LLM provider."""
        return getattr(self.provider, self.llm.provider)
