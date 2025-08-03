"""
agent/settings.py  (full file)
Validated configuration for the agent framework.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# —— Pydantic base import (v1 or v2) ————————————————————————————————
try:                                    # Pydantic-v2
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:             # Pydantic-v1 fallback
    from pydantic import BaseSettings   # type: ignore

from pydantic import Field, PositiveInt, ValidationError, model_validator

# —— Colour helpers (optional) ————————————————————————————————————————
_ISATTY = sys.stdout.isatty()
_colour = lambda t, c: f"\033[{c}m{t}\033[0m" if _ISATTY else t
RED, YEL, GRN, BLU, BLD = (
    lambda s: _colour(s, "31"),
    lambda s: _colour(s, "33"),
    lambda s: _colour(s, "32"),
    lambda s: _colour(s, "34"),
    lambda s: _colour(s, "1"),
)

# ————————————————————————————————————————————————————————————————————
# Sub-models
# ————————————————————————————————————————————————————————————————————

class ProviderSettings(BaseSettings):
    type: str = Field(default="openai")
    model_name: str = Field(default="gpt-4o-mini")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    max_tokens: PositiveInt = 1500
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    class Config:
        env_prefix = "PROVIDER_"
        extra = "ignore"

class ShellAgentSettings(BaseSettings):
    approve_shell_commands: bool = True
    command_timeout: PositiveInt = 60
    class Config:
        env_prefix = "SHELL_"
        extra = "ignore"

class PathSettings(BaseSettings):
    base_dir: Path = Path(os.getenv("SANDBOX_DIR", "~/sandbox")).expanduser()
    logs_dir: Path = Field(
        default_factory=lambda: Path("~/sandbox/logs").expanduser()
    )

    @model_validator(mode="after")
    def _ensure_dirs(self):
        # create the two directories that still exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self

    class Config:
        env_prefix = "PATH_"
        extra = "ignore"

class LoggingSettings(BaseSettings):
    root_dir: Path = Path("~/sandbox/logs").expanduser()
    flush_every: PositiveInt = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _ensure_root(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self

    class Config:
        env_prefix = "LOG_"
        extra = "ignore"

class DebugSettings(BaseSettings):               # ← NEW
    verbose: bool = Field(default=False, env="DEBUG_VERBOSE")
    mock_mode: bool = Field(default=False, env="MOCK_MODE")
    class Config:
        env_prefix = "DEBUG_"
        extra = "ignore"

# ————————————————————————————————————————————————————————————————————
# Root Settings model
# ————————————————————————————————————————————————————————————————————

class Settings(BaseSettings):
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    provider: ProviderSettings = ProviderSettings()
    shell_agent: ShellAgentSettings = ShellAgentSettings()
    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    debug: DebugSettings = DebugSettings()        # ← restored object style
    default_system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        env="DEFAULT_SYSTEM_PROMPT",
    )

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"

    # —— Nicety helper ————————————————————————————————————————————
    def coloured(self, text: str, colour: str = "green") -> str:
        return {"red": RED, "yellow": YEL, "green": GRN, "blue": BLU, "bold": BLD}.get(
            colour, lambda x: x
        )(text)

    # —— Legacy singleton for old code (memoised) ————————————————————
    @classmethod
    @lru_cache(maxsize=1)
    def current(cls) -> "Settings":
        try:
            return cls()
        except ValidationError as err:
            print(RED(f"[settings] invalid config: {err}"), file=sys.stderr)
            raise SystemExit(1)

    # —— Resolver: None | dict | Settings → fresh Settings ——————————
    @classmethod
    def resolve(cls, overrides: dict | "Settings" | None = None) -> "Settings":
        if overrides is None:
            return cls()
        if isinstance(overrides, Settings):
            return overrides
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be dict, Settings, or None")
        merged = cls().model_dump(mode="python")
        Settings._deep_merge(merged, overrides)
        return cls(**merged)

    # —— Deep-merge helper (static) ————————————————————————————————
    @staticmethod
    def _deep_merge(dst: dict, src: Mapping) -> dict:
        for k, v in src.items():
            if (
                k in dst
                and isinstance(dst[k], Mapping)
                and isinstance(v, Mapping)
            ):
                Settings._deep_merge(dst[k], v)
            else:
                dst[k] = deepcopy(v)
        return dst

