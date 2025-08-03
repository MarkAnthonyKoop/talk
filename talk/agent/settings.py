"""
Centralised, validated configuration for the agentic framework.

• Works on Pydantic v1 (falls back to `pydantic.BaseSettings`)
• Works on Pydantic v2 (prefers `pydantic_settings.BaseSettings`)
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Robust BaseSettings import
# --------------------------------------------------------------------------- #
try:                                    # Pydantic v2 + pydantic‑settings
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:             # Pydantic v1 fallback
    from pydantic import BaseSettings   # type: ignore

from pydantic import Field, PositiveInt, ValidationError, model_validator

# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #
_ISATTY = sys.stdout.isatty()
_style = lambda txt, code: f"\033[{code}m{txt}\033[0m" if _ISATTY else txt
RED = lambda s: _style(s, "31")
YELLOW = lambda s: _style(s, "33")
GREEN = lambda s: _style(s, "32")
BLUE = lambda s: _style(s, "34")
BOLD = lambda s: _style(s, "1")

# --------------------------------------------------------------------------- #
# Settings models
# --------------------------------------------------------------------------- #
class ProviderSettings(BaseSettings):
    type: str = Field(default="openai")
    model_name: str = Field(default="gpt-4o-mini")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    max_tokens: PositiveInt = Field(default=1500)
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
    conversations_dir: Path = Field(
        default_factory=lambda: Path("~/sandbox/conversations").expanduser()
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("~/sandbox/logs").expanduser()
    )

    @model_validator(mode="after")
    def _ensure_dirs(self):
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self

    class Config:
        env_prefix = "PATH_"
        extra = "ignore"


class Settings(BaseSettings):
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    provider: ProviderSettings = ProviderSettings()
    shell_agent: ShellAgentSettings = ShellAgentSettings()
    paths: PathSettings = PathSettings()

    default_system_prompt: str = Field(
        default="You are a helpful AI assistant.", env="DEFAULT_SYSTEM_PROMPT"
    )
    mockup_mode: bool = Field(default=False, env="MOCKUP_MODE")

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def coloured(self, text: str, colour: str = "green") -> str:
        return {
            "red": RED,
            "yellow": YELLOW,
            "green": GREEN,
            "blue": BLUE,
            "bold": BOLD,
        }.get(colour, lambda x: x)(text)

    @classmethod
    @lru_cache(maxsize=1)
    def current(cls) -> "Settings":
        try:
            return cls()
        except ValidationError as err:
            print(RED(f"[settings] invalid configuration: {err}"), file=sys.stderr)
            raise SystemExit(1)

    def resolve(self, overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        return self if not overrides else self.model_copy(update=overrides)


# Singleton instance
settings: Settings = Settings.current()

# --------------------------------------------------------------------------- #
# Legacy helper attached to singleton
# --------------------------------------------------------------------------- #
def get_config_value(
    dotted_key: str,
    default: Any = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    node: Any = settings.resolve(overrides)
    for part in dotted_key.split("."):
        if isinstance(node, BaseSettings):
            node = getattr(node, part, default)
        elif isinstance(node, dict):
            node = node.get(part, default)
        else:
            return default
    return node


# expose for old code (OpenAI/Anthropic back‑ends)
import types  # noqa: E402

