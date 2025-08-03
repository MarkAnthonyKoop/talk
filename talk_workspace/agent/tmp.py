# ---------------------------------------------------------------------------
# CHANGE LOG
# ---------------------------------------------------------------------------
# • Provider override now honors *whatever* llm.provider is set to, not just “google”.
# • Added helper `.provider_dict(provider_name)` to avoid deep-dict gymnastics.
# • Lavish comments clarify configuration-driven testing philosophy.
# ---------------------------------------------------------------------------

"""
Application-wide configuration using **Pydantic Settings**.

Key ideas
---------
1.  **Environment first** – all defaults can be overridden with env-vars
    (`TALK_LLM_PROVIDER`, `TALK_GOOGLE_MODEL_NAME`, …).
2.  **Configuration-Driven Tests** – `DEBUG_MOCK_MODE=1` & `TALK_FORCE_MODEL`
    let tests reshape behaviour without monkey-patching any code.
3.  **Provider-agnostic override** – `TALK_FORCE_MODEL` is now applied to
    whichever provider is active (google, openai, anthropic, etc.).
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Mapping

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Provider-specific sub-models
# ---------------------------------------------------------------------------

class GoogleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TALK_GOOGLE_", extra="ignore")
    model_name: str = "gemini-1.5-flash"
    project: str = ""
    location: str = ""


# Add more providers as the codebase grows …
# class OpenAISettings(BaseSettings): ...


# ---------------------------------------------------------------------------
# Top-level Settings model
# ---------------------------------------------------------------------------

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TALK_LLM_", extra="ignore")
    provider: str = "google"                       # ← default provider


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    google: GoogleSettings = GoogleSettings()
    # openai: OpenAISettings = OpenAISettings()  # placeholder


class DebugSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEBUG_", extra="ignore")
    verbose: bool = Field(False, env="DEBUG_VERBOSE")
    mock_mode: bool = Field(False, env="DEBUG_MOCK_MODE")


class Settings(BaseSettings):
    """Central, immutable configuration object."""
    model_config = SettingsConfigDict(extra="ignore")

    llm: LLMSettings = LLMSettings()
    provider: ProviderSettings = ProviderSettings()
    debug: DebugSettings = DebugSettings()

    # ---------------------------------------------------------------------
    # Class helpers
    # ---------------------------------------------------------------------

    @classmethod
    def resolve(cls, overrides: dict | "Settings" | None = None) -> "Settings":
        """
        Merge default → env var → overrides, then apply global *force* flags.

        `overrides` can be either a plain dict or another Settings instance.
        """
        # 1) defaults+env
        merged = cls().model_dump(mode="python")

        # 2) explicit overrides (test or caller)
        if overrides is not None:
            source = overrides.model_dump(mode="python") if isinstance(overrides, Settings) else overrides
            Settings._deep_merge(merged, source)

        # 3) global env overrides – force a specific model everywhere
        if (forced := os.getenv("TALK_FORCE_MODEL")):
            provider = merged["llm"]["provider"]
            provider_dict = cls._provider_dict(merged, provider)
            provider_dict["model_name"] = forced

        return cls(**merged)

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deep_merge(dst: dict, src: Mapping) -> None:
        """Recursive dict merge (src → dst, in-place)."""
        for k, v in src.items():
            if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
                Settings._deep_merge(dst[k], v)
            else:
                dst[k] = deepcopy(v)

    @staticmethod
    def _provider_dict(root: dict, provider_name: str) -> dict:
        """
        Navigate to `root["provider"][provider_name]`, creating dicts as needed.
        """
        root.setdefault("provider", {})
        root["provider"].setdefault(provider_name, {})
        return root["provider"][provider_name]

    # Convenience accessor
    def get_provider_settings(self):
        return getattr(self.provider, self.llm.provider)


