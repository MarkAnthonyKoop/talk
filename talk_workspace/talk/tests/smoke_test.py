#!/usr/bin/env python3
"""
backend_smoke_test.py
─────────────────────
Queries every supported LLM backend once and prints a short response.

Usage:
    python backend_smoke_test.py
"""

from textwrap import shorten

from agent.agent import Agent

PROMPT = "In one sentence, say hello and identify which LLM provider you are."

# List of provider configs to test
PROVIDERS = [
    # OpenAI default (gpt‑4o‑mini)
    {"name": "openai", "cfg": {"provider": {"type": "openai"}}},
    # Anthropic Claude 3 Haiku
    {
        "name": "anthropic",
        "cfg": {
            "provider": {
                "type": "anthropic",
                "model_name": "claude-3-haiku-20240307",
            }
        },
    },
    # Perplexity PPLX‑7B
    {
        "name": "perplexity",
        "cfg": {
            "provider": {
                "type": "perplexity",
                "model_name": "pplx-7b-online",
            }
        },
    },
    # OpenRouter proxy (route to gpt‑3.5‑turbo)
    {
        "name": "openrouter",
        "cfg": {
            "provider": {
                "type": "openrouter",
                "model_name": "openrouter/openai/gpt-3.5-turbo",
            }
        },
    },
    # Gemini Pro
    {"name": "gemini", "cfg": {"provider": {"type": "gemini", "model_name": "gemini-pro"}}},
    # Fireworks – DeepSeek‑v3
    {
        "name": "fireworks",
        "cfg": {
            "provider": {
                "type": "fireworks",
                "model_name": "accounts/fireworks/models/deepseek-v3",
            }
        },
    },
]

# --------------------------------------------------------------------------- #
def try_backend(entry):
    name = entry["name"]
    cfg = entry["cfg"]
    print(f"\n{name.upper():=^80}")

    try:
        agent = Agent(cfg_overrides=cfg)
        reply = agent.run(PROMPT)
        print(shorten(reply.replace("\n", " "), width=120, placeholder="…"))
    except Exception as exc:
        print(f"[ERROR] {exc}")


def main():
    for entry in PROVIDERS:
        try_backend(entry)


if __name__ == "__main__":
    main()

