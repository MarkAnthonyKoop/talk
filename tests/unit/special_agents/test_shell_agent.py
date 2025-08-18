"""
Unit tests for ShellAgent using configuration-driven mock mode.
"""

from special_agents.shell_agent import ShellAgent


def test_shell_agent_mock_mode(monkeypatch):
    # Enable full mock mode for all agents via env var.
    monkeypatch.setenv("DEBUG_MOCK_MODE", "1")

    sa = ShellAgent(max_cycles=1)

    result = sa.run("Please create a python script that prints 'hello'")

    # Validate successful test result (TestAgent mock returns EXIT 0 automatically)
    assert "EXIT 0" in result, "ShellAgent did not propagate test success"


def test_force_model_propagates(monkeypatch):
    # Only set force model -- no mock mode!
    monkeypatch.setenv("TALK_FORCE_MODEL", "gemini")

    sa = ShellAgent(max_cycles=1)

    # This test purely checks config plumbing -- model name should be overridden
    assert sa.agents["code"].cfg.get_provider_settings().model_name == "gemini"

