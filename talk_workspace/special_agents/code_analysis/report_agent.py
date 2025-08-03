#!/usr/bin/env python3
# special_agents/report_agent.py

"""
ReportAgent - Specialized agent for summarising test results.

This agent takes raw test output (for example, from the TestAgent) and
produces a concise human‑readable summary.  The summary contains
high‑level information such as how many tests were run, how many passed,
failed or were skipped, whether the test run timed out and the exit code.

Unlike CodeAgent or FileAgent this class does not interact with any LLM
back‑end.  It operates entirely on the provided input and therefore
requires no external dependencies.  Because it inherits from the base
``Agent`` class it participates in the same conversation logging
mechanism used throughout the system, making it easy to trace when and
how summaries were generated.

Typical usage:

    from special_agents.report_agent import ReportAgent
    raw_results = "... output from TestAgent ..."
    agent = ReportAgent()
    summary = agent.run(raw_results)
    print(summary)

The agent will attempt to parse a JSON payload after a ``JSON_RESULT:``
marker if present.  If no structured data is found the agent falls back
to simple heuristics to determine success or failure.

"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from agent.agent import Agent

log = logging.getLogger(__name__)


class ReportAgent(Agent):
    """Summarise test results into a concise report."""

    def __init__(self, **kwargs: Any) -> None:
        # This agent does not require any system prompts or LLM, so pass
        # through any overrides but leave roles empty.
        super().__init__(roles=[], **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, input_text: str) -> str:
        """
        Parse test output and return a human‑readable summary.

        Parameters
        ----------
        input_text : str
            The raw output string from a test run.  This can be the
            complete output of ``TestAgent`` or any string containing a
            ``JSON_RESULT:`` section with structured data.

        Returns
        -------
        str
            A concise summary describing test success, counts and exit
            codes.  In the event of an error parsing the input a
            descriptive error message is returned instead.
        """
        # Record the incoming request
        self._append("user", input_text)

        try:
            summary = self._parse_summary(input_text)
        except Exception as exc:  # pylint: disable=broad-except
            # Unexpected parse errors should not bubble up.  We log the
            # exception and return an error message instead.
            log.exception("ReportAgent failed to summarise test results: %s", exc)
            summary = f"ERROR: Failed to summarise test results: {exc}"

        # Record the assistant response
        self._append("assistant", summary)
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_summary(self, text: str) -> str:
        """Internal helper to produce a summary from raw test output."""
        # Look for JSON payload after a JSON_RESULT marker.  Many of the
        # ``TestAgent`` outputs end with something like ``JSON_RESULT: { ... }``.
        marker = "JSON_RESULT:"
        pos = text.find(marker)
        if pos != -1:
            json_str = text[pos + len(marker):].strip()
            try:
                data: Dict[str, Any] = json.loads(json_str)
                return self._format_from_json(data)
            except json.JSONDecodeError:
                # If JSON parsing fails fall back to heuristics
                pass

        # Fallback heuristic: determine success or failure based on keywords
        lowered = text.lower()
        if "success" in lowered and "failed" not in lowered:
            return "TEST SUMMARY: All tests passed."
        if "failure" in lowered or "error" in lowered:
            return "TEST SUMMARY: Some tests failed."
        if "timed out" in lowered:
            return "TEST SUMMARY: Test run timed out."
        # As a last resort, report that the outcome is unknown
        return "TEST SUMMARY: Unable to determine test result from input."

    def _format_from_json(self, data: Dict[str, Any]) -> str:
        """Build a summary string from parsed JSON results."""
        # Extract common fields with sensible defaults
        success = bool(data.get("success"))
        exit_code = data.get("exit_code", "n/a")
        timed_out = bool(data.get("timed_out", False))
        tests_run = data.get("tests_run")
        tests_passed = data.get("tests_passed")
        tests_failed = data.get("tests_failed")
        tests_skipped = data.get("tests_skipped")

        parts = []
        parts.append("TEST SUMMARY:")
        parts.append(f"Success: {success}")
        parts.append(f"Exit code: {exit_code}")
        if timed_out:
            parts.append("Status: TIMED OUT")
        if tests_run is not None:
            parts.append(f"Tests run: {tests_run}")
        if tests_passed is not None:
            parts.append(f"Passed: {tests_passed}")
        if tests_failed is not None:
            parts.append(f"Failed: {tests_failed}")
        if tests_skipped is not None:
            parts.append(f"Skipped: {tests_skipped}")

        return "\n".join(parts)


__all__ = ["ReportAgent"]