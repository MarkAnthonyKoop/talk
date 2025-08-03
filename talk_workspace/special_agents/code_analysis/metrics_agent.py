#!/usr/bin/env python3
# special_agents/metrics_agent.py

"""
MetricsAgent - analyse Python source files and report simple metrics.

This agent performs lightweight static analysis on Python code to
produce metrics such as the number of functions, classes and total
lines of code.  It operates entirely locally and does not rely on
external tools or LLMs.  The agent accepts either a path to a single
Python file, a directory containing multiple Python files or a raw
string of Python code.  When given a directory, the agent will
recursively traverse it and aggregate metrics across all ``.py`` files
found.

Example usage::

    from special_agents.metrics_agent import MetricsAgent
    agent = MetricsAgent()
    report = agent.run("/path/to/project")
    print(report)

The resulting report is a human‑readable summary of the metrics.  It
also includes a JSON representation at the end, similar to
``TestAgent``, to allow downstream agents to parse the data programmatically.

This agent is intended to be simple and independent, aligning with
the guiding principle of composing complex behaviours from small
specialised components.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

from agent.agent import Agent

log = logging.getLogger(__name__)


class MetricsAgent(Agent):
    """Analyse Python code and return basic metrics."""

    def __init__(self, **kwargs: Any) -> None:
        # No prompts are required for static analysis
        super().__init__(roles=[], **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, input_text: str) -> str:
        """
        Analyse the provided Python source(s) and return metrics.

        Parameters
        ----------
        input_text : str
            Either a path to a Python file, a directory containing
            Python files or a raw string containing Python source code.

        Returns
        -------
        str
            A human‑readable report summarising code metrics along
            with a machine‑readable ``JSON_RESULT`` section.
        """
        # Record the request for provenance
        self._append("user", input_text)

        try:
            metrics = self._collect_metrics(input_text)
            report = self._format_report(metrics)
        except Exception as exc:  # pragma: no cover - defensive catch
            log.exception("MetricsAgent failed to analyse code: %s", exc)
            report = f"ERROR: Failed to analyse code: {exc}"

        # Append assistant reply
        self._append("assistant", report)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_metrics(self, input_text: str) -> Dict[str, Any]:
        """Collect metrics from a file, directory or code string."""
        path = Path(input_text).expanduser()
        metrics = {
            "files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "per_file": {},
        }

        if path.exists():
            if path.is_dir():
                # Walk directory and accumulate metrics
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.endswith(".py"):
                            fpath = Path(root) / fname
                            file_metrics = self._analyse_file(fpath)
                            metrics["per_file"][str(fpath)] = file_metrics
                            metrics["files"] += 1
                            metrics["total_lines"] += file_metrics["lines"]
                            metrics["total_functions"] += file_metrics["functions"]
                            metrics["total_classes"] += file_metrics["classes"]
            else:
                # Single file
                file_metrics = self._analyse_file(path)
                metrics["per_file"][str(path)] = file_metrics
                metrics["files"] = 1
                metrics["total_lines"] = file_metrics["lines"]
                metrics["total_functions"] = file_metrics["functions"]
                metrics["total_classes"] = file_metrics["classes"]
        else:
            # Treat input as raw code string
            file_metrics = self._analyse_source(input_text)
            metrics["per_file"]["<string>"] = file_metrics
            metrics["files"] = 1
            metrics["total_lines"] = file_metrics["lines"]
            metrics["total_functions"] = file_metrics["functions"]
            metrics["total_classes"] = file_metrics["classes"]

        return metrics

    def _analyse_file(self, path: Path) -> Dict[str, int]:
        """Read and analyse a single Python file."""
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            source = path.read_text(errors="ignore")
        return self._analyse_source(source)

    def _analyse_source(self, source: str) -> Dict[str, int]:
        """Analyse a Python source string and return metrics."""
        lines = source.splitlines()
        metrics = {
            "lines": len(lines),
            "functions": 0,
            "classes": 0,
        }
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
        except SyntaxError:
            # If code is invalid, we still return line count
            pass
        return metrics

    def _format_report(self, metrics: Dict[str, Any]) -> str:
        """Format a human‑readable report and embed JSON result."""
        lines = []
        lines.append("CODE METRICS REPORT")
        lines.append(f"Files analysed: {metrics['files']}")
        lines.append(f"Total lines of code: {metrics['total_lines']}")
        lines.append(f"Total functions: {metrics['total_functions']}")
        lines.append(f"Total classes: {metrics['total_classes']}")
        # Optionally include per‑file breakdown if multiple files
        if metrics["files"] > 1:
            lines.append("")
            lines.append("Per‑file breakdown:")
            for fname, m in metrics["per_file"].items():
                lines.append(f"- {fname}: {m['lines']} lines, {m['functions']} functions, {m['classes']} classes")
        # Build summary string
        report = "\n".join(lines)
        # Append machine readable JSON
        report += f"\n\nJSON_RESULT: {json.dumps(metrics, indent=2)}\n"
        return report


__all__ = ["MetricsAgent"]