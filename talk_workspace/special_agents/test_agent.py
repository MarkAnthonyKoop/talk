#!/usr/bin/env python3
# special_agents/test_agent.py

"""
TestAgent - Specialized agent for running tests and reporting results.

This agent takes test commands or file patterns as input and executes
the appropriate test runner (pytest, unittest, etc.). It captures test
output, exit codes, and failures, and formats the results in a structured way.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from agent.agent import Agent

log = logging.getLogger(__name__)

class TestAgent(Agent):
    """
    Specialized agent for running tests and reporting results.
    
    This agent executes test commands, captures output and results,
    and formats them in a structured way. It supports different test
    runners like pytest and unittest.
    """
    
    def __init__(
        self, 
        base_dir: Optional[str] = None,
        default_runner: str = "pytest",
        default_timeout: int = 60,
        **kwargs
    ):
        """
        Initialize the TestAgent.
        
        Args:
            base_dir: The base directory for test execution (defaults to current directory)
            default_runner: The default test runner to use (pytest, unittest, etc.)
            default_timeout: Default timeout in seconds for test execution
            **kwargs: Additional arguments passed to the parent Agent class
        """
        # Initialize with empty roles since this agent doesn't use LLM prompting
        super().__init__(roles=[], **kwargs)
        
        # Set the base directory for test operations
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        log.info(f"TestAgent initialized with base directory: {self.base_dir}")
        
        # Set default test runner and timeout
        self.default_runner = default_runner
        self.default_timeout = default_timeout
        
        # Map of supported test runners and their commands
        self.runners = {
            "pytest": ["pytest", "-v"],
            "unittest": ["python", "-m", "unittest", "discover", "-v"],
            "nose": ["nosetests", "-v"],
            "django": ["python", "manage.py", "test"],
        }
    
    def run(self, input_text: str) -> str:
        """
        Run tests based on the input text.
        
        Args:
            input_text: Test command or JSON with test configuration
            
        Returns:
            A structured string with test results
        """
        # Record the operation in the conversation log for provenance
        self._append("user", f"Request to run tests:\n{input_text}")
        
        try:
            # Parse the input to get test configuration
            config = self._parse_input(input_text)
            
            # Run the tests and get results
            result = self._run_tests(config)
            
            # Record the result
            self._append("assistant", result)
            return result
            
        except Exception as e:
            error_msg = f"ERROR: Failed to run tests: {str(e)}"
            log.error(error_msg)
            self._append("assistant", error_msg)
            return error_msg
    
    def _parse_input(self, input_text: str) -> Dict[str, Any]:
        """
        Parse the input text to get test configuration.
        
        Args:
            input_text: Test command or JSON with test configuration
            
        Returns:
            A dictionary with test configuration
        """
        # Default configuration
        config = {
            "runner": self.default_runner,
            "timeout": self.default_timeout,
            "pattern": None,
            "args": [],
            "env": {},
        }
        
        # Try to parse as JSON
        try:
            input_data = json.loads(input_text)
            if isinstance(input_data, dict):
                # Update config with provided values
                config.update(input_data)
                return config
        except (json.JSONDecodeError, TypeError):
            pass
        
        # If not JSON, treat as command line
        input_text = input_text.strip()
        
        # Check if input specifies a runner
        for runner in self.runners:
            if input_text.startswith(runner):
                config["runner"] = runner
                input_text = input_text[len(runner):].strip()
                break
        
        # Extract timeout if specified
        timeout_match = re.search(r'--timeout[=\s]+(\d+)', input_text)
        if timeout_match:
            config["timeout"] = int(timeout_match.group(1))
            input_text = re.sub(r'--timeout[=\s]+\d+', '', input_text).strip()
        
        # Remaining text is args
        if input_text:
            config["args"] = input_text.split()
        
        return config
    
    def _run_tests(self, config: Dict[str, Any]) -> str:
        """
        Run tests with the specified configuration.
        
        Args:
            config: Test configuration dictionary
            
        Returns:
            A structured string with test results
        """
        runner = config["runner"]
        timeout = config["timeout"]
        pattern = config["pattern"]
        args = config["args"]
        env = {**os.environ, **config["env"]}
        
        # Get the base command for the runner
        if runner in self.runners:
            cmd = self.runners[runner].copy()
        else:
            # If runner not recognized, use it directly
            cmd = [runner]
        
        # Add pattern if specified
        if pattern:
            cmd.append(pattern)
        
        # Add additional args
        cmd.extend(args)
        
        log.info(f"Running tests with command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            # Run the tests with timeout
            process = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit
                timeout=timeout,
                env=env
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Parse the output
            return self._format_results(
                process.returncode,
                process.stdout,
                process.stderr,
                execution_time,
                cmd
            )
            
        except subprocess.TimeoutExpired:
            # Handle timeout
            execution_time = time.time() - start_time
            return self._format_results(
                -1,  # Special code for timeout
                "",
                f"Test execution timed out after {timeout} seconds",
                execution_time,
                cmd,
                timed_out=True
            )
    
    def _format_results(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time: float,
        cmd: List[str],
        timed_out: bool = False
    ) -> str:
        """
        Format test results in a structured way.
        
        Args:
            exit_code: The exit code of the test process
            stdout: Standard output from the test process
            stderr: Standard error from the test process
            execution_time: Test execution time in seconds
            cmd: The command that was executed
            timed_out: Whether the test execution timed out
            
        Returns:
            A structured string with test results
        """
        # Create a result dictionary
        result = {
            "success": exit_code == 0 and not timed_out,
            "exit_code": exit_code,
            "execution_time": round(execution_time, 2),
            "command": " ".join(cmd),
            "timed_out": timed_out,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Parse test output based on the runner
        if cmd[0] == "pytest" or "pytest" in cmd:
            result.update(self._parse_pytest_output(stdout))
        elif "unittest" in cmd:
            result.update(self._parse_unittest_output(stdout))
        
        # Add raw output for debugging
        result["stdout"] = stdout
        result["stderr"] = stderr
        
        # Create a summary string
        summary = f"TEST_RESULTS: {'SUCCESS' if result['success'] else 'FAILURE'}\n"
        summary += f"Exit Code: {exit_code}\n"
        summary += f"Execution Time: {result['execution_time']} seconds\n"
        
        if timed_out:
            summary += "Status: TIMED OUT\n"
        
        if "tests_run" in result:
            summary += f"Tests Run: {result['tests_run']}\n"
        if "tests_passed" in result:
            summary += f"Tests Passed: {result['tests_passed']}\n"
        if "tests_failed" in result:
            summary += f"Tests Failed: {result['tests_failed']}\n"
        if "tests_skipped" in result:
            summary += f"Tests Skipped: {result['tests_skipped']}\n"
        
        # Add output
        summary += "\n--- STDOUT ---\n"
        summary += stdout
        
        if stderr:
            summary += "\n--- STDERR ---\n"
            summary += stderr
        
        # Add JSON result for machine parsing
        summary += f"\n\nJSON_RESULT: {json.dumps(result, indent=2)}\n"
        
        return summary
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """
        Parse pytest output to extract test statistics.
        
        Args:
            output: Standard output from pytest
            
        Returns:
            A dictionary with test statistics
        """
        result = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "failures": [],
        }
        
        # Extract test summary
        summary_match = re.search(
            r'=+ ([\d]+) (passed|failed|skipped|xfailed|xpassed)(, ([\d]+) (passed|failed|skipped|xfailed|xpassed))*(, ([\d]+) (passed|failed|skipped|xfailed|xpassed))*(, ([\d]+) (passed|failed|skipped|xfailed|xpassed))* in [\d\.]+s',
            output
        )
        
        if summary_match:
            # Parse the summary line
            summary_line = summary_match.group(0)
            passed_match = re.search(r'(\d+) passed', summary_line)
            failed_match = re.search(r'(\d+) failed', summary_line)
            skipped_match = re.search(r'(\d+) skipped', summary_line)
            
            if passed_match:
                result["tests_passed"] = int(passed_match.group(1))
            if failed_match:
                result["tests_failed"] = int(failed_match.group(1))
            if skipped_match:
                result["tests_skipped"] = int(skipped_match.group(1))
            
            result["tests_run"] = result["tests_passed"] + result["tests_failed"] + result["tests_skipped"]
        
        # Extract failure details
        failure_sections = re.finditer(
            r'_{3,} (.*) _{3,}.*?\n(.*?)(?=_{3,}|$)',
            output,
            re.DOTALL
        )
        
        for section in failure_sections:
            test_name = section.group(1).strip()
            failure_details = section.group(2).strip()
            result["failures"].append({
                "test": test_name,
                "details": failure_details
            })
        
        return result
    
    def _parse_unittest_output(self, output: str) -> Dict[str, Any]:
        """
        Parse unittest output to extract test statistics.
        
        Args:
            output: Standard output from unittest
            
        Returns:
            A dictionary with test statistics
        """
        result = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "failures": [],
        }
        
        # Extract test summary
        summary_match = re.search(
            r'Ran (\d+) tests? in [\d\.]+s',
            output
        )
        
        if summary_match:
            result["tests_run"] = int(summary_match.group(1))
            
            # Check for failures
            if "FAILED (failures=" in output:
                failures_match = re.search(r'failures=(\d+)', output)
                if failures_match:
                    result["tests_failed"] = int(failures_match.group(1))
            
            if "FAILED (errors=" in output:
                errors_match = re.search(r'errors=(\d+)', output)
                if errors_match:
                    result["tests_failed"] += int(errors_match.group(1))
            
            if "FAILED (skipped=" in output:
                skipped_match = re.search(r'skipped=(\d+)', output)
                if skipped_match:
                    result["tests_skipped"] = int(skipped_match.group(1))
            
            # Calculate passed tests
            result["tests_passed"] = result["tests_run"] - result["tests_failed"] - result["tests_skipped"]
        
        # Extract failure details
        failure_sections = re.finditer(
            r'(ERROR|FAIL): (\w+) \((.*?)\)(.*?)(?=ERROR|FAIL|\Z)',
            output,
            re.DOTALL
        )
        
        for section in failure_sections:
            result_type = section.group(1)
            test_name = section.group(2)
            test_class = section.group(3)
            details = section.group(4).strip()
            
            result["failures"].append({
                "type": result_type,
                "test": f"{test_class}.{test_name}",
                "details": details
            })
        
        return result
    
    def discover_tests(self, pattern: str = "*_test.py") -> List[str]:
        """
        Discover test files in the base directory.
        
        Args:
            pattern: Glob pattern for test files
            
        Returns:
            A list of discovered test files
        """
        test_files = []
        
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if Path(file).match(pattern):
                    rel_path = os.path.relpath(os.path.join(root, file), self.base_dir)
                    # Normalize path separators
                    rel_path = rel_path.replace('\\', '/')
                    test_files.append(rel_path)
        
        return sorted(test_files)
    
    def run_specific_test(self, test_path: str, test_name: Optional[str] = None) -> str:
        """
        Run a specific test file or test case.
        
        Args:
            test_path: Path to the test file
            test_name: Optional specific test name/method to run
            
        Returns:
            Test results as a string
        """
        config = {
            "runner": self.default_runner,
            "timeout": self.default_timeout,
            "args": [test_path]
        }
        
        if test_name:
            if self.default_runner == "pytest":
                config["args"] = [f"{test_path}::{test_name}"]
            elif self.default_runner == "unittest":
                config["args"] = [f"{test_path}.{test_name}"]
        
        return self._run_tests(config)
