#!/usr/bin/env python3
"""
test_talk_orchestrator.py - Comprehensive tests for the Talk system

This test suite covers:
1. TalkOrchestrator initialization and configuration
2. CodeAgent diff generation functionality
3. FileAgent file operations and backup functionality
4. TestAgent test execution and result parsing
5. Blackboard communication between agents
6. PlanRunner execution with specialized agents
7. Timeout functionality
8. Interactive and non-interactive modes
"""

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

# Import Talk components
from agent.agent import Agent
from agent.messages import Message, Role
from plan_runner.blackboard import Blackboard, BlackboardEntry
from plan_runner.plan_runner import PlanRunner
from plan_runner.step import Step
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent
from talk.talk import TalkOrchestrator


# Helper functions and fixtures
def create_mock_agent(name="MockAgent"):
    """Create a mock agent that returns predefined responses."""
    mock_agent = Mock(spec=Agent)
    mock_agent.name = name
    mock_agent.id = f"{name}-12345678"
    mock_agent.run.return_value = f"Response from {name}"
    return mock_agent


def create_temp_directory():
    """Create a temporary directory for testing."""
    return tempfile.TemporaryDirectory()


# Test classes
class TestTalkOrchestratorInitialization(unittest.TestCase):
    """Test the initialization and configuration of TalkOrchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = create_temp_directory()
        # Mock signal.alarm to prevent actual alarms during tests
        self.alarm_patcher = patch('signal.alarm')
        self.mock_alarm = self.alarm_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        self.alarm_patcher.stop()

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    def test_initialization(self, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test that TalkOrchestrator initializes correctly."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        task = "Test task"
        working_dir = self.temp_dir.name
        model = "test-model"
        timeout_minutes = 10
        interactive = True

        # Act
        orchestrator = TalkOrchestrator(
            task=task,
            working_dir=working_dir,
            model=model,
            timeout_minutes=timeout_minutes,
            interactive=interactive
        )

        # Assert
        self.assertEqual(orchestrator.task, task)
        self.assertEqual(orchestrator.timeout_minutes, timeout_minutes)
        self.assertEqual(orchestrator.interactive, interactive)
        self.assertIsInstance(orchestrator.blackboard, Blackboard)
        self.assertEqual(len(orchestrator.plan), 4)  # Verify the plan has 4 steps
        
        # Verify agents were created with correct parameters
        mock_code_agent.assert_called_once()
        mock_file_agent.assert_called_once()
        mock_test_agent.assert_called_once()
        
        # Verify timeout was set
        self.mock_alarm.assert_called_once_with(timeout_minutes * 60)

    @patch('os.makedirs')
    def test_versioned_directory_creation(self, mock_makedirs):
        """Test that versioned directories are created correctly."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump') as mock_json_dump:
            
            # Act
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name
            )
            
            # Assert
            mock_makedirs.assert_called()
            mock_json_dump.assert_called_once()
            self.assertTrue(str(orchestrator.working_dir).endswith('talk1'))

    @patch('os.makedirs')
    def test_versioned_directory_increments(self, mock_makedirs):
        """Test that versioned directory numbers increment when directories exist."""
        # Arrange
        with patch('pathlib.Path.exists', side_effect=[True, True, False]), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump') as mock_json_dump:
            
            # Act
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name
            )
            
            # Assert
            self.assertTrue(str(orchestrator.working_dir).endswith('talk3'))


class TestCodeAgent(unittest.TestCase):
    """Test the CodeAgent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the LLM backend to avoid actual API calls
        self.backend_patcher = patch('agent.agent.Agent._setup_backend')
        self.mock_backend = self.backend_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.backend_patcher.stop()

    @patch('agent.agent.Agent.call_ai')
    def test_run_generates_diff(self, mock_call_ai):
        """Test that CodeAgent.run generates a unified diff."""
        # Arrange
        mock_call_ai.return_value = """
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    return "Hello"
+    return "Hello, World!"
+
"""
        code_agent = CodeAgent()
        input_text = "Task: Update hello function\ntest.py\ndef hello():\n    return \"Hello\"\n"

        # Act
        result = code_agent.run(input_text)

        # Assert
        self.assertIn("--- a/test.py", result)
        self.assertIn("+++ b/test.py", result)
        self.assertIn("+    return \"Hello, World!\"", result)

    @patch('agent.agent.Agent.call_ai')
    def test_handles_markdown_code_blocks(self, mock_call_ai):
        """Test that CodeAgent can extract diffs from markdown code blocks."""
        # Arrange
        mock_call_ai.return_value = """
Here's the diff:

```diff
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    return "Hello"
+    return "Hello, World!"
+
```
"""
        code_agent = CodeAgent()
        input_text = "Task: Update hello function\ntest.py\ndef hello():\n    return \"Hello\"\n"

        # Act
        result = code_agent.run(input_text)

        # Assert
        self.assertIn("--- a/test.py", result)
        self.assertIn("+++ b/test.py", result)
        self.assertIn("+    return \"Hello, World!\"", result)

    @patch('agent.agent.Agent.call_ai')
    def test_creates_diff_from_full_code(self, mock_call_ai):
        """Test that CodeAgent can create a diff when LLM returns full code."""
        # Arrange
        mock_call_ai.return_value = """
Here's the updated code:

```python
def hello():
    return "Hello, World!"
```
"""
        # Setup conversation history
        code_agent = CodeAgent()
        code_agent.conversation.append(Message(
            role=Role.USER,
            content="Current content:\n```\ndef hello():\n    return \"Hello\"\n```"
        ))
        
        input_text = "Task: Update hello function\ntest.py\ndef hello():\n    return \"Hello\"\n"

        # Act
        with patch.object(code_agent, '_extract_and_validate_diff', 
                          side_effect=code_agent._create_diff_from_response):
            result = code_agent.run(input_text)

        # Assert
        self.assertIn("--- a/", result)
        self.assertIn("+++ b/", result)
        self.assertIn("+    return \"Hello, World!\"", result)


class TestFileAgent(unittest.TestCase):
    """Test the FileAgent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = create_temp_directory()
        self.file_agent = FileAgent(base_dir=self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch('subprocess.run')
    def test_apply_diff(self, mock_subprocess):
        """Test that FileAgent can apply a diff."""
        # Arrange
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="patching file test.py",
            stderr=""
        )
        diff_text = """
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    return "Hello"
+    return "Hello, World!"
+
"""
        # Create the test file
        test_file_path = os.path.join(self.temp_dir.name, "test.py")
        with open(test_file_path, 'w') as f:
            f.write("def hello():\n    return \"Hello\"\n")

        # Act
        with patch('tempfile.NamedTemporaryFile', mock.mock_open(read_data=diff_text)):
            result = self.file_agent._apply_diff(diff_text)

        # Assert
        self.assertIn("PATCH_APPLIED", result)
        mock_subprocess.assert_called_once()

    def test_extract_file_paths(self):
        """Test that FileAgent can extract file paths from a diff."""
        # Arrange
        diff_text = """
--- a/test1.py
+++ b/test1.py
@@ -1,3 +1,4 @@
 def hello():
-    return "Hello"
+    return "Hello, World!"
+
--- a/test2.py
+++ b/test2.py
@@ -1,3 +1,4 @@
 def goodbye():
-    return "Goodbye"
+    return "Goodbye, World!"
+
"""
        # Act
        file_paths = self.file_agent._extract_file_paths(diff_text)

        # Assert
        self.assertEqual(len(file_paths), 2)
        self.assertIn("test1.py", file_paths)
        self.assertIn("test2.py", file_paths)

    @patch('shutil.copy2')
    @patch('os.makedirs')
    def test_backup_files(self, mock_makedirs, mock_copy):
        """Test that FileAgent can backup files."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True):
            file_paths = ["test1.py", "test2.py"]
            
            # Act
            backup_paths = self.file_agent._backup_files(file_paths)
            
            # Assert
            self.assertEqual(len(backup_paths), 2)
            mock_makedirs.assert_called()
            self.assertEqual(mock_copy.call_count, 2)

    def test_read_file(self):
        """Test that FileAgent can read files."""
        # Arrange
        test_content = "Test content"
        test_file_path = os.path.join(self.temp_dir.name, "test.py")
        with open(test_file_path, 'w') as f:
            f.write(test_content)

        # Act
        content = self.file_agent.read_file("test.py")

        # Assert
        self.assertEqual(content, test_content)

    def test_file_exists(self):
        """Test that FileAgent can check if files exist."""
        # Arrange
        test_file_path = os.path.join(self.temp_dir.name, "test.py")
        with open(test_file_path, 'w') as f:
            f.write("Test content")

        # Act & Assert
        self.assertTrue(self.file_agent.file_exists("test.py"))
        self.assertFalse(self.file_agent.file_exists("nonexistent.py"))

    def test_list_files(self):
        """Test that FileAgent can list files."""
        # Arrange
        test_files = ["test1.py", "test2.py", "subdir/test3.py"]
        for file_path in test_files:
            full_path = os.path.join(self.temp_dir.name, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write("Test content")

        # Act
        files = self.file_agent.list_files()

        # Assert
        self.assertEqual(len(files), 3)
        for file_path in test_files:
            self.assertIn(file_path.replace('\\', '/'), files)


class TestTestAgent(unittest.TestCase):
    """Test the TestAgent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = create_temp_directory()
        self.test_agent = TestAgent(base_dir=self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch('subprocess.run')
    def test_run_tests_pytest(self, mock_subprocess):
        """Test that TestAgent can run pytest tests."""
        # Arrange
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="===== 2 passed in 0.05s =====",
            stderr=""
        )
        
        # Act
        result = self.test_agent.run("pytest")
        
        # Assert
        self.assertIn("TEST_RESULTS: SUCCESS", result)
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_run_tests_unittest(self, mock_subprocess):
        """Test that TestAgent can run unittest tests."""
        # Arrange
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Ran 2 tests in 0.001s\n\nOK",
            stderr=""
        )
        
        # Act
        result = self.test_agent.run("unittest")
        
        # Assert
        self.assertIn("TEST_RESULTS: SUCCESS", result)
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_subprocess):
        """Test that TestAgent correctly reports test failures."""
        # Arrange
        mock_subprocess.return_value = Mock(
            returncode=1,
            stdout="===== 1 failed, 1 passed in 0.05s =====",
            stderr=""
        )
        
        # Act
        result = self.test_agent.run("pytest")
        
        # Assert
        self.assertIn("TEST_RESULTS: FAILURE", result)
        self.assertIn("Exit Code: 1", result)

    @patch('subprocess.run')
    def test_run_tests_timeout(self, mock_subprocess):
        """Test that TestAgent correctly handles test timeouts."""
        # Arrange
        mock_subprocess.side_effect = subprocess.TimeoutExpired("pytest", 10)
        
        # Act
        result = self.test_agent.run("pytest --timeout=10")
        
        # Assert
        self.assertIn("TEST_RESULTS: FAILURE", result)
        self.assertIn("TIMED OUT", result)

    def test_parse_pytest_output(self):
        """Test that TestAgent correctly parses pytest output."""
        # Arrange
        pytest_output = """
============================= test session starts ==============================
platform win32 -- Python 3.9.5, pytest-7.0.0, pluggy-1.0.0
rootdir: C:\\test
collected 2 items

test_example.py::test_pass PASSED                                       [ 50%]
test_example.py::test_fail FAILED                                       [100%]

=================================== FAILURES ===================================
_________________________________ test_fail __________________________________

    def test_fail():
>       assert 1 == 2
E       assert 1 == 2

test_example.py:6: AssertionError
=========================== short test summary info ===========================
FAILED test_example.py::test_fail - assert 1 == 2
========================= 1 failed, 1 passed in 0.12s =========================
"""
        # Act
        result = self.test_agent._parse_pytest_output(pytest_output)
        
        # Assert
        self.assertEqual(result["tests_run"], 2)
        self.assertEqual(result["tests_passed"], 1)
        self.assertEqual(result["tests_failed"], 1)
        self.assertEqual(len(result["failures"]), 1)

    def test_parse_unittest_output(self):
        """Test that TestAgent correctly parses unittest output."""
        # Arrange
        unittest_output = """
test_fail (test_example.TestExample) ... FAIL
test_pass (test_example.TestExample) ... ok

======================================================================
FAIL: test_fail (test_example.TestExample)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\\test\\test_example.py", line 6, in test_fail
    self.assertEqual(1, 2)
AssertionError: 1 != 2

----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)
"""
        # Act
        result = self.test_agent._parse_unittest_output(unittest_output)
        
        # Assert
        self.assertEqual(result["tests_run"], 2)
        self.assertEqual(result["tests_passed"], 1)
        self.assertEqual(result["tests_failed"], 1)
        self.assertEqual(len(result["failures"]), 1)

    def test_discover_tests(self):
        """Test that TestAgent can discover test files."""
        # Arrange
        test_files = ["test_one.py", "test_two.py", "not_a_test.py"]
        for file_path in test_files:
            with open(os.path.join(self.temp_dir.name, file_path), 'w') as f:
                f.write("# Test file")
        
        # Act
        discovered = self.test_agent.discover_tests("test_*.py")
        
        # Assert
        self.assertEqual(len(discovered), 2)
        self.assertIn("test_one.py", discovered)
        self.assertIn("test_two.py", discovered)
        self.assertNotIn("not_a_test.py", discovered)


class TestBlackboardCommunication(unittest.TestCase):
    """Test the communication between agents via the blackboard."""

    def test_agent_communication(self):
        """Test that agents can communicate through the blackboard."""
        # Arrange
        blackboard = Blackboard()
        code_agent = create_mock_agent("CodeAgent")
        file_agent = create_mock_agent("FileAgent")
        test_agent = create_mock_agent("TestAgent")
        
        # Act - Simulate CodeAgent writing to blackboard
        code_result = "Diff content"
        blackboard.add_sync("generate_code", code_result, author=code_agent.id)
        
        # Simulate FileAgent reading from blackboard and writing result
        code_entries = blackboard.query_sync(label="generate_code")
        file_result = "PATCH_APPLIED"
        blackboard.add_sync("apply_changes", file_result, author=file_agent.id)
        
        # Simulate TestAgent reading from blackboard and writing result
        file_entries = blackboard.query_sync(label="apply_changes")
        test_result = "TEST_RESULTS: SUCCESS"
        blackboard.add_sync("run_tests", test_result, author=test_agent.id)
        
        # Assert
        self.assertEqual(len(blackboard.entries()), 3)
        self.assertEqual(code_entries[0].content, code_result)
        self.assertEqual(file_entries[0].content, file_result)
        
        # Verify provenance tracking
        test_entries = blackboard.query_sync(author=test_agent.id)
        self.assertEqual(len(test_entries), 1)
        self.assertEqual(test_entries[0].label, "run_tests")


class TestPlanRunner(unittest.TestCase):
    """Test the PlanRunner execution with specialized agents."""

    def setUp(self):
        """Set up test fixtures."""
        self.blackboard = Blackboard()
        self.code_agent = create_mock_agent("CodeAgent")
        self.file_agent = create_mock_agent("FileAgent")
        self.test_agent = create_mock_agent("TestAgent")
        
        self.agents = {
            "coder": self.code_agent,
            "file": self.file_agent,
            "tester": self.test_agent
        }

    def test_run_linear_plan(self):
        """Test running a linear plan with PlanRunner."""
        # Arrange
        steps = [
            Step(label="generate_code", agent_key="coder", on_success="apply_changes"),
            Step(label="apply_changes", agent_key="file", on_success="run_tests"),
            Step(label="run_tests", agent_key="tester")
        ]
        
        runner = PlanRunner(steps, self.agents, self.blackboard)
        
        # Act
        result = runner.run("Initial prompt")
        
        # Assert
        self.assertEqual(result, "Response from TestAgent")
        self.assertEqual(self.code_agent.run.call_count, 1)
        self.assertEqual(self.file_agent.run.call_count, 1)
        self.assertEqual(self.test_agent.run.call_count, 1)
        
        # Verify blackboard entries
        entries = self.blackboard.entries()
        self.assertEqual(len(entries), 3)

    def test_run_with_parallel_steps(self):
        """Test running a plan with parallel steps."""
        # Arrange
        parallel_steps = [
            Step(label="parallel_1", agent_key="coder"),
            Step(label="parallel_2", agent_key="file")
        ]
        
        steps = [
            Step(
                label="parallel_wrapper",
                agent_key="coder",
                parallel_steps=parallel_steps,
                on_success="final"
            ),
            Step(label="final", agent_key="tester")
        ]
        
        runner = PlanRunner(steps, self.agents, self.blackboard)
        
        # Act
        result = runner.run("Initial prompt")
        
        # Assert
        self.assertEqual(result, "Response from TestAgent")
        
        # Verify all parallel steps were executed
        entries = self.blackboard.entries()
        self.assertEqual(len(entries), 4)  # wrapper + 2 parallel + final
        
        parallel_entries = [e for e in entries if e.label in ["parallel_1", "parallel_2"]]
        self.assertEqual(len(parallel_entries), 2)

    def test_run_with_error_handling(self):
        """Test running a plan with error handling."""
        # Arrange
        self.file_agent.run.side_effect = Exception("File operation failed")
        
        steps = [
            Step(label="generate_code", agent_key="coder", on_success="apply_changes", on_fail="error_handler"),
            Step(label="apply_changes", agent_key="file", on_success="run_tests", on_fail="error_handler"),
            Step(label="run_tests", agent_key="tester"),
            Step(label="error_handler", agent_key="coder")
        ]
        
        runner = PlanRunner(steps, self.agents, self.blackboard)
        
        # Act & Assert
        with self.assertRaises(Exception):
            runner.run("Initial prompt")
        
        self.assertEqual(self.code_agent.run.call_count, 1)
        self.assertEqual(self.file_agent.run.call_count, 1)
        self.assertEqual(self.test_agent.run.call_count, 0)


class TestTalkOrchestratorExecution(unittest.TestCase):
    """Test the execution of TalkOrchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = create_temp_directory()
        # Mock signal.alarm to prevent actual alarms during tests
        self.alarm_patcher = patch('signal.alarm')
        self.mock_alarm = self.alarm_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        self.alarm_patcher.stop()

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    @patch('talk.talk.PlanRunner.run')
    def test_non_interactive_mode(self, mock_run, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test running TalkOrchestrator in non-interactive mode."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        mock_run.return_value = "Final result"
        
        # Act
        with patch('builtins.print'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump'):
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name,
                interactive=False
            )
            result = orchestrator.run()
        
        # Assert
        self.assertEqual(result, 0)  # Success exit code
        mock_run.assert_called_once()

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    @patch('builtins.input', side_effect=["", "y", "y", "y"])
    def test_interactive_mode(self, mock_input, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test running TalkOrchestrator in interactive mode."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        
        # Act
        with patch('builtins.print'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump'):
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name,
                interactive=True
            )
            orchestrator._get_next_step = MagicMock(side_effect=[
                orchestrator.plan[1],  # apply_changes
                orchestrator.plan[2],  # run_tests
                orchestrator.plan[3],  # check_results
                None  # End of plan
            ])
            result = orchestrator._interactive_mode()
        
        # Assert
        self.assertEqual(mock_input.call_count, 4)  # Initial + 3 steps
        self.assertEqual(orchestrator.agents["coder"].run.call_count, 2)  # generate_code + check_results
        self.assertEqual(orchestrator.agents["file"].run.call_count, 1)  # apply_changes
        self.assertEqual(orchestrator.agents["tester"].run.call_count, 1)  # run_tests

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    def test_timeout_handler(self, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test the timeout handler."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        
        with patch('builtins.print'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump'), \
             patch('sys.exit') as mock_exit:
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name
            )
            
            # Simulate timeout by calling the handler directly
            orchestrator._timeout_handler(signal.SIGALRM, None)
            
            # Assert
            mock_exit.assert_called_once_with(1)
            
            # Verify timeout was recorded in blackboard
            timeout_entries = orchestrator.blackboard.query_sync(label="timeout")
            self.assertEqual(len(timeout_entries), 1)
            self.assertIn("timed out", timeout_entries[0].content)

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    def test_error_handling(self, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test error handling in TalkOrchestrator."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        
        # Make PlanRunner.run raise an exception
        with patch('talk.talk.PlanRunner.run', side_effect=Exception("Test error")), \
             patch('builtins.print'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump'), \
             patch('logging.error'):
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name,
                interactive=False
            )
            result = orchestrator._non_interactive_mode()
        
        # Assert
        self.assertEqual(result, 1)  # Error exit code

    @patch('talk.talk.CodeAgent')
    @patch('talk.talk.FileAgent')
    @patch('talk.talk.TestAgent')
    def test_keyboard_interrupt(self, mock_test_agent, mock_file_agent, mock_code_agent):
        """Test handling of keyboard interrupts."""
        # Arrange
        mock_code_agent.return_value = create_mock_agent("CodeAgent")
        mock_file_agent.return_value = create_mock_agent("FileAgent")
        mock_test_agent.return_value = create_mock_agent("TestAgent")
        
        # Make _non_interactive_mode raise KeyboardInterrupt
        with patch.object(TalkOrchestrator, '_non_interactive_mode', side_effect=KeyboardInterrupt()), \
             patch('builtins.print'), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('builtins.open', mock.mock_open()), \
             patch('json.dump'):
            orchestrator = TalkOrchestrator(
                task="Test task",
                working_dir=self.temp_dir.name,
                interactive=False
            )
            result = orchestrator.run()
        
        # Assert
        self.assertEqual(result, 130)  # SIGINT exit code


if __name__ == '__main__':
    unittest.main()
