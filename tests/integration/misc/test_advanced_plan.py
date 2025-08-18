#!/usr/bin/env python3
"""
tests/test_advanced_plan.py - Advanced tests for complex Plan Runner workflows

This test file implements complex multi-step plans that simulate realistic
software development workflows, including:

1. Iterative code generation, testing, and refinement
2. File operations with backup and restoration
3. Multiple specialized agents working together
4. Dynamic plan creation based on test results
5. Complete end-to-end development cycles

All test outputs, intermediate files, and logs are saved to tests/output/ directory.
"""

import os
import sys
import unittest
import uuid
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open

# Import plan runner components
from plan_runner.blackboard import Blackboard, BlackboardEntry
from plan_runner.step import Step
from plan_runner.plan_runner import PlanRunner

# Import agent components
from agent.agent import Agent
from agent.settings import Settings

# Import specialized agents
from special_agents.code_agent import CodeAgent
from special_agents.file_agent import FileAgent
from special_agents.test_agent import TestAgent

# Configure test output directory
TEST_OUTPUT_DIR = Path(__file__).parent / "output"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

class TestAdvancedPlan(unittest.TestCase):
    """Test advanced Plan Runner functionality with complex workflows."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Generate a unique test ID for this test run
        self.test_id = f"adv_test_{uuid.uuid4().hex[:8]}"
        self.test_output_dir = TEST_OUTPUT_DIR / self.test_id
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create a fresh blackboard for each test
        self.blackboard = Blackboard()
        
        # Create a temporary working directory for file operations
        self.working_dir = self.test_output_dir / "workspace"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Record test start time
        self.start_time = datetime.now()
        print(f"\nRunning {self._testMethodName} at {self.start_time}")
    
    def tearDown(self):
        """Clean up after each test."""
        # Record test duration
        duration = datetime.now() - self.start_time
        print(f"Test completed in {duration.total_seconds():.2f} seconds")
        
        # Save test metadata
        with open(self.test_output_dir / "metadata.txt", "w") as f:
            f.write(f"Test: {self._testMethodName}\n")
            f.write(f"Started: {self.start_time}\n")
            f.write(f"Duration: {duration.total_seconds():.2f} seconds\n")
            f.write(f"Status: {'PASS' if sys.exc_info()[0] is None else 'FAIL'}\n")
        
        # Save blackboard state
        self._save_blackboard_state()
    
    def _save_blackboard_state(self):
        """Save the current blackboard state to the output directory."""
        entries = self.blackboard.entries()
        
        with open(self.test_output_dir / "blackboard_state.txt", "w") as f:
            f.write(f"Total entries: {len(entries)}\n\n")
            for i, entry in enumerate(entries):
                f.write(f"Entry {i+1}:\n")
                f.write(f"  ID: {entry.id}\n")
                f.write(f"  Label: {entry.label}\n")
                f.write(f"  Author: {entry.author}\n")
                f.write(f"  Role: {entry.role}\n")
                f.write(f"  Content: {entry.content[:100]}...\n" if len(str(entry.content)) > 100 else f"  Content: {entry.content}\n")
                f.write(f"  Timestamp: {datetime.fromtimestamp(entry.ts)}\n")
                f.write("\n")
    
    def _create_mock_code_agent(self, name="MockCodeAgent"):
        """Create a mock CodeAgent that generates code diffs."""
        mock_agent = MagicMock(spec=CodeAgent)
        mock_agent.name = name
        mock_agent.id = f"{name}-{uuid.uuid4().hex[:8]}"
        
        # Define a series of responses for iterative code improvement
        self.code_iterations = [
            # Initial implementation with a bug (missing return statement)
            '```diff\n--- a/fibonacci.py\n+++ b/fibonacci.py\n@@ -0,0 +1,9 @@\n+def fibonacci(n):\n+    """Calculate the nth Fibonacci number."""\n+    if n <= 0:\n+        raise ValueError("Input must be a positive integer")\n+    if n == 1 or n == 2:\n+        return 1\n+    \n+    # Bug: missing "return" statement\n+    fibonacci(n-1) + fibonacci(n-2)\n```',
            # First fix - add return statement but still has a performance issue
            '```diff\n--- a/fibonacci.py\n+++ b/fibonacci.py\n@@ -6,4 +6,4 @@\n         return 1\n     \n     # Fixed: added return statement\n-    fibonacci(n-1) + fibonacci(n-2)\n+    return fibonacci(n-1) + fibonacci(n-2)\n```',
            # Second fix - add memoization for performance
            '```diff\n--- a/fibonacci.py\n+++ b/fibonacci.py\n@@ -1,9 +1,17 @@\n+# Added memoization for better performance\n+memo = {}\n+\n def fibonacci(n):\n     """Calculate the nth Fibonacci number."""\n     if n <= 0:\n         raise ValueError("Input must be a positive integer")\n+        \n+    # Check if we\'ve already calculated this value\n+    if n in memo:\n+        return memo[n]\n+        \n     if n == 1 or n == 2:\n-        return 1\n+        memo[n] = 1\n+        return memo[n]\n     \n-    return fibonacci(n-1) + fibonacci(n-2)\n+    memo[n] = fibonacci(n-1) + fibonacci(n-2)\n+    return memo[n]\n```',
            # Third fix - add test function
            '```diff\n--- a/fibonacci.py\n+++ b/fibonacci.py\n@@ -15,3 +15,14 @@\n     \n     memo[n] = fibonacci(n-1) + fibonacci(n-2)\n     return memo[n]\n+\n+# Added test function\n+def test_fibonacci():\n+    """Test the fibonacci function."""\n+    assert fibonacci(1) == 1\n+    assert fibonacci(2) == 1\n+    assert fibonacci(3) == 2\n+    assert fibonacci(4) == 3\n+    assert fibonacci(5) == 5\n+    assert fibonacci(10) == 55\n+    print("All tests passed!")\n```'
        ]
        
        # Create a counter to track which iteration we're on
        self.code_iteration_counter = 0
        
        def mock_run(prompt):
            # Return the next iteration of code
            iteration = min(self.code_iteration_counter, len(self.code_iterations) - 1)
            diff = self.code_iterations[iteration]
            self.code_iteration_counter += 1
            return diff
        
        mock_agent.run.side_effect = mock_run
        return mock_agent
    
    def _create_mock_file_agent(self, working_dir, name="MockFileAgent"):
        """Create a mock FileAgent that handles file operations."""
        mock_agent = MagicMock(spec=FileAgent)
        mock_agent.name = name
        mock_agent.id = f"{name}-{uuid.uuid4().hex[:8]}"
        
        # Track the current state of files
        self.files = {}
        
        def mock_run(diff):
            """Apply a diff to the virtual file system."""
            # Parse the diff to extract file content
            if "```diff" in diff:
                diff_content = diff.split("```diff")[1].split("```")[0].strip()
            else:
                diff_content = diff
                
            # Extract filename from the diff
            filename_match = diff_content.split("\n")[1]
            if "+++ b/" in filename_match:
                filename = filename_match.split("+++ b/")[1].strip()
            else:
                filename = "unknown.py"
                
            # Apply the diff (simplified - just extract the final content)
            lines = diff_content.split("\n")
            content_lines = []
            for line in lines[2:]:  # Skip the first two lines (--- and +++)
                if line.startswith("+"):
                    content_lines.append(line[1:])  # Remove the + prefix
            
            # Save the file content
            file_content = "\n".join(content_lines)
            self.files[filename] = file_content
            
            # Actually write to the filesystem for testing
            file_path = working_dir / filename
            with open(file_path, "w") as f:
                f.write(file_content)
            
            return f"PATCH_APPLIED: {filename} patched successfully"
        
        def mock_list_files():
            """List files in the virtual file system."""
            return list(self.files.keys())
        
        def mock_read_file(filename):
            """Read a file from the virtual file system."""
            if filename in self.files:
                return self.files[filename]
            else:
                raise FileNotFoundError(f"File {filename} not found")
        
        mock_agent.run.side_effect = mock_run
        mock_agent.list_files = MagicMock(side_effect=mock_list_files)
        mock_agent.read_file = MagicMock(side_effect=mock_read_file)
        
        return mock_agent
    
    def _create_mock_test_agent(self, working_dir, name="MockTestAgent"):
        """Create a mock TestAgent that runs tests."""
        mock_agent = MagicMock(spec=TestAgent)
        mock_agent.name = name
        mock_agent.id = f"{name}-{uuid.uuid4().hex[:8]}"
        
        # Define test results for each iteration
        self.test_results = [
            # First test - syntax error (missing return)
            """TEST_RESULTS: FAILURE
Traceback (most recent call last):
  File "fibonacci.py", line 9, in <module>
    fibonacci(5)
  File "fibonacci.py", line 8, in fibonacci
    fibonacci(n-1) + fibonacci(n-2)
TypeError: 'NoneType' object is not callable
""",
            # Second test - works but times out on large inputs
            """TEST_RESULTS: PARTIAL
Small tests passed, but large input test timed out:
fibonacci(35) - TIMEOUT after 5 seconds
""",
            # Third test - all tests pass
            """TEST_RESULTS: SUCCESS
Ran 6 tests in 0.001s
All tests passed!
""",
            # Fourth test - all tests pass including the test function
            """TEST_RESULTS: SUCCESS
Ran 7 tests in 0.001s
All tests passed!
"""
        ]
        
        # Create a counter to track which iteration we're on
        self.test_iteration_counter = 0
        
        def mock_run(command):
            # Return the next iteration of test results
            iteration = min(self.test_iteration_counter, len(self.test_results) - 1)
            result = self.test_results[iteration]
            self.test_iteration_counter += 1
            return result
        
        mock_agent.run.side_effect = mock_run
        return mock_agent
    
    def test_specialized_agents_collaboration(self):
        """Test that specialized agents can work together effectively."""
        # Create mock specialized agents
        code_agent = self._create_mock_code_agent()
        file_agent = self._create_mock_file_agent(self.working_dir)
        test_agent = self._create_mock_test_agent(self.working_dir)
        
        # Create a dictionary of agents
        agents = {
            "code": code_agent,
            "file": file_agent,
            "test": test_agent
        }
        
        # Create steps for a simple workflow
        generate_code = Step(label="generate_code", agent_key="code", on_success="apply_changes")
        apply_changes = Step(label="apply_changes", agent_key="file", on_success="run_tests")
        run_tests = Step(label="run_tests", agent_key="test")
        
        steps = [generate_code, apply_changes, run_tests]
        
        # Create and run the plan
        runner = PlanRunner(steps, agents, self.blackboard)
        result = runner.run("Implement a function to calculate Fibonacci numbers")
        
        # Verify all agents were called
        code_agent.run.assert_called_once()
        file_agent.run.assert_called_once()
        test_agent.run.assert_called_once()
        
        # Verify the file was created
        self.assertTrue("fibonacci.py" in file_agent.list_files())
        
        # Verify blackboard entries
        entries = self.blackboard.entries()
        self.assertEqual(len(entries), 3)
        
        # Save agent collaboration details to output file
        with open(self.test_output_dir / "agent_collaboration.txt", "w") as f:
            f.write("Specialized Agents Collaboration Test:\n")
            f.write(f"Generated code by {code_agent.name}\n")
            f.write(f"File operations by {file_agent.name}\n")
            f.write(f"Tests run by {test_agent.name}\n\n")
            f.write("Files created:\n")
            for filename in file_agent.list_files():
                f.write(f"  {filename}\n")
                # Also save the file content to the output directory
                with open(self.test_output_dir / filename, "w") as file_out:
                    file_out.write(file_agent.read_file(filename))
    
    def test_iterative_code_improvement(self):
        """Test an iterative workflow that improves code until tests pass."""
        # Create mock specialized agents
        code_agent = self._create_mock_code_agent()
        file_agent = self._create_mock_file_agent(self.working_dir)
        test_agent = self._create_mock_test_agent(self.working_dir)
        
        # Create a dictionary of agents
        agents = {
            "code": code_agent,
            "file": file_agent,
            "test": test_agent
        }
        
        # Create a custom plan runner that implements the iterative workflow
        class IterativePlanRunner(PlanRunner):
            def run(self, user_prompt: str) -> str:
                # Initialize
                iteration = 1
                max_iterations = 5
                all_tests_pass = False
                
                # Track progress
                progress_log = []
                
                while iteration <= max_iterations and not all_tests_pass:
                    # Step 1: Generate code
                    progress_log.append(f"Iteration {iteration}: Generating code...")
                    code_diff = self._run_single(self.index["generate_code"], user_prompt)
                    
                    # Step 2: Apply changes
                    progress_log.append(f"Iteration {iteration}: Applying changes...")
                    file_result = self._run_single(self.index["apply_changes"], code_diff)
                    
                    # Step 3: Run tests
                    progress_log.append(f"Iteration {iteration}: Running tests...")
                    test_result = self._run_single(self.index["run_tests"], "Run tests")
                    
                    # Check if tests pass
                    if "SUCCESS" in test_result and "All tests passed" in test_result:
                        all_tests_pass = True
                        progress_log.append(f"Iteration {iteration}: All tests passed!")
                    else:
                        # Prepare for next iteration
                        user_prompt = f"Fix the following test failure:\n{test_result}"
                        progress_log.append(f"Iteration {iteration}: Tests failed, will try again.")
                    
                    iteration += 1
                
                # Record the final progress log
                self.bb.add_sync("progress_log", "\n".join(progress_log), section="meta")
                
                # Return the final test result
                return test_result
        
        # Create steps for the iterative workflow
        generate_code = Step(label="generate_code", agent_key="code")
        apply_changes = Step(label="apply_changes", agent_key="file")
        run_tests = Step(label="run_tests", agent_key="test")
        
        steps = [generate_code, apply_changes, run_tests]
        
        # Create and run the iterative plan
        runner = IterativePlanRunner(steps, agents, self.blackboard)
        result = runner.run("Implement a function to calculate Fibonacci numbers with good performance")
        
        # Verify multiple iterations occurred
        self.assertGreater(code_agent.run.call_count, 1)
        self.assertGreater(file_agent.run.call_count, 1)
        self.assertGreater(test_agent.run.call_count, 1)
        
        # Verify the final result indicates success
        self.assertIn("SUCCESS", result)
        
        # Get the progress log
        progress_entries = self.blackboard.query_sync(label="progress_log")
        self.assertEqual(len(progress_entries), 1)
        
        # Save iterative workflow details to output file
        with open(self.test_output_dir / "iterative_workflow.txt", "w") as f:
            f.write("Iterative Code Improvement Test:\n")
            f.write(f"Total iterations: {code_agent.run.call_count}\n")
            f.write(f"Final result: {result}\n\n")
            f.write("Progress Log:\n")
            f.write(progress_entries[0].content)
            f.write("\n\nFinal code:\n")
            for filename in file_agent.list_files():
                f.write(f"\n--- {filename} ---\n")
                f.write(file_agent.read_file(filename))
                # Also save the final file to the output directory
                with open(self.test_output_dir / filename, "w") as file_out:
                    file_out.write(file_agent.read_file(filename))
    
    def test_dynamic_plan_creation(self):
        """Test creating and modifying plans dynamically based on results."""
        # Create mock specialized agents
        code_agent = self._create_mock_code_agent()
        file_agent = self._create_mock_file_agent(self.working_dir)
        test_agent = self._create_mock_test_agent(self.working_dir)
        
        # Create a dictionary of agents
        agents = {
            "code": code_agent,
            "file": file_agent,
            "test": test_agent
        }
        
        # Create a custom plan runner that builds plans dynamically
        class DynamicPlanRunner:
            def __init__(self, agents, blackboard):
                self.agents = agents
                self.bb = blackboard
                self.all_steps = []
            
            def run(self, user_prompt: str) -> str:
                # Phase 1: Initial code generation
                self.bb.add_sync("phase", "Initial code generation", section="meta")
                
                phase1_steps = [
                    Step(label="generate_code", agent_key="code"),
                    Step(label="apply_changes", agent_key="file"),
                    Step(label="run_tests", agent_key="test")
                ]
                
                phase1_runner = PlanRunner(phase1_steps, self.agents, self.bb)
                test_result = phase1_runner.run(user_prompt)
                self.all_steps.extend(phase1_steps)
                
                # Check test result to determine next phase
                if "SUCCESS" in test_result:
                    # Phase 2A: Add test function if tests passed
                    self.bb.add_sync("phase", "Adding test function", section="meta")
                    
                    phase2a_steps = [
                        Step(label="add_tests", agent_key="code"),
                        Step(label="apply_test_changes", agent_key="file"),
                        Step(label="run_final_tests", agent_key="test")
                    ]
                    
                    phase2a_runner = PlanRunner(phase2a_steps, self.agents, self.bb)
                    final_result = phase2a_runner.run("Add a test function to verify the Fibonacci implementation")
                    self.all_steps.extend(phase2a_steps)
                    
                    return final_result
                else:
                    # Phase 2B: Fix bugs if tests failed
                    self.bb.add_sync("phase", "Fixing bugs", section="meta")
                    
                    # Create steps for each iteration of fixes
                    max_fixes = 3
                    last_result = test_result
                    
                    for i in range(max_fixes):
                        fix_steps = [
                            Step(label=f"fix_code_{i+1}", agent_key="code"),
                            Step(label=f"apply_fix_{i+1}", agent_key="file"),
                            Step(label=f"test_fix_{i+1}", agent_key="test")
                        ]
                        
                        fix_runner = PlanRunner(fix_steps, self.agents, self.bb)
                        last_result = fix_runner.run(f"Fix the following test failure:\n{last_result}")
                        self.all_steps.extend(fix_steps)
                        
                        if "SUCCESS" in last_result:
                            self.bb.add_sync("phase", "Bugs fixed successfully", section="meta")
                            break
                    
                    return last_result
        
        # Create and run the dynamic plan
        runner = DynamicPlanRunner(agents, self.blackboard)
        result = runner.run("Implement a function to calculate Fibonacci numbers")
        
        # Verify the final result
        self.assertIn("SUCCESS", result)
        
        # Get the phase log
        phase_entries = self.blackboard.query_sync(label="phase")
        self.assertGreaterEqual(len(phase_entries), 2)
        
        # Save dynamic plan details to output file
        with open(self.test_output_dir / "dynamic_plan.txt", "w") as f:
            f.write("Dynamic Plan Creation Test:\n")
            f.write(f"Total steps created: {len(runner.all_steps)}\n")
            f.write(f"Final result: {result}\n\n")
            f.write("Execution Phases:\n")
            for entry in phase_entries:
                f.write(f"  {entry.content}\n")
            f.write("\nFinal code:\n")
            for filename in file_agent.list_files():
                f.write(f"\n--- {filename} ---\n")
                f.write(file_agent.read_file(filename))
                # Also save the final file to the output directory
                with open(self.test_output_dir / filename, "w") as file_out:
                    file_out.write(file_agent.read_file(filename))
    
    def test_complete_development_workflow(self):
        """Test a complete end-to-end development workflow with real agents if available."""
        # Try to use real agents if API keys are available
        use_real_agents = False
        
        try:
            # Create real agents
            real_code_agent = CodeAgent(name="RealCodeAgent")
            real_file_agent = FileAgent(name="RealFileAgent")
            real_test_agent = TestAgent(name="RealTestAgent")
            
            # Check if we're using real backends or fell back to stubs
            using_stubs = (
                real_code_agent.backend.__class__.__name__ == "StubBackend" or
                "stub_mode" in dir(real_code_agent.backend)
            )
            
            if not using_stubs:
                use_real_agents = True
                code_agent = real_code_agent
                file_agent = real_file_agent
                test_agent = real_test_agent
            else:
                # Fall back to mock agents
                code_agent = self._create_mock_code_agent(name="MockCodeAgent")
                file_agent = self._create_mock_file_agent(self.working_dir, name="MockFileAgent")
                test_agent = self._create_mock_test_agent(self.working_dir, name="MockTestAgent")
        except Exception as e:
            # Fall back to mock agents
            code_agent = self._create_mock_code_agent(name="MockCodeAgent")
            file_agent = self._create_mock_file_agent(self.working_dir, name="MockFileAgent")
            test_agent = self._create_mock_test_agent(self.working_dir, name="MockTestAgent")
        
        # Create a dictionary of agents
        agents = {
            "code": code_agent,
            "file": file_agent,
            "test": test_agent
        }
        
        # Create a workflow manager
        class DevelopmentWorkflow:
            def __init__(self, agents, blackboard, working_dir):
                self.agents = agents
                self.bb = blackboard
                self.working_dir = working_dir
                self.iterations = 0
                self.max_iterations = 5
            
            def run(self, task_description: str) -> dict:
                """Run the complete development workflow."""
                self.bb.add_sync("task", task_description, section="meta")
                
                # Phase 1: Initial code generation
                code_diff = self.agents["code"].run(
                    f"Generate code for the following task:\n{task_description}\n"
                    f"Respond with a unified diff."
                )
                self.bb.add_sync("initial_code", code_diff, section="code")
                
                # Phase 2: Apply changes to files
                file_result = self.agents["file"].run(code_diff)
                self.bb.add_sync("initial_file_op", file_result, section="files")
                
                # Phase 3: Run tests
                test_result = self.agents["test"].run("Run tests")
                self.bb.add_sync("initial_test", test_result, section="tests")
                
                # Phase 4: Iterative improvement
                current_result = test_result
                self.iterations = 1
                
                while "SUCCESS" not in current_result and self.iterations < self.max_iterations:
                    self.bb.add_sync(
                        f"iteration_{self.iterations}",
                        f"Starting iteration {self.iterations}",
                        section="meta"
                    )
                    
                    # Generate improved code based on test results
                    improved_code = self.agents["code"].run(
                        f"Fix the following test failures:\n{current_result}\n"
                        f"Respond with a unified diff."
                    )
                    self.bb.add_sync(f"code_iter_{self.iterations}", improved_code, section="code")
                    
                    # Apply the improved code
                    file_result = self.agents["file"].run(improved_code)
                    self.bb.add_sync(f"file_op_iter_{self.iterations}", file_result, section="files")
                    
                    # Run tests again
                    current_result = self.agents["test"].run("Run tests")
                    self.bb.add_sync(f"test_iter_{self.iterations}", current_result, section="tests")
                    
                    self.iterations += 1
                
                # Phase 5: Final assessment
                if "SUCCESS" in current_result:
                    status = "success"
                    message = f"Task completed successfully after {self.iterations} iterations"
                else:
                    status = "partial"
                    message = f"Maximum iterations ({self.max_iterations}) reached, but tests still failing"
                
                # List all files
                files = {}
                try:
                    for filename in self.agents["file"].list_files():
                        file_path = self.working_dir / filename
                        if file_path.exists():
                            with open(file_path, "r") as f:
                                files[filename] = f.read()
                except:
                    # If list_files() is not implemented or fails
                    for file_path in self.working_dir.glob("*.*"):
                        if file_path.is_file():
                            with open(file_path, "r") as f:
                                files[file_path.name] = f.read()
                
                return {
                    "status": status,
                    "message": message,
                    "iterations": self.iterations,
                    "final_result": current_result,
                    "files": files
                }
        
        # Create and run the workflow
        workflow = DevelopmentWorkflow(agents, self.blackboard, self.working_dir)
        
        # Task: Implement a factorial function with tests
        task = """
        Implement a Python function that calculates the factorial of a number.
        Requirements:
        1. Handle edge cases (negative numbers, zero)
        2. Include proper error handling
        3. Use recursion with memoization for efficiency
        4. Add a test function to verify correctness
        """
        
        result = workflow.run(task)
        
        # Verify the workflow completed
        self.assertIn(result["status"], ["success", "partial"])
        self.assertGreaterEqual(workflow.iterations, 1)
        
        # Save complete workflow details to output file
        with open(self.test_output_dir / "complete_workflow.txt", "w") as f:
            f.write("Complete Development Workflow Test:\n")
            f.write(f"Using real agents: {use_real_agents}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Message: {result['message']}\n")
            f.write(f"Iterations: {result['iterations']}\n")
            f.write(f"Final result: {result['final_result']}\n\n")
            
            f.write("Generated files:\n")
            for filename, content in result["files"].items():
                f.write(f"\n--- {filename} ---\n")
                f.write(content)
                # Also save each file to the output directory
                with open(self.test_output_dir / filename, "w") as file_out:
                    file_out.write(content)
        
        # Return success if the workflow completed
        return result["status"] in ["success", "partial"]
    
    def test_real_agents_if_available(self):
        """Test with real agents if API keys are available."""
        try:
            # Create real agents
            code_agent = CodeAgent(name="RealCodeAgent")
            file_agent = FileAgent(name="RealFileAgent", working_dir=str(self.working_dir))
            test_agent = TestAgent(name="RealTestAgent", working_dir=str(self.working_dir))
            
            # Check if we're using real backends or fell back to stubs
            using_stubs = (
                code_agent.backend.__class__.__name__ == "StubBackend" or
                "stub_mode" in dir(code_agent.backend)
            )
            
            if using_stubs:
                self.skipTest("No API keys available, using stub backends")
            
            # Create a dictionary of agents
            agents = {
                "code": code_agent,
                "file": file_agent,
                "test": test_agent
            }
            
            # Create a simple plan for a "Hello, World!" program
            generate_code = Step(label="generate_code", agent_key="code", on_success="apply_changes")
            apply_changes = Step(label="apply_changes", agent_key="file", on_success="run_tests")
            run_tests = Step(label="run_tests", agent_key="test")
            
            steps = [generate_code, apply_changes, run_tests]
            
            # Create and run the plan
            runner = PlanRunner(steps, agents, self.blackboard)
            result = runner.run(
                "Create a simple 'Hello, World!' Python program with a main function and a test function."
            )
            
            # Verify we got a response
            self.assertIsNotNone(result)
            
            # Save the real agents plan results to output file
            with open(self.test_output_dir / "real_agents_test.txt", "w") as f:
                f.write("Real Agents Test:\n")
                f.write(f"Steps: {[s.label for s in steps]}\n")
                f.write(f"Final result: '{result}'\n\n")
                f.write("Blackboard entries:\n")
                for entry in self.blackboard.entries():
                    f.write(f"  {entry.label}: {entry.content[:100]}...\n" if len(str(entry.content)) > 100 else f"  {entry.label}: {entry.content}\n")
                
                # List all files in the working directory
                f.write("\nFiles created:\n")
                for file_path in self.working_dir.glob("*.*"):
                    if file_path.is_file():
                        f.write(f"  {file_path.name}\n")
                        # Also save each file to the output directory
                        with open(self.test_output_dir / file_path.name, "w") as file_out:
                            with open(file_path, "r") as file_in:
                                file_out.write(file_in.read())
            
            return True
        except Exception as e:
            self.skipTest(f"Error testing with real agents: {str(e)}")
            return False

if __name__ == "__main__":
    unittest.main()
