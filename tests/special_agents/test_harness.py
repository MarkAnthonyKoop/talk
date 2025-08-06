#!/usr/bin/env python3
"""
Test Harness for Talk Framework Agents

This harness validates that agents follow the core architectural principles:
1. Prompt in → Completion out
2. Use LLM via call_ai() 
3. Handle errors gracefully
4. Support inter-agent communication via .talk_scratch
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.agent import Agent
from plan_runner.step import Step

log = logging.getLogger(__name__)


class AgentTestHarness:
    """Base test harness for validating Talk agents."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
        self.mock_llm_responses = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log test progress."""
        if self.verbose:
            print(f"[{level}] {message}")
            
    def add_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        
        status = "✓ PASS" if passed else "✗ FAIL"
        self.log(f"{status}: {test_name} {details}")
    
    def test_agent_contract(self, agent: Agent, agent_name: str) -> bool:
        """Test that an agent follows the basic contract."""
        self.log(f"\nTesting {agent_name} contract compliance...")
        
        all_passed = True
        
        # Test 1: Has run method
        try:
            assert hasattr(agent, 'run'), "Agent missing run() method"
            assert callable(agent.run), "run() is not callable"
            self.add_test_result(f"{agent_name}.run() exists", True)
        except AssertionError as e:
            self.add_test_result(f"{agent_name}.run() exists", False, str(e))
            all_passed = False
        
        # Test 2: Has call_ai method (from parent Agent)
        try:
            assert hasattr(agent, 'call_ai'), "Agent missing call_ai() method"
            self.add_test_result(f"{agent_name}.call_ai() exists", True)
        except AssertionError as e:
            self.add_test_result(f"{agent_name}.call_ai() exists", False, str(e))
            all_passed = False
        
        # Test 3: Returns string from run()
        try:
            with patch.object(agent, 'call_ai', return_value="Test completion"):
                result = agent.run("Test input")
                assert isinstance(result, str), f"run() returned {type(result)}, expected str"
                self.add_test_result(f"{agent_name}.run() returns string", True)
        except Exception as e:
            self.add_test_result(f"{agent_name}.run() returns string", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_agent_uses_llm(self, agent: Agent, agent_name: str, test_input: str = "Test task") -> bool:
        """Test that agent actually uses the LLM."""
        self.log(f"\nTesting {agent_name} LLM usage...")
        
        # Mock call_ai to track if it's called
        call_ai_called = False
        original_call_ai = agent.call_ai
        
        def mock_call_ai():
            nonlocal call_ai_called
            call_ai_called = True
            return f"Mock LLM response for {agent_name}"
        
        try:
            agent.call_ai = mock_call_ai
            result = agent.run(test_input)
            
            # Check if agent is expected to use LLM
            # FileAgent and TestAgent might not use LLM
            if agent_name in ["FileAgent", "TestAgent"]:
                self.add_test_result(f"{agent_name} LLM usage", True, "Optional for this agent")
                return True
            
            assert call_ai_called, f"{agent_name} did not call call_ai()"
            self.add_test_result(f"{agent_name} uses LLM", True)
            return True
            
        except AssertionError as e:
            self.add_test_result(f"{agent_name} uses LLM", False, str(e))
            return False
        finally:
            agent.call_ai = original_call_ai
    
    def test_planning_agent(self) -> bool:
        """Test PlanningAgent specifically."""
        self.log("\n=== Testing PlanningAgent ===")
        
        from special_agents.planning_agent import PlanningAgent
        
        agent = PlanningAgent()
        
        # Basic contract test
        if not self.test_agent_contract(agent, "PlanningAgent"):
            return False
        
        # Test with mock LLM response
        mock_response = json.dumps({
            "todo_hierarchy": "[ ] Test task\n    [ ] Step 1\n    [ ] Step 2",
            "analysis": {
                "situation": "Starting new task",
                "action_needed": "generate_code",
                "confidence": "high"
            },
            "next_action": "generate_code",
            "recommendation": "Start by generating code"
        })
        
        with patch.object(agent, 'call_ai', return_value=mock_response):
            result = agent.run(json.dumps({
                "task_description": "Write hello world",
                "blackboard_state": {},
                "last_action": "",
                "last_result": ""
            }))
            
            # Verify it returns JSON-like completion
            try:
                parsed = json.loads(result)
                assert "next_action" in parsed, "Missing next_action in response"
                assert parsed["next_action"] in ["generate_code", "apply_files", "run_tests", "complete", "research", "error_recovery"]
                self.add_test_result("PlanningAgent output structure", True)
            except (json.JSONDecodeError, AssertionError) as e:
                self.add_test_result("PlanningAgent output structure", False, str(e))
                return False
        
        return True
    
    def test_branching_agent(self) -> bool:
        """Test BranchingAgent specifically."""
        self.log("\n=== Testing BranchingAgent ===")
        
        from special_agents.branching_agent import BranchingAgent
        
        # Create mock Step and plan
        step = Step(label="select_action", agent_key="branching")
        plan = [
            Step(label="plan_next", agent_key="planning"),
            step,  # The branching step itself
            Step(label="generate_code", agent_key="code"),
            Step(label="apply_files", agent_key="file"),
            Step(label="complete", agent_key=None)
        ]
        
        agent = BranchingAgent(step=step, plan=plan)
        
        # Basic contract test
        if not self.test_agent_contract(agent, "BranchingAgent"):
            return False
        
        # Test with mock LLM response
        mock_response = """Based on the planning recommendation to generate code,
        I'll select the generate_code step.
        
        SELECTED: generate_code"""
        
        with patch.object(agent, 'call_ai', return_value=mock_response):
            planning_input = json.dumps({
                "next_action": "generate_code",
                "recommendation": "Start by generating code"
            })
            
            result = agent.run(planning_input)
            
            # Verify it modified the step
            assert step.on_success == "generate_code", f"Step not modified correctly: {step.on_success}"
            self.add_test_result("BranchingAgent modifies step", True)
        
        return True
    
    def test_code_agent(self) -> bool:
        """Test CodeAgent specifically."""
        self.log("\n=== Testing CodeAgent ===")
        
        from special_agents.code_agent import CodeAgent
        
        agent = CodeAgent()
        
        # Basic contract test
        if not self.test_agent_contract(agent, "CodeAgent"):
            return False
        
        # Test with mock LLM response
        mock_response = """I'll create a simple hello world function.

```python
# filename: hello.py
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
```

This code should be saved as hello.py."""
        
        with patch.object(agent, 'call_ai', return_value=mock_response):
            result = agent.run("Write a hello world function")
            
            # Verify it returns code
            assert "```" in result, "No code block in response"
            assert "hello" in result.lower(), "No hello world content"
            self.add_test_result("CodeAgent generates code", True)
        
        return True
    
    def test_agent_chain(self) -> bool:
        """Test a chain of agents working together."""
        self.log("\n=== Testing Agent Chain (Planning → Branching → Code) ===")
        
        from special_agents.planning_agent import PlanningAgent
        from special_agents.branching_agent import BranchingAgent
        from special_agents.code_agent import CodeAgent
        
        # Create agents
        planning = PlanningAgent()
        
        step = Step(label="select_action", agent_key="branching")
        plan = [
            Step(label="generate_code", agent_key="code"),
            Step(label="apply_files", agent_key="file"),
            Step(label="complete", agent_key=None)
        ]
        branching = BranchingAgent(step=step, plan=plan)
        
        code = CodeAgent()
        
        # Mock responses
        planning_response = json.dumps({
            "todo_hierarchy": "[ ] Create hello world",
            "next_action": "generate_code",
            "recommendation": "Start with code generation"
        })
        
        branching_response = "I'll select generate_code.\n\nSELECTED: generate_code"
        
        code_response = """```python
def hello_world():
    print("Hello, World!")
```"""
        
        # Test the chain
        with patch.object(planning, 'call_ai', return_value=planning_response):
            planning_output = planning.run('{"task_description": "hello world"}')
            assert "generate_code" in planning_output
            self.add_test_result("Chain: Planning outputs", True)
        
        with patch.object(branching, 'call_ai', return_value=branching_response):
            branching_output = branching.run(planning_output)
            assert step.on_success == "generate_code"
            self.add_test_result("Chain: Branching selects", True)
        
        with patch.object(code, 'call_ai', return_value=code_response):
            code_output = code.run("Generate hello world based on plan")
            assert "hello_world" in code_output
            self.add_test_result("Chain: Code generates", True)
        
        return True
    
    def test_scratch_communication(self) -> bool:
        """Test .talk_scratch directory communication."""
        self.log("\n=== Testing Scratch Directory Communication ===")
        
        from special_agents.planning_agent import PlanningAgent
        from special_agents.code_agent import CodeAgent
        
        # Use temp directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                planning = PlanningAgent()
                code = CodeAgent()
                
                # Planning saves to scratch
                mock_response = json.dumps({
                    "next_action": "generate_code",
                    "todo_hierarchy": "[ ] Test"
                })
                
                with patch.object(planning, 'call_ai', return_value=mock_response):
                    planning.run('{"task_description": "test"}')
                
                # Check scratch file was created
                scratch_file = Path(".talk_scratch/latest_planning.json")
                assert scratch_file.exists(), "Planning didn't create scratch file"
                
                # Code agent should be able to read it
                with open(scratch_file) as f:
                    data = json.load(f)
                    assert data["next_action"] == "generate_code"
                
                self.add_test_result("Scratch communication", True)
                return True
                
            except AssertionError as e:
                self.add_test_result("Scratch communication", False, str(e))
                return False
            finally:
                os.chdir(old_cwd)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all agent tests."""
        self.log("\n" + "="*60)
        self.log("TALK AGENT TEST HARNESS")
        self.log("="*60)
        
        # Run individual agent tests
        planning_ok = self.test_planning_agent()
        branching_ok = self.test_branching_agent()
        code_ok = self.test_code_agent()
        
        # Run integration tests
        chain_ok = self.test_agent_chain()
        scratch_ok = self.test_scratch_communication()
        
        # Summary
        self.log("\n" + "="*60)
        self.log("TEST SUMMARY")
        self.log("="*60)
        
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = sum(1 for r in self.test_results if not r["passed"])
        
        self.log(f"\nTotal: {len(self.test_results)} tests")
        self.log(f"Passed: {passed} ✓")
        self.log(f"Failed: {failed} ✗")
        
        if failed > 0:
            self.log("\nFailed tests:", "ERROR")
            for r in self.test_results:
                if not r["passed"]:
                    self.log(f"  - {r['test']}: {r['details']}", "ERROR")
        
        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.test_results),
            "all_passed": failed == 0,
            "results": self.test_results
        }


def main():
    """Run the test harness."""
    harness = AgentTestHarness(verbose=True)
    results = harness.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()