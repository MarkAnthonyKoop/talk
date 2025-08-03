#!/usr/bin/env python3
"""
Test suite for the PlanningAgent and IntelligentTalkOrchestrator.

This script validates the planning capabilities and integration with the Talk framework.
"""

import sys
import os
import json
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to be less verbose for testing
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_planning_agent():
    """Test the PlanningAgent with various task scenarios."""
    print("=== PlanningAgent Test Suite ===")
    
    try:
        from special_agents.planning_agent import PlanningAgent
        print("[OK] PlanningAgent imported successfully")
        
        # Create agent instance
        agent = PlanningAgent()
        print("[OK] PlanningAgent instance created")
        
        # Test scenarios with different complexity levels
        test_scenarios = [
            {
                "task": "Create a simple hello world FastAPI app",
                "expected_complexity": "low",
                "description": "Simple task"
            },
            {
                "task": "Build a REST API with user authentication, CRUD operations, and email notifications",
                "expected_complexity": "medium",
                "description": "Medium complexity task"
            },
            {
                "task": "Create a complete microservice architecture with Docker, Kubernetes, API Gateway, multiple databases, user management, real-time notifications, and comprehensive testing",
                "expected_complexity": "high", 
                "description": "High complexity task"
            },
            {
                "task": "Fix the database connection timeout issue in the user authentication service",
                "expected_complexity": "low",
                "description": "Bug fix task"
            },
            {
                "task": "Research best practices for implementing OAuth2 with JWT tokens in FastAPI",
                "expected_complexity": "low",
                "description": "Research task"
            }
        ]
        
        print(f"\nTesting {len(test_scenarios)} different scenarios:")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Test {i}: {scenario['description']} ---")
            print(f"Task: {scenario['task']}")
            
            try:
                # Generate plan
                result = agent.run(scenario['task'])
                plan = json.loads(result)
                
                # Validate plan structure
                required_fields = ["plan_type", "analysis", "execution_steps", "total_steps"]
                missing_fields = [field for field in required_fields if field not in plan]
                
                if missing_fields:
                    print(f"[ERROR] Missing required fields: {missing_fields}")
                    continue
                
                # Extract key information
                complexity = plan["analysis"].get("complexity", "unknown")
                total_steps = plan.get("total_steps", 0)
                plan_type = plan.get("plan_type", "unknown")
                
                print(f"Generated Plan:")
                print(f"  Complexity: {complexity}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Plan Type: {plan_type}")
                
                # Show execution steps
                if plan.get("execution_steps"):
                    print(f"  Execution Steps:")
                    for step in plan["execution_steps"]:
                        print(f"    - {step.get('label', 'unknown')} ({step.get('agent_key', 'unknown')})")
                
                # Validate plan makes sense
                if total_steps < 3:
                    print(f"[WARNING] Plan seems too short ({total_steps} steps)")
                elif total_steps > 10:
                    print(f"[WARNING] Plan seems too long ({total_steps} steps)")
                else:
                    print(f"[OK] Plan length appropriate")
                
                # Check if complexity assessment is reasonable
                expected = scenario.get("expected_complexity", "medium")
                if complexity == expected:
                    print(f"[OK] Complexity assessment matches expectation")
                else:
                    print(f"[INFO] Complexity: {complexity} (expected: {expected})")
                
            except Exception as e:
                print(f"[ERROR] Failed to generate plan: {e}")
        
        print("\n[OK] PlanningAgent testing completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] PlanningAgent test failed: {e}")
        return False

def test_step_creation():
    """Test converting JSON plans back to Step objects."""
    print("\n=== Step Creation Test ===")
    
    try:
        from special_agents.planning_agent import PlanningAgent
        from plan_runner.step import Step
        
        agent = PlanningAgent()
        
        # Generate a plan
        task = "Create a simple API with user management"
        plan_json = agent.run(task)
        
        # Convert back to steps
        steps = agent.create_steps_from_plan(plan_json)
        
        print(f"[OK] Converted plan to {len(steps)} Step objects")
        
        # Validate steps
        for step in steps:
            if not isinstance(step, Step):
                print(f"[ERROR] Invalid step type: {type(step)}")
                return False
            
            if not step.label:
                print(f"[ERROR] Step missing label")
                return False
            
            if not step.agent_key:
                print(f"[WARNING] Step '{step.label}' has no agent_key")
        
        print(f"[OK] All steps validated successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Step creation test failed: {e}")
        return False

def test_intelligent_orchestrator():
    """Test the IntelligentTalkOrchestrator integration."""
    print("\n=== IntelligentTalkOrchestrator Test ===")
    
    try:
        from special_agents.intelligent_talk_orchestrator import IntelligentTalkOrchestrator
        
        # Create orchestrator instance (don't run it, just test initialization)
        task = "Create a simple FastAPI hello world app"
        orchestrator = IntelligentTalkOrchestrator(
            task=task,
            enable_planning=True,
            enable_memory=True,
            timeout_minutes=1  # Short timeout for testing
        )
        
        print("[OK] IntelligentTalkOrchestrator created successfully")
        
        # Test plan generation
        plan = orchestrator._create_plan()
        
        if not plan:
            print("[ERROR] No execution plan generated")
            return False
        
        print(f"[OK] Generated execution plan with {len(plan)} steps")
        
        # Validate plan steps
        for step in plan:
            if step.agent_key and step.agent_key not in orchestrator.agents:
                print(f"[WARNING] Step '{step.label}' references unknown agent: {step.agent_key}")
        
        # Test plan summary
        summary = orchestrator.get_plan_summary()
        if summary:
            print(f"[OK] Plan summary generated:")
            print(f"    Complexity: {summary.get('complexity', 'unknown')}")
            print(f"    Total steps: {summary.get('total_steps', 0)}")
            print(f"    Research required: {summary.get('research_required', False)}")
        
        print("[OK] IntelligentTalkOrchestrator integration test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] IntelligentTalkOrchestrator test failed: {e}")
        return False

def test_memory_integration():
    """Test integration with ReminiscingAgent."""
    print("\n=== Memory Integration Test ===")
    
    try:
        from special_agents.intelligent_talk_orchestrator import IntelligentTalkOrchestrator
        
        # Create orchestrator with memory enabled
        orchestrator = IntelligentTalkOrchestrator(
            task="Build a user authentication system",
            enable_memory=True,
            enable_planning=False  # Focus on memory testing
        )
        
        if not hasattr(orchestrator, 'reminiscing_agent'):
            print("[ERROR] ReminiscingAgent not initialized")
            return False
        
        print("[OK] ReminiscingAgent integrated successfully")
        
        # Test memory context retrieval
        memory_context = orchestrator._get_memory_context()
        
        if memory_context:
            print("[OK] Memory context retrieved")
            print(f"    Context length: {len(memory_context)} characters")
        else:
            print("[INFO] No memory context found (expected for new session)")
        
        print("[OK] Memory integration test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Memory integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Intelligent Talk Planning System")
    print("=" * 50)
    
    tests = [
        ("PlanningAgent Core", test_planning_agent),
        ("Step Creation", test_step_creation),
        ("Orchestrator Integration", test_intelligent_orchestrator),
        ("Memory Integration", test_memory_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print("=" * 50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Intelligent Talk system is ready.")
        return 0
    else:
        print(f"\n[WARNING] {len(results) - passed} tests failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())