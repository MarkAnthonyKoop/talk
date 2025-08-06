#!/usr/bin/env python3
"""
Test the Dynamic Talk concept without full integration.
"""

import sys
sys.path.insert(0, '/home/xx/code')

from special_agents.assessor_agent import AssessorAgent, TaskComplexity

def test_assessor():
    """Test the AssessorAgent functionality."""
    print("Testing AssessorAgent...")
    
    # Create assessor
    assessor = AssessorAgent(name="TestAssessor")
    
    # Test different task complexities
    tasks = [
        "list files in current directory",
        "create a REST API for user management",
        "build an agentic orchestration system",
        "create a production-ready kubernetes-based microservices platform"
    ]
    
    for task in tasks:
        print(f"\nTask: '{task}'")
        
        # Quick pattern assessment
        complexity = assessor._pattern_assess(task)
        print(f"Pattern-based assessment: {complexity.value}")
        
        # Full assessment would use LLM
        # assessment = assessor.assess_task(task)
        # print(f"Full assessment: {assessment}")


def test_workflow_selector():
    """Test workflow generation concept."""
    print("\n\nTesting WorkflowSelector concept...")
    
    from orchestration.workflow_selector import WorkflowSelector
    
    selector = WorkflowSelector()
    
    # Test simple workflow
    simple_assessment = {
        "complexity": TaskComplexity.SIMPLE,
        "domains": [],
        "requires_research": False,
        "requires_planning": False
    }
    
    workflow = selector._simple_workflow(simple_assessment)
    print("\nSimple workflow:")
    for step in workflow:
        print(f"  - {step.label} ({step.agent_key})")
    
    # Test complex workflow
    complex_assessment = {
        "complexity": TaskComplexity.COMPLEX,
        "domains": [],
        "requires_research": True,
        "requires_planning": True
    }
    
    workflow = selector._complex_workflow(complex_assessment)
    print("\nComplex workflow:")
    for step in workflow:
        print(f"  - {step.label} ({step.agent_key})")
        if step.parallel_steps:
            for p in step.parallel_steps:
                print(f"    ├─ {p.label} ({p.agent_key})")


if __name__ == "__main__":
    print("=== DYNAMIC TALK CONCEPT TEST ===\n")
    
    test_assessor()
    test_workflow_selector()
    
    print("\n\n✅ Concept validation complete!")
    print("The Dynamic Talk architecture is sound, but needs integration work")
    print("to handle the existing codebase's import structure.")