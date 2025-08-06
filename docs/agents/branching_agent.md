# BranchingAgent

The BranchingAgent makes intelligent control flow decisions in the orchestration workflow, analyzing current state and results to determine the optimal next step.

## Overview

This agent enables conditional branching and intelligent flow control by analyzing results from previous steps and making decisions about workflow continuation.

## Decision Types

The BranchingAgent can make these decisions:

- **CONTINUE**: Proceed to the next step in linear flow
- **LOOP_REFINEMENT**: Jump back to the refinement cycle
- **COMPLETE**: Task is finished successfully
- **ESCALATE**: Requires human intervention or expert review
- **RESTART**: Start over from planning phase

## Usage

```python
from special_agents.branching_agent import BranchingAgent

agent = BranchingAgent()
decision_json = agent.run(current_workflow_state)
decision = json.loads(decision_json)

print(f"Decision: {decision['decision']}")
print(f"Target: {decision.get('target')}")
print(f"Reason: {decision['reason']}")
```

## Return Format

The agent returns JSON with:
- `decision`: One of the decision types above
- `target`: Label of the step to jump to (if applicable)
- `reason`: Human-readable explanation of the decision
- `confidence`: Confidence level (0.0-1.0) in the decision

## Integration with Orchestration

The BranchingAgent works with labeled Steps in the workflow:

```python
# Workflow with labeled steps for branching
steps = [
    Step("assessor"),
    Step("refinement", label="development_cycle"),
    Step("branching", label="decision_point"),
    Step("file", label="final_apply")
]
```

When the BranchingAgent returns `loop_refinement` with target `development_cycle`, the orchestrator jumps back to that labeled step.

## Decision Logic

The agent uses both heuristic rules and LLM analysis:

1. **Parse Structured Results**: If input contains JSON from RefinementAgent, uses predefined rules
2. **LLM Analysis**: For unstructured input, uses LLM to analyze state and make decisions
3. **Fallback**: Conservative default to continue workflow if analysis fails

This dual approach ensures reliable decisions while maintaining flexibility for complex scenarios.