# RefinementAgent

The RefinementAgent encapsulates the iterative code development cycle, managing the process of generating code, applying it, testing it, and refining based on test results until the code meets quality standards.

## Overview

The RefinementAgent coordinates between three sub-agents:
- **CodeAgent**: Generates and improves code
- **FileAgent**: Applies code changes to the filesystem
- **TestAgent**: Runs tests and validates implementations

## Key Features

- **Iterative Refinement**: Automatically loops through code->test->evaluate->refine cycles
- **Quality Gates**: Uses LLM evaluation to determine when code meets standards
- **Maximum Iterations**: Prevents infinite loops with configurable max attempts
- **Structured Results**: Returns detailed JSON with status, iterations, and improvements made

## Usage

```python
from special_agents.refinement_agent import RefinementAgent

agent = RefinementAgent(
    base_dir="/path/to/workspace",
    max_iterations=5
)

result_json = agent.run("Create a function that calculates factorial")
result = json.loads(result_json)

print(f"Status: {result['status']}")
print(f"Iterations: {result['iterations']}")
```

## Return Values

The agent returns JSON with these fields:
- `status`: "success", "needs_improvement", "failed", or "max_iterations" 
- `iterations`: Number of refinement cycles performed
- `final_output`: The last code generation output
- `test_results`: Results from the final test run
- `improvements_made`: List of improvements attempted

## Architecture

The RefinementAgent implements a finite state machine:

1. **Generate**: CodeAgent creates or improves code
2. **Apply**: FileAgent writes code to filesystem
3. **Test**: TestAgent runs validation tests
4. **Evaluate**: LLM analyzes test results for success/failure
5. **Decide**: Continue refining or declare success/failure

This encapsulation allows the higher-level orchestration to treat iterative development as a single atomic operation.