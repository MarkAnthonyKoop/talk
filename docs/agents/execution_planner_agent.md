# ExecutionPlannerAgent

The ExecutionPlannerAgent is a critical component of the Talk orchestration system, responsible for generating concrete execution plans as `List[Step]` objects that can be directly executed by the PlanRunner.

## Architecture Role

This agent serves as the bridge between high-level task descriptions and concrete workflow execution:

```
Task Description → ExecutionPlannerAgent → List[Step] → PlanRunner → Execution
```

## Core Responsibility

**CRITICAL**: The ExecutionPlannerAgent MUST return `List[Step]`, not string descriptions. This is fundamental to the Talk orchestration architecture and changing this breaks Pydantic validation.

### Correct Interface
```python
def run(self, input_text: str) -> List[Step]:
    # Analyze task and generate steps
    return [Step(...), Step(...), ...]
```

### Incorrect Interface (Causes Pydantic Errors)
```python
def run(self, input_text: str) -> str:
    # This breaks the orchestration system
    return "Generated plan description"
```

## Workflow Generation

The agent analyzes task requirements and generates optimized Step sequences:

1. **Task Analysis**: Determines complexity and requirements
2. **Pattern Selection**: Chooses appropriate workflow pattern
3. **Step Generation**: Creates concrete Step objects with proper agent assignments
4. **Optimization**: Adds error handling and transition logic

## Available Workflow Patterns

- **Simple**: Basic linear workflows for straightforward tasks
- **Research**: Includes web search for domain knowledge
- **Iterative**: Code→test→refine cycles for complex development
- **Complex**: Multi-stage workflows with branching logic

## Step Generation Example

For a simple Python function task, generates:
1. `Step("reminiscing", label="recall_memories")`
2. `Step("coder", label="generate_code")`
3. `Step("file", label="apply_changes")`
4. `Step("tester", label="run_tests")`
5. `Step("coder", label="check_results")`

## Integration Points

- **Input**: Task descriptions from TalkOrchestrator
- **Output**: List of Step objects for PlanRunner execution
- **Agents**: Coordinates CodeAgent, FileAgent, TestAgent, and others
- **Error Handling**: Returns empty list `[]` on failure, never strings

## Critical Architecture Notes

1. **Never modify return type**: Always return `List[Step]`
2. **Pydantic validation**: The orchestration system validates Step objects
3. **Error handling**: Return empty list on failure, log errors separately
4. **Agent ecosystem**: Must understand available agents and their capabilities

This agent embodies the Talk framework's principle: complex internal logic with simple interface contracts.