# Talk Framework Execution Plans (List[Step])

## Step Class Definition
```python
@dataclass(slots=True)
class Step:
    label: Optional[str] = None        # Step identifier for jumps
    agent_key: str = ""                 # Which agent executes this
    on_success: Optional[str] = None    # Jump to label on success
    on_fail: Optional[str] = None       # Jump to label on failure
    steps: List["Step"] = []            # Serial sub-steps
    parallel_steps: List["Step"] = []   # Parallel sub-steps
```

## Talk v10 Plan (Simplified/Current)

The most streamlined plan with fast-path optimization:

```python
def _create_plan(self) -> List[Step]:
    steps = [
        Step(
            label="analyze",
            agent_key="planning",
            on_success="implement",
            on_fail="complete"
        ),
        Step(
            label="implement",
            agent_key="code",  # or "refinement" if enabled
            on_success="persist",
            on_fail="complete"
        ),
        Step(
            label="persist",
            agent_key=None,  # Handled by file manager
            on_success="complete",
            on_fail="complete"
        ),
        Step(
            label="complete",
            agent_key=None,
            on_success=None,
            on_fail=None
        )
    ]
```

**Flow**: `analyze → implement → persist → complete`

**Issues with this plan**:
- Only ONE code generation step
- No iteration or refinement loop
- No validation or testing
- Linear, no branching logic

## Talk v3/v4 Plan (With Branching)

More sophisticated with dynamic branching:

```python
steps = [
    # Planning checkpoint
    Step(
        label="plan_next",
        agent_key="planning",
        on_success="select_action",
        on_fail="error_recovery"
    ),
    
    # Branch selection - DYNAMIC
    Step(
        label="select_action",
        agent_key="branching",
        on_success="dynamic",  # Set by BranchingAgent at runtime
        on_fail="error_recovery"
    ),
    
    # Work steps (branching targets)
    Step(
        label="generate_code",
        agent_key="code",
        on_success="plan_next",  # Loop back to planning
        on_fail="plan_next"
    ),
    Step(
        label="apply_files",
        agent_key="file",
        on_success="plan_next",
        on_fail="plan_next"
    ),
    Step(
        label="run_tests",
        agent_key="test",
        on_success="plan_next",
        on_fail="plan_next"
    ),
    
    # Terminal states
    Step(
        label="complete",
        agent_key=None,
        on_success=None,
        on_fail=None
    ),
    Step(
        label="error_recovery",
        agent_key="planning",
        on_success="select_action",
        on_fail="manual_intervention"
    )
]
```

**Flow**: `plan_next → select_action → [generate_code|apply_files|run_tests] → plan_next → ...`

**Key Feature**: BranchingAgent dynamically sets `on_success` to jump to different steps

## Talk v5 Plan (With Memory)

Adds memory retrieval at the start:

```python
steps = [
    # Memory retrieval (NEW)
    Step(
        label="retrieve_memories",
        agent_key="reminiscing",
        on_success="memory_aware_planning",
        on_fail="memory_aware_planning"
    ),
    
    # Memory-aware planning
    Step(
        label="memory_aware_planning",
        agent_key="planning",
        on_success="select_action",
        on_fail="error_recovery"
    ),
    
    # ... rest same as v3/v4 ...
]
```

## Original Talk Plan (Standard)

The classic generate-apply-test loop:

```python
steps = [
    Step(
        label="generate_code",
        agent_key="code",
        on_success="apply_changes",
        on_fail="complete"
    ),
    Step(
        label="apply_changes", 
        agent_key="file",
        on_success="run_tests",
        on_fail="complete"
    ),
    Step(
        label="run_tests",
        agent_key="test",
        on_success="check_results",
        on_fail="complete"
    ),
    Step(
        label="check_results",
        agent_key="validation",
        on_success="complete",
        on_fail="generate_code"  # Loop on test failure
    ),
    Step(
        label="complete",
        agent_key=None,
        on_success=None,
        on_fail=None
    )
]
```

**Flow**: `generate_code → apply_changes → run_tests → check_results → [complete OR loop]`

## What's Missing for Large Code Generation

### 1. No Multi-Component Steps
Current plans have ONE code generation step. Need:
```python
Step(
    label="generate_components",
    agent_key=None,
    steps=[  # Serial sub-steps
        Step(label="gen_storage", agent_key="code"),
        Step(label="gen_cache", agent_key="code"),
        Step(label="gen_api", agent_key="code"),
        # ... 10+ more components
    ]
)
```

### 2. No Parallel Generation
Could use `parallel_steps` for concurrent generation:
```python
Step(
    label="generate_all",
    agent_key=None,
    parallel_steps=[
        Step(label="gen_models", agent_key="code"),
        Step(label="gen_utils", agent_key="code"),
        Step(label="gen_tests", agent_key="code"),
    ]
)
```

### 3. No Context Accumulation
Each step is isolated. Need:
```python
Step(
    label="generate_with_context",
    agent_key="context_aware_code",
    # Agent should see all previous components
)
```

### 4. No Iterative Refinement
Missing:
```python
Step(
    label="refine_until_complete",
    agent_key="refinement",
    on_success="check_completeness",
    on_fail="refine_until_complete"  # Self-loop
)
```

## Ideal Plan for Large Projects

```python
steps = [
    # Phase 1: Comprehensive Planning
    Step(
        label="create_detailed_plan",
        agent_key="comprehensive_planner",
        on_success="validate_plan",
        on_fail="retry_planning"
    ),
    
    # Phase 2: Parallel Component Generation
    Step(
        label="generate_all_components",
        agent_key=None,
        parallel_steps=[
            Step("gen_core", "code_agent"),
            Step("gen_data", "code_agent"),
            Step("gen_api", "code_agent"),
            Step("gen_utils", "code_agent"),
            Step("gen_tests", "code_agent"),
        ],
        on_success="integrate_components"
    ),
    
    # Phase 3: Integration
    Step(
        label="integrate_components",
        agent_key="integration_agent",
        on_success="validate_system",
        on_fail="fix_integration"
    ),
    
    # Phase 4: Validation & Refinement
    Step(
        label="validate_system",
        agent_key="validation_agent",
        on_success="run_tests",
        on_fail="refine_components"
    ),
    
    # Phase 5: Testing
    Step(
        label="run_tests",
        agent_key="test",
        on_success="check_quality",
        on_fail="fix_failures"
    ),
    
    # Phase 6: Quality Check
    Step(
        label="check_quality",
        agent_key="quality_agent",
        on_success="complete",
        on_fail="refine_components"
    ),
    
    # Refinement Loop
    Step(
        label="refine_components",
        agent_key="refinement",
        on_success="validate_system",
        on_fail="manual_intervention"
    )
]
```

## Summary

Current Talk plans are designed for **single-file, iterative development**, not large-scale code generation. They lack:

1. **Multiple code generation steps** - Only one "generate_code" step
2. **Parallel execution** - Everything is sequential
3. **Context accumulation** - Each step starts fresh
4. **Component awareness** - No concept of multi-file projects
5. **Comprehensive planning** - Planning agent gives single next action

The v11/v12 attempts bypass the plan structure entirely, manually iterating in the orchestrator instead of using the Step system's capabilities.