# Talk v2 Architecture

Talk v2 represents a significant architectural evolution, introducing proper iterative development cycles and intelligent control flow while simplifying the orchestration syntax.

## Key Improvements

### 1. Simplified Step Syntax
```python
# Old way - verbose and repetitive
Step(label="generate_code", agent_key="coder", on_success="apply_changes")

# New way - clean and intuitive  
Step("coder")  # Linear flow by default
Step("refinement", label="dev_cycle")  # Only add labels for branching
```

### 2. Encapsulated Iterative Development
The RefinementAgent encapsulates the entire code->test->refine loop:
- Eliminates the need for explicit file->test->check workflows
- Handles quality gates internally
- Returns structured results for decision making

### 3. Intelligent Control Flow
The BranchingAgent enables conditional branching:
- Analyzes results to determine next action
- Supports looping, completion, escalation
- Makes the orchestration truly dynamic

## Architecture Flow

```
Task Input
    ↓
[Assessor] → Analyze task complexity
    ↓
[Planner] → Generate execution strategy  
    ↓
[Research] → Optional domain research (if needed)
    ↓
[Refinement] → Iterative development cycle
    ↓           (code→file→test→evaluate→refine)
[Branching] → Intelligent flow decision
    ↓
Decision Logic:
├─ COMPLETE → [Final Apply] → Done
├─ LOOP → Jump back to [Refinement]  
├─ ESCALATE → Human review
└─ RESTART → Jump back to [Planner]
```

## Benefits

1. **Cleaner Code**: Simplified Step syntax reduces boilerplate
2. **Better Separation**: Each agent has a clear, focused responsibility
3. **Intelligent Loops**: Dynamic branching based on actual results
4. **Robust Execution**: Proper error handling and escalation paths
5. **Testable Components**: Each agent can be tested in isolation

## Workflow Comparison

### Talk v1 (Linear)
```
research → generate → apply → test → check → END
```
Issues: 
- No loops or refinement
- Files generated in `check` step aren't applied
- Fixed linear flow regardless of results

### Talk v2 (Dynamic)  
```
assess → plan → research → refinement → branching
                                ↑         ↓
                                └─────────┘ (loop)
                                          ↓
                                    final_apply → END
```
Benefits:
- Proper iterative refinement
- Dynamic flow based on results  
- Complete file application
- Intelligent escalation

## Agent Responsibilities

- **AssessorAgent**: Task complexity analysis
- **ExecutionPlannerAgent**: Strategy generation
- **WebSearchAgent**: Domain research (optional)
- **RefinementAgent**: Complete development cycle
- **BranchingAgent**: Flow control decisions  
- **FileAgent**: Final file operations

This architecture achieves the goal of intelligent orchestration while maintaining simplicity and testability.