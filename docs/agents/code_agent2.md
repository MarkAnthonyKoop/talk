# CodeAgent (Enhanced Version)

The enhanced CodeAgent is a sophisticated project-level orchestrator that generates structured file operations for complex code generation tasks.

## Overview

This version represents a significant evolution from the original simple diff generator to a comprehensive code orchestration agent that handles:

- Project-level file planning and creation
- Structured file operations (CREATE_FILE, MODIFY_FILE)
- Multiple parsing strategies with fallback mechanisms
- Context-aware filename generation
- Integration with FileAgent through structured commands

## Architecture Comparison

### Original CodeAgent (tmp/code_agent.py)
- **Purpose**: Simple unified diff generator
- **Input**: Task description
- **Output**: Unified diff text
- **Size**: 65 lines
- **Philosophy**: Unix-style tool - one job, do it well

### Enhanced CodeAgent
- **Purpose**: Project-level file operation planner
- **Input**: Project context and task description
- **Output**: Structured file operations
- **Size**: 350 lines
- **Philosophy**: Intelligent orchestrator with multiple capabilities

## Key Features

### Structured Output Format
The agent generates FILE_OPERATIONS blocks:

```
FILE_OPERATIONS:
CREATE: filename.ext
```
[file content]
```

MODIFY: existing_file.ext
```diff
[unified diff]
```
```

### Multiple Parsing Strategies
1. **Structured Parser**: Handles FILE_OPERATIONS format
2. **Fallback Parser**: Extracts code blocks when structure is missing
3. **Filename Inference**: Guesses appropriate filenames based on content

### Error Handling
- Multiple fallback strategies prevent total failures
- Graceful degradation when LLM output doesn't match expected format
- Extensive logging for debugging complex parsing scenarios

## Integration

Works seamlessly with the enhanced FileAgent through structured commands:
- `CREATE_FILE: filename` followed by content
- `MODIFY_FILE: filename` followed by diff or replacement content

## When to Use

The enhanced CodeAgent is ideal for:
- Multi-file project generation
- Complex code refactoring across multiple files
- Integration with sophisticated FileAgent operations
- Tasks requiring intelligent file organization

The original CodeAgent remains useful for:
- Simple single-file modifications
- Direct patch application workflows
- Minimal overhead scenarios