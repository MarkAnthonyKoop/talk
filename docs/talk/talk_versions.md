# Talk Versions Documentation

## Development Workflow

### Version Management

The Talk framework has multiple versions in `talk/versions/`, each implementing different orchestration strategies:

- **talk_v2.py** - Enhanced orchestration with RefinementAgent and BranchingAgent
- **talk_v3_planning.py** - Planning-focused approach
- **talk_v4_validated.py** - Validation and testing emphasis
- **talk_v5_reminiscing.py** - Memory and context management
- **talk_v10_refinement.py** - Advanced refinement cycles
- **talk_v11_comprehensive.py** - Comprehensive code generation
- **talk_v12_tracked.py** - Progress tracking features
- **talk_v13_codebase.py** - Codebase-scale generation
- **talk_v14.py** - Platform-scale orchestration
- **talk_v15.py** - Enterprise-scale (50,000+ lines)
- **talk_v16.py** - Meta orchestration (200,000+ lines)
- **talk_v17.py** - Singularity (1,000,000+ lines)

### Testing a Version

To test a specific Talk version:

1. **Copy the version to talk.py**:
   ```bash
   cp talk/versions/talk_v2.py talk/talk.py
   ```

2. **Run using the talk CLI**:
   ```bash
   talk "Your task description" --dir /path/to/working/dir
   ```

3. **Optional: Specify a model**:
   ```bash
   talk "Your task" --model gemini-2.0-flash --dir /tmp/workspace
   ```

### Model Configuration

#### Default Behavior (as of current configuration)
- **Provider**: Google (set in `agent/settings.py`)
- **Model**: gemini-2.0-flash
- **Override Priority**:
  1. CLI `--model` flag (highest)
  2. `TALK_FORCE_MODEL` environment variable
  3. Provider-specific env vars (e.g., `TALK_GOOGLE_MODEL_NAME`)
  4. Defaults in settings.py (lowest)

#### Setting Model via CLI
```bash
# Use GPT-4
talk "task" --model gpt-4o

# Use Claude
talk "task" --model claude-3-5-sonnet-20241022

# Use Gemini
talk "task" --model gemini-2.0-flash
```

#### Setting Model via Environment
```bash
# Force all agents to use a specific model
export TALK_FORCE_MODEL=claude-3-5-sonnet-20241022
talk "task"

# Change default provider
export TALK_LLM_PROVIDER=openai
export TALK_OPENAI_MODEL_NAME=gpt-4o-mini
talk "task"
```

### Important Notes

1. **Always test in a safe directory**: Use `--dir /tmp/test` or similar
2. **Version consistency**: Ensure talk.py matches the version you're testing
3. **Model availability**: Ensure you have API keys for the models you're using:
   - `GEMINI_API_KEY` for Google models
   - `ANTHROPIC_API_KEY` for Claude models
   - `OPENAI_API_KEY` for OpenAI models

### Debugging

To see which model is being used:
```bash
# Check the logs - they show the selected backend
talk "test" --dir /tmp/test 2>&1 | grep "Selected LLM backend"
```

To test with stub backend (no API calls):
```bash
# Don't set any API keys - Talk will fall back to stub
unset GEMINI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY
talk "test" --dir /tmp/test
```

## Version Selection Guide

- **v2**: Best for iterative refinement with control flow
- **v3-v5**: Specialized approaches (planning, validation, memory)
- **v10-v12**: Medium-scale projects (1,000-10,000 lines)
- **v13-v14**: Large-scale projects (10,000-100,000 lines)
- **v15-v17**: Enterprise/civilization scale (50,000+ lines)

Choose based on your project size and requirements.