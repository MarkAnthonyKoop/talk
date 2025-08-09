# Repository Structure Guidelines

## Directory Categories and Rules

### Tests Directory (`tests/`)

#### Categories (Orthogonal & Clear)
```
tests/
├── unit/              # Single component tests
├── integration/       # Multi-component interaction tests  
├── e2e/              # Full system end-to-end tests
├── performance/      # Performance & benchmark tests
└── data/             # Test data (gitignored)
    ├── input/        # Input data for tests
    └── output/       # Test results & logs
```

#### Test Data Organization

**Input Data** (`tests/data/input/`)
- `fixtures/` - Static test data that rarely changes
- `generated/` - Dynamically generated test data
- `samples/` - Example data for testing

**Output Data** (`tests/data/output/`)
```
output/
├── latest/           # Symlinks to most recent runs
│   ├── unit         -> ../archive/2025/08-aug/2025-08-09_1423_unit_agents/
│   ├── integration  -> ../archive/2025/08-aug/2025-08-09_1156_integration_workflow/
│   └── e2e          -> ../archive/2025/08-aug/2025-08-08_0934_e2e_scenarios/
│
└── archive/         # Historical test runs
    └── 2025/
        └── 08-aug/
            ├── 2025-08-09_1423_unit_agents/
            ├── 2025-08-09_1156_integration_workflow/
            └── 2025-08-08_0934_e2e_scenarios/
```

**Naming Convention:** `YYYY-MM-DD_HHMM_category_subcategory/`
- ISO date format for sorting
- 24-hour time for multiple runs per day
- Category matches test type
- Subcategory for specific component

### Documentation Directory (`docs/`)

#### Categories (By Purpose)
```
docs/
├── architecture/     # System design, diagrams, decisions
├── api/             # API specs, endpoints, schemas
├── guides/          # How-to guides, tutorials
├── reference/       # Configuration, CLI, options
├── development/     # Contributing, setup, workflows
└── releases/        # Changelogs, migration guides
```

#### Documentation Rules
1. **One topic per file** - Don't mix concepts
2. **Descriptive names** - `agent_communication.md` not `comm.md`
3. **Version specific docs** - Include version in filename if needed
4. **Keep flat** - Avoid deep nesting beyond 2 levels

### Special Agents Directory (`special_agents/`)

#### Categories (By Function)
```
special_agents/
├── core/           # Essential system agents
│   ├── orchestrator/
│   ├── communicator/
│   └── validator/
│
├── research/       # Experimental agents
│   ├── youtube/
│   ├── web/
│   └── analysis/
│
├── tools/          # Utility agents
│   ├── file_handler/
│   ├── code_generator/
│   └── test_runner/
│
└── examples/       # Reference implementations
    ├── simple/
    └── advanced/
```

## Categorization Decision Tree

When adding a new file, ask:

### For Tests:
```
Is it testing a single component in isolation?
  → tests/unit/{component}/

Is it testing interaction between components?
  → tests/integration/{interaction_type}/

Is it testing the full system workflow?
  → tests/e2e/{scenario}/

Is it measuring performance?
  → tests/performance/{metric}/
```

### For Documentation:
```
Is it explaining how the system is built?
  → docs/architecture/

Is it describing how to use an API?
  → docs/api/

Is it teaching how to do something?
  → docs/guides/

Is it a reference for configuration/options?
  → docs/reference/

Is it about contributing or development?
  → docs/development/
```

### For Agents:
```
Is it required for basic system operation?
  → special_agents/core/

Is it experimental or research?
  → special_agents/research/

Is it a utility for other agents?
  → special_agents/tools/

Is it a demo or reference?
  → special_agents/examples/
```

## Orthogonality Check

Categories are orthogonal if an item clearly belongs in ONE category:

✅ **Good:** `unit/` vs `integration/` vs `e2e/`
- A test is EITHER testing one component OR multiple OR the full system

❌ **Bad:** `agents/` vs `communication/` vs `core/`  
- An agent that handles communication could go in any of these

## Auto-Cleanup Rules

### Tests Output
- Keep `latest/` symlinks for quick access
- Archive runs older than 30 days to `archive/`
- Compress archives older than 90 days
- Delete archives older than 1 year

### Generated Files
- Never commit files in `tests/data/`
- Clean `tests/data/output/` before test runs
- Use timestamps to prevent conflicts

## File Naming Conventions

### Tests
- `test_*.py` - Test files
- `conftest.py` - Pytest configuration
- `fixtures.py` - Shared test fixtures

### Documentation
- `UPPERCASE.md` - Important docs (README, INSTALL)
- `lowercase_with_underscores.md` - Regular docs
- `topic_v2.md` - Versioned docs

### Agents
- `agent_name.py` - Main agent file
- `config.yaml` - Agent configuration
- `requirements.txt` - Agent dependencies

## Migration Plan

To migrate existing files:

1. **Map current locations** to new categories
2. **Git mv** files to preserve history
3. **Update imports** in affected files
4. **Run tests** to ensure nothing broke
5. **Update documentation** references

## Enforcement

This structure is enforced by:
- `.gitignore` patterns
- Pre-commit hooks (to be added)
- CLAUDE.md instructions
- Code review guidelines