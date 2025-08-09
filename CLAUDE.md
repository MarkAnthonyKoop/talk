# IMPORTANT: Repository Structure Guidelines for Claude/AI Assistants

## CRITICAL: READ THIS FIRST

This repository has a **STRICT** directory structure that MUST be followed. AI models have a tendency to create files in the wrong locations. This document enforces the correct structure.

## ABSOLUTE RULES - NO EXCEPTIONS

### 1. TOP-LEVEL DIRECTORY (`~/code/`)
**NEVER** create these files at the top level:
- ❌ NO test files
- ❌ NO output files  
- ❌ NO result files
- ❌ NO log files
- ❌ NO temporary files
- ❌ NO data files
- ❌ NO documentation (except README.md and CLAUDE.md)

**ONLY** these belong at top level:
- ✅ `README.md` (points to docs/README.md)
- ✅ `CLAUDE.md` (this file)
- ✅ `.gitignore`
- ✅ `Makefile`
- ✅ `setup.py` / `pyproject.toml` / `package.json` (project config)
- ✅ Main directories (see structure below)

### 2. MANDATORY DIRECTORY STRUCTURE

**See [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) for complete details**

```
~/code/
├── docs/           # Documentation (categorized by purpose)
│   ├── architecture/   # System design
│   ├── api/           # API documentation
│   ├── guides/        # How-to guides
│   └── reference/     # Reference docs
│
├── talk/           # Main Talk framework code
│   ├── talk.py     # Current version
│   └── talk_v*.py  # Version history
│
├── tests/          # Tests (categorized by type)
│   ├── unit/          # Single component tests
│   ├── integration/   # Multi-component tests
│   ├── e2e/          # End-to-end tests
│   └── data/         # Test data (gitignored)
│       ├── input/    # Input fixtures
│       └── output/   # Results (date-organized)
│
├── special_agents/ # Agents (categorized by function)
│   ├── core/         # Essential agents
│   ├── research/     # Experimental agents
│   ├── tools/        # Utility agents
│   └── examples/     # Reference implementations
│
├── miniapps/       # Mini applications
└── scripts/        # Utility scripts
```

### 3. CATEGORIZATION RULES

Before creating ANY file, **FIND THE RIGHT CATEGORY:**

1. **Check existing categories** - Look for a directory that fits
2. **Use orthogonal categories** - File should clearly belong in ONE place
3. **Follow the decision tree** in [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

#### For Test Output:
```
tests/data/output/archive/YYYY/MM-month/YYYY-MM-DD_HHMM_category_subcategory/
```
Example: `tests/data/output/archive/2025/08-aug/2025-08-09_1423_unit_agents/`

### 4. FILE PLACEMENT RULES

#### Documentation
- **ALWAYS** place in `docs/`
- Talk-specific docs go in `docs/talk/`
- NEVER create .md files at top level (except README.md, CLAUDE.md)

#### Test Files
- **ALWAYS** place test code in `tests/`
- Test inputs go in `tests/input/` (gitignored)
- Test outputs go in `tests/output/` (gitignored)
- NEVER create test files at top level

#### Output Files
- **ALWAYS** place in `tests/output/` or appropriate subdirectory
- NEVER create *_result.json, *_output.json at top level
- Log files go in `tests/output/logs/`

#### Data Files
- Input data goes in `tests/input/`
- Output data goes in `tests/output/`
- NEVER commit large data files (.npy, .csv, .dat)

### 4. BEFORE CREATING ANY FILE

**ASK YOURSELF:**
1. Am I creating this in the correct directory?
2. Does this belong in docs/, tests/, or a subdirectory?
3. Will this file be gitignored if it's output/temporary?

**IF UNSURE:** Place it in the most specific subdirectory possible, NOT at top level.

### 5. COMMON MISTAKES TO AVOID

❌ **WRONG:**
```python
# Creating test file at top level
with open('test_results.json', 'w') as f:
    json.dump(results, f)
```

✅ **CORRECT:**
```python
# Creating test file in proper location
import os
os.makedirs('tests/output', exist_ok=True)
with open('tests/output/test_results.json', 'w') as f:
    json.dump(results, f)
```

❌ **WRONG:**
```python
# Creating documentation at top level
with open('analysis.md', 'w') as f:
    f.write(analysis)
```

✅ **CORRECT:**
```python
# Creating documentation in docs
with open('docs/analysis.md', 'w') as f:
    f.write(analysis)
```

### 6. GIT COMMIT GUIDELINES

When committing:
- NEVER commit output files
- NEVER commit temporary files
- NEVER commit large data files
- Check .gitignore is working properly

### 7. CLEANUP COMMANDS

If you've made a mess:
```bash
# Remove all untracked files (careful!)
git clean -fd

# Check what would be removed first
git clean -fdn

# Remove specific patterns
find . -name "*_result.json" -type f -delete
find . -name "*.log" -type f -delete
```

## ENFORCEMENT

This file is automatically loaded by Claude Code. Following these rules is MANDATORY, not optional. The repository maintainer will reject any changes that violate this structure.

## Questions?

See `docs/README.md` for project documentation.

---

**Remember:** When in doubt, put files in subdirectories, NOT at the top level!