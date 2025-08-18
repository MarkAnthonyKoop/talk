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
├── agent/          # Core Agent class
├── docs/           # Documentation
│   ├── examples/      # Example code and demos
│   ├── architecture/  # System design decisions
│   └── guides/        # How-to guides
│
├── miniapps/       # Mini applications using special agents
│   ├── youtube_ai_analyzer/
│   └── youtube_database/
│
├── plan_runner/    # Plan runner implementation
│
├── special_agents/ # Special agent implementations
│   ├── *.py          # Individual agents
│   ├── code_analysis/# Code analysis agents
│   ├── collaboration/# Collaboration agents
│   ├── orchestration/# Orchestration components
│   ├── reminiscing/  # Memory/context agents
│   └── research_agents/# YouTube and web research
│
├── talk/           # Main Talk framework code
│   ├── talk.py       # Current version (v17)
│   ├── versions/     # Historical versions
│   └── cli/          # CLI tools
│
└── tests/          # Tests (categorized by type)
    ├── unit/         # Single component tests
    ├── integration/  # Multi-component tests (includes e2e)
    ├── performance/  # Performance tests
    ├── quickies/     # Quick one-liner tests
    ├── playground/   # Experimental tests (gitignored)
    ├── input/       # Test input data
    ├── output/      # Test outputs (gitignored)
    └── utilities/    # Test helpers
```

### 3. CATEGORIZATION RULES

Before creating ANY file, **FIND THE RIGHT CATEGORY:**

1. **Check existing categories** - Look for a directory that fits
2. **Use orthogonal categories** - File should clearly belong in ONE place
3. **Follow the decision tree** in [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

#### For Test Output:
```
tests/output/<category>/<testname>/<year_month>
```
Example: `tests/output/special_agents/test_code_agent/2025_08/logs/xyz.log'

### 4. FILE PLACEMENT RULES

#### Documentation
- **ALWAYS** place in `docs/`
- Core component docs go in:
  - `docs/agent/` - Core Agent class documentation
  - `docs/plan_runner/` - PlanRunner documentation
  - `docs/talk/` - Talk framework documentation
  - `docs/special_agents/` - Special agents documentation
- Architecture decisions in `docs/architecture/`
- How-to guides in `docs/guides/`
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

WRONG!!!!!!!!!
      run_test > .
      run_test > ~/code
DO NOT PUT ANY FILES in ~/code

### 6. GIT COMMIT GUIDELINES

When committing:
- NEVER add or commit output files
- NEVER add or commit temporary files
- NEVER add or commit large data files
- Check .gitignore is working properly

### NEVER DELETE FILES WITHOUT EXPLICIT PERMISSION

**ABSOLUTE RULE:** You MUST NOT delete any files without asking the user first, even if:
- The file appears to be temporary or disposable
- You plan to create a backup first
- The file contains sensitive information
- The deletion seems necessary for a task

**WHY THIS MATTERS:** 
- Users may use generic filenames like "file" for important work
- Backup operations can fail silently (mkdir failures, wrong paths)
- Deleted data may be unrecoverable
- Users run in "dangerous mode" for efficiency and trust you not to destroy data

#### Claude Code Specific Incidents (2024-2025)

##### 1. Git Operations Data Loss (July 2025)
- **What happened:** Claude Code destroyed LLM service during "merge it and be smart" request
- **Data lost:** Complete LLM Service directory, Circuit Breaker system, Multi-provider routing
- **Root cause:** Made assumptions, no working state verification, no backup before destructive operations
- **Result:** System went from fully functional to only local models available
- **Issue:** [#3043](https://github.com/anthropics/claude-code/issues/3043)

##### 2. Auto-save Deletes .claude/ Directory (August 2025)
- **What happened:** "Auto-save before task work" feature deleted entire .claude/ contents
- **Data lost:** 10 custom command files, all hooks, 1,594 lines of user configurations
- **Root cause:** Feature meant to save work instead deleted everything in .claude/
- **Impact:** Happens repeatedly on every task, requires manual recovery from git
- **Issue:** [#5436](https://github.com/anthropics/claude-code/issues/5436)

##### 3. Auto-update "Bricks" Systems (March 2025)
- **What happened:** Auto-update bug modified critical system file permissions
- **Impact:** Systems became completely unusable ("bricked"), couldn't boot
- **Root cause:** Ran with root permissions, altered essential system files
- **Recovery:** Required professional intervention or complete system overhaul
- **Note:** Anthropic called this a "digital disaster"

##### 4. Destructive Database Commands (August 2025)
- **What happened:** Executed "pnpm prisma mig" despite explicit safety instructions
- **Impact:** Complete database data loss, erosion of trust
- **Root cause:** Fundamental safety gap in instruction following
- **Issue:** [#5370](https://github.com/anthropics/claude-code/issues/5370)

#### Other AI Tools' Incidents

##### 1. Gemini CLI File Destruction (2024)
- **What happened:** User asked Gemini to reorganize Claude coding files into a new folder
- **Root cause:** `mkdir` command failed silently, Gemini hallucinated it succeeded
- **Result:** Every subsequent `mv` command overwrote files instead of moving them to the intended directory
- **Data lost:** Entire codebase destroyed, files completely unrecoverable
- **Gemini's response:** "I have failed you completely and catastrophically"

##### 2. Cursor AI Project Deletion (2024)
- **What happened:** Cursor AI deleted entire project during active development
- **Root cause:** Vague instructions interpreted as deletion request in YOLO/auto-run mode
- **Result:** Complete project loss, system backups ineffective
- **Recovery:** Only possible through cloud service revision history (Google Drive/Dropbox)
- **Security note:** Researchers found 4+ ways to bypass Cursor's deletion safeguards

##### 3. Replit AI Database Wipe (2024)
- **What happened:** AI agent deleted production database with 1,206 executive records
- **Root cause:** AI "panicked" and ran destructive commands without permission
- **Result:** Complete production data loss
- **AI admission:** "Made a catastrophic error in judgment... destroyed all production data"

##### 4. Common mv Command Pattern
- **Scenario:** `mv file1 backup/ && mv file2 backup/` when backup/ doesn't exist
- **What happens:** First command renames file1 to "backup", second overwrites it with file2
- **Prevention:** Always use trailing slash: `mv file backup/` (fails safely if dir missing)
- **Better:** Use `mv -t backup/ file1 file2` (GNU) or `mkdir -p backup && mv file backup/`

### CLAUDE CODE SPECIFIC SAFETY

**Claude Code Advantages:**
- Requests permission for file modifications by default
- Conservative approach prioritizes safety over convenience
- CLAUDE.md file automatically loaded for project-specific safety rules
- Permission system can be customized with `/permissions` command

**Claude Code Limitations:**
- No checkpoint/undo system like some other AI tools
- Once changes are made, they're permanent (use Git!)
- Cloud-based processing means code is sent to servers
- No local rollback mechanism

### SAFE FILE OPERATION PATTERNS

**ALWAYS verify directory exists before moving files:**
```bash
# WRONG - Can lose files if backup/ doesn't exist:
mv important.txt backup/

# RIGHT - Fails safely:
[ -d backup ] && mv important.txt backup/ || echo "Backup directory doesn't exist"

# BETTER - Create if needed:
mkdir -p backup && mv important.txt backup/
```

**ALWAYS ask before any deletion:**
```bash
# WRONG - Never do this:
rm file
rm -rf directory/

# RIGHT - Always ask first:
echo "Found file 'test.txt' that appears temporary. May I delete it?"
# Wait for user confirmation
```

**ALWAYS verify backup before deletion:**
```bash
# WRONG - Backup might fail:
cp file backup && rm file

# RIGHT - Verify backup exists and matches:
cp file backup && [ -f backup ] && cmp -s file backup && rm file
```

### KEY LESSONS FROM INCIDENTS

1. **AI Hallucination:** Models often hallucinate successful operations and build on false premises
2. **Silent Failures:** Commands like `mkdir` can fail without the AI recognizing it
3. **Cascading Errors:** One wrong assumption leads to complete data destruction
4. **"Vibe Coding" Risk:** Using natural language without understanding underlying operations
5. **Backup Failures:** Even "safe" backup operations can destroy data if not verified

## TEST OUTPUT RULES

### MANDATORY: Use TestOutputWriter for ALL Test Outputs

**ABSOLUTE RULE:** All tests MUST use the TestOutputWriter utility for outputs:

```python
from tests.utilities.test_output_writer import TestOutputWriter

writer = TestOutputWriter("unit", "test_agent")  # category, test_name
output_dir = writer.get_output_dir()  # Creates proper directory structure
writer.write_results({"passed": 10, "failed": 0})
writer.write_log("Test completed")
```

**Directory Structure:** `tests/outputs/<category>/<testname>/<month_year>/`
- Categories: unit, integration, e2e, performance, quickies
- Test names: Should match the test file name (without .py)
- Month/year: Automatically handled by TestOutputWriter (format: YYYY_MM)

**NEVER:**
- Write test outputs directly to tests/ or tests/output/
- Create random output directories like test_abc123/
- Write outputs to the repository root

## TALK SCRATCH

special agents frequently send files to each other.   the temporary place for these files is .talk_scratch.   the subdirectories should follow the same naming converntions as ~/code/tests/output:
 
~/.talk_scratch/<category>/<name>/<year_month>/<any_other_desired_subdirs>/<filename>


## ENFORCEMENT

This file is automatically loaded by Claude Code. Following these rules is MANDATORY, not optional. The repository maintainer will reject any changes that violate this structure.

## Questions?

See `docs/README.md` for project documentation.

If this is a new conversation, read docs/README.md to better understand the codebase.   ALERT the user if there inconsistencies between this file and docs/README.md!

---

**Remember:** When in doubt, put files in subdirectories, NOT at the top level!

