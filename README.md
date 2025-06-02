# Talk

_Autonomous CLI for continuous, unattended improvement of any codebase._

---

## 1. Project Overview
Talk is a command-line companion that **analyses, plans, and rewrites source code on its own**.  
Point it at a repository and Talk will:

1. Parse and understand the code (syntax, dependencies, tests).
2. Generate an improvement plan (refactors, lint fixes, performance tweaks, test additions).
3. Apply the edits, run checks, and iterate until no further actionable improvements remain.

Think of it as an “autopilot” for your codebase—capable of evolving a project while you focus on bigger ideas.

---

## 2. Quick Start

### Prerequisites
* Python **3.10+**
* `git` installed and the target project committed
* (Optional) Virtualenv, `pipx`, or similar isolation tool

### Installation

```bash
# Clone the repository
git clone https://github.com/MarkAnthonyKoop/talk.git
cd talk

# Install in editable / dev mode
pip install -e .

# Or install globally with pipx
pipx install git+https://github.com/MarkAnthonyKoop/talk.git
```

### First Run

```bash
# Analyse and improve the current directory
talk run .

# Dry-run (prints proposed patches without applying)
talk run . --dry

# Target a different branch
talk run /path/to/repo --branch feature/new-api

# Restrict to specific files or globs
talk run . --include "src/**/*.py"

# Roll back last automated commit
talk undo
```

---

## 3. Commands

| Command | Description |
|---------|-------------|
| `talk run <path>` | Iterate on codebase at `<path>` until exhausted or stopped. |
| `talk plan` | Generate and display the next set of improvements without applying them. |
| `talk undo` | Revert the most recent Talk-generated commit. |
| `talk config` | Edit or print current configuration file. |
| `talk plugins` | List, add, or remove strategy plugins. |

Run `talk --help` for the full CLI reference.

---

## 4. Configuration

A `.talk.yaml` file at the repo root controls behaviour:

```yaml
strategy: conservative   # conservative | balanced | aggressive
max_iterations: 5
lint: true               # run ruff / black after each iteration
tests:
  enabled: true
  command: pytest
backup: git              # git | filesystem
```

---

## 5. Architecture

```
┌──────────┐   AST & FS   ┌────────────┐  Diff/Patches  ┌────────────┐
│ Analyzer ├────────────►│   Planner  ├───────────────►│  Executor  │
└──────────┘              └────────────┘               └────────────┘
      ▲                         │                           │
      │  Metrics & Logs         │ Commits / Undo            ▼
┌────────────┐            ┌────────────┐             ┌────────────┐
│  Reporter  │◄───────────┤  Versioner │◄────────────┤  CLI (Typer)│
└────────────┘            └────────────┘             └────────────┘
```

1. **CLI (Typer)** – Parses user flags and kicks off a `Session`.
2. **Analyzer** – Builds an **AST**, dependency graph, and coverage map.
3. **Planner** – Chooses the next “improvement task” (refactor, test, etc.) via rule-based heuristics or optional LLM plugin.
4. **Executor** – Applies edits with `libcst`, `refactor`, or template generators.
5. **Versioner** – Creates checkpoint commits and enables `talk undo`.
6. **Reporter** – Streams progress, diffs, metrics, and warnings.

The loop (Analyzer → Planner → Executor) repeats until:
* No further improvements detected
* Maximum iterations reached
* A failure threshold is exceeded

---

## 6. Extending Talk

* **Plugins** – Drop a Python file in `talk/plugins/` exposing `plan(repo)` for custom strategies.
* **Hooks** – Pre/post iteration hooks allow integration with CI, Slack, etc.
* **Language Support** – Python first; additional languages can be added by implementing an `Analyzer`/`Executor` pair.

---

## 7. Roadmap

- [ ] JavaScript/TypeScript support  
- [ ] AI-assisted documentation writer  
- [ ] Interactive “explain my diff” mode  
- [ ] Web dashboard for progress visualization  

---

## 8. Contributing

1. Fork the repo and create a feature branch.
2. Run `pre-commit install` to enable linters.
3. Ensure `pytest` passes and `ruff` shows no issues.
4. Submit a PR—screenshots and context welcome!

---

## 9. License

Distributed under the MIT License. See `LICENSE` for details.

---

> Made with ❤️ to give your code a voice—let it Talk.
