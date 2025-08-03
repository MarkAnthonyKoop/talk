# Talk

**Talk** is a lightweight, batteries-included orchestration engine that lets multiple LLM-powered agents collaborate on a codebase autonomously.  
It was inspired by systems like [Aider](https://github.com/paul-gauthier/aider) but rebuilt around a compositional “blackboard” pattern for maximum transparency and hackability.

## ✨ Key Ideas

* **Blackboard pattern** – every agent writes its output to a shared, structured datastore. Nothing is hidden in “agent-to-agent” chats.  
* **Specialised agents** – minimal, single-responsibility agents (CodeAgent, FileAgent, TestAgent) do one thing well.  
* **Deterministic workflow** – an explicit `PlanRunner` executes a graph of Steps so you can see *exactly* what happens and when.  
* **Filesystem-first** – Talk works on plain files, no remote sandboxes or proprietary formats.

---

## 🏗 Architecture

```text
┌────────────┐        write          ┌──────────────┐       read
│ CodeAgent  │ ───────────────▶     │              │◀─────────────┐
│ (LLM diff) │                      │   Blackboard │              │
└────────────┘◀───────────────      │  (in-memory) │      write   │
         ▲          read            └──────────────┘              │
         │                                                     ┌──▼─────────┐
         │                      read / write                   │ TestAgent  │
         │                                                     │ (pytest)   │
   ┌─────▼───────┐       apply diff      ┌────────────┐        └────────────┘
   │ FileAgent   │──────────────────────▶│   Files    │
   │ (patch)     │                       └────────────┘
   └─────────────┘
```

1. **CodeAgent** takes the task & current files → emits a **unified diff**.  
2. **FileAgent** applies the diff to disk (with automatic backups).  
3. **TestAgent** runs tests (`pytest` by default) and reports structured results.  
4. The **PlanRunner** decides which step comes next based on success / failure.

Everything each agent does is appended to the **Blackboard** (`BlackboardEntry`), giving you full provenance and an easy way to inspect or replay sessions.

---

## 🔑 Features

* Autonomous code generation (LLM-driven diffs).
* Safe patch application with automatic file backups.
* Test execution with timeout & detailed parsing of results.
* Versioned working directories (`talk1/`, `talk2/`, …) so nothing is overwritten.
* Interactive **or** fully automatic mode.
* 30-minute default timeout guard.
* Works with any OpenAI-compatible LLM; configurable provider list out-of-the-box (OpenAI, Anthropic, Gemini, Perplexity, Fireworks, …).

---

## ⚙️ Configuration

Talk uses a thin Pydantic-powered settings layer (`agent/settings.py`).  
Configuration values come from **four** sources — in order of precedence (highest → lowest):

1. **Runtime overrides** – e.g. `CodeAgent(overrides={...})`  
2. **Environment variables** – prefixed with `TALK_…` (see below)  
3. **Global force override** – `TALK_FORCE_MODEL` short-circuits every other model field  
4. **Built-in defaults** – provider=`google`, model=`gemini-1.5-flash`

### Default model

*Built-in*:  
```python
# agent/settings.py
class GoogleSettings(BaseSettings):
    model_name: str = "gemini-1.5-flash"
```

### Environment variable overrides

```bash
# change only Google provider’s default
export TALK_GOOGLE_MODEL_NAME="gemini-1.5-pro"

# override *all* providers everywhere
export TALK_FORCE_MODEL="gemini-2.0-flash"
```

### CLI override

```bash
# one-off per run
python talk/talk.py --task "Add logging" --model gemini-1.5-pro
```

### Putting it together

Priority order for the model used by `CodeAgent`:
`--model` CLI flag ▶ `TALK_FORCE_MODEL` ▶ `TALK_GOOGLE_MODEL_NAME` ▶ built-in default.

If you prefer OpenAI models:
```bash
export TALK_LLM_PROVIDER="openai"
export TALK_OPENAI_MODEL_NAME="gpt-4o-mini"
```

Everything else (paths, logging cadence, debug flags) follows the same pattern:  
`TALK_<SECTION>_<FIELD>` environment variables override the defaults.

---

## 📦 Installation

```bash
git clone https://github.com/yourname/talk.git
cd talk
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure you have a C compiler (for `patch`) and that your `patch` command is on `PATH` (Linux/macOS already OK; Windows users can install [GnuWin32 patch](http://gnuwin32.sourceforge.net/packages/patch.htm) or use Git Bash).

Set API keys as env vars, e.g.:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```

---

## 🚀 Usage

### Automatic mode (fire-and-forget)

```bash
python talk/talk.py --task "Add a Fibonacci function with tests"
```

Talk will:

1. Create a directory `talkN/`
2. Generate diffs
3. Apply them
4. Run tests until they pass or the timeout hits

### Interactive mode

```bash
python talk/talk.py -t "Refactor utils.py for PEP-8" -i
```

You’ll be prompted before each step and can inspect / edit files in another editor before continuing.

### Custom model & directory

```bash
python talk/talk.py \
  --task "Implement CLI argument parsing" \
  --model gpt-4o-mini \
  --dir ./my_project
```

---

## 🧩 Specialized Agents

| Agent       | Responsibility                        | Important Methods |
|-------------|---------------------------------------|-------------------|
| `CodeAgent` | Prompt an LLM to output **unified diffs** | `run(prompt_str)` |
| `FileAgent` | Apply diffs, backup originals, list/read files | `run(diff)`, `list_files()` |
| `TestAgent` | Run pytest/unittest, parse results     | `run("pytest")`, `discover_tests()` |

All inherit from the generic `Agent` facade and therefore support `switch_provider`, conversation logging, and provenance IDs.

---

## 🔄 Execution Plan

```text
generate_code  ─▶ apply_changes ─▶ run_tests ─▶ check_results
       ▲                            │
       └──────── (on failure) ◀─────┘
```

*Implementation*: four `Step` objects wired together; executed by `runtime.plan_runner.PlanRunner`.

---

## 🧪 Running Tests

The suite uses **pytest** + **unittest.mock**.

```bash
pytest -q
```

Coverage:

```bash
pytest --cov=talk --cov-report=term-missing
```

---

## 🛠 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `patch: command not found` | Install `patch` and make sure it’s on PATH |
| `OPENAI_API_KEY missing`   | Export the key in your shell or `.env` file |
| “Execution timed out”      | Increase `--timeout` or break your task into smaller units |
| Diff fails to apply        | Check conflicting local edits; run in `-i` mode and fix manually |

Enable verbose logging by setting env var:

```bash
export TALK_LOG_LEVEL=DEBUG
```

---

## 🤝 Contributing

1. Fork the repo & create a branch.
2. Run `pre-commit install` to get black/ruff hooks.
3. Add or adjust **tests** for every new feature or bug-fix.
4. Ensure `pytest && pytest -q` passes.
5. Submit a PR — include a concise description and reference any issues.

All contributions welcome: features, bug reports, docs, or just ideas!  
Feel free to open a discussion if you’re unsure.

---

## 📄 License

MIT – see `LICENSE` file for details.
