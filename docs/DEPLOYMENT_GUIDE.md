# Talk System – Deployment Guide

> Version 1.0 · July 2025  
> Applies to repository root `C:/…/code/`

---

## 1. System Overview & Architecture

Talk is a **multi-agent orchestration framework** that automates software development tasks through a blackboard-style communication hub.

```
 ┌────────────┐     writes         reads ┌────────────┐
 │ CodeAgent  │ ───────────────►        │            │
 ├────────────┤ ◄───────────────         │            │
 │ FileAgent  │      Blackboard          │  PlanRunner│
 ├────────────┤                          │            │
 │ TestAgent  │ ───────────────►         │            │
 └────────────┘                          └────────────┘
                                              │
                                              ▼
                                        TalkOrchestrator
```

* **agent/** – generic chat agent façade + back-ends  
* **special_agents/** – `CodeAgent`, `FileAgent`, `TestAgent`  
* **plan_runner/** – `Step`, `PlanRunner`, `Blackboard`  
* **talk/** – CLI entry-point (`talk.py`) providing interactive & batch modes  
* **tests/** – 1 700 + lines of validation

---

## 2. Prerequisites & Setup

| Requirement | Recommended | Notes |
|-------------|-------------|-------|
| OS          | Linux, macOS, Windows 10/11 | Windows uses `git-bash`/PowerShell |
| Python      | 3.11+       | CPython only, venv required |
| Disk        | >1 GB       | temp build & test artefacts |
| RAM         | >4 GB       | LLM calls stream in-memory |
| Optional    | CUDA 11+    | for local GPU back-ends |

### 2.1 Clone & create venv

```bash
git clone https://github.com/your-org/talk.git
cd talk
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 System packages

* `patch` (GNU) – required by `FileAgent`  
  * Ubuntu: `sudo apt install patch`  
  * macOS: pre-installed  
  * Windows: install via **Git for Windows** or **GnuWin** and add to `PATH`.

---

## 3. Configuration Options

All configuration lives in **agent/settings.py** (Pydantic):

| Env var / `.env` key | Default | Description |
|----------------------|---------|-------------|
| `LLM_PROVIDER`       | `google`| stub/`openai`/`anthropic`/… |
| `LLM_MODEL`          | provider default | Model name |
| `LOGS_DIR`           | `./logs`| Conversation transcripts |
| `DEBUG_MOCK_MODE`    | `0`     | `1` forces stub backend |
| `TIMEOUT_MINUTES`    | `30`    | Orchestrator run-time cap |

Example **.env**

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LOGS_DIR=/var/log/talk
TIMEOUT_MINUTES=20
```

---

## 4. Running the System

### 4.1 Non-interactive batch

```bash
python -m talk.talk \
  --task "Implement a REST endpoint returning Fibonacci" \
  --working-dir /absolute/path/to/project \
  --model gpt-4o-mini \
  --timeout 20
```

Exit codes: `0` success, `1` failure, `130` interrupted.

### 4.2 Interactive walk-through

```bash
python -m talk.talk --interactive \
  --task "Refactor utils for Python 3.12" \
  --working-dir .
```

At every step you can accept (`y`), skip (`n`) or abort (`q`).

---

## 5. API Key Setup

| Provider | Required key | Env variable |
|----------|--------------|--------------|
| Google Gemini | `GEMINI_API_KEY` | GEMINI_API_KEY |
| OpenAI | `OPENAI_API_KEY` | OPENAI_API_KEY |
| Anthropic | `ANTHROPIC_API_KEY` | ANTHROPIC_API_KEY |

1. Create **.env** or export in shell.  
2. Verify with:

```bash
python - <<'PY'
from agent.settings import Settings
print(Settings.resolve().provider)
PY
```

If no keys are found Talk falls back to **StubBackend** – safe for CI.

---

## 6. Testing & Validation

### 6.1 Unit + integration

```bash
export PYTHONPATH=$PWD      # Windows: set PYTHONPATH=%CD%
cd tests
python -m unittest discover -v
```

19 tests should pass (3 skipped without keys).

### 6.2 End-to-end smoke

```bash
python -m talk.talk \
  --task "Create hello_world.py saying 'Hi'" \
  --working-dir ./demo \
  --interactive false
```

Inspect *demo/talk1/blackboard.json* and generated file.

---

## 7. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `LLMBackendError: API key not set` | Export key or set `DEBUG_MOCK_MODE=1` |
| `patch: command not found` | Install GNU patch and ensure it’s on `PATH` |
| `signal.alarm` not available on Windows | The orchestrator swaps to polling loop automatically; ignore warning |
| Unified diff rejected | Check diff context lines – or delete target file and rerun `FileAgent` |
| CLI exits with `1` quickly | Look at `logs/YYMMDD_HHMMSS/error.log` for stacktrace |

---

## 8. Usage Examples

### 8.1 Quick Fix Workflow

```bash
python -m talk.talk \
  --task "Fix failing pytest tests in current repo" \
  --working-dir $(pwd) \
  --interactive false
```

The system loops: generate diff → apply → run tests until green.

### 8.2 Interactive Code Review

```bash
python -m talk.talk \
  --task "Improve docstrings in src/*" \
  --working-dir ./my_lib \
  --interactive
```

Accept or reject each diff interactively.

---

## 9. Extension Points

### 9.1 Add a new agent

1. Create `special_agents/my_agent.py`:

```python
from agent.agent import Agent

class MyAgent(Agent):
    def run(self, input_text: str) -> str:
        # custom logic
        return super().run(input_text)
```

2. Register in your plan:

```python
from special_agents.my_agent import MyAgent
agents = {"reviewer": MyAgent()}
steps = [
    Step(label="review", agent_key="reviewer")
]
```

### 9.2 Custom back-end

Implement `agent/llm_backends/my_backend.py` exposing `.complete()`, then add to `agent/llm_backends/__init__.py` mapping.

---

## Appendix A – Command Cheatsheet

| Task | Command |
|------|---------|
| Run unit tests | `python -m unittest -v` |
| Generate coverage | `pytest --cov=.` |
| Clean artefacts | `rm -rf **/__pycache__ tests/output` |
| Lint | `ruff check .` |

---

Happy hacking 🎉
