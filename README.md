# Talk Agent Framework

A lightweight, **agent-centric architecture** for building autonomous,
LLM-powered workflows.  
Instead of a monolithic CLI, the framework exposes composable parts:

| Layer                 | Key Modules (folder)                        | Purpose |
|-----------------------|---------------------------------------------|---------|
| **Message Primitives**| `agent/messages.py`                         | Typed chat turns (`Message`, `Role`, `MessageList`) |
| **Agent Core**        | `agent/agent.py` + `agent/storage.py`       | Conversation logging, settings management, pluggable LLM backend |
| **LLM Back-ends**     | `agent/llm_backends/…`                      | OpenAI, Anthropic, Gemini, Perplexity, Fireworks, OpenRouter, Shell, Stub |
| **Runtime Execution** | `runtime/*`                                 | Blackboard (shared memory), `PlanRunner`, `Step` graph |
| **Special Agents**    | `special_agents/*`                          | Code diff generator, patch applier, test runner, branching/loop control, shell orchestrator |
| **CLI Wrapper**       | `talk/talk.py`                              | Minimal wrapper to chat with any configured backend |

---

## ⚡ Quick Start

```bash
git clone https://github.com/MarkAnthonyKoop/talk.git
cd talk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"        # extras: openai, anthropic, google-generativeai …
export OPENAI_API_KEY="sk-..."
python -m talk.talk -p "Hello, agent!"
```

`talk.talk` is only a demo—real power comes from composing agents in
`runtime.PlanRunner`.

---

## 🧩 Core Concepts

### 1. **Message**
Validated Pydantic model that guarantees each chat turn is well-formed
*before* hitting a provider API or disk log.

### 2. **Agent**
Holds its own `Settings`, appends messages to `ConversationLog`,
delegates completions to a selected backend.

```python
from agent.agent import Agent
bot = Agent(overrides={"provider": {"type": "openrouter"}})
print(bot.run("How are you?"))
```

### 3. **LLMBackend**
Uniform interface (`complete(messages: List[Message])`) implemented by
provider-specific classes.  Add a new backend by dropping a file in
`agent/llm_backends/` and registering its `type` name.

### 4. **Runtime Pipeline**
`PlanRunner` executes an ordered list of `Step` objects, writing each
agent’s output to a concurrent-safe `Blackboard`.

```python
from runtime.plan_runner import PlanRunner
from runtime.step import Step
from runtime.blackboard import Blackboard

steps = [Step("ask", "assistant")]
agents = {"assistant": Agent()}
out = asyncio.run(PlanRunner(steps, agents, Blackboard()).run("Hi"))
```

### 5. **Special Agents**
Higher-level behaviours built on `Agent`:

* **CodeAgent** – LLM → unified diff  
* **FileAgent** – apply diff via `patch`  
* **TestAgent** – run pytest, return exit code  
* **BranchingAgent** – in-place control-flow rewrites  
* **ShellAgent** – orchestrates an iterative *code → apply → test* loop

---

## 🛠 Extending

* **Add a provider**: implement `MyBackend(LLMBackend)` and return it in
  `llm_backends.get_backend`.
* **Custom workflow**: define `Step` graph, create agents dict, run
  `PlanRunner`.
* **Persistent state**: read & write `BlackboardEntry(meta=…)` for
  arbitrary scratch data.

---

## 🧪 Testing

```bash
pytest -q              # unit tests
python tests/smoke_test.py   # hit every configured backend once
```

Set the relevant `*_API_KEY` env vars before running smoke tests.

---

## 📂 Directory Layout

```
agent/
  ├─ agent.py            # Core Agent façade
  ├─ messages.py         # Message + MessageList models
  ├─ settings.py         # Pydantic-validated config
  ├─ storage.py          # ConversationLog (JSONL)
  └─ llm_backends/       # Provider implementations
runtime/
  ├─ blackboard.py       # Shared async KV store
  ├─ step.py             # Workflow node
  └─ plan_runner.py      # Orchestrator
special_agents/          # Higher-level agents
talk/
  └─ talk.py             # Minimal CLI wrapper
tests/                   # Unit & smoke tests
```

---

## ✨ Roadmap

- [ ] Streaming token support
- [ ] Web dashboard for conversation & plan visualisation
- [ ] Docker image with all optional deps pre-installed
- [ ] Fine-tuned diff critic agent for safer code edits

PRs welcome—let your **agents** do the talking! 🤖
