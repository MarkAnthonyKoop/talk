# Getting Started with the **Talk Agent Framework**

Welcome to **Talk** – a lightweight, agent-centric architecture for building autonomous, LLM-powered workflows.  
This guide walks you from first install to writing production-ready pipelines.

---

## 1. Installation

### Prerequisites
* Python ≥ 3.9
* `git`, `pip`, (optionally) a virtual environment tool such as `venv` or `conda`

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/MarkAnthonyKoop/talk.git
cd talk

# 2. Create & activate a virtual-env  (recommended)
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate                             # Windows (PowerShell)

# 3. Install core library in editable mode
pip install -e .

# 4. (Optional) Provider extras
# pip install -e .[openai]       # for OpenAI GPT models
# pip install -e .[anthropic]    # for Claude models
# pip install -e .[gemini]       # for Gemini models
```

> **Tip:** Skip the extras if you just want to play – the framework automatically falls back to a *stub* backend that returns mock responses.

---

## 2. Quick Start

### 2.1 One-liner chat from the command line

```bash
python -m talk.talk -m stub -p "Hello, framework!"
```

Output
```
Assistant: [stub] This is a mock response from the stub backend.
```

Replace `stub` with `openai`, `anthropic`, etc. once you have API keys.

### 2.2 Hello-World in Python

```python
from agent.agent import Agent

bot = Agent(overrides={"provider": {"type": "stub"}})
print(bot.run("What can you do?"))
```

---

## 3. Configuration Guide

Talk loads settings in the following priority order:

1. **Runtime overrides** (Python dictionaries)  
2. **Environment variables** – all `TALK_*` vars  
3. **YAML / JSON** config files  
4. **Defaults** from `agent/settings.py`

### 3.1 Environment variables

```bash
export OPENAI_API_KEY="sk-•••"
export TALK_PROVIDER_TYPE=openai
export TALK_MODEL_NAME=gpt-3.5-turbo
```

### 3.2 YAML example

```yaml
# my_config.yaml
provider:
  type: stub
  model_name: gpt-4-turbo-like
  temperature: 0.3
conversation:
  log_enabled: true
  log_path: ./logs/tutorial.jsonl
```

Load it:

```python
from agent.settings import Settings
from agent.agent import Agent

settings = Settings.from_file("my_config.yaml")
agent = Agent(settings=settings)
```

### 3.3 Programmatic override

```python
Agent(overrides={
    "provider": {"type": "stub", "temperature": 0.2},
    "conversation": {"system_prompt": "You are a helpful assistant."}
})
```

---

## 4. Basic Usage Patterns

### 4.1 Stateful chat

```python
bot = Agent(overrides={"provider": {"type": "stub"}})

bot.run("Remind me to drink water.")
bot.run("What did I ask you to remind me?")
```

### 4.2 Building a workflow (PlanRunner)

```python
import asyncio
from runtime.plan_runner import PlanRunner
from runtime.step import Step
from runtime.blackboard import Blackboard
from agent.agent import Agent

steps = [
    Step("greet", "greeter", prompt="Say hello"),
    Step("analyse", "analyst", prompt="Summarise: {input}"),
]
agents = {
    "greeter": Agent(overrides={"provider": {"type": "stub"}}),
    "analyst": Agent(overrides={"provider": {"type": "stub"}}),
}

result = asyncio.run(
    PlanRunner(steps, agents, Blackboard()).run("Talk is awesome!")
)
print(result)
```

### 4.3 Using special agents

```python
from special_agents.code_agent import CodeAgent
diff = CodeAgent(overrides={"provider": {"type": "stub"}}).run(
    "Create a diff that renames variable 'x' to 'total'."
)
print(diff)
```

---

## 5. Common Use Cases

| Use case                 | Suggested pieces                                      |
|--------------------------|-------------------------------------------------------|
| Ad-hoc Q&A / chat        | `Agent` with desired backend                          |
| Automated code editing   | `CodeAgent` + `FileAgent` in a PlanRunner pipeline    |
| Continuous integration   | `TestAgent` to run pytest, gate merges                |
| Data analysis pipeline   | Multi-step PlanRunner, store artifacts on `Blackboard`|
| Conversational UI server | Wrap `Agent.run` in FastAPI / Flask endpoint          |

---

## 6. Troubleshooting

| Symptom                                   | Fix |
|-------------------------------------------|-----|
| `LLMBackendError: environment variable ... not set` | Export the required API key or use `provider.type: stub` |
| `Backend class 'OpenaiBackend' not found` | Upgrade to latest `talk` – naming was fixed in v0.1.1 |
| Getting the *stub* response unexpectedly  | Framework fell back after backend error – check your keys & network |
| Unicode errors in diff application        | Ensure file is UTF-8 encoded before applying with `FileAgent` |
| Long-running PlanRunner hangs             | Verify each step’s agent returns; add timeouts or use async backends |

Enable debug logs:

```bash
export TALK_LOG_LEVEL=DEBUG
```

---

## 7. Next Steps (Advanced)

1. **Streaming tokens** – fork and implement `LLMBackend.stream()`  
2. **Custom backends** – drop `my_backend.py` in `agent/llm_backends` and register the type.  
3. **Parallel execution** – define independent `Step`s; PlanRunner executes concurrently.  
4. **Persisted blackboards** – swap `Blackboard` for a Redis / SQLite implementation.  
5. **Web dashboard** – track conversation & workflow state visually (roadmap item).  
6. **CI/CD integration** – pair `ShellAgent` with Git hooks to auto-patch PRs.  
7. **Security** – integrate content moderation & PII redaction middleware.  

---  

Happy building — let your **agents** do the talking! 🤖
