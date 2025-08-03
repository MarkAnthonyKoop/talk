# Test Suite Summary

This document consolidates everything generated and executed during the `tests/` phase of the **Talk** project ‑ from the very first “hello-agent” check to the fully-fledged iterative development workflow.

---

## 1  Overview of Test Files

| File | Lines | Primary Purpose |
|------|-------|-----------------|
| **tests/test_simple_agent.py** | ~370 | Validate the *Agent* façade: instantiation, conversation logging, prompt/response cycle, provider switching & graceful degradation without API keys. |
| **tests/test_simple_plan.py**  | ~450 | Exercise the synchronous `PlanRunner`, `Step` graph semantics and blackboard interactions – linear, nested and parallel flows. |
| **tests/test_advanced_plan.py** | ~950 | Simulate realistic multi-agent development loops using mocked `CodeAgent`, `FileAgent`, `TestAgent`; includes dynamic plans, iterative repair cycles and an optional run with real models if keys are present. |

Total lines under **tests/** ≈ 1 770.

---

## 2  What Each File Covers

### test_simple_agent.py
* Agent construction with:
  * default settings
  * custom name / id
  * explicit role messages
  * provider overrides (`google`, `openai`, `anthropic`).
* Prompt–response sanity checks in **mock** and real-backend modes.
* Conversation history integrity (`Role.user`, `Role.assistant`, manual `system` appends).
* `switch_provider(**kw)` success / failure paths.
* Absence of API keys → fallback to *StubBackend*.
* All artefacts saved in `tests/output/<test_id>/`.

### test_simple_plan.py
* Dataclass `Step` creation (auto-labelling, child / parallel steps).
* Linear 3-step execution path with assertion on call order & blackboard entries.
* Blackboard read/write across steps (mini ETL example).
* Simplified error propagation test – ensures runner stops on exception.
* Nested serial children and fork–join parallelism validation.
* Skeleton branching runner that chooses path1 vs path2.
* Skips real-agent test when keys or working_dir args are missing.

### test_advanced_plan.py
* Mock specialised agents that **generate code diffs, apply patches, run tests**.
* End-to-end workflow:
  1. Generate buggy Fibonacci,
  2. Apply diff,
  3. Fail tests,
  4. Iterate until `SUCCESS`.
* Dynamic plan builder that adds fix-steps at runtime based on test output.
* “Complete Development Workflow” orchestrator that keeps iterating ≤ 5 times.
* Optional **real** agents used when `GEMINI_API_KEY` / `OPENAI_API_KEY` et al. are set.
* Large bundle of artefacts (memoised `fibonacci.py`, progress logs, etc.) written to per-run sub-directories.

---

## 3  Results & Status

All tests were executed on **Windows 11 / Python 3.11**.

| Suite | Tests run | Failures | Errors | Skipped | Time |
|-------|-----------|----------|--------|---------|------|
| test_simple_agent | 6 | 0 | 0 | 1 (real-model) | 0.04 s |
| test_simple_plan  | 8 | 0 | 0 | 1 (real-agents plan) | 0.04 s |
| test_advanced_plan | 5 | 0 | 0 | 1 (real-agents) | 0.06 s |

Overall **PASSED**  (`OK (skipped=3)`).

Skipped cases occur only when API keys are absent or DEBUG_MOCK_MODE is enabled – ensuring CI always passes.

---

## 4  Example Outputs & Files

All artefacts live under `tests/output/`.

Typical structure:

```
tests/output/
 └─ adv_test_a60c04fa/
     ├─ metadata.txt
     ├─ blackboard_state.txt
     ├─ iterative_workflow.txt
     ├─ fibonacci.py                 ← final file emitted by FileAgent
     ├─ agent_collaboration.txt
     └─ workspace/…                  ← temp working directory
```

Sample excerpt from `iterative_workflow.txt`:

```
Iterative Code Improvement Test:
Total iterations: 4
Final result: TEST_RESULTS: SUCCESS
…
Progress Log:
Iteration 1: Generating code...
Iteration 1: Applying changes...
Iteration 1: Running tests...
Iteration 1: Tests failed, will try again.
…
Iteration 4: All tests passed!
```

---

## 5  Functionality Verified

* Agent layer
  * Settings hierarchy & stub fallback.
  * Provider switching API.
  * Conversation persistence.
* Plan execution engine
  * Linear, branching, nested & parallel step traversal.
  * `on_success` label jumping.
  * Blackboard CRUD & query helpers.
* Specialised agents collaboration
  * Diff generation → patch application → test execution loop.
* Iterative repair logic & dynamic plan augmentation.
* File I/O isolation into per-test temp dirs.

---

## 6  Running the Tests

From project root:

```bash
export PYTHONPATH=$PWD          # Windows: set PYTHONPATH=%CD%
cd tests

# Run everything
python -m unittest discover -s . -p "test_*.py" -v

# Run a single suite
python test_simple_agent.py

# Quiet mode
pytest -q   # if pytest is installed; auto-discovers same tests
```

Artefacts are regenerated on each run; clean up with:

```bash
rm -rf tests/output/*
```

---

## 7  Expected Behaviour With vs Without API Keys

| Scenario | Backend | Skips | Assertions |
|----------|---------|-------|------------|
| **No API keys** (default CI) | Agents fallback to `StubBackend` (mock mode) | Any test requiring real completions is auto-skipped | All other logic must still succeed. |
| **Keys provided** (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) | Real LLM calls | No skips | Additional numeric/string checks verify deterministic prompts (e.g., “2+2 → 4”). |

The environment variable `DEBUG_MOCK_MODE=1` forces stub mode even if keys are present.

---

## 8  Future Enhancements

1. **Coverage reporting** – integrate `pytest --cov` & badge in README.
2. Bring in **property-based tests** (Hypothesis) for Step graph invariants.
3. **Integration smoke tests** invoking the CLI (`talk/talk.py`) in real shell.
4. Parametrise suites across providers (OpenAI, Anthropic, Gemini) with `pytest.mark.parametrize`.
5. Containerised test matrix via GitHub Actions (Windows + Linux).
6. Add fuzzing of unified-diff parsing to harden FileAgent.
7. Persist blackboard to SQLite and assert resumption correctness.
8. Stress-test parallel PlanRunner with 100+ concurrent steps.

---

*(Generated automatically by Talk’s test automation on 2025-07-30.)*
