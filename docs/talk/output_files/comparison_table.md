# Talk Output Directory Comparison Tables

## Table 1: Output Location Scenarios

| Scenario | User Type | Current Directory | Command | Output Location | Compliance |
|----------|-----------|------------------|---------|-----------------|------------|
| Development Test | Claude Code | `/home/xx/code/tests/output/integration/talk/2025_08/` | `talk "task"` | `/home/xx/code/tests/output/integration/talk/2025_08/.talk/[session]/` | ✅ Compliant |
| Development Test | Claude Code | `/home/xx/code/` | `talk "task"` | `/home/xx/code/.talk/[session]/` | ❌ Violates CLAUDE.md |
| User Project | End User | `~/projects/myapp/` | `talk "add feature"` | `~/projects/myapp/.talk/[session]/` | ✅ Expected |
| New Project | End User | `~/new_project/` | `talk "create app"` | `~/new_project/.talk/[session]/workspace/` | ✅ Expected |
| Override Path | Any | Anywhere | `PATH_OUTPUT_ROOT=/tmp talk "task"` | `/tmp/[session]/` | ✅ Configurable |

## Table 2: Directory Structure Components

| Component | Purpose | Location | Contents | Visibility |
|-----------|---------|----------|----------|------------|
| Session Directory | Store session metadata | `.talk/[timestamp]_[type]_[name]/` | Logs, info, artifacts | Hidden |
| Workspace | Generated code | `[session]/workspace/` | Source files | Hidden (indirect) |
| Logs | Execution logs | `[session]/logs/` | `.log` files | Hidden (indirect) |
| Artifacts | Additional outputs | `[session]/artifacts/` | Reports, data | Hidden (indirect) |
| Scratch | Temp agent communication | `.talk_scratch/` | Agent exchange files | Hidden |

## Table 3: Configuration Methods

| Method | Priority | Scope | Example | Use Case |
|--------|----------|-------|---------|----------|
| Environment Variable | Highest | Session | `PATH_OUTPUT_ROOT=/custom/path` | CI/CD, Testing |
| CLI Flag | High | Session | `--dir ./src` | Specific work directory |
| Config File | Medium | Project | `.talk/config.yaml` | Project defaults |
| Code Default | Lowest | Global | `Path.cwd() / ".talk"` | Fallback |

## Table 4: User vs Developer Requirements

| Aspect | Regular User Need | Claude Code Need | Current Support | Gap |
|--------|------------------|------------------|-----------------|-----|
| Output Location | Project directory | `tests/output/` | ✅ Both work | None |
| Visibility | Visible outputs | Hidden OK | ⚠️ Hidden by default | Make visible option |
| Cleanup | Auto or manual | Auto required | ❌ Manual only | Add auto-cleanup |
| Structure | Simple | CLAUDE.md compliant | ✅ Compliant structure | None |
| Configuration | Simple CLI | Enforced paths | ⚠️ No enforcement | Add validation |

## Table 5: Proposed Improvements Impact

| Improvement | Complexity | Breaking Change | User Benefit | Developer Benefit |
|-------------|------------|-----------------|--------------|-------------------|
| Visible directory option | Low | No | Easier to find outputs | Better debugging |
| Path validation | Low | No | Avoid mistakes | Ensure compliance |
| Auto-cleanup | Medium | No | Less disk usage | Cleaner test runs |
| Config file support | Medium | No | Project settings | Consistent testing |
| Session management CLI | Medium | No | Better control | Easier testing |
| Separate session/work | High | Yes | Clearer structure | Better organization |

## Table 6: Output Directory Patterns Across Talk Versions

| Version | Output Pattern | Session Structure | Working Directory | Notes |
|---------|---------------|-------------------|-------------------|-------|
| v2 | `.talk/[timestamp]_talk_v2_[task]/` | Standard 3-dir | `[session]/workspace/` | Current |
| v3-v10 | `.talk/[timestamp]_talk_v[N]_[task]/` | Standard 3-dir | `[session]/workspace/` | Consistent |
| v11-v14 | `.talk/[timestamp]_[task]/` | Variable | Multiple workspaces | Complex |
| v15-v17 | `.talk/[timestamp]_enterprise_[task]/` | Extended | Galaxy structure | Massive scale |

## Table 7: Common Issues and Solutions

| Issue | Cause | Current Behavior | Proposed Solution |
|-------|-------|------------------|-------------------|
| Can't find outputs | Hidden `.talk` directory | User confused | Add `--show-output` flag |
| Disk space usage | No cleanup | Accumulates forever | Auto-cleanup after 30 days |
| Wrong location | No validation | Creates anywhere | Add path validation |
| Long paths | Timestamp + type + name | Path too long errors | Shorten format |
| WSL paths | Windows/Linux mixing | Special handling | Improve path resolution |

## Table 8: Compliance Check Matrix

| Check | Condition | Pass | Fail | Action |
|-------|-----------|------|------|--------|
| Repository Root | Output under `/home/xx/code/` | Not at root level | At root level | Move to tests/output/ |
| Test Structure | Under `tests/output/` | ✅ | ❌ | Change directory |
| Category | Has category dir | `tests/output/[category]/` | Direct in output/ | Add category |
| Date Format | Year_Month format | `2025_08` | Other format | Use YYYY_MM |
| Hidden Files | In `.talk/` | Acceptable | At top level | Use .talk/ directory |

## Summary Metrics

| Metric | Current State | Target State | Gap |
|--------|--------------|--------------|-----|
| Compliance Rate | 75% (when used correctly) | 100% | Needs enforcement |
| User Friendliness | 6/10 | 9/10 | Visibility, cleanup |
| Developer Experience | 7/10 | 9/10 | Validation, tools |
| Flexibility | 9/10 | 9/10 | Already good |
| Maintainability | 5/10 | 8/10 | Config, cleanup |