# Agent Usage by Talk Version

## Summary Table

| Talk Version | Key Agents Used | Notes |
|-------------|-----------------|-------|
| **talk.py (v17)** | MetaMetaOrchestratorAgent | Current production - orchestrates v16 instances |
| **v16_meta** | MetaOrchestratorAgent | Orchestrates v15 instances |
| **v15_enterprise** | EnterpriseCodebaseAgent | Enterprise-scale generation |
| **v14_enhanced** | EnhancedCodebaseAgent | Enhanced quality version |
| **v13_codebase** | CodebaseAgent | First codebase-focused version |
| **v12_tracked** | PlanningAgent, CodeAgent, FileAgent, TestAgent, RefinementAgent, WebSearchAgent | Added tracking |
| **v11_comprehensive** | PlanningAgent, CodeAgent, FileAgent, TestAgent, RefinementAgent, WebSearchAgent | Comprehensive features |
| **v10_refinement** | PlanningAgent, BranchingAgent, CodeAgent, FileAgent, TestAgent, WebSearchAgent, RefinementAgent | Added refinement |
| **v5_reminiscing** | PlanningAgent, BranchingAgent, CodeAgent, FileAgent, TestAgent, WebSearchAgent, ReminiscingAgent | Added memory/context |
| **v4_validated** | PlanningAgent, BranchingAgent, CodeAgent, FileAgent, TestAgent, WebSearchAgent | Validation focus |
| **v3_planning** | PlanningAgent, BranchingAgent, CodeAgent, FileAgent, TestAgent, ExecutionPlanningAgent, WebSearchAgent | Planning focus |
| **v2** | AssessorAgent, ExecutionPlannerAgent, RefinementAgent, BranchingAgent, FileAgent, WebSearchAgent | Early architecture |
| **talk_beast** | AssessorAgent, TaskAnalysisAgent, ExecutionPlannerAgent, CompletionVerifierAgent, CodeAgent, FileAgent, TestAgent, WebSearchAgent, MetricsAgent, EnhancedWorkflowSelector | "Beast mode" - many agents |
| **dynamic_talk** | AssessorAgent, CodeAgent, FileAgent, TestAgent, WebSearchAgent, WorkflowSelector | Dynamic workflow selection |
| **intelligent_talk** | IntelligentTalkOrchestrator | Intelligent orchestration |
| **talk_curr** | CodeAgent, FileAgent, TestAgent, WebSearchAgent | Basic current version |

## Core Agents (Used Across Multiple Versions)

### Essential Agents (Most Used)
- **CodeAgent** - Code generation (10+ versions)
- **FileAgent** - File operations (10+ versions)
- **TestAgent** - Test execution (9+ versions)
- **WebSearchAgent** - Web search capabilities (11+ versions)
- **PlanningAgent** - Task planning (7+ versions)

### Evolution-Specific Agents
- **BranchingAgent** - Branching logic (v3-v5, v10)
- **RefinementAgent** - Code refinement (v2, v10-v12)
- **AssessorAgent** - Task assessment (v2, beast, dynamic)
- **ExecutionPlannerAgent** - Execution planning (v2, v3, beast)

### Specialized Agents (Single Version)
- **ReminiscingAgent** - Memory/context (v5 only)
- **TaskAnalysisAgent** - Task analysis (beast only)
- **CompletionVerifierAgent** - Completion verification (beast only)
- **MetricsAgent** - Code metrics (beast only)

### Hierarchical Orchestrators
1. **CodebaseAgent** → **EnhancedCodebaseAgent** → **EnterpriseCodebaseAgent** (v13→v14→v15)
2. **MetaOrchestratorAgent** → **MetaMetaOrchestratorAgent** (v16→v17)
3. **IntelligentTalkOrchestrator** (intelligent_talk variant)

## Agents That Can Be Archived/Deprecated

Based on usage analysis, these agents might be candidates for archiving:

1. **Never used in any Talk version:**
   - galaxy_decomposer_agent.py
   - naming_agent.py
   - shell_agent.py (only used in CLI tool)
   - youtube_agent.py (duplicate of research_agents/youtube/youtube_agent.py?)

2. **Only used in older versions (v2-v5):**
   - BranchingAgent (unless still needed for backward compatibility)
   - ExecutionPlanningAgent (replaced by ExecutionPlannerAgent)

3. **Duplicate/Evolution candidates:**
   - Consider archiving CodebaseAgent and EnhancedCodebaseAgent if v15's EnterpriseCodebaseAgent supersedes them
   - reminiscing_agent.py vs reminiscing_agent_enhanced.py (only basic version used)