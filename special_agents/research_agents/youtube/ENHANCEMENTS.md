# YouTube Research CLI - v2.0 Enhancements

## Overview
The YouTube Research CLI has been significantly enhanced with AI-powered planning, context awareness, and full Talk framework integration.

## Key Features Added

### 1. Context File Loading (`.talk/yr/`)
- **Automatic Loading**: Loads most recent context file on startup
- **Conversation Continuity**: Maintains context across sessions
- **Format**: JSON files with pattern `conversation_YYYYMMDD_HHMMSS_uuid.json`
- **Location**: `.talk/yr/` directory in current working directory

### 2. AI-Powered Planning
- **ResearchPlan Dataclass**: Structured planning with intent, requirements, and steps
- **Planning Agent**: Dedicated agent for creating research plans
- **Dynamic Execution**: Plans adapt based on query type and context
- **Step-by-Step Breakdown**: Clear phases (gather, analyze, synthesize)

### 3. Full Talk Framework Integration
- **Step Objects**: Uses `plan_runner.step.Step` for execution
- **Parallel Execution**: `ThreadPoolExecutor` for concurrent tasks
- **Agent Registry**: Multiple specialized agents for different tasks
- **Error Handling**: Fallback plans and graceful degradation

### 4. Specialized Agents
- **Planner**: Creates research plans
- **History Analyzer**: Searches viewing history
- **Transcript Analyzer**: Processes video transcripts
- **Web Researcher**: Performs web searches
- **Pattern Detector**: Identifies patterns in data
- **Synthesizer**: Combines results into comprehensive reports

## Commands

### Standard Enhanced Version (`yr`)
```bash
yr "What Claude videos have I watched?"
yr "What should I learn about AI agents?"
yr "https://youtube.com/watch?v=VIDEO_ID"
```

### Orchestrated Version (`yro`)
```bash
yro "Complex query requiring parallel processing"
```
*Note: Requires `pydantic-settings` package*

## Performance Improvements

### Response Quality
- **Before**: Generic suggestions, often missing actual data
- **After**: Specific video titles, actionable recommendations

### Example Query: "What Claude videos have I watched?"

**Before (v1.0)**:
```
No specific Claude videos found in your history.
Consider watching Claude tutorials online.
```

**After (v2.0)**:
```
You've watched several Claude videos:
1. "Claude Code Expert in 30 Minutes (With Real App Build!)"
2. "Building headless automation with Claude Code"
3. "Claude 3.7 goes hard for programmers…"
[... specific recommendations based on viewing patterns]
```

## Testing

### Test Coverage
- 13 comprehensive tests
- Context loading/saving
- AI planning features
- Error handling
- End-to-end integration

### Run Tests
```bash
python tests/test_orchestrated_research.py
python tests/test_youtube_search.py
```

## Architecture

```
YouTubeResearchCLI
├── Context Manager (load/save .talk/yr/)
├── Planning Agent (creates ResearchPlan)
├── Agent Registry
│   ├── History Analyzer
│   ├── Transcript Analyzer
│   ├── Web Researcher
│   ├── Pattern Detector
│   └── Synthesizer
├── Parallel Executor (ThreadPoolExecutor)
└── Database Interface (SQLite)
```

## Database Schema
- **Table**: `videos`
- **Fields**: `video_id`, `title`, `channel`, `url`, `ai_score`, `categories`
- **Size**: 20,806 videos
- **Note**: All channels show as "Unknown" (data extraction limitation)

## Future Enhancements
1. Fix Python version compatibility issues
2. Add more sophisticated pattern detection
3. Implement caching for transcript fetches
4. Add support for playlist analysis
5. Create visualization of viewing patterns

## Known Issues
1. **Channel Names**: All show as "Unknown" due to takeout data format
2. **Python Versions**: Some environments require `pydantic-settings`
3. **AI Score**: Field has false positives, using keyword matching instead

## Configuration

### Environment Variables
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

### Custom Database
```bash
yr --db /path/to/custom.db "your query"
```

## Context File Example
```json
{
  "query": "What Claude videos have I watched?",
  "plan": {
    "intent": "analyze viewing history",
    "requires_history": true,
    "requires_web_search": false,
    "requires_transcript": false,
    "expected_output": "List of Claude videos"
  },
  "results_summary": {
    "history_matches": 15,
    "web_search_performed": false,
    "transcript_fetched": false
  },
  "timestamp": "2024-08-07T12:09:51.123456"
}
```

## Impact
The enhancements transform the YouTube Research CLI from a basic search tool to an intelligent research assistant that:
- Remembers previous conversations
- Plans research strategies
- Executes tasks in parallel
- Provides specific, actionable insights
- Learns from your viewing patterns

This represents a significant step toward autonomous research agents that can handle complex, multi-faceted queries with minimal user intervention.