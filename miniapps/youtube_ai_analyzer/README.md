# YouTube AI Content Analyzer

A Talk framework-based application that analyzes your YouTube viewing history to identify and categorize AI/coding content that could enable better codebase analysis and design.

## Features

- **Multi-Agent Orchestration**: Uses Talk framework's PlanRunner to coordinate multiple specialized agents
- **Content Categorization**: Identifies 8+ categories of AI/coding content:
  - LLM tutorials and prompt engineering
  - AI-assisted coding tools
  - Codebase analysis techniques
  - Software architecture principles
  - AI agent frameworks
  - Machine learning tutorials
  - DevOps automation
  - Testing and quality assurance

- **Intelligent Recommendations**: 
  - Identifies under-explored high-value categories
  - Recommends top channels based on relevance scoring
  - Suggests specific searches to fill knowledge gaps

- **Learning Path Generation**: Creates customized learning paths based on your viewing history
- **Export Capabilities**: Generates both JSON data and Markdown reports

## Architecture

The app uses Talk framework's orchestration pattern:

```
YouTubeAIAnalyzer (Main Orchestrator)
    ├── YoutubeAgent (Data Extraction)
    ├── WebSearchAgent (Channel Research)  
    ├── PlanningAgent (Execution Planning)
    ├── CodeAgent (Analysis Generation)
    └── FileAgent (Result Storage)
```

### Key Components

1. **AIContentCategorizer**: Categorizes videos into AI/coding topics
2. **YouTubeAIAnalyzer**: Main orchestrator using Talk's PlanRunner
3. **Multi-Step Pipeline**:
   - Extract YouTube data (watch history, subscriptions, searches)
   - Categorize content by relevance
   - Identify top AI/coding channels
   - Generate recommendations
   - Create learning paths
   - Export results

## Usage

### Basic Usage

```bash
cd ~/code/miniapps/youtube_ai_analyzer
python youtube_ai_analyzer.py
```

### With Custom Takeout File

```bash
python youtube_ai_analyzer.py --takeout /path/to/takeout.zip
```

### Specify Output Directory

```bash
python youtube_ai_analyzer.py --output ./my_results
```

## Output

The analyzer generates:

1. **JSON Results** (`ai_content_analysis_[timestamp].json`):
   - Complete categorization data
   - Channel rankings with scores
   - Detailed recommendations
   - Learning path definitions

2. **Markdown Report** (`ai_content_report_[timestamp].md`):
   - Human-readable summary
   - Top channels and categories
   - Actionable recommendations
   - Progress tracking for learning paths

## Example Output

### Categories Found
- **ai_coding**: 15 videos - AI-assisted coding tools and techniques
- **llm_tutorials**: 8 videos - LLM usage and prompt engineering
- **codebase_analysis**: 3 videos - Codebase analysis tools (under-explored!)

### Top Channels
1. **ThePrimeagen** - Score: 8.4, Categories: ai_coding, software_architecture
2. **Fireship** - Score: 7.2, Categories: ai_coding, devops_automation
3. **TechLead** - Score: 5.1, Categories: software_architecture, testing_quality

### Recommendations
- **Explore codebase_analysis**: High-value category with only 3 videos watched
- **Follow ThePrimeagen**: High relevance for AI coding topics
- **Search for**: "AST parsing tutorials", "static analysis tools"

### Learning Paths
- **AI-Assisted Development Mastery** (40% complete)
  1. LLM Fundamentals
  2. Prompt Engineering
  3. AI Coding Tools
  4. Multi-Agent Systems
  5. Codebase Analysis with AI

## Integration with Talk Framework

This app demonstrates Talk framework patterns:

- **Step-based Execution**: Uses `Step` objects for orchestration
- **Blackboard Pattern**: Shares data between agents via Blackboard
- **PlanRunner**: Executes multi-step plans with proper sequencing
- **Agent Contract**: All agents follow "prompt in → completion out" pattern
- **Result Persistence**: Saves to `.talk/scratch/` for inter-agent communication

## Requirements

- Python 3.11+
- Talk framework installed
- YouTube takeout data (zip file)
- Required agents:
  - YoutubeAgent
  - WebSearchAgent
  - PlanningAgent
  - CodeAgent
  - FileAgent

## Testing

Run the test script:

```bash
python test_analyzer.py
```

This will:
- Verify all agents are available
- Test categorization logic
- Run a simplified analysis
- Generate sample output

## Future Enhancements

- [ ] Real-time YouTube API integration
- [ ] Video transcript analysis
- [ ] Playlist generation for learning paths
- [ ] Integration with course platforms
- [ ] Collaborative filtering with other users
- [ ] Export to Notion/Obsidian
- [ ] Weekly progress tracking
- [ ] Custom category definitions