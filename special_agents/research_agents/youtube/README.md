# YouTube Research CLI

A powerful command-line tool for analyzing YouTube viewing history with AI-powered insights, transcript fetching, and web research capabilities.

## Features

- 📊 **Database Analysis**: Query your YouTube viewing history with natural language
- 📝 **Transcript Fetching**: Download and analyze video transcripts
- 🌐 **Web Research**: Research topics found in videos using web search
- 🤖 **AI-Powered Insights**: Get personalized learning recommendations
- 📚 **Learning Paths**: Generate comprehensive learning paths for any topic

## Installation

Install in editable mode for development:

```bash
pip install -e /path/to/youtube-research-cli
```

## Commands

### Research a Video
Fetch transcript and research topics from a specific video:
```bash
youtube-research research-video https://youtube.com/watch?v=VIDEO_ID
```

### Research a Topic
Research any topic using your viewing history and web search:
```bash
youtube-research research-topic "LangChain RAG systems"
```

### Analyze and Get Recommendations
Get AI-powered analysis and recommendations:
```bash
youtube-research analyze "What should I learn about AI agents?"
```

### Basic YouTube CLI
The package also includes the original YouTube CLI:
```bash
youtube-cli stats
youtube-cli query "What Python videos have I watched?"
youtube-cli ai-videos --months 3
youtube-cli transcript https://youtube.com/watch?v=VIDEO_ID
```

## Requirements

- Python 3.8+
- YouTube viewing history database (created from Google Takeout)
- API keys for AI providers (Anthropic, OpenAI, or Google)

## Database Setup

First, create a database from your YouTube takeout:
```bash
python build_db_fast.py --takeout path/to/takeout.zip
```

## Dependencies

- `youtube-transcript-api` - For fetching video transcripts
- `duckduckgo-search` - For web research
- `beautifulsoup4` - For HTML parsing
- `requests` - For HTTP requests

## License

MIT