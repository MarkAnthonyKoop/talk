# YouTube Research CLI (`yr`)

A powerful AI-powered command-line tool for analyzing YouTube viewing history with intelligent routing, transcript fetching, and web research capabilities.

## 🚀 Quick Start

```bash
# Install dependencies
pip install youtube-transcript-api ddgs beautifulsoup4 requests pydantic-settings

# Install the tool
cd ~/code/special_agents/research_agents/youtube
bash install.sh

# Use the yr command from anywhere!
yr "hello"
yr "What AI videos have I watched?"
yr "prompt engineering"
yr "https://youtube.com/watch?v=VIDEO_ID"
```

## ✨ Features

### 🤖 Smart Routing (`yr`)
The `yr` command uses AI to understand your intent and automatically routes to the right tool:

```bash
# Just type naturally - AI decides what to do
yr "What Claude videos have I watched?"     # → analyzes your history
yr "machine learning fundamentals"          # → researches the topic
yr "https://youtube.com/watch?v=abc123"    # → fetches transcript & researches
yr "hello"                                  # → friendly greeting & help
```

### 📊 Database Analysis
- Query 20,000+ videos from your YouTube history
- Natural language search: "What AI videos have I watched?"
- Pattern analysis and viewing insights
- Accurate AI/tech content detection

### 📝 Transcript Fetching
- Download full transcripts from any YouTube video
- Extract key topics using AI
- Save transcripts for offline analysis

### 🌐 Web Research
- Research topics with DuckDuckGo search
- Combine viewing history with current information
- Generate learning paths and recommendations

### 🎯 Intelligent Features
- **No unnecessary web searches**: History queries stay local
- **Context-aware responses**: Greetings, questions, and topics handled differently
- **Personalized insights**: Based on your actual viewing history

## 📦 Installation

### Quick Install
```bash
# From the youtube directory
bash install.sh

# Creates global command 'yr' accessible from anywhere
```

### Manual Install
```bash
# Create symlink manually
ln -sf /path/to/youtube-research ~/.local/bin/yr

# Ensure ~/.local/bin is in your PATH
export PATH="$HOME/.local/bin:$PATH"
```

## 🎮 Usage Examples

### Smart Mode (Recommended)
```bash
# Let AI decide what to do
yr "What AI coding tools have I watched?"
yr "Tell me about LangChain"
yr "https://youtube.com/watch?v=p0UOYuA5RdU"
yr "What should I learn next?"
```

### Explicit Commands
```bash
# Research a specific video
yr research-video https://youtube.com/watch?v=VIDEO_ID --deep

# Research a topic with your history + web
yr research-topic "prompt engineering"

# Analyze viewing history
yr analyze "What patterns are in my AI video watching?"
```

## 🗄️ Database Setup

Create a database from your YouTube takeout:

```bash
# Download your YouTube data from Google Takeout
# Then build the database:
python build_db_fast.py --takeout ~/Downloads/takeout.zip

# This creates youtube_fast.db with all your viewing history
```

## 🔧 Configuration

### Environment Variables
```bash
# Set AI provider API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Custom Database Path
```bash
# Use a different database
yr --db /path/to/custom.db "your query"
```

## 📊 What It Can Do

### Viewing History Analysis
- Find specific videos: "What Claude tutorials have I watched?"
- Identify patterns: "How has my AI interest evolved?"
- Get statistics: "How many coding videos have I watched?"

### Learning Recommendations
- Knowledge gaps: "What AI concepts am I missing?"
- Next steps: "What should I learn after LangChain?"
- Learning paths: "Create a roadmap for prompt engineering"

### Research & Discovery
- Transcript analysis: Research any video with full transcript
- Topic deep-dives: Combine history with web research
- Current trends: Get up-to-date information on any topic

## 🛠️ Technical Details

### Architecture
- **Smart Router**: AI-powered intent detection
- **Database**: SQLite with 20,000+ videos indexed
- **Agents**: Specialized agents for different tasks
- **Web Search**: DuckDuckGo integration for research

### Key Files
- `youtube_research_cli.py` - Main CLI with smart routing
- `youtube_agent.py` - Core YouTube agent
- `youtube_history_agent.py` - History analysis agent
- `build_db_fast.py` - Database builder

### Dependencies
```
youtube-transcript-api  # Fetch video transcripts
ddgs                   # Web search (updated from duckduckgo-search)
beautifulsoup4         # HTML parsing
requests               # HTTP requests
pydantic-settings      # Configuration management
```

## 🐛 Troubleshooting

### "Database not found"
```bash
# Build the database first
python build_db_fast.py --takeout ~/Downloads/takeout.zip
```

### "No AI videos found"
The tool now uses keyword matching instead of the faulty ai_score field, so it should find your actual AI content.

### Web search warnings
```bash
# Update to latest search package
pip install ddgs  # (replaces duckduckgo-search)
```

## 📈 Recent Improvements

- ✅ Smart routing with AI intent detection
- ✅ Fixed AI video detection (no more false positives)
- ✅ No unnecessary web searches for history queries
- ✅ Accurate Claude/GPT/LangChain video finding
- ✅ Context-aware responses (greetings vs queries)
- ✅ Global `yr` command for quick access

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! The tool integrates with the Talk framework for agent orchestration.

---

**Pro tip**: Just type `yr` followed by whatever you're thinking about. The AI will figure out what you need!