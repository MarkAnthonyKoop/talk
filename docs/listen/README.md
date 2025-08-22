# Listen - Real-time Audio-Driven Task Execution

Listen is an experimental application that combines real-time audio transcription with the Talk framework to enable voice-driven software development and task execution.

## Overview

Listen continuously captures audio from the microphone, transcribes it to text in real-time, filters for task-relevant content, and can automatically create and execute development plans based on what it hears.

## Components

### 1. AudioListenerAgent (`audio_listener_agent.py`)
- Captures audio from microphone continuously
- Transcribes audio to text using speech recognition
- Maintains transcription history with timestamps
- Calculates basic relevance scores

### 2. RelevanceAgent (`relevance_agent.py`)
- Filters transcriptions for task-relevant content
- Extracts keywords, concepts, and actionable items
- Scores relevance on multiple dimensions
- Identifies trigger phrases for execution

### 3. ListenOrchestrator (`listen_v1.py`)
- Main orchestration system
- Integrates audio listening with Talk framework
- Creates execution plans using PlanningAgent
- Executes plans through PlanRunner

## Installation

```bash
# Install required audio libraries
pip install SpeechRecognition pyaudio

# For Linux users, you may also need:
sudo apt-get install python3-pyaudio portaudio19-dev
```

## Usage

### Basic Usage

```bash
# Listen for a specific task
python3 listen/listen_v1.py "Create a REST API for managing users"

# With custom working directory
python3 listen/listen_v1.py "Build a web scraper" --dir ~/projects/scraper

# With auto-execution enabled
python3 listen/listen_v1.py "Fix the bug in authentication" --auto-execute

# With custom thresholds
python3 listen/listen_v1.py "Optimize database queries" \
  --relevance-threshold 0.3 \
  --action-threshold 0.7
```

### Command-Line Options

- `task`: The task description to listen for and execute
- `--dir`: Working directory for file operations (default: current)
- `--relevance-threshold`: Minimum score to consider content relevant (0.0-1.0, default: 0.4)
- `--action-threshold`: Minimum score to trigger actions (0.0-1.0, default: 0.6)
- `--model`: LLM model to use (default: gemini-2.0-flash)
- `--auto-execute`: Automatically execute plans when relevant content is detected
- `--single`: Single capture mode (capture once and exit)
- `--verbose`: Enable verbose logging

## How It Works

1. **Audio Capture**: Continuously listens to microphone input
2. **Transcription**: Converts speech to text using Google Speech Recognition
3. **Relevance Filtering**: Analyzes transcriptions for task relevance
4. **Action Detection**: Identifies actionable items and trigger phrases
5. **Plan Creation**: Uses PlanningAgent to create execution plans
6. **Execution**: Runs plans through the Talk framework

## Execution Triggers

When auto-execute is disabled, say these phrases to trigger execution:
- "go ahead"
- "start"
- "begin"
- "proceed"
- "yes"
- "do it"
- "execute"
- "run it"

## Testing

Run the test suite to verify components work correctly:

```bash
python3 listen/test_listen.py
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Audio     │────>│ Transcription│────>│  Relevance  │
│   Input     │     │              │     │  Filtering  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                v
                                          ┌─────────────┐
                                          │  Planning   │
                                          │   Agent     │
                                          └─────────────┘
                                                │
                                                v
                                          ┌─────────────┐
                                          │    Plan     │
                                          │   Runner    │
                                          └─────────────┘
```

## Example Session

```
$ python3 listen/listen_v1.py "Create a Python CLI tool for file management"

==================================================
LISTEN v1 - Audio-Driven Task Execution
==================================================

Task: Create a Python CLI tool for file management

Listening for audio input...

Speak clearly about your task to provide context.
Say 'go ahead' or 'start' to trigger execution

Press Ctrl+C to stop listening
==================================================

[You speak]: "I need a CLI tool that can list files, copy them, move them, and delete them safely with confirmation prompts."

HIGH RELEVANCE (0.72): I need a CLI tool that can list files, copy them...

=== PENDING ACTIONS ===
1. Create CLI tool structure
2. Implement file listing functionality
3. Add copy operation with safety checks
4. Add move operation with confirmation
5. Implement safe delete with prompts
=======================

[You speak]: "Go ahead and start building it"

Triggering plan execution...
Creating execution plan for: Create a Python CLI tool for file management...
Executing 8 steps...
Plan execution completed
```

## Limitations

- Requires microphone access and audio libraries
- Speech recognition accuracy depends on audio quality
- Currently uses Google Speech Recognition (requires internet)
- Real-time transcription may have slight delays
- Background noise can affect accuracy

## Future Enhancements

- [ ] Support for multiple speech recognition engines
- [ ] Offline speech recognition option
- [ ] Wake word detection
- [ ] Multi-language support
- [ ] Audio command history and replay
- [ ] Integration with other input sources (video, screen capture)
- [ ] Voice feedback and confirmation
- [ ] Noise cancellation and audio enhancement

## Troubleshooting

### No Audio Input Detected
- Check microphone permissions
- Verify microphone is connected and working
- Try adjusting the ambient noise calibration time

### Speech Recognition Errors
- Ensure internet connection for Google Speech Recognition
- Speak clearly and at a moderate pace
- Reduce background noise
- Try adjusting relevance thresholds

### Installation Issues
- On macOS: `brew install portaudio`
- On Linux: `sudo apt-get install python3-pyaudio`
- On Windows: Download PyAudio wheel from unofficial binaries