# Listen v4 - Intelligent Conversational Assistant

## Overview
Listen v4 builds upon v3's speaker identification capabilities to create a fully interactive conversational assistant that knows when and how to respond naturally.

## New Features in v4

### 1. **Context Relevance Detection**
- Intelligent decision-making about when to respond
- Wake word detection ("Hey Listen")
- Question recognition
- Command/request identification
- Conversation flow analysis

### 2. **Voice Synthesis (TTS)**
- Real-time text-to-speech responses
- Multiple TTS engine support:
  - System TTS (pyttsx3) - fastest, local
  - Coqui TTS - neural voices
  - RealtimeTTS framework - automatic fallback
- Natural conversational speech output

### 3. **Intelligent Response Generation**
- Context-aware replies using LLM
- Maintains conversation history
- Speaker-specific responses (uses names when known)
- Concise, conversational tone
- Follow-up question handling

### 4. **Enhanced Conversation Management**
- Response cooldown to prevent over-talking
- Confidence thresholds for response triggering
- Interjection handling for important information
- Multi-turn conversation tracking

## Architecture

```
Listen v4 Architecture:

Audio Input (Mic/File)
    ↓
Speech Recognition
    ↓
Speaker Identification ←→ Speaker Database
    ↓
Context Analysis ← Conversation History
    ↓
Should Respond? (Decision)
    ├─ No → Continue Listening
    └─ Yes → Generate Response
             ↓
         LLM Response
             ↓
         Voice Synthesis
             ↓
         Audio Output
```

## Usage

### Basic Commands
```bash
# Start with microphone input
listen

# Process a WAV file
listen --file conversation.wav

# Disable TTS (text output only)
listen --no-tts

# Adjust response sensitivity
listen --confidence 0.8

# Specify speaker database
listen --db /path/to/speakers.db
```

### Trigger Phrases
The assistant responds to:
- **Wake words**: "Hey Listen", "OK Listen", "Listen up"
- **Questions**: Any utterance ending with "?"
- **Polite requests**: "Can you...", "Could you...", "Please..."
- **Commands**: "Tell me...", "Show me...", "Help me..."
- **Follow-ups**: Short responses after assistant speaks

### Voice Enrollment
Users can enroll their voice for personalized interactions:
1. Say "Enroll my voice" or "This is [name]"
2. Follow the prompts to provide voice samples
3. System remembers speaker for future conversations

## Technical Components

### Context Relevance Agent
- Analyzes each utterance for response triggers
- Maintains cooldown between responses
- Uses both rule-based and LLM-based detection
- Confidence scoring for response decisions

### Response Generator Agent
- Generates contextually appropriate responses
- Maintains conversation continuity
- Incorporates speaker information
- Follows conversational guidelines

### TTS Integration
- Automatic engine selection and fallback
- Asynchronous playback for smooth interaction
- Queue-based response management
- Error handling for TTS failures

## Configuration

### Environment Variables
```bash
# Set Hugging Face token for pyannote
export HF_TOKEN="your_token_here"

# Set Python path for imports
export PYTHONPATH=/home/tony/talk
```

### Dependencies
```bash
# Core audio
pip install SpeechRecognition pyaudio

# TTS support
pip install RealtimeTTS pyttsx3

# Speaker identification (if not already installed)
pip install pyannote.audio speechbrain
```

## Performance

### Response Times
- Wake word detection: <100ms
- Context analysis: ~200ms
- Response generation: ~500ms (LLM dependent)
- TTS latency: <250ms (system TTS)

### Resource Usage
- CPU: Moderate (spike during model inference)
- Memory: ~2GB (with models loaded)
- Disk: ~500MB (model cache)

## Example Conversation

```
[User]: Hey Listen, what can you help me with?
[Listen v4]: I can help you with various tasks like answering questions, 
            managing information, and having natural conversations.

[User]: Can you remember who I am?
[Listen v4]: Would you like to enroll your voice so I can recognize you 
            in future conversations?

[User]: Yes, enroll my voice. This is John.
[Listen v4]: Starting voice enrollment for John. Please say: 
            "The quick brown fox jumps over the lazy dog"

[User]: The quick brown fox jumps over the lazy dog
[Listen v4]: Sample 1 accepted. Please say: "Python is a powerful 
            programming language"

... (enrollment continues) ...

[Listen v4]: Enrollment complete! I'll remember you as John.

[John]: Thanks! What's the weather like?
[Listen v4]: I'd need to check current weather data for your location, John.
```

## Future Enhancements (v5 ideas)

1. **Multi-modal understanding**: Process visual inputs alongside audio
2. **Emotion detection**: Respond appropriately to user's emotional state
3. **Task execution**: Direct integration with system commands
4. **Memory persistence**: Long-term conversation memory
5. **Multi-language support**: Automatic language detection and switching
6. **Custom wake words**: User-configurable activation phrases
7. **Voice cloning**: Match user's speaking style
8. **Proactive assistance**: Anticipate needs based on context

## Troubleshooting

### TTS Not Working
- Check pyttsx3 installation: `pip install pyttsx3`
- Verify system has audio output device
- Try with `--no-tts` flag to confirm other features work

### No Response Generated
- Check confidence threshold (lower with `--confidence 0.5`)
- Verify LLM agent is configured properly
- Check conversation context is being maintained

### Speaker Not Recognized
- Ensure adequate voice samples during enrollment
- Check microphone quality and background noise
- Verify speaker database path is correct

## Summary

Listen v4 represents a significant evolution from v3, transforming a speaker identification system into a fully interactive conversational assistant. The addition of context-aware response generation and voice synthesis creates a natural, hands-free interaction experience suitable for various applications from personal assistants to meeting facilitators.