# Listen v4 Setup Guide: TTS and Context Detection

## Overview
This guide provides step-by-step instructions for setting up the TTS (Text-to-Speech) and context relevance detection capabilities for Listen v4.

## Prerequisites
- Python 3.9+ (3.11 recommended)
- Audio input/output capability
- Internet connection for cloud TTS services (optional)

## Installation Steps

### 1. Core TTS Libraries

#### RealtimeTTS (Primary - Real-time Performance)
```bash
# Install with all engines
pip install realtimetts[all]

# Or minimal installation
pip install realtimetts

# Additional engines (install as needed)
pip install elevenlabs  # For ElevenLabs TTS
pip install azure-cognitiveservices-speech  # For Azure TTS
pip install openai  # For OpenAI TTS
```

#### Coqui TTS (Secondary - Voice Cloning)
```bash
pip install TTS

# For CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### MeloTTS (Alternative - Multilingual)
```bash
# Install from source for latest features
pip install git+https://github.com/myshell-ai/MeloTTS.git

# Or via pip (may be older version)
pip install melotts
```

### 2. Wake Word Detection

#### openWakeWord (Recommended)
```bash
pip install openwakeword

# Download pre-trained models
python -c "import openwakeword; openwakeword.utils.download_models()"
```

#### Porcupine (Commercial Alternative)
```bash
# Free tier available
pip install pvporcupine

# Get access key from https://picovoice.ai/
```

### 3. Speech Processing

```bash
# Core speech recognition
pip install SpeechRecognition pyaudio

# Audio processing
pip install librosa soundfile numpy

# Voice Activity Detection
pip install silero-vad

# If pyaudio installation fails on macOS:
brew install portaudio
pip install pyaudio

# If pyaudio installation fails on Linux:
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
pip install pyaudio
```

### 4. Intent Detection & NLP

```bash
# Transformers for advanced intent detection
pip install transformers torch tokenizers

# Download a conversation model
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium'); AutoModel.from_pretrained('microsoft/DialoGPT-medium')"

# Alternative: spaCy for lightweight NLP
pip install spacy
python -m spacy download en_core_web_sm
```

### 5. Cloud TTS APIs (Optional)

```bash
# Deepgram (fastest latency)
pip install deepgram-sdk

# Azure Cognitive Services
pip install azure-cognitiveservices-speech

# Google Cloud TTS
pip install google-cloud-texttospeech

# OpenAI
pip install openai>=1.0.0

# Amazon Polly
pip install boto3
```

### 6. Audio Playback

```bash
# Cross-platform option 1: pygame
pip install pygame

# Cross-platform option 2: pydub + simpleaudio
pip install pydub simpleaudio

# Linux additional requirements for audio
sudo apt-get install ffmpeg
```

## Configuration

### 1. Environment Variables

Create a `.env` file in your Listen v4 directory:

```bash
# Cloud TTS API Keys (optional)
OPENAI_API_KEY=your_openai_key_here
AZURE_SPEECH_KEY=your_azure_key_here
AZURE_SPEECH_REGION=your_azure_region
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json
DEEPGRAM_API_KEY=your_deepgram_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# TTS Settings
DEFAULT_TTS_ENGINE=realtimetts
TTS_RESPONSE_TIMEOUT=5.0
TTS_QUALITY_LEVEL=normal  # immediate, normal, high_quality

# Wake Word Settings
WAKE_WORD_THRESHOLD=0.5
WAKE_WORDS=hey assistant,listen,computer

# Context Detection Settings
RESPONSE_CONFIDENCE_THRESHOLD=0.6
CONVERSATION_CONTEXT_WINDOW=10
INTERJECTION_COOLDOWN_SECONDS=30
```

### 2. Voice Profiles Configuration

Create `voice_profiles.json`:

```json
{
  "default": {
    "engine": "realtimetts",
    "voice_id": "default",
    "speed": 1.0,
    "pitch": 1.0,
    "emotion": "neutral"
  },
  "user_profiles": {
    "user1": {
      "engine": "azure",
      "voice_id": "en-US-JennyNeural",
      "speed": 1.1,
      "pitch": 1.0,
      "emotion": "friendly"
    },
    "user2": {
      "engine": "deepgram",
      "voice_id": "aura-asteria-en",
      "speed": 0.9,
      "pitch": 0.95,
      "emotion": "professional"
    }
  }
}
```

### 3. Wake Words Setup

Train custom wake words with openWakeWord:

```python
# train_wake_word.py
import openwakeword
from openwakeword.model import Model

# Generate training data for custom wake word
wake_word = "hey assistant"
model = Model()

# This would typically involve generating many audio samples
# For production, use Piper TTS to generate training data
model.train_new_model(
    wake_word_phrase=wake_word,
    training_samples_dir="./wake_word_samples/",
    output_model_path=f"./models/{wake_word.replace(' ', '_')}.onnx"
)
```

## Testing the Setup

### 1. Basic TTS Test

```python
# test_tts.py
import asyncio
from docs.examples.listen_v4_tts_integration import TTSManager, ResponseUrgency

async def test_tts():
    manager = TTSManager()
    
    # Test different engines
    test_text = "Hello, this is a test of the TTS system."
    
    for urgency in ResponseUrgency:
        print(f"Testing {urgency.value} urgency...")
        try:
            audio_data, engine = await manager.synthesize_response(
                test_text, urgency
            )
            print(f"✅ Success with {engine.value}")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
```

### 2. Wake Word Detection Test

```python
# test_wake_word.py
import openwakeword
import numpy as np

# Load model
model = openwakeword.Model(wakeword_models=["hey_jarvis"])

# Test with mock audio
mock_audio = np.random.random(1600).astype(np.float32)  # 16kHz, 0.1s
prediction = model.predict(mock_audio)

print("Wake word predictions:", prediction)
```

### 3. Full Integration Test

```python
# Run the demo from the integration example
python docs/examples/listen_v4_tts_integration.py
```

## Performance Optimization

### 1. Latency Optimization

```python
# In your Listen v4 configuration
TTS_CONFIG = {
    "primary_engine": "realtimetts",  # Fastest
    "fallback_engines": ["system_tts", "melo_tts"],
    "max_latency_ms": 250,  # Fail if slower
    "enable_streaming": True,  # Stream audio as generated
    "buffer_size": 1024  # Audio buffer size
}
```

### 2. Memory Optimization

```python
# Model loading optimization
MODELS_CONFIG = {
    "lazy_loading": True,  # Load models on first use
    "cache_models": True,  # Keep models in memory
    "max_models_cached": 3,  # Limit memory usage
    "unload_timeout": 300  # Unload after 5 min idle
}
```

### 3. Quality vs Speed Trade-offs

```python
# Dynamic quality adjustment based on urgency
QUALITY_SETTINGS = {
    ResponseUrgency.IMMEDIATE: {
        "sample_rate": 16000,
        "bit_depth": 16,
        "quality": "fast"
    },
    ResponseUrgency.NORMAL: {
        "sample_rate": 22050, 
        "bit_depth": 16,
        "quality": "balanced"
    },
    ResponseUrgency.HIGH_QUALITY: {
        "sample_rate": 44100,
        "bit_depth": 24,
        "quality": "highest"
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Audio Not Playing
```bash
# Test audio output
python -c "import pygame; pygame.mixer.init(); pygame.mixer.music.load('test.wav'); pygame.mixer.music.play()"

# Check audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

#### 2. High Latency
- Switch to local TTS engines (RealtimeTTS, MeloTTS)
- Reduce audio buffer size
- Use lower quality settings for real-time responses
- Check network latency for cloud APIs

#### 3. Wake Word Not Detecting
- Adjust threshold in configuration
- Train custom wake word models
- Check microphone input levels
- Test with different wake word phrases

#### 4. Poor Voice Quality
- Use higher quality TTS engines (Azure, OpenAI)
- Increase sample rate and bit depth
- Check audio output device capabilities
- Test different voice profiles

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable TTS debug info
os.environ["TTS_DEBUG"] = "1"
os.environ["WAKE_WORD_DEBUG"] = "1"
```

## Security Considerations

### API Key Management
- Store API keys in environment variables, not code
- Use different keys for development/production
- Rotate keys regularly
- Monitor API usage and costs

### Privacy
- Audio data handling: Consider local-only processing
- Cloud TTS: Review data retention policies
- Voice profiles: Encrypt stored voice characteristics
- Wake word data: Keep training data secure

### Rate Limiting
```python
# Implement rate limiting for cloud APIs
RATE_LIMITS = {
    "openai_tts": {"requests_per_minute": 50},
    "azure_tts": {"characters_per_minute": 100000},
    "deepgram": {"requests_per_second": 10}
}
```

## Next Steps

1. **Integration**: Integrate with existing Listen v3 architecture
2. **Customization**: Train custom wake words and voice profiles
3. **Monitoring**: Add metrics and logging for performance tracking
4. **Optimization**: Fine-tune for your specific use case and hardware
5. **Testing**: Comprehensive testing with real users and scenarios

## References

- [RealtimeTTS Documentation](https://github.com/KoljaB/RealtimeTTS)
- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [openWakeWord Documentation](https://github.com/dscripka/openWakeWord)
- [Deepgram API Documentation](https://developers.deepgram.com/)
- [Azure Speech Services](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)