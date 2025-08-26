#!/usr/bin/env python3
"""
TTS Synthesis Agent

Converts text to speech.
Part of Listen v7's agentic architecture.
"""

import logging
from typing import Dict, Any
from agent.agent import Agent

log = logging.getLogger(__name__)


class TTSSynthesisAgent(Agent):
    """
    Agent for text-to-speech synthesis.
    """
    
    def __init__(self, **kwargs):
        roles = [
            "You synthesize natural sounding speech",
            "You control voice parameters and emotion",
            "You handle multi-language synthesis",
            "You optimize for clarity and naturalness"
        ]
        
        super().__init__(roles=roles, **kwargs)
        self.tts_engine = None
        self._init_tts()
        log.info("TTSSynthesisAgent initialized")
    
    def _init_tts(self):
        """Initialize TTS engine."""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except:
            log.warning("TTS engine not available")
    
    async def synthesize(self, text: str) -> Dict[str, Any]:
        """Synthesize speech from text."""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return {"success": True, "text": text}
            except Exception as e:
                log.error(f"TTS failed: {e}")
                print(f"🤖 {text}")
                return {"success": False, "text": text, "fallback": "console"}
        else:
            print(f"🤖 {text}")
            return {"success": True, "text": text, "method": "console"}
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self.tts_engine else "degraded"}
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        return f"TTSSynthesisAgent: {prompt}"
    
    async def cleanup(self):
        if self.tts_engine:
            self.tts_engine.stop()
        log.info("TTSSynthesisAgent cleanup complete")