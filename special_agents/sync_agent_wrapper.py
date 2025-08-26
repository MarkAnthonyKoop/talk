#!/usr/bin/env python3
"""
Synchronous Agent Wrapper for PlanRunner compatibility.

Makes async agents work with the synchronous PlanRunner.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


class SyncAgentWrapper:
    """
    Wrapper to make async agents work with sync PlanRunner.
    
    This wrapper runs async methods in a separate thread to avoid
    event loop conflicts when called from synchronous code.
    """
    
    def __init__(self, async_agent, agent_name: str):
        self.async_agent = async_agent
        self.agent_name = agent_name
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def run(self, prompt: str) -> str:
        """
        Synchronous run method for PlanRunner.
        
        Routes to the appropriate agent method based on agent type.
        """
        # Parse prompt for context
        context = self._parse_prompt_context(prompt)
        
        try:
            if self.agent_name == "voice_processor":
                return self._run_voice_processor(context)
            elif self.agent_name == "speaker_identifier":
                return self._run_speaker_identifier(context)
            elif self.agent_name == "intent_detector":
                return self._run_intent_detector(context)
            elif self.agent_name == "security_validator":
                return self._run_security_validator(context)
            elif self.agent_name == "mcp_executor":
                return self._run_mcp_executor(context)
            elif self.agent_name == "response_generator":
                return self._run_response_generator(context)
            elif self.agent_name == "tts_synthesizer":
                return self._run_tts_synthesizer(context)
            elif self.agent_name == "orchestrator":
                return self._run_orchestrator(prompt, context)
            else:
                return self._run_generic(prompt, context)
                
        except Exception as e:
            log.error(f"Error in {self.agent_name}: {e}")
            return json.dumps({
                "error": str(e),
                "agent": self.agent_name,
                "prompt": prompt[:100]
            })
    
    def _run_async(self, coro):
        """Run an async coroutine in a thread pool."""
        future = self.executor.submit(asyncio.run, coro)
        return future.result(timeout=10)
    
    def _run_voice_processor(self, context: Dict) -> str:
        """Run voice processor agent."""
        # Get audio data - try cache first, then context
        audio_data = getattr(self.async_agent, '_audio_data_cache', None)
        if audio_data is None:
            audio_data = context.get("audio_data", b"")
            # Handle case where audio_data might be a length integer from JSON
            if isinstance(audio_data, int):
                # Create mock audio data of specified length
                audio_data = b"mock_audio" * (audio_data // 10)
        
        result = self._run_async(self.async_agent.transcribe(audio_data))
        
        return json.dumps({
            "transcript": result.get("transcript", ""),
            "confidence": result.get("confidence", 0.0),
            "service_used": result.get("service_used", "unknown")
        })
    
    def _run_speaker_identifier(self, context: Dict) -> str:
        """Run speaker identification agent."""
        # Get audio data - try cache first, then context
        audio_data = getattr(self.async_agent, '_audio_data_cache', None)
        if audio_data is None:
            audio_data = context.get("audio_data", b"")
            if isinstance(audio_data, int):
                audio_data = b"mock_audio" * (audio_data // 10)
        
        result = self._run_async(self.async_agent.identify(audio_data))
        
        return json.dumps({
            "speaker_id": result.get("speaker_id", "unknown"),
            "confidence": result.get("confidence", 0.0)
        })
    
    def _run_intent_detector(self, context: Dict) -> str:
        """Run intent detection agent."""
        transcript = context.get("transcript", context.get("raw_prompt", ""))
        
        result = self._run_async(self.async_agent.classify(transcript))
        
        return json.dumps({
            "intent_type": result.get("intent_type", "UNKNOWN"),
            "confidence": result.get("confidence", 0.0),
            "entities": result.get("entities", [])
        })
    
    def _run_security_validator(self, context: Dict) -> str:
        """Run security validation agent."""
        command = context.get("command", context.get("raw_prompt", ""))
        
        result = self._run_async(self.async_agent.validate(command))
        
        return json.dumps({
            "safe": result.get("safe", False),
            "risk_level": result.get("risk_level", "unknown"),
            "warnings": result.get("warnings", [])
        })
    
    def _run_mcp_executor(self, context: Dict) -> str:
        """Run MCP executor agent."""
        command = context.get("command", context.get("raw_prompt", ""))
        
        result = self._run_async(self.async_agent.execute(command))
        
        return json.dumps({
            "success": result.get("success", False),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "command": command
        })
    
    def _run_response_generator(self, context: Dict) -> str:
        """Run response generation agent."""
        transcript = context.get("transcript", "")
        command_result = context.get("command_result", {})
        
        response = self._run_async(
            self.async_agent.generate(
                transcript,
                command_result=command_result
            )
        )
        
        return response  # Already a string
    
    def _run_tts_synthesizer(self, context: Dict) -> str:
        """Run TTS synthesis agent."""
        text = context.get("text", context.get("raw_prompt", ""))
        
        result = self._run_async(self.async_agent.synthesize(text))
        
        return json.dumps({
            "success": result.get("success", False),
            "method": result.get("method", "unknown"),
            "text": text
        })
    
    def _run_orchestrator(self, prompt: str, context: Dict) -> str:
        """Run orchestrator agent."""
        if hasattr(self.async_agent, 'run'):
            result = self._run_async(self.async_agent.run(prompt, context))
            return result
        else:
            return f"Orchestrator received: {prompt}"
    
    def _run_generic(self, prompt: str, context: Dict) -> str:
        """Run generic agent."""
        if hasattr(self.async_agent, 'run'):
            result = self._run_async(self.async_agent.run(prompt, context))
            return result
        else:
            return f"Agent {self.agent_name} processed: {prompt}"
    
    def _parse_prompt_context(self, prompt: str) -> Dict[str, Any]:
        """Parse context from prompt string."""
        # Try to parse as JSON first
        if prompt.startswith("{"):
            try:
                return json.loads(prompt)
            except:
                pass
        
        # Simple key extraction
        context = {"raw_prompt": prompt}
        
        # Extract common patterns from blackboard-enhanced prompts
        if "Original Task:" in prompt:
            parts = prompt.split("Original Task:", 1)
            if len(parts) > 1:
                task_line = parts[1].split("\n")[0].strip()
                context["original_task"] = task_line
        
        if "Instruction:" in prompt:
            parts = prompt.split("Instruction:", 1)
            if len(parts) > 1:
                context["instruction"] = parts[1].strip()
        
        if "Transcribed:" in prompt:
            parts = prompt.split("Transcribed:", 1)
            if len(parts) > 1:
                context["transcript"] = parts[1].strip()
        
        return context
    
    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)