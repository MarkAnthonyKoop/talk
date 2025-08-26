#!/usr/bin/env python3
"""
Response Generation Agent

Generates natural language responses.
Part of Listen v7's agentic architecture.
"""

import logging
from typing import Dict, Any, List
from agent.agent import Agent

log = logging.getLogger(__name__)


class ResponseGenerationAgent(Agent):
    """
    Agent for generating natural language responses.
    """
    
    def __init__(self, service_tier: str = "standard", **kwargs):
        roles = [
            "You generate natural, helpful responses",
            "You contextualize responses based on conversation history",
            "You explain command results clearly",
            "You maintain conversational flow",
            "You adapt tone and style to the user"
        ]
        
        super().__init__(roles=roles, **kwargs)
        self.service_tier = service_tier
        log.info("ResponseGenerationAgent initialized")
    
    async def generate(self, transcript: str, intent: Any = None, 
                       command_result: Dict = None, speaker: Dict = None,
                       conversation_history: List = None) -> str:
        """Generate contextual response."""
        
        # Build context
        context_parts = []
        
        if transcript:
            context_parts.append(f"User said: {transcript}")
        
        if intent and hasattr(intent, "intent_type"):
            context_parts.append(f"Intent: {intent.intent_type}")
        
        if command_result:
            if command_result.get("success"):
                output = command_result.get("stdout", "")[:200]
                context_parts.append(f"Command executed successfully: {output}")
            else:
                error = command_result.get("error", command_result.get("stderr", "Unknown error"))
                context_parts.append(f"Command failed: {error}")
        
        # Generate response
        if command_result and command_result.get("success"):
            if "ls" in command_result.get("command", ""):
                files = command_result.get("stdout", "").split("\n")[:5]
                return f"I found {len(files)} items in your directory. Here are some: {', '.join(files)}"
            else:
                return f"Command executed successfully. {command_result.get('stdout', '')[:100]}"
        elif command_result:
            return f"I encountered an error: {command_result.get('error', 'Unknown error')}"
        else:
            return f"I understand you said: {transcript}. How can I help you?"
    
    async def generate_conversational(self, transcript: str, speaker: Dict = None,
                                     conversation_history: List = None) -> str:
        """Generate pure conversational response."""
        return f"You said: {transcript}. That's interesting!"
    
    async def generate_brief(self, result: Dict) -> str:
        """Generate brief response for quick actions."""
        if result.get("success"):
            return "Done!"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "service_tier": self.service_tier}
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        return f"ResponseGenerationAgent: {prompt}"
    
    async def cleanup(self):
        log.info("ResponseGenerationAgent cleanup complete")