#!/usr/bin/env python3
"""
Intent Detection Agent

Classifies user intent from transcribed text.
Part of Listen v7's agentic architecture.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from agent.agent import Agent

log = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intent."""
    ACTION = "action"
    CONVERSATION = "conversation"
    MIXED = "mixed"
    UNCLEAR = "unclear"


@dataclass
class Intent:
    """Structured intent representation."""
    intent_type: IntentType
    confidence: float
    action_category: Optional[str] = None
    command: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    entities: Optional[List[str]] = None
    raw_text: str = ""


class IntentDetectionAgent(Agent):
    """
    Specialized agent for intent classification.
    
    Determines whether user input is:
    - An action/command to execute
    - A conversational query
    - A mixed intent
    - Unclear/ambiguous
    """
    
    def __init__(self, **kwargs):
        roles = [
            "You classify user intent from natural language",
            "You identify commands, questions, and conversational elements",
            "You extract entities and parameters from user input",
            "You disambiguate unclear requests",
            "You provide confidence scores for classifications"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        # Intent patterns
        self.action_patterns = {
            "filesystem": {
                "keywords": ["list", "show", "create", "delete", "move", "copy", "rename"],
                "entities": ["file", "files", "folder", "directory", "document"],
                "regex": [r"(list|show|display)\s+(my\s+)?files?", r"(create|make)\s+.+\s+(file|folder)"]
            },
            "system": {
                "keywords": ["run", "execute", "launch", "start", "stop", "restart", "kill"],
                "entities": ["program", "process", "application", "service"],
                "regex": [r"(run|execute|launch)\s+\w+", r"(start|stop|restart)\s+.+service"]
            },
            "git": {
                "keywords": ["commit", "push", "pull", "merge", "branch", "status", "diff"],
                "entities": ["repository", "repo", "changes", "commits"],
                "regex": [r"git\s+\w+", r"(commit|push|pull)\s+.+changes?"]
            },
            "development": {
                "keywords": ["build", "compile", "test", "debug", "install", "deploy"],
                "entities": ["project", "code", "package", "dependency"],
                "regex": [r"(build|compile)\s+.+project", r"run\s+tests?"]
            }
        }
        
        self.conversation_patterns = [
            r"^(what|who|where|when|why|how)\s+",
            r"^(can you|could you|would you)\s+",
            r"^(tell me|explain|describe)\s+",
            r"^(hello|hi|hey|goodbye|bye)",
            r"(what do you think|how about|what if)",
            r"^(thanks|thank you|please)"
        ]
        
        log.info("IntentDetectionAgent initialized")
    
    async def detect(self, text: str, context: List[Dict] = None) -> Intent:
        """
        Detect intent from text.
        
        This is the main action called by PlanRunner.
        """
        text_lower = text.lower().strip()
        
        # Calculate scores
        action_score, action_category, command = self._score_action_intent(text_lower)
        conversation_score = self._score_conversation_intent(text_lower)
        
        # Consider context
        if context:
            context_modifier = self._analyze_context(context)
            action_score *= context_modifier.get("action_weight", 1.0)
            conversation_score *= context_modifier.get("conversation_weight", 1.0)
        
        # Determine intent type
        if action_score > 0.7 and conversation_score < 0.3:
            intent_type = IntentType.ACTION
            confidence = action_score
        elif conversation_score > 0.7 and action_score < 0.3:
            intent_type = IntentType.CONVERSATION
            confidence = conversation_score
        elif action_score > 0.4 and conversation_score > 0.4:
            intent_type = IntentType.MIXED
            confidence = (action_score + conversation_score) / 2
        else:
            intent_type = IntentType.UNCLEAR
            confidence = max(action_score, conversation_score)
        
        # Extract parameters if action
        parameters = None
        entities = []
        if intent_type in [IntentType.ACTION, IntentType.MIXED] and action_category:
            parameters = self._extract_parameters(text_lower, action_category)
            entities = self._extract_entities(text_lower, action_category)
        
        return Intent(
            intent_type=intent_type,
            confidence=confidence,
            action_category=action_category,
            command=command,
            parameters=parameters,
            entities=entities,
            raw_text=text
        )
    
    def _score_action_intent(self, text: str) -> tuple:
        """Score how likely the text is an action/command."""
        best_score = 0.0
        best_category = None
        best_command = None
        
        for category, patterns in self.action_patterns.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for kw in patterns["keywords"] if kw in text)
            if keyword_matches > 0:
                score += min(keyword_matches * 0.3, 1.0)
            
            # Check entities
            entity_matches = sum(1 for entity in patterns["entities"] if entity in text)
            if entity_matches > 0:
                score += min(entity_matches * 0.2, 0.5)
            
            # Check regex patterns
            for pattern in patterns.get("regex", []):
                if re.search(pattern, text):
                    score += 0.5
                    # Extract command from pattern match
                    match = re.search(pattern, text)
                    if match and not best_command:
                        best_command = match.group(0)
            
            if score > best_score:
                best_score = score
                best_category = category
        
        # Normalize score
        best_score = min(best_score, 1.0)
        
        # If no command extracted, use first verb as command
        if best_score > 0 and not best_command:
            verbs = re.findall(r'\b(list|show|create|delete|run|execute|commit|build)\b', text)
            if verbs:
                best_command = verbs[0]
        
        return best_score, best_category, best_command
    
    def _score_conversation_intent(self, text: str) -> float:
        """Score how likely the text is conversational."""
        score = 0.0
        
        # Check conversation patterns
        for pattern in self.conversation_patterns:
            if re.search(pattern, text):
                score += 0.4
        
        # Check for question marks
        if "?" in text:
            score += 0.3
        
        # Check for opinion/feeling words
        opinion_words = ["think", "feel", "believe", "seems", "maybe", "probably"]
        if any(word in text for word in opinion_words):
            score += 0.2
        
        # Normalize
        return min(score, 1.0)
    
    def _analyze_context(self, context: List[Dict]) -> Dict[str, float]:
        """Analyze conversation context to adjust intent weights."""
        if not context:
            return {"action_weight": 1.0, "conversation_weight": 1.0}
        
        # Look at recent intents
        recent_intents = [turn.get("intent_type") for turn in context[-3:] if "intent_type" in turn]
        
        # If recent context was all actions, boost action weight
        if recent_intents and all(intent == "ACTION" for intent in recent_intents):
            return {"action_weight": 1.2, "conversation_weight": 0.8}
        
        # If recent context was conversational, boost conversation weight
        if recent_intents and all(intent == "CONVERSATION" for intent in recent_intents):
            return {"action_weight": 0.8, "conversation_weight": 1.2}
        
        return {"action_weight": 1.0, "conversation_weight": 1.0}
    
    def _extract_parameters(self, text: str, category: str) -> Dict[str, Any]:
        """Extract parameters for the detected action category."""
        params = {}
        
        if category == "filesystem":
            # Extract operation
            if "list" in text or "show" in text:
                params["operation"] = "list"
            elif "create" in text or "make" in text:
                params["operation"] = "create"
            elif "delete" in text or "remove" in text:
                params["operation"] = "delete"
            
            # Extract target
            if "file" in text:
                params["target"] = "files"
            elif "folder" in text or "directory" in text:
                params["target"] = "directories"
            else:
                params["target"] = "."  # Current directory
            
            # Extract path if mentioned
            path_match = re.search(r'(?:in|at|from)\s+([/\w\-\.]+)', text)
            if path_match:
                params["path"] = path_match.group(1)
        
        elif category == "system":
            # Extract command to run
            cmd_match = re.search(r'(run|execute|launch)\s+(\w+)', text)
            if cmd_match:
                params["command"] = cmd_match.group(2)
        
        elif category == "git":
            # Extract git operation
            git_match = re.search(r'(commit|push|pull|merge|status|diff)', text)
            if git_match:
                params["operation"] = git_match.group(1)
            
            # Extract message if commit
            if "commit" in text:
                msg_match = re.search(r'["\'](.+)["\']', text)
                if msg_match:
                    params["message"] = msg_match.group(1)
        
        return params
    
    def _extract_entities(self, text: str, category: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        # Extract file/folder names
        if category == "filesystem":
            # Look for quoted strings
            quoted = re.findall(r'["\']([^"\']+)["\']', text)
            entities.extend(quoted)
            
            # Look for paths
            paths = re.findall(r'[/\w\-\.]+(?:/[/\w\-\.]+)+', text)
            entities.extend(paths)
        
        # Extract program names
        elif category == "system":
            # Common program names
            programs = re.findall(r'\b(python|node|npm|docker|git)\b', text)
            entities.extend(programs)
        
        return list(set(entities))  # Remove duplicates
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of intent detection."""
        return {
            "status": "healthy",
            "categories_configured": list(self.action_patterns.keys()),
            "conversation_patterns": len(self.conversation_patterns)
        }
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Agent interface for LLM-based intent analysis.
        
        Can provide more sophisticated intent detection using LLM.
        """
        if "detect intent" in prompt.lower():
            text = context.get("text", "") if context else ""
            intent = await self.detect(text)
            return f"Intent: {intent.intent_type.value} (confidence: {intent.confidence:.2f})"
        
        return f"IntentDetectionAgent: {prompt}"
    
    async def classify(self, transcript: str) -> Dict[str, Any]:
        """
        Classify intent from transcript.
        
        This is the main action for intent classification.
        """
        intent = await self.detect(transcript)
        
        return {
            "intent_type": intent.intent_type.value,
            "confidence": intent.confidence,
            "action_category": intent.action_category,
            "command": intent.command,
            "entities": intent.entities or [],
            "raw_text": transcript
        }
    
    async def cleanup(self):
        """Clean up intent detection resources."""
        log.info("IntentDetectionAgent cleanup complete")