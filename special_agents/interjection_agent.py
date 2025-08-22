"""
InterjectionAgent - Provides confident, contextual interjections.

This agent monitors conversations and information flow, interjecting
with relevant information when confidence is high that the user would
find it valuable.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import json

from agent.agent import Agent

log = logging.getLogger(__name__)


class InterjectionAgent(Agent):
    """
    Agent that provides timely, relevant interjections during conversations.
    
    This agent monitors the conversation context and available information,
    deciding when to interject with helpful information, reminders, or insights.
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 cooldown_seconds: int = 30,
                 **kwargs):
        """
        Initialize the InterjectionAgent.
        
        Args:
            confidence_threshold: Minimum confidence to interject (0-1)
            cooldown_seconds: Minimum seconds between interjections
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are a helpful assistant that provides timely, relevant information.",
            "You monitor conversations and interject only when you have highly relevant information.",
            "You are concise and to the point, respecting the flow of conversation.",
            "You provide context-aware suggestions, reminders, and insights.",
            "You learn from patterns to improve your timing and relevance."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_interjection = None
        
        # Context tracking
        self.conversation_context = deque(maxlen=20)
        self.information_context = deque(maxlen=50)
        self.user_preferences = {}
        self.interjection_history = []
        
        # Interjection patterns
        self.interjection_types = {
            "reminder": {
                "triggers": ["remind", "remember", "don't forget", "schedule", "appointment"],
                "confidence_boost": 0.1
            },
            "clarification": {
                "triggers": ["what", "how", "why", "explain", "confused", "unclear"],
                "confidence_boost": 0.15
            },
            "suggestion": {
                "triggers": ["should", "recommend", "suggest", "advice", "best"],
                "confidence_boost": 0.1
            },
            "correction": {
                "triggers": ["wrong", "incorrect", "mistake", "error", "actually"],
                "confidence_boost": 0.2
            },
            "information": {
                "triggers": ["tell me", "what is", "define", "meaning", "information"],
                "confidence_boost": 0.1
            },
            "connection": {
                "triggers": ["related", "similar", "like", "connection", "link"],
                "confidence_boost": 0.05
            }
        }
        
        log.info(f"Initialized InterjectionAgent (threshold: {confidence_threshold})")
    
    def should_interject(self,
                        conversation_turn: Dict[str, Any],
                        available_info: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Determine if the agent should interject.
        
        Args:
            conversation_turn: Current conversation turn
            available_info: Available information that might be relevant
            
        Returns:
            Tuple of (should_interject, confidence, interjection_type)
        """
        # Check cooldown
        if self.last_interjection:
            time_since = (datetime.now() - self.last_interjection).total_seconds()
            if time_since < self.cooldown_seconds:
                return False, 0, None
        
        # Analyze conversation for interjection triggers
        text = conversation_turn.get("text", "").lower()
        speaker = conversation_turn.get("speaker", "unknown")
        
        best_type = None
        best_confidence = 0
        
        for int_type, config in self.interjection_types.items():
            # Check for triggers
            trigger_found = any(trigger in text for trigger in config["triggers"])
            
            if trigger_found:
                # Calculate base confidence
                confidence = 0.5 + config["confidence_boost"]
                
                # Boost confidence based on available information relevance
                if available_info:
                    relevance_scores = [
                        info.get("relevance_score", 0) 
                        for info in available_info[:5]
                    ]
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)
                    confidence += avg_relevance * 0.3
                
                # Boost confidence for direct questions
                if "?" in text:
                    confidence += 0.1
                
                # Reduce confidence if speaker is the assistant
                if speaker.lower() in ["assistant", "system", "bot"]:
                    confidence *= 0.5
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_type = int_type
        
        # Check against threshold
        should_interject = best_confidence >= self.confidence_threshold
        
        if should_interject:
            log.debug(f"Interjection decision: YES (type: {best_type}, confidence: {best_confidence:.2f})")
        
        return should_interject, best_confidence, best_type
    
    def generate_interjection(self,
                             interjection_type: str,
                             context: Dict[str, Any],
                             available_info: List[Dict[str, Any]]) -> str:
        """
        Generate an appropriate interjection.
        
        Args:
            interjection_type: Type of interjection to generate
            context: Current conversation context
            available_info: Relevant information available
            
        Returns:
            The interjection text
        """
        # Build prompt based on interjection type
        if interjection_type == "reminder":
            prompt = self._build_reminder_prompt(context, available_info)
        elif interjection_type == "clarification":
            prompt = self._build_clarification_prompt(context, available_info)
        elif interjection_type == "suggestion":
            prompt = self._build_suggestion_prompt(context, available_info)
        elif interjection_type == "correction":
            prompt = self._build_correction_prompt(context, available_info)
        elif interjection_type == "information":
            prompt = self._build_information_prompt(context, available_info)
        elif interjection_type == "connection":
            prompt = self._build_connection_prompt(context, available_info)
        else:
            prompt = self._build_general_prompt(context, available_info)
        
        # Generate interjection
        interjection = self.run(prompt)
        
        # Record interjection
        self.last_interjection = datetime.now()
        self.interjection_history.append({
            "timestamp": self.last_interjection.isoformat(),
            "type": interjection_type,
            "context": context.get("text", "")[:100],
            "interjection": interjection
        })
        
        return interjection
    
    def _build_reminder_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for reminder interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""Based on the conversation about: "{recent_context}"
        
Relevant information from memory:
{json.dumps(relevant_items, indent=2)}

Provide a helpful reminder or relevant information the user might have forgotten.
Be concise and specific. Start with a gentle indicator like "Just a reminder:" or "Don't forget:"."""
    
    def _build_clarification_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for clarification interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""The user seems confused or asking about: "{recent_context}"
        
Relevant information that might help:
{json.dumps(relevant_items, indent=2)}

Provide a clear, helpful clarification. Be direct and informative.
Start with "To clarify:" or "Here's what you need to know:"."""
    
    def _build_suggestion_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for suggestion interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""The user is discussing: "{recent_context}"
        
Based on available information:
{json.dumps(relevant_items, indent=2)}

Provide a helpful suggestion or recommendation. Be actionable and specific.
Start with "You might want to:" or "I suggest:"."""
    
    def _build_correction_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for correction interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""There may be an error or misconception in: "{recent_context}"
        
Correct information:
{json.dumps(relevant_items, indent=2)}

Provide a gentle, respectful correction with the accurate information.
Start with "Actually:" or "Just to clarify:"."""
    
    def _build_information_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for information interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""The user is asking about: "{recent_context}"
        
Relevant information:
{json.dumps(relevant_items, indent=2)}

Provide the requested information clearly and concisely.
Focus on being helpful and informative."""
    
    def _build_connection_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build prompt for connection interjection."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""Current topic: "{recent_context}"
        
Related information that might be relevant:
{json.dumps(relevant_items, indent=2)}

Point out interesting connections or related information.
Start with "This relates to:" or "You might also be interested in:"."""
    
    def _build_general_prompt(self, context: Dict, info: List[Dict]) -> str:
        """Build general interjection prompt."""
        recent_context = context.get("text", "")
        relevant_items = [item.get("content", "") for item in info[:3]]
        
        return f"""Current conversation: "{recent_context}"
        
Available context:
{json.dumps(relevant_items, indent=2)}

Provide a helpful, relevant interjection that adds value to the conversation.
Be concise and respectful of the conversation flow."""
    
    def update_context(self, conversation_turn: Dict[str, Any]):
        """
        Update the agent's context with new conversation data.
        
        Args:
            conversation_turn: New conversation turn
        """
        self.conversation_context.append(conversation_turn)
    
    def add_information(self, info: Dict[str, Any]):
        """
        Add new information to the agent's knowledge.
        
        Args:
            info: New information item
        """
        self.information_context.append(info)
    
    def learn_from_feedback(self, interjection_id: int, was_helpful: bool):
        """
        Learn from user feedback on interjections.
        
        Args:
            interjection_id: ID of the interjection
            was_helpful: Whether the interjection was helpful
        """
        if 0 <= interjection_id < len(self.interjection_history):
            self.interjection_history[interjection_id]["feedback"] = was_helpful
            
            # Adjust confidence threshold based on feedback
            if was_helpful:
                # Lower threshold slightly to interject more
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
            else:
                # Raise threshold to be more selective
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
            
            log.info(f"Adjusted confidence threshold to {self.confidence_threshold:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interjection statistics."""
        total = len(self.interjection_history)
        
        if total == 0:
            return {"total_interjections": 0}
        
        # Count by type
        type_counts = {}
        helpful_count = 0
        
        for interjection in self.interjection_history:
            int_type = interjection.get("type", "unknown")
            type_counts[int_type] = type_counts.get(int_type, 0) + 1
            
            if interjection.get("feedback") == True:
                helpful_count += 1
        
        return {
            "total_interjections": total,
            "interjections_by_type": type_counts,
            "helpful_percentage": (helpful_count / total) * 100 if total > 0 else 0,
            "current_threshold": self.confidence_threshold,
            "cooldown_seconds": self.cooldown_seconds
        }
    
    def run(self, prompt: str) -> str:
        """
        Generate an interjection based on the prompt.
        
        Args:
            prompt: The prompt describing what to interject
            
        Returns:
            The interjection text
        """
        try:
            # Use the LLM to generate the interjection
            response = self.send_message(prompt)
            
            # Clean up the response
            interjection = response.strip()
            
            # Ensure it's concise
            if len(interjection) > 200:
                # Truncate to first complete sentence under 200 chars
                sentences = interjection.split(". ")
                interjection = sentences[0] + "."
            
            return interjection
        
        except Exception as e:
            log.error(f"Error generating interjection: {e}")
            return "I have relevant information that might help."
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Provide timely, relevant interjections during conversations"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest interjection might be needed."""
        all_triggers = []
        for config in self.interjection_types.values():
            all_triggers.extend(config["triggers"])
        return all_triggers