#!/usr/bin/env python3
"""
RelevanceAgent - Filters and prioritizes content based on task relevance.

This agent analyzes transcribed content and determines its relevance to
the specified task, helping to filter noise and focus on actionable information.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from agent.agent import Agent

log = logging.getLogger(__name__)


class RelevanceAgent(Agent):
    """
    Agent that evaluates and filters content based on task relevance.
    
    This agent:
    1. Analyzes content for task-relevant keywords and concepts
    2. Scores relevance on multiple dimensions
    3. Identifies actionable items from content
    4. Maintains context of what has been processed
    """
    
    def __init__(self, 
                 task_description: str,
                 relevance_threshold: float = 0.3,
                 **kwargs):
        """
        Initialize the RelevanceAgent.
        
        Args:
            task_description: The task to evaluate relevance against
            relevance_threshold: Minimum score to consider content relevant
            **kwargs: Additional arguments for base Agent
        """
        roles = [
            "You are a relevance filtering and prioritization agent.",
            "You analyze content to determine if it relates to a specific task.",
            "You identify actionable items and important information.",
            "You filter out noise and irrelevant content."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.task_description = task_description
        self.relevance_threshold = relevance_threshold
        self.processed_content = []
        self.action_items = []
        
        # Extract task components for analysis
        self.task_keywords = self._extract_keywords(task_description)
        self.task_concepts = self._extract_concepts(task_description)
        self.task_actions = self._extract_actions(task_description)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'that',
            'this', 'these', 'those', 'it', 'its', 'we', 'our', 'us', 'them'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add n-grams for compound terms
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                keywords.append(f"{words[i]}_{words[i+1]}")
        
        return list(set(keywords))
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract high-level concepts from task description."""
        concepts = []
        
        # Technology-related concepts
        tech_patterns = [
            r'\b(api|rest|graphql|websocket)\b',
            r'\b(database|sql|nosql|mongodb|postgres)\b',
            r'\b(web|app|application|service|server)\b',
            r'\b(frontend|backend|fullstack)\b',
            r'\b(react|vue|angular|django|flask|express)\b',
            r'\b(docker|kubernetes|cloud|aws|gcp|azure)\b',
            r'\b(machine learning|ml|ai|neural|model)\b',
            r'\b(test|testing|unit|integration|e2e)\b'
        ]
        
        text_lower = text.lower()
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.extend(matches)
        
        # Action concepts
        action_patterns = [
            r'\b(create|build|implement|develop|design)\b',
            r'\b(fix|debug|resolve|repair|patch)\b',
            r'\b(optimize|improve|enhance|refactor)\b',
            r'\b(analyze|review|evaluate|assess)\b',
            r'\b(deploy|release|publish|ship)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract actionable items from task description."""
        actions = []
        
        # Look for imperative verbs and action phrases
        action_verbs = [
            'create', 'build', 'implement', 'add', 'write', 'develop',
            'fix', 'debug', 'resolve', 'update', 'modify', 'change',
            'test', 'verify', 'validate', 'check', 'ensure',
            'deploy', 'install', 'configure', 'setup', 'initialize',
            'analyze', 'review', 'document', 'explain', 'describe'
        ]
        
        text_lower = text.lower()
        for verb in action_verbs:
            if verb in text_lower:
                # Try to extract the full action phrase
                pattern = rf'\b{verb}\s+[\w\s]{{1,20}}'
                matches = re.findall(pattern, text_lower)
                actions.extend(matches)
        
        return list(set(actions))
    
    def evaluate_relevance(self, content: str) -> Dict[str, Any]:
        """
        Evaluate the relevance of content to the task.
        
        Args:
            content: The content to evaluate
            
        Returns:
            Dictionary with relevance scores and analysis
        """
        content_lower = content.lower()
        
        # Calculate keyword match score
        keyword_matches = sum(1 for kw in self.task_keywords if kw in content_lower)
        keyword_score = keyword_matches / max(len(self.task_keywords), 1)
        
        # Calculate concept match score
        concept_matches = sum(1 for concept in self.task_concepts if concept in content_lower)
        concept_score = concept_matches / max(len(self.task_concepts), 1)
        
        # Calculate action relevance
        action_matches = sum(1 for action in self.task_actions if action in content_lower)
        action_score = action_matches / max(len(self.task_actions), 1)
        
        # Check for specific triggers
        triggers = self._check_triggers(content_lower)
        
        # Calculate overall relevance score (weighted average)
        overall_score = (
            keyword_score * 0.3 +
            concept_score * 0.3 +
            action_score * 0.2 +
            (0.2 if triggers else 0)
        )
        
        # Identify specific actionable items
        actionable_items = self._extract_actionable_items(content)
        
        return {
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "keyword_score": keyword_score,
            "concept_score": concept_score,
            "action_score": action_score,
            "has_triggers": bool(triggers),
            "triggers": triggers,
            "actionable_items": actionable_items,
            "is_relevant": overall_score >= self.relevance_threshold,
            "matched_keywords": [kw for kw in self.task_keywords if kw in content_lower],
            "matched_concepts": [c for c in self.task_concepts if c in content_lower]
        }
    
    def _check_triggers(self, content: str) -> List[str]:
        """Check for specific trigger phrases that indicate high relevance."""
        triggers = []
        
        # Direct instruction triggers
        instruction_triggers = [
            "please", "could you", "can you", "i need", "we need",
            "let's", "let me", "go ahead", "start", "begin",
            "yes", "correct", "that's right", "exactly", "proceed"
        ]
        
        # Question triggers
        question_triggers = [
            "how do", "how can", "what is", "what are", "where is",
            "when should", "why does", "can we", "should we"
        ]
        
        # Confirmation triggers
        confirmation_triggers = [
            "confirm", "verify", "check", "make sure", "ensure"
        ]
        
        all_triggers = instruction_triggers + question_triggers + confirmation_triggers
        
        for trigger in all_triggers:
            if trigger in content:
                triggers.append(trigger)
        
        return triggers
    
    def _extract_actionable_items(self, content: str) -> List[str]:
        """Extract specific actionable items from content."""
        items = []
        
        # Look for numbered lists
        numbered_items = re.findall(r'\d+\.\s+([^\n]+)', content)
        items.extend(numbered_items)
        
        # Look for bullet points
        bullet_items = re.findall(r'[-*•]\s+([^\n]+)', content)
        items.extend(bullet_items)
        
        # Look for imperative sentences
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(sentence.lower().startswith(verb) for verb in 
                   ['create', 'add', 'implement', 'fix', 'update', 'test', 'deploy']):
                items.append(sentence)
        
        return list(set(items))
    
    def filter_transcriptions(self, transcriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of transcriptions for relevance.
        
        Args:
            transcriptions: List of transcription dictionaries
            
        Returns:
            Filtered list of relevant transcriptions with scores
        """
        relevant = []
        
        for trans in transcriptions:
            # Get the text content
            text = trans.get('text', '')
            if not text:
                continue
            
            # Evaluate relevance
            evaluation = self.evaluate_relevance(text)
            
            if evaluation['is_relevant']:
                # Merge transcription data with evaluation
                enhanced_trans = {**trans, **evaluation}
                relevant.append(enhanced_trans)
                
                # Track processed content
                self.processed_content.append({
                    "timestamp": trans.get('timestamp', datetime.now().isoformat()),
                    "content": text,
                    "relevance_score": evaluation['overall_score']
                })
                
                # Extract action items
                if evaluation['actionable_items']:
                    self.action_items.extend(evaluation['actionable_items'])
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return relevant
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get a summary of identified action items."""
        return {
            "total_action_items": len(self.action_items),
            "unique_actions": list(set(self.action_items)),
            "processed_count": len(self.processed_content),
            "average_relevance": (
                sum(p['relevance_score'] for p in self.processed_content) / 
                len(self.processed_content)
            ) if self.processed_content else 0
        }
    
    def run(self, prompt: str) -> str:
        """
        Process relevance filtering request.
        
        Args:
            prompt: Instructions or content to evaluate
            
        Returns:
            Relevance evaluation results
        """
        try:
            # Check if prompt contains transcriptions to filter
            if prompt.startswith('{') or prompt.startswith('['):
                # Assume JSON input of transcriptions
                try:
                    data = json.loads(prompt)
                    if isinstance(data, list):
                        filtered = self.filter_transcriptions(data)
                        return json.dumps({
                            "relevant_count": len(filtered),
                            "total_count": len(data),
                            "relevant_transcriptions": filtered,
                            "action_summary": self.get_action_summary()
                        }, indent=2)
                    elif isinstance(data, dict) and 'transcriptions' in data:
                        filtered = self.filter_transcriptions(data['transcriptions'])
                        return json.dumps({
                            "relevant_count": len(filtered),
                            "total_count": len(data['transcriptions']),
                            "relevant_transcriptions": filtered,
                            "action_summary": self.get_action_summary()
                        }, indent=2)
                except json.JSONDecodeError:
                    pass
            
            # Single content evaluation
            evaluation = self.evaluate_relevance(prompt)
            
            return json.dumps(evaluation, indent=2)
        
        except Exception as e:
            log.error(f"Error in RelevanceAgent: {e}")
            return f"Error evaluating relevance: {e}"
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Filter and prioritize content based on task relevance"
    
    @property  
    def triggers(self) -> List[str]:
        """Words that suggest relevance filtering is needed."""
        return ["filter", "relevant", "prioritize", "important", "actionable", "focus"]