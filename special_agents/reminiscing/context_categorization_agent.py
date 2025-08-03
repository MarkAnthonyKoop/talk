#!/usr/bin/env python3
"""
ContextCategorizationAgent - Classifies prompts to determine memory search strategy.

This agent analyzes incoming prompts/contexts and categorizes them to determine
the most appropriate memory search strategy. It acts as a routing mechanism for
the memory system, similar to how human attention mechanisms work.

Categories:
- architectural: High-level design decisions, system architecture
- implementation: Specific coding tasks, function implementation  
- debugging: Error analysis, troubleshooting, bug fixes
- research: Information gathering, learning new concepts
- general: General programming questions, unclear intent

Search Strategies:
- graph_traversal: For architectural decisions requiring relationship analysis
- code_similarity: For implementation tasks needing similar code examples
- error_similarity: For debugging tasks requiring similar error patterns
- semantic_search: For research and general questions
- temporal_search: For tasks requiring recent context or progression
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from agent.agent import Agent
from agent.messages import Message, Role

log = logging.getLogger(__name__)

class ContextCategorizationAgent(Agent):
    """
    Specialized agent for categorizing prompts and determining memory search strategy.
    
    This agent uses LLM analysis combined with keyword detection to classify
    incoming contexts and determine the optimal memory retrieval approach.
    """
    
    def __init__(self, **kwargs):
        """Initialize with specialized system prompt for context categorization."""
        roles = kwargs.pop("roles", [])
        
        # Add categorization-specific system prompts
        categorization_prompts = [
            "You are a context analysis specialist that categorizes programming tasks.",
            "Your job is to analyze prompts and determine the best memory search strategy.",
            "Always respond with a structured format specifying category and strategy.",
            "Be precise and consistent in your categorizations."
        ]
        
        # Combine with any existing roles
        roles = categorization_prompts + roles
        
        # Initialize the base agent
        super().__init__(roles=roles, **kwargs)
        
        # Define category patterns for fallback classification
        self.category_patterns = {
            'architectural': [
                r'\b(architect|design|structure|system|framework|pattern)\b',
                r'\b(component|module|service|api|interface)\b', 
                r'\b(scalab|maintain|extend|refactor)\b',
                r'\b(microservice|monolith|database|schema)\b'
            ],
            'implementation': [
                r'\b(implement|code|function|method|class)\b',
                r'\b(algorithm|logic|calculate|process)\b',
                r'\b(feature|functionality|requirement)\b',
                r'\b(loop|condition|variable|parameter)\b'
            ],
            'debugging': [
                r'\b(error|bug|fix|debug|troubleshoot)\b',
                r'\b(exception|crash|fail|issue|problem)\b',
                r'\b(trace|log|stack|breakpoint)\b',
                r'\b(wrong|incorrect|unexpected|broken)\b'
            ],
            'research': [
                r'\b(learn|understand|explain|tutorial)\b',
                r'\b(what is|how to|why does|best practice)\b',
                r'\b(documentation|example|guide|reference)\b',
                r'\b(compare|evaluate|choose|recommend)\b'
            ]
        }
        
        # Define strategy mappings
        self.category_strategies = {
            'architectural': 'graph_traversal',
            'implementation': 'code_similarity', 
            'debugging': 'error_similarity',
            'research': 'semantic_search',
            'general': 'semantic_search'
        }
    
    def run(self, input_text: str) -> str:
        """
        Analyze the input context and determine category and search strategy.
        
        Args:
            input_text: The context or prompt to categorize
            
        Returns:
            Structured response with category and recommended search strategy
        """
        try:
            # First attempt: Use LLM for intelligent categorization
            llm_result = self._llm_categorize(input_text)
            
            # Parse LLM result
            category, strategy, confidence = self._parse_llm_result(llm_result)
            
            # If LLM result is unclear, use pattern-based fallback
            if confidence < 0.7:
                log.info("LLM categorization unclear, using pattern-based fallback")
                fallback_category = self._pattern_based_categorize(input_text)
                if fallback_category != 'general':
                    category = fallback_category
                    strategy = self.category_strategies[category]
                    confidence = 0.8
            
            # Format structured response
            return self._format_response(category, strategy, confidence, input_text)
            
        except Exception as e:
            log.error(f"Error categorizing context: {e}")
            # Safe fallback
            return self._format_response('general', 'semantic_search', 0.5, input_text)
    
    def _llm_categorize(self, input_text: str) -> str:
        """Use LLM to categorize the input context."""
        prompt = f"""
Analyze this programming context and categorize it:

CONTEXT: {input_text}

Determine:
1. CATEGORY: One of [architectural, implementation, debugging, research, general]
2. STRATEGY: Recommended memory search approach
3. CONFIDENCE: How certain you are (0.0-1.0)

Categories:
- architectural: High-level design, system architecture, frameworks
- implementation: Specific coding tasks, algorithms, feature development
- debugging: Error analysis, troubleshooting, bug fixes
- research: Learning, explanations, best practices, comparisons
- general: Unclear intent or general programming questions

Strategies:
- graph_traversal: For architectural decisions needing relationship analysis
- code_similarity: For implementation needing similar code examples
- error_similarity: For debugging needing similar error patterns  
- semantic_search: For research and general knowledge questions
- temporal_search: For tasks requiring recent context progression

Respond EXACTLY in this format:
CATEGORY: [category]
STRATEGY: [strategy] 
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        
        self._append("user", prompt)
        response = self.call_ai()
        self._append("assistant", response)
        
        return response
    
    def _parse_llm_result(self, result: str) -> Tuple[str, str, float]:
        """Parse the LLM categorization result."""
        try:
            category = 'general'
            strategy = 'semantic_search'
            confidence = 0.5
            
            # Extract structured fields
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith('CATEGORY:'):
                    category = line.split(':', 1)[1].strip().lower()
                elif line.startswith('STRATEGY:'):
                    strategy = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
            
            # Validate category
            valid_categories = ['architectural', 'implementation', 'debugging', 'research', 'general']
            if category not in valid_categories:
                category = 'general'
                confidence = 0.3
            
            # Validate strategy
            valid_strategies = ['graph_traversal', 'code_similarity', 'error_similarity', 'semantic_search', 'temporal_search']
            if strategy not in valid_strategies:
                strategy = self.category_strategies.get(category, 'semantic_search')
                confidence = min(confidence, 0.6)
            
            return category, strategy, confidence
            
        except Exception as e:
            log.warning(f"Error parsing LLM result: {e}")
            return 'general', 'semantic_search', 0.3
    
    def _pattern_based_categorize(self, input_text: str) -> str:
        """Categorize using pattern matching as fallback."""
        text_lower = input_text.lower()
        category_scores = {}
        
        # Score each category based on pattern matches
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if not category_scores or max(category_scores.values()) == 0:
            return 'general'
        
        return max(category_scores, key=category_scores.get)
    
    def _format_response(self, category: str, strategy: str, confidence: float, context: str) -> str:
        """Format the structured response."""
        return f"""CATEGORY: {category}
STRATEGY: {strategy}
CONFIDENCE: {confidence:.2f}
CONTEXT_PREVIEW: {context[:100]}...

INTERPRETATION:
- Task Type: {category.title()}
- Memory Approach: {strategy.replace('_', ' ').title()}
- Confidence Level: {self._confidence_description(confidence)}

MEMORY_SEARCH_PARAMETERS:
- Primary Dimension: {self._get_primary_dimension(category)}
- Secondary Dimensions: {self._get_secondary_dimensions(category)}
- Search Depth: {self._get_search_depth(category)}
- Temporal Weight: {self._get_temporal_weight(category)}"""
    
    def _confidence_description(self, confidence: float) -> str:
        """Convert confidence score to human-readable description."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _get_primary_dimension(self, category: str) -> str:
        """Get the primary search dimension for a category."""
        dimension_map = {
            'architectural': 'system_design',
            'implementation': 'code_similarity',
            'debugging': 'error_patterns',
            'research': 'semantic_concepts',
            'general': 'semantic_similarity'
        }
        return dimension_map.get(category, 'semantic_similarity')
    
    def _get_secondary_dimensions(self, category: str) -> List[str]:
        """Get secondary search dimensions for a category."""
        secondary_map = {
            'architectural': ['component_relationships', 'design_patterns', 'scalability_concerns'],
            'implementation': ['algorithm_patterns', 'language_features', 'code_structure'],
            'debugging': ['exception_types', 'failure_modes', 'diagnostic_traces'],
            'research': ['conceptual_relationships', 'best_practices', 'comparative_analysis'],
            'general': ['contextual_similarity', 'keyword_matching']
        }
        return secondary_map.get(category, ['contextual_similarity'])
    
    def _get_search_depth(self, category: str) -> str:
        """Get recommended search depth for a category."""
        depth_map = {
            'architectural': 'deep',  # Need broader context for design decisions
            'implementation': 'medium',  # Need similar examples
            'debugging': 'focused',  # Need specific error patterns
            'research': 'broad',  # Need comprehensive understanding
            'general': 'medium'
        }
        return depth_map.get(category, 'medium')
    
    def _get_temporal_weight(self, category: str) -> str:
        """Get temporal weighting for a category."""
        temporal_map = {
            'architectural': 'low',  # Design principles are timeless
            'implementation': 'medium',  # Recent patterns matter
            'debugging': 'high',  # Recent errors most relevant
            'research': 'low',  # Knowledge is generally stable
            'general': 'medium'
        }
        return temporal_map.get(category, 'medium')