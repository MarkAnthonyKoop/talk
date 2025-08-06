#!/usr/bin/env python3
"""
NamingAgent - Generates concise, descriptive names for various contexts.

This agent follows the Talk framework contract: prompt in ==> completion out.
It specializes in generating appropriate names with specified constraints.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from agent.agent import Agent

log = logging.getLogger(__name__)

class NamingAgent(Agent):
    """
    Agent specialized in generating contextually appropriate names.
    
    Follows Talk contract:
    - Input: Description of what needs naming + constraints
    - Output: Appropriate name within constraints
    - No side effects: pure naming function
    """
    
    def __init__(self, **kwargs):
        """Initialize with naming-focused roles."""
        super().__init__(roles=[
            "You are a naming specialist that generates concise, descriptive names.",
            "You create names that are contextually appropriate and follow specified constraints.",
            "You consider readability, clarity, and convention when generating names.",
            "You output only the requested name, nothing else.",
            "You never add explanations, just the clean name."
        ], **kwargs)
    
    def run(self, input_text: str) -> str:
        """
        Generate a name based on the input description and constraints.
        
        Args:
            input_text: Description of what to name + constraints
            
        Returns:
            Clean name that meets the requirements
        """
        try:
            # Build a focused prompt for naming
            naming_prompt = self._build_naming_prompt(input_text)
            
            # Get LLM to generate the name
            self._append("user", naming_prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Clean and return the name
            clean_name = self._extract_clean_name(completion)
            
            return clean_name
            
        except Exception as e:
            log.error(f"Error generating name: {e}")
            # Return a safe fallback name
            return "unnamed_item"
    
    def _build_naming_prompt(self, input_text: str) -> str:
        """Build a focused prompt for name generation."""
        return f"""Generate a name based on this description and constraints:

{input_text}

Rules:
1. Output ONLY the name, no explanation
2. Use lowercase with underscores for multi-word names
3. Be descriptive but concise
4. Follow any size/character constraints specified
5. Make it readable and clear

Name:"""
    
    def _extract_clean_name(self, completion: str) -> str:
        """Extract clean name from LLM completion."""
        # Remove common prefixes/suffixes
        clean = completion.strip()
        
        # Remove quotes if present
        if clean.startswith('"') and clean.endswith('"'):
            clean = clean[1:-1]
        if clean.startswith("'") and clean.endswith("'"):
            clean = clean[1:-1]
        
        # Remove any explanation text (keep only first line)
        clean = clean.split('\n')[0].strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Name:", "name:", "Result:", "Output:"]
        for prefix in prefixes_to_remove:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        
        # Ensure it's a valid identifier-like name
        # Keep only alphanumeric and underscores
        import re
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', clean)
        
        # Remove multiple underscores
        clean = re.sub(r'_+', '_', clean)
        
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        
        # Ensure it's not empty
        if not clean:
            return "generated_name"
        
        return clean