#!/usr/bin/env python3
"""
ActiveListeningAgent - Manages active listening strategies.

This agent adjusts listening behavior based on context and tasks.
"""

import logging
from typing import Dict, List, Any, Optional
from agent.agent import Agent

log = logging.getLogger(__name__)


class ActiveListeningAgent(Agent):
    """
    Manages active listening and attention strategies.
    
    This agent:
    - Adjusts listening sensitivity based on context
    - Manages attention focus for specific tasks
    - Optimizes audio processing parameters
    """
    
    def __init__(self, **kwargs):
        """Initialize the active listening agent."""
        roles = [
            "You manage active listening strategies.",
            "You adjust attention based on context and tasks.",
            "You optimize audio processing for current needs."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.listening_mode = "normal"
        self.attention_targets = []
        
        log.info("ActiveListeningAgent initialized")
    
    def set_listening_mode(self, mode: str):
        """Set the current listening mode."""
        valid_modes = ["passive", "normal", "active", "focused"]
        if mode in valid_modes:
            self.listening_mode = mode
            log.info(f"Listening mode set to: {mode}")
        else:
            log.warning(f"Invalid listening mode: {mode}")
    
    def add_attention_target(self, target: str):
        """Add something to pay special attention to."""
        self.attention_targets.append(target)
        log.info(f"Added attention target: {target}")
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get current processing configuration."""
        configs = {
            "passive": {"sensitivity": 0.3, "timeout": 10},
            "normal": {"sensitivity": 0.5, "timeout": 5}, 
            "active": {"sensitivity": 0.7, "timeout": 2},
            "focused": {"sensitivity": 0.9, "timeout": 1}
        }
        
        return configs.get(self.listening_mode, configs["normal"])