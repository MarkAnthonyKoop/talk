#!/usr/bin/env python3
"""
MultiSourceOrchestrator - Coordinates multiple input sources.

This agent manages audio from different sources and coordinates processing.
"""

import logging
from typing import Dict, List, Any, Optional
from agent.agent import Agent

log = logging.getLogger(__name__)


class MultiSourceOrchestrator(Agent):
    """
    Orchestrates multiple audio input sources.
    
    This agent:
    - Manages microphone, file, and streaming inputs
    - Coordinates processing pipelines
    - Handles source prioritization
    """
    
    def __init__(self, **kwargs):
        """Initialize the orchestrator."""
        roles = [
            "You coordinate multiple audio input sources.",
            "You manage processing pipelines for different inputs.",
            "You handle source prioritization and switching."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.active_sources = {}
        self.processing_queue = []
        
        log.info("MultiSourceOrchestrator initialized")
    
    def add_source(self, source_id: str, source_type: str, config: Dict[str, Any]):
        """Add an audio source."""
        self.active_sources[source_id] = {
            "type": source_type,
            "config": config,
            "active": True,
            "priority": config.get("priority", 0)
        }
        log.info(f"Added audio source: {source_id} ({source_type})")
    
    def process_audio(self, source_id: str, audio_data: Any) -> Dict[str, Any]:
        """Process audio from a specific source."""
        if source_id not in self.active_sources:
            return {"error": "Unknown source"}
        
        return {
            "source_id": source_id,
            "processed": True,
            "timestamp": audio_data.get("timestamp") if hasattr(audio_data, "get") else None
        }
    
    def get_active_sources(self) -> List[str]:
        """Get list of active sources."""
        return [sid for sid, src in self.active_sources.items() if src["active"]]