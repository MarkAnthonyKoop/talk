"""
Input sources for Listen - modular transducers for various data streams.

This module provides base classes and implementations for different input
sources that feed into the Listen personal assistant system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import asyncio
import logging

log = logging.getLogger(__name__)


class InputSource(ABC):
    """
    Base class for all input sources (transducers).
    
    Each input source represents a different way of capturing
    context about the user's life - audio, email, social media, etc.
    """
    
    def __init__(self, name: str, priority: int = 5):
        """
        Initialize an input source.
        
        Args:
            name: Unique identifier for this source
            priority: Priority level (1-10, higher = more important)
        """
        self.name = name
        self.priority = priority
        self.is_active = False
        self.metadata = {}
        self.start_time = None
        
    @abstractmethod
    async def start(self) -> None:
        """Start capturing from this source."""
        self.is_active = True
        self.start_time = datetime.now()
        log.info(f"Started input source: {self.name}")
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop capturing from this source."""
        self.is_active = False
        log.info(f"Stopped input source: {self.name}")
    
    @abstractmethod
    async def capture(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Capture data from this source.
        
        Yields:
            Dictionary containing captured data with standard fields:
            - source: Name of this source
            - timestamp: When captured
            - data: The actual content
            - metadata: Additional context
            - confidence: Confidence score (0-1)
            - category: Suggested category
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that this source is available and configured.
        
        Returns:
            True if source is ready to use
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of this input source."""
        return {
            "name": self.name,
            "active": self.is_active,
            "priority": self.priority,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "metadata": self.metadata
        }
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for this source."""
        self.metadata[key] = value
    
    def format_output(self, 
                     data: Any, 
                     category: Optional[str] = None,
                     confidence: float = 1.0,
                     **kwargs) -> Dict[str, Any]:
        """
        Format captured data in standard format.
        
        Args:
            data: The captured content
            category: Suggested category for this data
            confidence: Confidence score (0-1)
            **kwargs: Additional metadata
            
        Returns:
            Standardized data dictionary
        """
        return {
            "source": self.name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "category": category,
            "confidence": confidence,
            "priority": self.priority,
            "metadata": {**self.metadata, **kwargs}
        }


class MultiSourceOrchestrator:
    """
    Orchestrates multiple input sources for the Listen system.
    
    This class manages multiple input sources, merges their streams,
    and provides a unified interface for consuming data from all sources.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.sources: Dict[str, InputSource] = {}
        self.is_running = False
        self.data_queue = asyncio.Queue()
        self.tasks = []
        
    def add_source(self, source: InputSource) -> None:
        """
        Add an input source to the orchestrator.
        
        Args:
            source: The input source to add
        """
        if source.validate():
            self.sources[source.name] = source
            log.info(f"Added input source: {source.name}")
        else:
            log.warning(f"Failed to add invalid source: {source.name}")
    
    def remove_source(self, name: str) -> None:
        """Remove an input source by name."""
        if name in self.sources:
            del self.sources[name]
            log.info(f"Removed input source: {name}")
    
    async def start(self) -> None:
        """Start all input sources."""
        self.is_running = True
        
        # Start each source
        for source in self.sources.values():
            await source.start()
            
            # Create capture task for each source
            task = asyncio.create_task(
                self._capture_from_source(source)
            )
            self.tasks.append(task)
        
        log.info(f"Orchestrator started with {len(self.sources)} sources")
    
    async def stop(self) -> None:
        """Stop all input sources."""
        self.is_running = False
        
        # Stop each source
        for source in self.sources.values():
            await source.stop()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        log.info("Orchestrator stopped")
    
    async def _capture_from_source(self, source: InputSource) -> None:
        """
        Capture data from a single source and queue it.
        
        Args:
            source: The source to capture from
        """
        try:
            async for data in source.capture():
                if not self.is_running:
                    break
                await self.data_queue.put(data)
        except Exception as e:
            log.error(f"Error capturing from {source.name}: {e}")
    
    async def get_next_data(self) -> Dict[str, Any]:
        """
        Get the next piece of data from any source.
        
        Returns:
            Data dictionary from one of the sources
        """
        return await self.data_queue.get()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all sources."""
        return {
            "running": self.is_running,
            "sources": {
                name: source.get_status() 
                for name, source in self.sources.items()
            },
            "queue_size": self.data_queue.qsize()
        }
    
    def prioritize_sources(self) -> List[str]:
        """
        Get list of source names ordered by priority.
        
        Returns:
            List of source names in priority order
        """
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        return [name for name, _ in sorted_sources]