#!/usr/bin/env python3
"""
ReminiscingAgent - Main orchestrator for memory-based contextual retrieval.

This agent implements a human-like memory system that provides contextual traces
when considering new tasks. It coordinates multiple sub-agents to:
1. Categorize the current context/prompt 
2. Search through conversation and code history
3. Return relevant "memory traces" that might inform the current task

The agent uses LangGraph to orchestrate the workflow and manage state.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any, TypedDict
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    add_messages = None

from agent.agent import Agent
from agent.messages import Message, Role

from .context_categorization_agent import ContextCategorizationAgent
from .memory_trace_agent import MemoryTraceAgent
from .vector_store import ConversationVectorStore

log = logging.getLogger(__name__)

class ReminiscingState(TypedDict):
    """State structure for the reminiscing workflow."""
    context: str
    category: Optional[str]
    search_strategy: Optional[str]
    memory_traces: List[Dict[str, Any]]
    confidence: float
    final_response: str

class ReminiscingAgent(Agent):
    """
    Orchestrates memory-based contextual retrieval using human-like memory traces.
    
    This agent mimics human memory by:
    - Analyzing context to determine what kind of memory search is needed
    - Performing multi-dimensional searches through past conversations and code
    - Using spreading activation to find associatively related content
    - Returning contextual traces that "ring a bell" for the current task
    """
    
    def __init__(self, storage_path=None, **kwargs):
        """Initialize with memory system components.
        
        Args:
            storage_path: Optional path for persistent memory storage
            **kwargs: Additional arguments for Agent base class
        """
        # Extract storage_path before passing to parent
        super().__init__(roles=[
            "You are a memory specialist that helps recall relevant past context.",
            "You analyze new tasks and find related experiences from conversation history.",
            "You provide contextual traces that inform better decision making."
        ], **kwargs)
        
        # Initialize sub-agents and components
        self.categorization_agent = ContextCategorizationAgent(**kwargs)
        self.memory_trace_agent = MemoryTraceAgent(**kwargs)
        self.vector_store = ConversationVectorStore(storage_path=storage_path)
        
        # Initialize LangGraph workflow if available
        self.workflow = None
        if LANGGRAPH_AVAILABLE:
            self._setup_workflow()
        else:
            log.warning("LangGraph not available. Using simplified workflow.")
    
    def _setup_workflow(self):
        """Set up the LangGraph workflow for memory retrieval."""
        workflow = StateGraph(ReminiscingState)
        
        # Add nodes for each step
        workflow.add_node("categorize", self._categorize_context)
        workflow.add_node("search_memory", self._search_memory)
        workflow.add_node("format_response", self._format_response)
        
        # Define the workflow edges
        workflow.set_entry_point("categorize")
        workflow.add_edge("categorize", "search_memory")
        workflow.add_edge("search_memory", "format_response")
        workflow.add_edge("format_response", END)
        
        self.workflow = workflow.compile()
    
    def run(self, input_text: str) -> str:
        """
        Process input and return relevant memory traces.
        
        Args:
            input_text: Context or task description to find memories for
            
        Returns:
            Formatted response with relevant memory traces and insights
        """
        try:
            if self.workflow:
                # Use LangGraph workflow
                return self._run_with_langgraph(input_text)
            else:
                # Use simplified workflow
                return self._run_simplified(input_text)
                
        except Exception as e:
            log.error(f"Error in reminiscing process: {e}")
            return f"MEMORY_ERROR: Could not retrieve relevant memories: {str(e)}"
    
    def _run_with_langgraph(self, input_text: str) -> str:
        """Run the full LangGraph workflow."""
        initial_state = ReminiscingState(
            context=input_text,
            category=None,
            search_strategy=None,
            memory_traces=[],
            confidence=0.0,
            final_response=""
        )
        
        result = self.workflow.invoke(initial_state)
        return result["final_response"]
    
    def _run_simplified(self, input_text: str) -> str:
        """Run a simplified workflow without LangGraph."""
        # Step 1: Categorize context
        category_result = self.categorization_agent.run(input_text)
        
        # Step 2: Search memory based on category
        memory_result = self.memory_trace_agent.run(f"Context: {input_text}\nCategory: {category_result}")
        
        # Step 3: Format response
        return self._format_simple_response(input_text, category_result, memory_result)
    
    def _categorize_context(self, state: ReminiscingState) -> ReminiscingState:
        """Categorize the context to determine search strategy."""
        try:
            category_result = self.categorization_agent.run(state["context"])
            
            # Parse the category result to extract category and strategy
            category, strategy = self._parse_category_result(category_result)
            
            state["category"] = category
            state["search_strategy"] = strategy
            
            log.info(f"Context categorized as: {category} with strategy: {strategy}")
            
        except Exception as e:
            log.warning(f"Context categorization failed: {e}")
            state["category"] = "general"
            state["search_strategy"] = "semantic_search"
        
        return state
    
    def _search_memory(self, state: ReminiscingState) -> ReminiscingState:
        """Search memory using the determined strategy."""
        try:
            # Prepare search input
            search_input = {
                "context": state["context"],
                "category": state["category"],
                "strategy": state["search_strategy"]
            }
            
            # Perform memory search
            memory_result = self.memory_trace_agent.run(json.dumps(search_input))
            
            # Parse memory traces
            traces, confidence = self._parse_memory_result(memory_result)
            
            state["memory_traces"] = traces
            state["confidence"] = confidence
            
            log.info(f"Found {len(traces)} memory traces with confidence {confidence:.2f}")
            
        except Exception as e:
            log.warning(f"Memory search failed: {e}")
            state["memory_traces"] = []
            state["confidence"] = 0.0
        
        return state
    
    def _format_response(self, state: ReminiscingState) -> ReminiscingState:
        """Format the final response with memory traces and insights."""
        try:
            if not state["memory_traces"]:
                state["final_response"] = "No relevant memory traces found for this context."
                return state
            
            # Build formatted response
            response_parts = []
            response_parts.append("MEMORY_TRACES:")
            response_parts.append(f"Context Category: {state['category']}")
            response_parts.append(f"Search Strategy: {state['search_strategy']}")
            response_parts.append(f"Confidence: {state['confidence']:.2f}")
            response_parts.append("")
            
            # Add each memory trace
            for i, trace in enumerate(state["memory_traces"], 1):
                response_parts.append(f"Trace {i}: {trace.get('description', 'Unknown')}")
                response_parts.append(f"  Relevance: {trace.get('relevance', 0.0):.2f}")
                response_parts.append(f"  Context: {trace.get('context', 'No context')[:200]}...")
                response_parts.append("")
            
            state["final_response"] = "\n".join(response_parts)
            
        except Exception as e:
            log.error(f"Error formatting response: {e}")
            state["final_response"] = f"Error formatting memory traces: {str(e)}"
        
        return state
    
    def _parse_category_result(self, result: str) -> tuple[str, str]:
        """Parse categorization result into category and strategy."""
        try:
            # Look for structured output
            if "CATEGORY:" in result and "STRATEGY:" in result:
                lines = result.split('\n')
                category = None
                strategy = None
                
                for line in lines:
                    if line.startswith("CATEGORY:"):
                        category = line.split(":", 1)[1].strip()
                    elif line.startswith("STRATEGY:"):
                        strategy = line.split(":", 1)[1].strip()
                
                return category or "general", strategy or "semantic_search"
            
            # Fallback: classify based on keywords
            result_lower = result.lower()
            if "architectural" in result_lower or "design" in result_lower:
                return "architectural", "graph_traversal"
            elif "debugging" in result_lower or "error" in result_lower:
                return "debugging", "error_similarity"
            elif "implementation" in result_lower or "code" in result_lower:
                return "implementation", "code_similarity"
            else:
                return "general", "semantic_search"
                
        except Exception as e:
            log.warning(f"Error parsing category result: {e}")
            return "general", "semantic_search"
    
    def _parse_memory_result(self, result: str) -> tuple[List[Dict], float]:
        """Parse memory search result into traces and confidence."""
        try:
            # Try to parse JSON result first
            if result.startswith("{") or result.startswith("["):
                data = json.loads(result)
                if isinstance(data, dict):
                    return data.get("traces", []), data.get("confidence", 0.0)
                elif isinstance(data, list):
                    return data, 0.8  # Default confidence for list results
            
            # Fallback: parse text-based result
            traces = []
            confidence = 0.0
            
            # Simple parsing for text-based results
            if "TRACE:" in result:
                trace_sections = result.split("TRACE:")
                for section in trace_sections[1:]:  # Skip first empty part
                    trace = {
                        "description": section.split('\n')[0].strip(),
                        "context": section,
                        "relevance": 0.5  # Default relevance
                    }
                    traces.append(trace)
                confidence = min(len(traces) * 0.2, 1.0)
            
            return traces, confidence
            
        except Exception as e:
            log.warning(f"Error parsing memory result: {e}")
            return [], 0.0
    
    def _format_simple_response(self, context: str, category: str, memory: str) -> str:
        """Format a simple response for non-LangGraph workflow."""
        return f"""MEMORY_TRACES:
Context: {context[:100]}...
Category: {category}

{memory}

Note: Using simplified workflow (LangGraph not available)"""

    def store_conversation(self, conversation_data: Dict[str, Any]):
        """Store conversation data for future memory retrieval."""
        try:
            self.vector_store.store_conversation(conversation_data)
            log.info("Conversation stored in memory system")
        except Exception as e:
            log.error(f"Failed to store conversation: {e}")
    
    def store_code_context(self, code_data: Dict[str, Any]):
        """Store code context for future memory retrieval."""
        try:
            self.vector_store.store_code_context(code_data)
            log.info("Code context stored in memory system")
        except Exception as e:
            log.error(f"Failed to store code context: {e}")