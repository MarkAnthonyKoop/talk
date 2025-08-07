#!/usr/bin/env python3
"""
Enhanced ReminiscingAgent with optional Serena semantic search.

This version adds:
- Optional Serena integration for semantic code analysis
- Choice of vector store (basic or enhanced with real embeddings)
- Dual-memory architecture (conversation + code)
- Smart routing to decide when to use Serena
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from pathlib import Path

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from agent.agent import Agent

from .context_categorization_agent import ContextCategorizationAgent
from .memory_trace_agent import MemoryTraceAgent
from .vector_store import ConversationVectorStore
from .enhanced_vector_store_v2 import EnhancedVectorStoreV2

log = logging.getLogger(__name__)


class EnhancedReminiscingState(TypedDict):
    """Enhanced state structure for dual-memory workflow."""
    context: str
    category: Optional[str]
    search_strategy: Optional[str]
    
    # Conversation memory
    memory_traces: List[Dict[str, Any]]
    confidence: float
    
    # Semantic code context (NEW)
    code_context: Optional[str]
    semantic_results: List[Dict[str, Any]]
    semantic_confidence: float
    use_serena: bool
    
    # Combined response
    final_response: str
    response_type: str  # "conversation_only", "code_only", "merged"
    performance_metrics: Dict[str, float]


class EnhancedReminiscingAgent(Agent):
    """
    Enhanced ReminiscingAgent with optional Serena semantic search.
    
    Features:
    - Dual-memory architecture (conversation + semantic code)
    - Optional Serena integration (off by default)
    - Enhanced vector store with real embeddings
    - Smart routing based on query type
    - Performance tracking
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 enable_semantic_search: bool = False,  # OFF by default as requested
                 use_enhanced_vector_store: bool = True,
                 auto_route_to_serena: bool = False,
                 **kwargs):
        """
        Initialize enhanced reminiscing agent.
        
        Args:
            storage_path: Path for persistent memory storage
            enable_semantic_search: Enable Serena integration (default: False)
            use_enhanced_vector_store: Use real embeddings (default: True)
            auto_route_to_serena: Automatically use Serena for code queries (default: False)
            **kwargs: Additional arguments for Agent base class
        """
        super().__init__(roles=[
            "You are an enhanced memory specialist with dual-memory capabilities.",
            "You search both conversation history and semantic code context.",
            "You provide rich contextual traces combining past experiences and code understanding."
        ], **kwargs)
        
        # Initialize sub-agents
        self.categorization_agent = ContextCategorizationAgent(**kwargs)
        self.memory_trace_agent = MemoryTraceAgent(**kwargs)
        
        # Choose vector store implementation
        if use_enhanced_vector_store:
            log.info("Using enhanced vector store with real embeddings")
            self.vector_store = EnhancedVectorStoreV2(storage_path=storage_path)
        else:
            log.info("Using basic vector store with hash-based embeddings")
            self.vector_store = ConversationVectorStore(storage_path=storage_path)
        
        # Serena integration
        self.semantic_search_enabled = enable_semantic_search
        self.auto_route_to_serena = auto_route_to_serena
        self.serena_agent = None
        
        if self.semantic_search_enabled:
            try:
                from special_agents.reminiscing.serena_agent import SerenaAgent
                self.serena_agent = SerenaAgent(**kwargs)
                log.info("Serena semantic search enabled successfully")
            except ImportError as e:
                log.warning(f"Serena not available: {e}. Falling back to standard search.")
                self.semantic_search_enabled = False
        
        # Performance tracking
        self.performance_metrics = {
            "conversation_search_time": [],
            "semantic_search_time": [],
            "total_queries": 0,
            "serena_uses": 0
        }
        
        # Initialize workflow
        self.workflow = None
        if LANGGRAPH_AVAILABLE:
            self._setup_enhanced_workflow()
        else:
            log.warning("LangGraph not available. Using simplified workflow.")
    
    def _setup_enhanced_workflow(self):
        """Set up enhanced workflow with optional semantic search."""
        workflow = StateGraph(EnhancedReminiscingState)
        
        # Add nodes
        workflow.add_node("categorize", self._categorize_context)
        workflow.add_node("search_conversation", self._search_conversation_memory)
        workflow.add_node("search_semantic", self._search_code_semantically)
        workflow.add_node("merge_contexts", self._merge_memory_contexts)
        workflow.add_node("format_response", self._format_enhanced_response)
        
        # Define edges
        workflow.set_entry_point("categorize")
        workflow.add_edge("categorize", "search_conversation")
        workflow.add_edge("search_conversation", "search_semantic")
        workflow.add_edge("search_semantic", "merge_contexts")
        workflow.add_edge("merge_contexts", "format_response")
        workflow.add_edge("format_response", END)
        
        self.workflow = workflow.compile()
    
    def _should_use_serena(self, context: str, category: Optional[str] = None) -> bool:
        """
        Determine if Serena's overhead is worth it for this query.
        
        Args:
            context: The query context
            category: Query category from categorization
            
        Returns:
            True if Serena should be used
        """
        if not self.semantic_search_enabled or not self.serena_agent:
            return False
        
        if not self.auto_route_to_serena:
            # Only use if explicitly code-related
            return self._is_code_query(context, category)
        
        # Auto-routing heuristics
        context_lower = context.lower()
        
        # High-value Serena cases
        code_indicators = [
            "class", "function", "method", "implementation",
            "bug", "error", "debug", "fix",
            "import", "dependency", "module",
            "refactor", "optimize", "performance"
        ]
        
        if any(indicator in context_lower for indicator in code_indicators):
            return True
        
        # Check for specific code patterns
        if re.search(r'\b[A-Z][a-z]+[A-Z]\w*\b', context):  # CamelCase
            return True
        if re.search(r'\b\w+\(\)', context):  # Function calls
            return True
        if re.search(r'\.\w+\(', context):  # Method calls
            return True
        
        # Category-based decision
        if category in ["implementation", "debugging", "architectural", "code_review"]:
            return True
        
        return False
    
    def _is_code_query(self, context: str, category: Optional[str] = None) -> bool:
        """Check if query is explicitly code-related."""
        code_keywords = ["code", "function", "class", "file", "implementation", "bug"]
        context_lower = context.lower()
        return any(kw in context_lower for kw in code_keywords)
    
    def _categorize_context(self, state: EnhancedReminiscingState) -> EnhancedReminiscingState:
        """Categorize the context and decide on search strategy."""
        result = self.categorization_agent.run(state["context"])
        
        # Parse categorization result
        category = "general"
        search_strategy = "similarity"
        
        if "Category:" in result:
            lines = result.split('\n')
            for line in lines:
                if "Category:" in line:
                    category = line.split("Category:")[-1].strip().lower()
                elif "Strategy:" in line:
                    search_strategy = line.split("Strategy:")[-1].strip().lower()
        
        state["category"] = category
        state["search_strategy"] = search_strategy
        state["use_serena"] = self._should_use_serena(state["context"], category)
        
        log.info(f"Categorized as: {category}, Strategy: {search_strategy}, Use Serena: {state['use_serena']}")
        
        return state
    
    def _search_conversation_memory(self, state: EnhancedReminiscingState) -> EnhancedReminiscingState:
        """Search conversation memory using vector store."""
        start_time = time.time()
        
        try:
            # Use enhanced search if available
            if isinstance(self.vector_store, EnhancedVectorStoreV2):
                results = self.vector_store.search_memories(
                    state["context"],
                    memory_type="conversation",
                    limit=5,
                    use_code_structure=True
                )
            else:
                results = self.vector_store.search_memories(
                    state["context"],
                    memory_type="conversation",
                    limit=5
                )
            
            state["memory_traces"] = results
            state["confidence"] = max([r.get("similarity_score", 0) for r in results], default=0.0)
            
            elapsed = time.time() - start_time
            self.performance_metrics["conversation_search_time"].append(elapsed)
            
            log.info(f"Found {len(results)} conversation memories in {elapsed:.2f}s")
            
        except Exception as e:
            log.error(f"Conversation search failed: {e}")
            state["memory_traces"] = []
            state["confidence"] = 0.0
        
        return state
    
    def _search_code_semantically(self, state: EnhancedReminiscingState) -> EnhancedReminiscingState:
        """Search for code context using Serena or enhanced vector store."""
        if not state["use_serena"]:
            # Use enhanced vector store for code search
            start_time = time.time()
            
            try:
                if isinstance(self.vector_store, EnhancedVectorStoreV2):
                    results = self.vector_store.search_memories(
                        state["context"],
                        memory_type="code",
                        limit=5,
                        use_code_structure=True
                    )
                    
                    # Format as code context
                    if results:
                        code_pieces = []
                        for r in results:
                            if "content" in r:
                                code_pieces.append(f"// From {r.get('memory_id', 'unknown')}:\n{r['content'][:500]}")
                        
                        state["code_context"] = "\n\n".join(code_pieces)
                        state["semantic_confidence"] = max([r.get("similarity_score", 0) for r in results], default=0.0)
                    else:
                        state["code_context"] = ""
                        state["semantic_confidence"] = 0.0
                else:
                    state["code_context"] = ""
                    state["semantic_confidence"] = 0.0
                
                elapsed = time.time() - start_time
                log.info(f"Vector store code search took {elapsed:.2f}s")
                
            except Exception as e:
                log.error(f"Vector store code search failed: {e}")
                state["code_context"] = ""
                state["semantic_confidence"] = 0.0
            
            state["semantic_results"] = []
            return state
        
        # Use Serena for semantic analysis
        start_time = time.time()
        
        try:
            log.info("Using Serena for semantic code analysis")
            
            # Build optimized query for Serena
            serena_query = self._build_serena_query(state["context"], state["category"])
            
            # Call Serena
            serena_result = self.serena_agent.run(serena_query)
            
            # Parse results
            code_context, semantic_data = self._parse_serena_results(serena_result)
            
            state["code_context"] = code_context
            state["semantic_results"] = semantic_data.get("results", [])
            state["semantic_confidence"] = semantic_data.get("confidence", 0.5)
            
            elapsed = time.time() - start_time
            self.performance_metrics["semantic_search_time"].append(elapsed)
            self.performance_metrics["serena_uses"] += 1
            
            log.info(f"Serena analysis completed in {elapsed:.2f}s")
            
        except Exception as e:
            log.error(f"Serena search failed: {e}")
            state["code_context"] = ""
            state["semantic_results"] = []
            state["semantic_confidence"] = 0.0
        
        return state
    
    def _merge_memory_contexts(self, state: EnhancedReminiscingState) -> EnhancedReminiscingState:
        """Merge conversation and code contexts."""
        has_conversation = len(state.get("memory_traces", [])) > 0
        has_code = bool(state.get("code_context", "").strip())
        
        if has_conversation and has_code:
            state["response_type"] = "merged"
        elif has_conversation:
            state["response_type"] = "conversation_only"
        elif has_code:
            state["response_type"] = "code_only"
        else:
            state["response_type"] = "none_found"
        
        # Calculate combined confidence
        conv_conf = state.get("confidence", 0.0)
        code_conf = state.get("semantic_confidence", 0.0)
        
        if conv_conf > 0 and code_conf > 0:
            combined = (conv_conf * 0.6 + code_conf * 0.4)  # Weight conversation higher
        else:
            combined = max(conv_conf, code_conf)
        
        state["confidence"] = combined
        
        log.info(f"Context merge: {state['response_type']} (confidence: {combined:.2f})")
        
        return state
    
    def _format_enhanced_response(self, state: EnhancedReminiscingState) -> EnhancedReminiscingState:
        """Format the enhanced response with both memory types."""
        response_parts = []
        response_parts.append("ENHANCED_MEMORY_TRACES:")
        response_parts.append(f"Category: {state.get('category', 'unknown')}")
        response_parts.append(f"Response Type: {state.get('response_type', 'none_found')}")
        
        if state.get("use_serena"):
            response_parts.append("Semantic Analysis: Serena")
        else:
            response_parts.append("Semantic Analysis: Vector Store")
        
        response_parts.append("")
        
        # Add conversation memories
        if state.get("memory_traces"):
            response_parts.append("CONVERSATION MEMORIES:")
            for i, trace in enumerate(state["memory_traces"][:3], 1):
                response_parts.append(f"  {i}. Score: {trace.get('similarity_score', 0):.2f}")
                content = trace.get("content", "")[:200]
                response_parts.append(f"     {content}...")
            response_parts.append("")
        
        # Add code context
        if state.get("code_context"):
            response_parts.append("CODE CONTEXT:")
            code = state["code_context"][:500]
            response_parts.append(f"  {code}...")
            response_parts.append("")
        
        # Add performance metrics
        if state.get("use_serena") and self.performance_metrics["semantic_search_time"]:
            last_serena_time = self.performance_metrics["semantic_search_time"][-1]
            response_parts.append(f"PERFORMANCE: Serena took {last_serena_time:.2f}s")
        
        response_parts.append(f"CONFIDENCE: {state.get('confidence', 0.0):.2f}")
        
        state["final_response"] = "\n".join(response_parts)
        state["performance_metrics"] = self.get_performance_summary()
        
        return state
    
    def _build_serena_query(self, context: str, category: Optional[str]) -> str:
        """Build optimized query for SerenaAgent."""
        prefixes = {
            "implementation": "Find implementation details for:",
            "debugging": "Find code related to debugging:",
            "architectural": "Analyze architecture for:",
            "code_review": "Find code to review:",
            "general": "Find relevant code for:"
        }
        
        prefix = prefixes.get(category, prefixes["general"])
        return f"{prefix} {context}"
    
    def _parse_serena_results(self, serena_result: str) -> Tuple[str, Dict]:
        """Parse SerenaAgent results."""
        code_context = ""
        semantic_data = {"results": [], "confidence": 0.5}
        
        # Extract structured data from Serena result
        if "SERENA_ANALYSIS_COMPLETE" in serena_result:
            # Parse file reference
            import re
            file_match = re.search(r'\.talk/serena/[^\s]+\.json', serena_result)
            
            if file_match:
                file_path = Path.cwd() / file_match.group(0)
                if file_path.exists():
                    try:
                        with open(file_path) as f:
                            data = json.load(f)
                        
                        # Extract relevant parts
                        if "results" in data:
                            semantic_data = data["results"]
                            
                            # Build code context
                            if "code_snippets" in semantic_data:
                                code_context = "\n\n".join(semantic_data["code_snippets"][:3])
                            elif "lsp_capabilities" in semantic_data:
                                code_context = f"LSP Analysis: {', '.join(semantic_data['lsp_capabilities'])}"
                        
                        semantic_data["confidence"] = 0.8
                        
                    except Exception as e:
                        log.warning(f"Failed to parse Serena results: {e}")
        
        # Fallback: extract from text
        if not code_context and "SEMANTIC_CAPABILITIES_USED:" in serena_result:
            lines = serena_result.split('\n')
            for i, line in enumerate(lines):
                if "SEMANTIC_CAPABILITIES_USED:" in line:
                    # Get next few lines
                    code_context = "\n".join(lines[i+1:i+4])
                    break
        
        return code_context, semantic_data
    
    def run(self, input_text: str) -> str:
        """
        Run enhanced reminiscing with optional Serena.
        
        Args:
            input_text: Query or context to search for
            
        Returns:
            Enhanced memory traces with performance metrics
        """
        self.performance_metrics["total_queries"] += 1
        
        if self.workflow:
            # Use LangGraph workflow
            initial_state = EnhancedReminiscingState(
                context=input_text,
                category=None,
                search_strategy=None,
                memory_traces=[],
                confidence=0.0,
                code_context=None,
                semantic_results=[],
                semantic_confidence=0.0,
                use_serena=False,
                final_response="",
                response_type="none_found",
                performance_metrics={}
            )
            
            result_state = self.workflow.invoke(initial_state)
            return result_state["final_response"]
        else:
            # Simplified workflow
            return self._run_simplified(input_text)
    
    def _run_simplified(self, input_text: str) -> str:
        """Simplified workflow without LangGraph."""
        # Categorize
        category_result = self.categorization_agent.run(input_text)
        category = "general"
        if "Category:" in category_result:
            category = category_result.split("Category:")[-1].split('\n')[0].strip().lower()
        
        use_serena = self._should_use_serena(input_text, category)
        
        # Search conversation memory
        conv_results = self.vector_store.search_memories(input_text, "conversation", 5)
        
        # Search code (Serena or vector store)
        code_context = ""
        if use_serena and self.serena_agent:
            try:
                serena_result = self.serena_agent.run(self._build_serena_query(input_text, category))
                code_context, _ = self._parse_serena_results(serena_result)
            except:
                pass
        elif isinstance(self.vector_store, EnhancedVectorStoreV2):
            code_results = self.vector_store.search_memories(input_text, "code", 3)
            if code_results:
                code_context = "\n".join([r.get("content", "")[:200] for r in code_results])
        
        # Format response
        response = f"""ENHANCED_MEMORY_TRACES:
Category: {category}
Method: {'Serena' if use_serena else 'Vector Store'}

CONVERSATION MEMORIES:
{len(conv_results)} found

CODE CONTEXT:
{'Available' if code_context else 'None'}

Performance: {self.get_performance_summary()}
"""
        
        return response
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        import numpy as np
        
        summary = {
            "total_queries": self.performance_metrics["total_queries"],
            "serena_uses": self.performance_metrics["serena_uses"],
            "serena_percentage": (self.performance_metrics["serena_uses"] / 
                                 max(1, self.performance_metrics["total_queries"])) * 100
        }
        
        if self.performance_metrics["conversation_search_time"]:
            summary["avg_conversation_search"] = np.mean(self.performance_metrics["conversation_search_time"])
        
        if self.performance_metrics["semantic_search_time"]:
            summary["avg_semantic_search"] = np.mean(self.performance_metrics["semantic_search_time"])
        
        return summary
    
    def store_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Store conversation in vector store."""
        return self.vector_store.store_conversation(conversation_data)
    
    def store_code_context(self, code_data: Dict[str, Any]) -> str:
        """Store code context in vector store."""
        return self.vector_store.store_code_context(code_data)
    
    @classmethod
    def create_enhanced(cls, storage_path=None, **kwargs):
        """Factory method for enhanced agent with Serena enabled."""
        return cls(
            storage_path=storage_path,
            enable_semantic_search=True,
            use_enhanced_vector_store=True,
            auto_route_to_serena=True,
            **kwargs
        )
    
    @classmethod
    def create_standard(cls, storage_path=None, **kwargs):
        """Factory method for standard agent (no Serena)."""
        return cls(
            storage_path=storage_path,
            enable_semantic_search=False,
            use_enhanced_vector_store=True,
            auto_route_to_serena=False,
            **kwargs
        )
    
    @classmethod
    def create_basic(cls, storage_path=None, **kwargs):
        """Factory method for basic agent (no Serena, basic embeddings)."""
        return cls(
            storage_path=storage_path,
            enable_semantic_search=False,
            use_enhanced_vector_store=False,
            auto_route_to_serena=False,
            **kwargs
        )