#!/usr/bin/env python3
"""
EnhancedVectorStore v2 - With real embeddings and code-aware processing.

This version improves on the basic vector store by:
- Using sentence-transformers for real semantic embeddings
- Adding code structure extraction
- Enhanced metadata for better search
"""

from __future__ import annotations

import ast
import json
import logging
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pickle

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .vector_store import ConversationVectorStore

log = logging.getLogger(__name__)


class EnhancedVectorStoreV2(ConversationVectorStore):
    """
    Enhanced vector store with real embeddings and code awareness.
    
    Improvements over base class:
    - Real semantic embeddings using sentence-transformers
    - Code structure extraction and indexing
    - Enhanced metadata for better filtering
    - Smarter similarity scoring
    """
    
    def __init__(self, storage_path: Optional[str] = None, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_real_embeddings: bool = True):
        """
        Initialize enhanced vector store.
        
        Args:
            storage_path: Path for persistent storage
            embedding_model: Name of sentence-transformer model to use
            use_real_embeddings: Whether to use real embeddings (vs hash-based)
        """
        super().__init__(storage_path)
        
        self.use_real_embeddings = use_real_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_real_embeddings:
            try:
                log.info(f"Loading embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                log.info(f"Embedding model loaded successfully (dim={self.embedding_dim})")
            except Exception as e:
                log.warning(f"Failed to load embedding model: {e}. Falling back to hash-based.")
                self.use_real_embeddings = False
                self.embedding_model = None
        else:
            self.embedding_model = None
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                log.warning("sentence-transformers not available. Using hash-based embeddings.")
        
        # Code structure index
        self.code_structures = {}  # memory_id -> code structure data
        
        # Performance metrics
        self.metrics = {
            "embedding_time": [],
            "search_time": [],
            "total_embeddings": 0
        }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate real semantic embedding for text.
        
        Falls back to hash-based if sentence-transformers unavailable.
        """
        if not text:
            return [0.0] * self.embedding_dim
        
        if self.use_real_embeddings and self.embedding_model:
            try:
                # Use real semantic embeddings
                import time
                start = time.time()
                
                # Generate embedding
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                
                # Track metrics
                elapsed = time.time() - start
                self.metrics["embedding_time"].append(elapsed)
                self.metrics["total_embeddings"] += 1
                
                return embedding.tolist()
                
            except Exception as e:
                log.warning(f"Embedding generation failed: {e}. Using fallback.")
        
        # Fallback to hash-based (from parent class)
        return super()._generate_embedding(text)
    
    def _extract_code_structure(self, code: str, language: str = None) -> Dict[str, Any]:
        """
        Extract semantic structure from code.
        
        Args:
            code: Source code string
            language: Programming language (auto-detected if None)
            
        Returns:
            Dictionary with code structure information
        """
        structure = {
            "language": language or self._detect_language(code),
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity": 0,
            "loc": len(code.splitlines()),
            "has_tests": False,
            "has_docstrings": False
        }
        
        # Python-specific extraction using AST
        if structure["language"] == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        structure["functions"].append({
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')],
                            "has_docstring": ast.get_docstring(node) is not None
                        })
                        if "test" in node.name.lower():
                            structure["has_tests"] = True
                            
                    elif isinstance(node, ast.ClassDef):
                        structure["classes"].append({
                            "name": node.name,
                            "bases": [base.id for base in node.bases if hasattr(base, 'id')],
                            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                        })
                        
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            structure["imports"].append(alias.name)
                            
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            structure["imports"].append(node.module)
                            
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                structure["variables"].append(target.id)
                
                # Calculate complexity (number of nodes)
                structure["complexity"] = len(list(ast.walk(tree)))
                
                # Check for docstrings
                if ast.get_docstring(tree):
                    structure["has_docstrings"] = True
                    
            except SyntaxError:
                log.debug("Failed to parse Python code with AST")
        
        # Generic extraction for other languages
        else:
            # Extract functions (basic regex)
            func_pattern = r'(?:def|function|func|fn)\s+(\w+)'
            structure["functions"] = re.findall(func_pattern, code)
            
            # Extract classes
            class_pattern = r'(?:class|struct|interface)\s+(\w+)'
            structure["classes"] = re.findall(class_pattern, code)
            
            # Check for tests
            if re.search(r'test_|_test|Test|describe\(|it\(', code):
                structure["has_tests"] = True
        
        return structure
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code content."""
        # Simple heuristic-based detection
        if "def " in code and "import " in code:
            return "python"
        elif "function " in code or "const " in code or "=>" in code:
            return "javascript"
        elif "public class" in code or "private " in code:
            return "java"
        elif "func " in code and "package " in code:
            return "go"
        elif "#include" in code:
            return "cpp"
        elif "fn " in code and "let " in code:
            return "rust"
        else:
            return "unknown"
    
    def store_code_context(self, code_data: Dict[str, Any]) -> str:
        """
        Enhanced code context storage with structure extraction.
        """
        # Extract code structure before storing
        if "code" in code_data:
            structure = self._extract_code_structure(
                code_data["code"],
                code_data.get("language")
            )
            code_data["structure"] = structure
        
        # Store using parent method
        memory_id = super().store_code_context(code_data)
        
        # Store structure separately for fast lookup
        if "structure" in code_data:
            self.code_structures[memory_id] = code_data["structure"]
        
        return memory_id
    
    def search_memories(self, query: str, memory_type: Optional[str] = None, 
                       limit: int = 10, use_code_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced search with code structure awareness.
        
        Args:
            query: Search query
            memory_type: Filter by type
            limit: Max results
            use_code_structure: Whether to boost code structure matches
            
        Returns:
            Sorted list of relevant memories
        """
        import time
        start = time.time()
        
        # Get base results from parent
        results = super().search_memories(query, memory_type, limit * 2)  # Get more for re-ranking
        
        if use_code_structure and results:
            # Extract query intent
            query_lower = query.lower()
            looking_for_function = "function" in query_lower or "method" in query_lower or "def" in query_lower
            looking_for_class = "class" in query_lower or "struct" in query_lower
            looking_for_import = "import" in query_lower or "dependency" in query_lower
            
            # Re-rank based on code structure matches
            for result in results:
                memory_id = result.get("memory_id")
                boost = 0.0
                
                if memory_id in self.code_structures:
                    structure = self.code_structures[memory_id]
                    
                    # Boost matches based on query intent
                    if looking_for_function and structure["functions"]:
                        # Check if any function name is in query
                        for func in structure["functions"]:
                            if isinstance(func, dict):
                                if func["name"].lower() in query_lower:
                                    boost += 0.3
                            elif isinstance(func, str) and func.lower() in query_lower:
                                boost += 0.3
                    
                    if looking_for_class and structure["classes"]:
                        for cls in structure["classes"]:
                            if isinstance(cls, dict):
                                if cls["name"].lower() in query_lower:
                                    boost += 0.3
                            elif isinstance(cls, str) and cls.lower() in query_lower:
                                boost += 0.3
                    
                    if looking_for_import and structure["imports"]:
                        for imp in structure["imports"]:
                            if imp.lower() in query_lower:
                                boost += 0.2
                    
                    # General quality boost
                    if structure["has_docstrings"]:
                        boost += 0.1
                    if structure["has_tests"]:
                        boost += 0.05
                
                # Apply boost to similarity score
                if "similarity_score" in result:
                    result["similarity_score"] = min(1.0, result["similarity_score"] + boost)
                    result["structure_boost"] = boost
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Track search time
        elapsed = time.time() - start
        self.metrics["search_time"].append(elapsed)
        
        # Return limited results
        return results[:limit]
    
    def _extract_metadata(self, data: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        """Enhanced metadata extraction."""
        # Get base metadata from parent
        metadata = super()._extract_metadata(data, memory_type)
        
        # Add enhanced metadata
        if memory_type == "code" and "code" in data:
            code = data["code"]
            metadata.update({
                "language": self._detect_language(code),
                "loc": len(code.splitlines()),
                "char_count": len(code),
                "has_tests": "test" in code.lower(),
                "has_comments": "#" in code or "//" in code or "/*" in code,
                "has_docstrings": '"""' in code or "'''" in code,
                "complexity_estimate": len(re.findall(r'\b(?:if|for|while|try|except|catch)\b', code))
            })
            
            # Add structure summary if available
            if "structure" in data:
                structure = data["structure"]
                metadata.update({
                    "function_count": len(structure.get("functions", [])),
                    "class_count": len(structure.get("classes", [])),
                    "import_count": len(structure.get("imports", []))
                })
        
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including performance metrics."""
        stats = super().get_stats()
        
        # Add enhanced stats
        stats.update({
            "using_real_embeddings": self.use_real_embeddings,
            "embedding_model": getattr(self.embedding_model, 'model_name', None) if self.embedding_model else None,
            "code_structures_indexed": len(self.code_structures),
            "metrics": {
                "avg_embedding_time": np.mean(self.metrics["embedding_time"]) if self.metrics["embedding_time"] else 0,
                "avg_search_time": np.mean(self.metrics["search_time"]) if self.metrics["search_time"] else 0,
                "total_embeddings": self.metrics["total_embeddings"]
            }
        })
        
        return stats
    
    def optimize_embeddings(self):
        """Re-generate all embeddings with the current model."""
        if not self.use_real_embeddings or not self.embedding_model:
            log.warning("Cannot optimize - real embeddings not available")
            return
        
        log.info("Optimizing embeddings for all stored memories...")
        
        # Re-generate conversation embeddings
        for conv in self.conversations:
            memory_id = conv["memory_id"]
            content = conv["content"]
            self.embeddings[memory_id] = self._generate_embedding(content)
        
        # Re-generate code embeddings
        for code_ctx in self.code_contexts:
            memory_id = code_ctx["memory_id"]
            content = code_ctx["content"]
            self.embeddings[memory_id] = self._generate_embedding(content)
        
        log.info(f"Optimized {len(self.embeddings)} embeddings")
        
        # Save if persistent
        if self.storage_path:
            self._save_to_disk()