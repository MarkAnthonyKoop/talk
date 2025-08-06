#!/usr/bin/env python3
"""
SemanticSearchAgent - Codebase semantic search and contextual retrieval.

This agent implements the concepts from the Serena MCP video to provide
intelligent, focused context retrieval from codebases. Instead of loading
entire files or directories, it uses semantic search to find only the
most relevant pieces of information for a given task or query.

Key features:
- Semantic similarity search through code and documentation
- Focused context window management  
- Automatic relevance scoring and filtering
- Multi-dimensional search (code, comments, documentation, tests)
- Memory-efficient retrieval that avoids token waste

This solves the "Claude working at 30% potential" problem by ensuring
the context window contains only highly relevant information.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib
import mimetypes

from agent.agent import Agent
from .enhanced_vector_store import EnhancedVectorStore, MemoryNode

log = logging.getLogger(__name__)

@dataclass
class CodeContext:
    """Represents a semantic unit of code context."""
    file_path: str
    content: str
    context_type: str  # 'function', 'class', 'file', 'test', 'doc'
    language: str
    start_line: int
    end_line: int
    symbols: Set[str]  # Functions, classes, variables mentioned
    semantic_score: float = 0.0
    relevance_score: float = 0.0

class SemanticSearchAgent(Agent):
    """
    Semantic search agent for intelligent codebase context retrieval.
    
    This agent mimics the Serena MCP functionality to:
    1. Index codebases semantically instead of just textually
    2. Provide focused, relevant context for tasks
    3. Avoid overwhelming the context window with irrelevant information
    4. Enable Claude to work at full potential with clean, targeted context
    """
    
    def __init__(self, **kwargs):
        """Initialize with semantic search capabilities."""
        super().__init__(roles=[
            "You are a semantic search specialist for code repositories.",
            "You find the most relevant code, documentation, and context for any given task.",
            "You use semantic similarity to avoid irrelevant information cluttering the context.",
            "You provide focused, precise context that maximizes LLM performance.",
            "You understand code structure, relationships, and semantic meaning."
        ], **kwargs)
        
        # Enhanced vector store for semantic search
        self.vector_store = EnhancedVectorStore()
        
        # Codebase indexing
        self.indexed_files: Dict[str, Dict] = {}
        self.code_contexts: List[CodeContext] = []
        self.symbol_index: Dict[str, Set[str]] = defaultdict(set)  # symbol -> file paths
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> dependencies
        
        # Search configuration
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        self.relevance_threshold = 0.3  # Minimum relevance to include in results
        self.max_context_items = 10  # Maximum context pieces to return
        
    def run(self, input_text: str) -> str:
        """
        Perform semantic search and return focused context.
        
        Args:
            input_text: Search query or task description
            
        Returns:
            Focused, relevant context for the task
        """
        try:
            # Parse the input to understand what kind of search is needed
            search_intent = self._analyze_search_intent(input_text)
            
            # Perform multi-dimensional semantic search
            results = self._perform_semantic_search(input_text, search_intent)
            
            # Filter and rank results for optimal context
            focused_context = self._create_focused_context(results, input_text)
            
            # Format the response for maximum utility
            return self._format_search_response(focused_context, search_intent)
            
        except Exception as e:
            log.error(f"Error in semantic search: {e}")
            return f"SEARCH_ERROR: Could not perform semantic search: {str(e)}"
    
    def index_codebase(self, root_path: str = None) -> Dict[str, Any]:
        """
        Index a codebase for semantic search.
        
        Args:
            root_path: Root directory to index (defaults to current directory)
            
        Returns:
            Indexing statistics
        """
        try:
            root = Path(root_path) if root_path else Path.cwd()
            
            log.info(f"Starting semantic indexing of {root}")
            
            # Clear existing indexes
            self.indexed_files.clear()
            self.code_contexts.clear()
            self.symbol_index.clear()
            self.import_graph.clear()
            
            # Find and process code files
            stats = {
                'files_processed': 0,
                'contexts_extracted': 0,
                'symbols_indexed': 0,
                'languages_found': set(),
                'errors': []
            }
            
            for file_path in self._find_code_files(root):
                try:
                    self._index_file(file_path, stats)
                except Exception as e:
                    stats['errors'].append(f"{file_path}: {str(e)}")
                    log.warning(f"Error indexing {file_path}: {e}")
            
            # Build import/dependency graph
            self._build_dependency_graph()
            
            # Store in vector store for semantic search
            self._store_contexts_in_vector_store()
            
            stats['languages_found'] = list(stats['languages_found'])
            log.info(f"Indexing complete: {stats}")
            
            return stats
            
        except Exception as e:
            log.error(f"Error indexing codebase: {e}")
            return {'error': str(e)}
    
    def _analyze_search_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the search query to understand intent and focus."""
        intent = {
            'type': 'general',
            'focus_areas': [],
            'exclude_areas': [],
            'language_hints': [],
            'context_size': 'medium'
        }
        
        query_lower = query.lower()
        
        # Determine search type
        if any(word in query_lower for word in ['implement', 'create', 'add', 'build']):
            intent['type'] = 'implementation'
        elif any(word in query_lower for word in ['fix', 'bug', 'error', 'debug', 'issue']):
            intent['type'] = 'debugging'
        elif any(word in query_lower for word in ['test', 'testing', 'unit test', 'spec']):
            intent['type'] = 'testing'
        elif any(word in query_lower for word in ['refactor', 'clean', 'optimize', 'improve']):
            intent['type'] = 'refactoring'
        elif any(word in query_lower for word in ['understand', 'explain', 'how does', 'what is']):
            intent['type'] = 'analysis'
        
        # Extract language hints
        for ext, lang in self.supported_languages.items():
            if lang in query_lower or ext in query:
                intent['language_hints'].append(lang)
        
        # Extract focus areas (classes, functions, modules mentioned)
        # Look for CamelCase (likely class names)
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', query)
        intent['focus_areas'].extend(camel_case)
        
        # Look for snake_case (likely function/variable names)
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', query)
        intent['focus_areas'].extend(snake_case)
        
        # Look for file paths
        file_paths = re.findall(r'\b[\w/\\]+\.[\w]+\b', query)
        intent['focus_areas'].extend(file_paths)
        
        # Determine context size needed
        if any(word in query_lower for word in ['full', 'complete', 'entire', 'all']):
            intent['context_size'] = 'large'
        elif any(word in query_lower for word in ['quick', 'brief', 'simple', 'just']):
            intent['context_size'] = 'small'
        
        return intent
    
    def _perform_semantic_search(self, query: str, intent: Dict) -> List[CodeContext]:
        """Perform multi-dimensional semantic search."""
        results = []
        
        # 1. Vector similarity search
        vector_results = self._vector_similarity_search(query)
        results.extend(vector_results)
        
        # 2. Symbol-based search
        symbol_results = self._symbol_search(query, intent)
        results.extend(symbol_results)
        
        # 3. Structural search (imports, dependencies)
        if intent['focus_areas']:
            structural_results = self._structural_search(intent['focus_areas'])
            results.extend(structural_results)
        
        # 4. Intent-specific search
        intent_results = self._intent_based_search(query, intent)
        results.extend(intent_results)
        
        # Deduplicate by file_path + start_line
        seen = set()
        unique_results = []
        
        for context in results:
            key = (context.file_path, context.start_line)
            if key not in seen:
                seen.add(key)
                unique_results.append(context)
        
        return unique_results
    
    def _vector_similarity_search(self, query: str) -> List[CodeContext]:
        """Search using vector embeddings for semantic similarity."""
        results = []
        
        try:
            # Use enhanced vector store search
            search_results = self.vector_store.search_enhanced(
                query=query,
                strategy='hybrid',
                limit=20,
                filters={'type': 'code'}
            )
            
            for result in search_results:
                if result['score'] < self.relevance_threshold:
                    continue
                
                # Find corresponding code context
                memory_id = result['memory_id']
                
                # Try to find matching context in our indexed contexts
                matching_contexts = [
                    ctx for ctx in self.code_contexts
                    if self._context_matches_result(ctx, result)
                ]
                
                for context in matching_contexts:
                    context.semantic_score = result['score']
                    results.append(context)
        
        except Exception as e:
            log.warning(f"Vector search failed: {e}")
        
        return results
    
    def _symbol_search(self, query: str, intent: Dict) -> List[CodeContext]:
        """Search based on symbols (functions, classes, variables)."""
        results = []
        query_words = set(query.lower().split())
        
        # Search through symbol index
        for symbol, file_paths in self.symbol_index.items():
            symbol_lower = symbol.lower()
            
            # Check if symbol matches query
            if (symbol_lower in query.lower() or 
                any(word in symbol_lower for word in query_words) or
                any(word in symbol for word in intent.get('focus_areas', []))):
                
                # Find contexts containing this symbol
                for file_path in file_paths:
                    matching_contexts = [
                        ctx for ctx in self.code_contexts
                        if ctx.file_path == file_path and symbol in ctx.symbols
                    ]
                    
                    for context in matching_contexts:
                        # Score based on symbol relevance
                        relevance = self._calculate_symbol_relevance(symbol, query, intent)
                        context.relevance_score = max(context.relevance_score, relevance)
                        results.append(context)
        
        return results
    
    def _structural_search(self, focus_areas: List[str]) -> List[CodeContext]:
        """Search based on code structure and dependencies."""
        results = []
        
        # Find files that import or are imported by focus areas
        for focus in focus_areas:
            # Try to find files containing this focus item
            related_files = set()
            
            for file_path, imports in self.import_graph.items():
                if focus in str(file_path) or any(focus in imp for imp in imports):
                    related_files.add(file_path)
            
            # Add contexts from related files
            for file_path in related_files:
                file_contexts = [ctx for ctx in self.code_contexts if ctx.file_path == file_path]
                for context in file_contexts:
                    context.relevance_score = max(context.relevance_score, 0.6)
                    results.append(context)
        
        return results
    
    def _intent_based_search(self, query: str, intent: Dict) -> List[CodeContext]:
        """Search based on the specific intent (debugging, testing, etc.)."""
        results = []
        
        if intent['type'] == 'testing':
            # Prioritize test files
            test_contexts = [
                ctx for ctx in self.code_contexts
                if 'test' in ctx.file_path.lower() or ctx.context_type == 'test'
            ]
            for context in test_contexts:
                context.relevance_score = max(context.relevance_score, 0.7)
                results.extend(test_contexts)
        
        elif intent['type'] == 'debugging':
            # Prioritize error handling, logging, exception contexts
            error_keywords = ['error', 'exception', 'try', 'catch', 'log', 'debug']
            for context in self.code_contexts:
                content_lower = context.content.lower()
                if any(keyword in content_lower for keyword in error_keywords):
                    context.relevance_score = max(context.relevance_score, 0.6)
                    results.append(context)
        
        elif intent['type'] == 'implementation':
            # Prioritize classes, main functions, and entry points
            for context in self.code_contexts:
                if (context.context_type in ['class', 'function'] and 
                    ('main' in context.content.lower() or 'init' in context.content.lower())):
                    context.relevance_score = max(context.relevance_score, 0.6)
                    results.append(context)
        
        return results
    
    def _create_focused_context(self, results: List[CodeContext], query: str) -> List[CodeContext]:
        """Create focused context by filtering and ranking results."""
        # Combine semantic and relevance scores
        for context in results:
            context.relevance_score = (
                context.semantic_score * 0.6 + 
                context.relevance_score * 0.4
            )
        
        # Sort by relevance
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Filter by minimum threshold
        filtered_results = [
            ctx for ctx in sorted_results 
            if ctx.relevance_score >= self.relevance_threshold
        ]
        
        # Limit to max context items
        final_results = filtered_results[:self.max_context_items]
        
        # Ensure diversity - avoid too many contexts from same file
        diverse_results = self._ensure_diversity(final_results)
        
        return diverse_results
    
    def _format_search_response(self, contexts: List[CodeContext], intent: Dict) -> str:
        """Format the search results for optimal LLM consumption."""
        if not contexts:
            return "SEMANTIC_SEARCH: No relevant code context found for the query."
        
        response_parts = []
        response_parts.append("SEMANTIC_SEARCH_RESULTS:")
        response_parts.append(f"Intent: {intent['type']}")
        response_parts.append(f"Contexts found: {len(contexts)}")
        response_parts.append("")
        
        for i, context in enumerate(contexts, 1):
            response_parts.append(f"CONTEXT {i}:")
            response_parts.append(f"File: {context.file_path}")
            response_parts.append(f"Type: {context.context_type}")
            response_parts.append(f"Language: {context.language}")
            response_parts.append(f"Lines: {context.start_line}-{context.end_line}")
            response_parts.append(f"Relevance: {context.relevance_score:.3f}")
            
            if context.symbols:
                symbols_str = ", ".join(list(context.symbols)[:5])
                if len(context.symbols) > 5:
                    symbols_str += f" (+{len(context.symbols) - 5} more)"
                response_parts.append(f"Symbols: {symbols_str}")
            
            response_parts.append("")
            response_parts.append("```" + context.language)
            response_parts.append(context.content)
            response_parts.append("```")
            response_parts.append("")
        
        # Add summary
        files_involved = set(ctx.file_path for ctx in contexts)
        languages_involved = set(ctx.language for ctx in contexts)
        
        response_parts.append("SUMMARY:")
        response_parts.append(f"- Files involved: {len(files_involved)}")
        response_parts.append(f"- Languages: {', '.join(languages_involved)}")
        response_parts.append(f"- Total lines of context: {sum(ctx.end_line - ctx.start_line + 1 for ctx in contexts)}")
        
        return "\n".join(response_parts)
    
    def _find_code_files(self, root: Path) -> List[Path]:
        """Find all code files in the directory tree."""
        code_files = []
        
        # Directories to skip
        skip_dirs = {
            '.git', '.svn', '.hg',
            'node_modules', '__pycache__', '.pytest_cache',
            'target', 'build', 'dist', '.venv', 'venv',
            '.idea', '.vscode', '.vs'
        }
        
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip files in ignored directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Check if it's a supported code file
            suffix = file_path.suffix.lower()
            if suffix in self.supported_languages:
                code_files.append(file_path)
        
        return sorted(code_files)
    
    def _index_file(self, file_path: Path, stats: Dict):
        """Index a single code file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self.supported_languages[file_path.suffix.lower()]
            stats['languages_found'].add(language)
            
            # Extract contexts based on file type
            if language == 'python':
                contexts = self._extract_python_contexts(file_path, content)
            elif language in ['javascript', 'typescript']:
                contexts = self._extract_js_contexts(file_path, content)
            elif language == 'markdown':
                contexts = self._extract_markdown_contexts(file_path, content)
            else:
                contexts = self._extract_generic_contexts(file_path, content, language)
            
            # Store contexts
            self.code_contexts.extend(contexts)
            stats['contexts_extracted'] += len(contexts)
            
            # Index symbols
            for context in contexts:
                for symbol in context.symbols:
                    self.symbol_index[symbol].add(str(file_path))
                    stats['symbols_indexed'] += 1
            
            # Store file metadata
            self.indexed_files[str(file_path)] = {
                'language': language,
                'contexts': len(contexts),
                'symbols': sum(len(ctx.symbols) for ctx in contexts),
                'last_indexed': str(file_path.stat().st_mtime)
            }
            
            stats['files_processed'] += 1
            
        except Exception as e:
            log.warning(f"Error indexing {file_path}: {e}")
            raise
    
    def _extract_python_contexts(self, file_path: Path, content: str) -> List[CodeContext]:
        """Extract semantic contexts from Python files."""
        contexts = []
        
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                context = None
                symbols = set()
                
                if isinstance(node, ast.FunctionDef):
                    # Extract function context
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line + 10
                    
                    func_content = '\n'.join(lines[start_line-1:end_line])
                    
                    symbols.add(node.name)
                    # Extract function arguments
                    for arg in node.args.args:
                        symbols.add(arg.arg)
                    
                    context = CodeContext(
                        file_path=str(file_path),
                        content=func_content,
                        context_type='function',
                        language='python',
                        start_line=start_line,
                        end_line=end_line,
                        symbols=symbols
                    )
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class context
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line + 20
                    
                    class_content = '\n'.join(lines[start_line-1:end_line])
                    
                    symbols.add(node.name)
                    # Extract method names
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            symbols.add(item.name)
                    
                    context = CodeContext(
                        file_path=str(file_path),
                        content=class_content,
                        context_type='class',
                        language='python',
                        start_line=start_line,
                        end_line=end_line,
                        symbols=symbols
                    )
                
                if context:
                    contexts.append(context)
            
            # Add full file context for small files
            if len(lines) < 100:
                contexts.append(CodeContext(
                    file_path=str(file_path),
                    content=content,
                    context_type='file',
                    language='python',
                    start_line=1,
                    end_line=len(lines),
                    symbols=self._extract_all_symbols(content)
                ))
        
        except SyntaxError:
            # If parsing fails, treat as generic text
            contexts.extend(self._extract_generic_contexts(file_path, content, 'python'))
        
        return contexts
    
    def _extract_js_contexts(self, file_path: Path, content: str) -> List[CodeContext]:
        """Extract contexts from JavaScript/TypeScript files."""
        # Simplified JS parsing using regex patterns
        contexts = []
        lines = content.splitlines()
        
        # Find function definitions
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*)?=>|(\w+)\s*:\s*(?:async\s+)?(?:function\s*)?\([^)]*\)\s*{)'
        
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1) or match.group(2) or match.group(3)
            if func_name:
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find function body end (simplified)
                brace_count = 0
                end_line = start_line
                for i, line in enumerate(lines[start_line-1:], start_line):
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0 and i > start_line:
                        end_line = i
                        break
                    if i > start_line + 50:  # Limit context size
                        end_line = i
                        break
                
                func_content = '\n'.join(lines[start_line-1:end_line])
                
                contexts.append(CodeContext(
                    file_path=str(file_path),
                    content=func_content,
                    context_type='function',
                    language='javascript',
                    start_line=start_line,
                    end_line=end_line,
                    symbols={func_name}
                ))
        
        return contexts
    
    def _extract_markdown_contexts(self, file_path: Path, content: str) -> List[CodeContext]:
        """Extract contexts from Markdown files."""
        contexts = []
        lines = content.splitlines()
        
        # Find code blocks
        in_code_block = False
        code_language = None
        code_start = 0
        code_lines = []
        
        for i, line in enumerate(lines, 1):
            if line.startswith('```'):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    code_language = line[3:].strip() or 'text'
                    code_start = i + 1
                    code_lines = []
                else:
                    # End of code block
                    if code_lines:
                        code_content = '\n'.join(code_lines)
                        contexts.append(CodeContext(
                            file_path=str(file_path),
                            content=code_content,
                            context_type='code_block',
                            language=code_language,
                            start_line=code_start,
                            end_line=i - 1,
                            symbols=self._extract_all_symbols(code_content)
                        ))
                    in_code_block = False
            elif in_code_block:
                code_lines.append(line)
        
        # Add full markdown context for documentation
        contexts.append(CodeContext(
            file_path=str(file_path),
            content=content,
            context_type='documentation',
            language='markdown',
            start_line=1,
            end_line=len(lines),
            symbols=set()
        ))
        
        return contexts
    
    def _extract_generic_contexts(self, file_path: Path, content: str, language: str) -> List[CodeContext]:
        """Extract contexts from generic text files."""
        lines = content.splitlines()
        
        # For small files, include everything
        if len(lines) <= 50:
            return [CodeContext(
                file_path=str(file_path),
                content=content,
                context_type='file',
                language=language,
                start_line=1,
                end_line=len(lines),
                symbols=self._extract_all_symbols(content)
            )]
        
        # For larger files, break into chunks
        contexts = []
        chunk_size = 30
        
        for start in range(0, len(lines), chunk_size):
            end = min(start + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[start:end])
            
            contexts.append(CodeContext(
                file_path=str(file_path),
                content=chunk_content,
                context_type='chunk',
                language=language,
                start_line=start + 1,
                end_line=end,
                symbols=self._extract_all_symbols(chunk_content)
            ))
        
        return contexts
    
    def _extract_all_symbols(self, content: str) -> Set[str]:
        """Extract symbol-like identifiers from content."""
        symbols = set()
        
        # Find CamelCase identifiers
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)
        symbols.update(camel_case)
        
        # Find snake_case identifiers
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content)
        symbols.update(snake_case)
        
        # Find function-like calls
        function_calls = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', content)
        symbols.update(function_calls)
        
        return symbols
    
    def _build_dependency_graph(self):
        """Build import/dependency graph from indexed files."""
        for file_path in self.indexed_files:
            imports = set()
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                language = self.indexed_files[file_path]['language']
                
                if language == 'python':
                    # Find Python imports
                    import_patterns = [
                        r'from\s+(\w+(?:\.\w+)*)\s+import',
                        r'import\s+(\w+(?:\.\w+)*)'
                    ]
                    for pattern in import_patterns:
                        matches = re.findall(pattern, content)
                        imports.update(matches)
                
                elif language in ['javascript', 'typescript']:
                    # Find JS/TS imports
                    import_patterns = [
                        r'from\s+["\']([^"\']+)["\']',
                        r'import\s+["\']([^"\']+)["\']',
                        r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
                    ]
                    for pattern in import_patterns:
                        matches = re.findall(pattern, content)
                        imports.update(matches)
                
                self.import_graph[file_path] = imports
                
            except Exception as e:
                log.debug(f"Could not build dependencies for {file_path}: {e}")
    
    def _store_contexts_in_vector_store(self):
        """Store extracted contexts in the vector store for semantic search."""
        for context in self.code_contexts:
            # Create metadata for the context
            metadata = {
                'file_path': context.file_path,
                'context_type': context.context_type,
                'language': context.language,
                'start_line': context.start_line,
                'end_line': context.end_line,
                'symbols': list(context.symbols),
                'memory_type': 'code'
            }
            
            # Store in vector store
            conversation_data = {
                'content': context.content,
                'metadata': metadata,
                'type': 'code_context'
            }
            
            try:
                self.vector_store.store_conversation_enhanced(conversation_data)
            except Exception as e:
                log.debug(f"Could not store context in vector store: {e}")
    
    def _context_matches_result(self, context: CodeContext, result: Dict) -> bool:
        """Check if a context matches a search result."""
        metadata = result.get('metadata', {})
        
        return (
            metadata.get('file_path') == context.file_path and
            metadata.get('start_line') == context.start_line and
            metadata.get('end_line') == context.end_line
        )
    
    def _calculate_symbol_relevance(self, symbol: str, query: str, intent: Dict) -> float:
        """Calculate how relevant a symbol is to the query."""
        relevance = 0.0
        
        # Exact match gets high score
        if symbol.lower() in query.lower():
            relevance += 0.8
        
        # Partial match gets medium score
        query_words = set(query.lower().split())
        if any(word in symbol.lower() for word in query_words):
            relevance += 0.5
        
        # Focus area match gets high score
        if symbol in intent.get('focus_areas', []):
            relevance += 0.9
        
        # Intent-based scoring
        if intent['type'] == 'testing' and 'test' in symbol.lower():
            relevance += 0.3
        elif intent['type'] == 'debugging' and any(word in symbol.lower() for word in ['error', 'debug', 'log']):
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _ensure_diversity(self, results: List[CodeContext]) -> List[CodeContext]:
        """Ensure diversity in results to avoid too many contexts from same file."""
        diverse_results = []
        file_counts = Counter()
        max_per_file = 3
        
        for context in results:
            if file_counts[context.file_path] < max_per_file:
                diverse_results.append(context)
                file_counts[context.file_path] += 1
        
        return diverse_results

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        return {
            'files_indexed': len(self.indexed_files),
            'contexts_extracted': len(self.code_contexts),
            'symbols_indexed': len(self.symbol_index),
            'languages_supported': len(self.supported_languages),
            'total_context_lines': sum(
                ctx.end_line - ctx.start_line + 1 for ctx in self.code_contexts
            ),
            'context_types': Counter(ctx.context_type for ctx in self.code_contexts),
            'languages_found': Counter(ctx.language for ctx in self.code_contexts)
        }