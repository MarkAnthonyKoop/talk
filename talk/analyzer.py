#!/usr/bin/env python3
"""
Analyzer module for Talk CLI.

This module provides functionality to parse and analyze a codebase,
build an Abstract Syntax Tree (AST), create a dependency graph,
and identify potential improvement areas.
"""

import ast
import os
import re
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

# Optional imports for enhanced functionality
try:
    import radon.complexity as radon_cc
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger("talk.analyzer")


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class ImprovementType(Enum):
    """Types of potential code improvements."""
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    UNUSED_CODE = "unused_code"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    STYLE = "style"
    SECURITY = "security"
    TESTING = "testing"


@dataclass
class CodeLocation:
    """Represents a location in the code."""
    file_path: Path
    start_line: int
    end_line: int
    start_column: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        """String representation of the location."""
        if self.start_column is not None and self.end_column is not None:
            return f"{self.file_path}:{self.start_line}:{self.start_column}-{self.end_line}:{self.end_column}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class Improvement:
    """Represents a potential code improvement."""
    type: ImprovementType
    location: CodeLocation
    description: str
    confidence: float  # 0.0 to 1.0
    before_snippet: Optional[str] = None
    after_snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the improvement."""
        return f"{self.type.value.capitalize()} ({self.confidence:.2f}): {self.description} at {self.location}"


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph."""
    name: str
    type: str  # 'module', 'class', 'function', 'variable'
    file_path: Optional[Path] = None
    location: Optional[CodeLocation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dependency:
    """Represents a dependency between two nodes."""
    source: DependencyNode
    target: DependencyNode
    type: str  # 'import', 'call', 'inheritance', 'reference'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Results of code analysis."""
    language: Language
    files_analyzed: List[Path]
    improvements: List[Improvement] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_improvements_by_type(self, improvement_type: ImprovementType) -> List[Improvement]:
        """Filter improvements by type."""
        return [i for i in self.improvements if i.type == improvement_type]

    def get_improvements_by_confidence(self, min_confidence: float = 0.0) -> List[Improvement]:
        """Filter improvements by minimum confidence level."""
        return [i for i in self.improvements if i.confidence >= min_confidence]

    def get_dependency_graph(self):
        """Get the dependency graph as a networkx graph if available."""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available. Cannot create dependency graph.")
            return None

        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.__dict__)
        
        for dep in self.dependencies:
            G.add_edge(dep.source.name, dep.target.name, type=dep.type, **dep.metadata)
        
        return G


class Analyzer:
    """Base class for code analyzers."""
    
    def __init__(self, root_path: Path, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        """Initialize the analyzer with root path and file patterns."""
        self.root_path = root_path
        self.include_patterns = include_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or ["**/venv/**", "**/.git/**", "**/__pycache__/**"]
        self.analysis_result = None

    def detect_language(self, file_path: Path) -> Language:
        """Detect the programming language of a file."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.py':
            return Language.PYTHON
        elif suffix == '.js':
            return Language.JAVASCRIPT
        elif suffix == '.ts':
            return Language.TYPESCRIPT
        else:
            # Try to detect by content for files without clear extensions
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars
                
                if re.search(r'import\s+[a-zA-Z_]|from\s+[a-zA-Z_]+\s+import', content):
                    return Language.PYTHON
                elif re.search(r'(const|let|var)\s+[a-zA-Z_]|function\s+[a-zA-Z_]', content):
                    # Check for TypeScript-specific syntax
                    if re.search(r':\s*[A-Z][a-zA-Z]*(\[\])?', content):
                        return Language.TYPESCRIPT
                    return Language.JAVASCRIPT
            except Exception as e:
                logger.debug(f"Error detecting language for {file_path}: {e}")
        
        return Language.UNKNOWN

    def find_files(self) -> List[Path]:
        """Find all relevant files in the codebase."""
        from glob import glob
        
        all_files = []
        for include_pattern in self.include_patterns:
            pattern = os.path.join(self.root_path, include_pattern)
            matched_files = [Path(p) for p in glob(pattern, recursive=True)]
            all_files.extend(matched_files)
        
        # Apply exclude patterns
        for exclude_pattern in self.exclude_patterns:
            pattern = os.path.join(self.root_path, exclude_pattern)
            excluded_files = set(Path(p) for p in glob(pattern, recursive=True))
            all_files = [f for f in all_files if f not in excluded_files]
        
        return [f for f in all_files if f.is_file()]

    def analyze(self) -> AnalysisResult:
        """Analyze the codebase and return results."""
        raise NotImplementedError("Subclasses must implement analyze()")

    def build_ast(self, file_path: Path) -> Any:
        """Build an Abstract Syntax Tree for the given file."""
        raise NotImplementedError("Subclasses must implement build_ast()")

    def build_dependency_graph(self, files: List[Path]) -> Tuple[List[Dependency], Dict[str, DependencyNode]]:
        """Build a dependency graph for the given files."""
        raise NotImplementedError("Subclasses must implement build_dependency_graph()")

    def identify_improvements(self, files: List[Path]) -> List[Improvement]:
        """Identify potential improvements in the given files."""
        raise NotImplementedError("Subclasses must implement identify_improvements()")


class PythonAnalyzer(Analyzer):
    """Analyzer for Python code."""
    
    def build_ast(self, file_path: Path) -> Union[ast.AST, cst.Module]:
        """Build an AST for a Python file using both ast and libcst."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Build standard library AST
            std_ast = ast.parse(content, filename=str(file_path))
            
            # Build libcst AST with metadata
            cst_module = cst.parse_module(content)
            wrapper = MetadataWrapper(cst_module)
            
            # Store both ASTs in the metadata for later use
            return {
                'std_ast': std_ast,
                'cst_module': cst_module,
                'cst_wrapper': wrapper
            }
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_imports(self, tree: ast.AST, file_path: Path) -> List[Tuple[str, CodeLocation]]:
        """Extract imports from a Python AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'module') and node.module:
                    module_name = node.module
                    for name in node.names:
                        full_name = f"{module_name}.{name.name}" if module_name else name.name
                        location = CodeLocation(
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                            end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
                        )
                        imports.append((full_name, location))
                else:
                    for name in node.names:
                        location = CodeLocation(
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                            end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
                        )
                        imports.append((name.name, location))
        
        return imports

    def _extract_definitions(self, tree: ast.AST, file_path: Path) -> List[Tuple[str, str, CodeLocation]]:
        """Extract class and function definitions from a Python AST."""
        definitions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                location = CodeLocation(
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                    end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
                )
                definitions.append((node.name, 'class', location))
            
            elif isinstance(node, ast.FunctionDef):
                location = CodeLocation(
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                    end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
                )
                definitions.append((node.name, 'function', location))
        
        return definitions

    def build_dependency_graph(self, files: List[Path]) -> Tuple[List[Dependency], Dict[str, DependencyNode]]:
        """Build a dependency graph for Python files."""
        dependencies = []
        nodes = {}
        
        # First pass: collect all definitions
        for file_path in files:
            if self.detect_language(file_path) != Language.PYTHON:
                continue
            
            ast_result = self.build_ast(file_path)
            if not ast_result:
                continue
            
            std_ast = ast_result['std_ast']
            
            # Extract definitions
            for name, node_type, location in self._extract_definitions(std_ast, file_path):
                # Create a unique ID for the node
                module_path = str(file_path.relative_to(self.root_path))
                node_id = f"{module_path}:{name}"
                
                node = DependencyNode(
                    name=name,
                    type=node_type,
                    file_path=file_path,
                    location=location
                )
                nodes[node_id] = node
        
        # Second pass: collect imports and references
        for file_path in files:
            if self.detect_language(file_path) != Language.PYTHON:
                continue
            
            ast_result = self.build_ast(file_path)
            if not ast_result:
                continue
            
            std_ast = ast_result['std_ast']
            
            # Extract imports
            for import_name, location in self._extract_imports(std_ast, file_path):
                # Create source node (the importing module)
                module_path = str(file_path.relative_to(self.root_path))
                source_id = module_path
                
                if source_id not in nodes:
                    source_node = DependencyNode(
                        name=module_path,
                        type='module',
                        file_path=file_path
                    )
                    nodes[source_id] = source_node
                else:
                    source_node = nodes[source_id]
                
                # Create target node (the imported module/object)
                # This is a simplification - in a real implementation we would resolve
                # the import to an actual file/definition
                target_node = DependencyNode(
                    name=import_name,
                    type='import',
                    file_path=None
                )
                target_id = import_name
                
                if target_id not in nodes:
                    nodes[target_id] = target_node
                
                # Create dependency
                dependency = Dependency(
                    source=source_node,
                    target=nodes[target_id],
                    type='import'
                )
                dependencies.append(dependency)
        
        return dependencies, nodes

    def _check_complexity(self, tree: ast.AST, file_path: Path) -> List[Improvement]:
        """Check for complex functions and methods."""
        improvements = []
        
        if not RADON_AVAILABLE:
            logger.debug("Radon not available, skipping complexity analysis")
            return improvements
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use radon to calculate cyclomatic complexity
            results = radon_cc.cc_visit(content)
            
            for result in results:
                if result.complexity > 10:  # High complexity threshold
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=result.lineno,
                        end_line=result.endline if hasattr(result, 'endline') else result.lineno + 10  # Estimate
                    )
                    
                    confidence = min(1.0, (result.complexity - 10) / 10)  # Scale confidence based on complexity
                    
                    improvement = Improvement(
                        type=ImprovementType.COMPLEXITY,
                        location=location,
                        description=f"Complex {result.type.lower()} '{result.name}' (complexity: {result.complexity})",
                        confidence=confidence,
                        metadata={
                            'complexity': result.complexity,
                            'type': result.type,
                            'name': result.name
                        }
                    )
                    improvements.append(improvement)
        
        except Exception as e:
            logger.error(f"Error analyzing complexity in {file_path}: {e}")
        
        return improvements

    def _check_documentation(self, tree: ast.AST, file_path: Path) -> List[Improvement]:
        """Check for missing or inadequate documentation."""
        improvements = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # Check for missing docstring
                if not ast.get_docstring(node):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.lineno,
                        start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                        end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
                    )
                    
                    node_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                    
                    # Higher confidence for public methods/classes
                    confidence = 0.9 if not node.name.startswith('_') else 0.6
                    
                    improvement = Improvement(
                        type=ImprovementType.DOCUMENTATION,
                        location=location,
                        description=f"Missing docstring for {node_type} '{node.name}'",
                        confidence=confidence,
                        metadata={
                            'node_type': node_type,
                            'name': node.name,
                            'is_private': node.name.startswith('_')
                        }
                    )
                    improvements.append(improvement)
        
        return improvements

    def _check_unused_imports(self, tree: ast.AST, file_path: Path) -> List[Improvement]:
        """Check for unused imports."""
        improvements = []
        
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = {}  # name -> node
                self.used_names = set()
            
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = node
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = node
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.used_names.add(node.id)
                self.generic_visit(node)
            
            def get_unused_imports(self):
                return {name: node for name, node in self.imports.items() 
                        if name not in self.used_names and not name.startswith('_')}
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        for name, node in visitor.get_unused_imports().items():
            location = CodeLocation(
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.lineno,
                start_column=node.col_offset if hasattr(node, 'col_offset') else None,
                end_column=node.end_col_offset if hasattr(node, 'end_col_offset') else None
            )
            
            improvement = Improvement(
                type=ImprovementType.UNUSED_CODE,
                location=location,
                description=f"Unused import '{name}'",
                confidence=0.9,  # High confidence for unused imports
                metadata={
                    'name': name,
                    'type': 'import'
                }
            )
            improvements.append(improvement)
        
        return improvements

    def identify_improvements(self, files: List[Path]) -> List[Improvement]:
        """Identify potential improvements in Python files."""
        all_improvements = []
        
        for file_path in files:
            if self.detect_language(file_path) != Language.PYTHON:
                continue
            
            logger.debug(f"Analyzing file for improvements: {file_path}")
            
            ast_result = self.build_ast(file_path)
            if not ast_result:
                continue
            
            std_ast = ast_result['std_ast']
            
            # Run various checks
            complexity_improvements = self._check_complexity(std_ast, file_path)
            documentation_improvements = self._check_documentation(std_ast, file_path)
            unused_imports_improvements = self._check_unused_imports(std_ast, file_path)
            
            all_improvements.extend(complexity_improvements)
            all_improvements.extend(documentation_improvements)
            all_improvements.extend(unused_imports_improvements)
            
            # Additional checks could be added here:
            # - Style checks (line length, naming conventions)
            # - Performance issues (inefficient data structures, algorithms)
            # - Security issues (use of eval, exec, etc.)
            # - Testing gaps (untested functions)
        
        return all_improvements

    def analyze(self) -> AnalysisResult:
        """Analyze Python code and return results."""
        logger.info(f"Starting Python code analysis in {self.root_path}")
        
        # Find all relevant files
        files = self.find_files()
        python_files = [f for f in files if self.detect_language(f) == Language.PYTHON]
        
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Build dependency graph
        dependencies, nodes = self.build_dependency_graph(python_files)
        
        # Identify improvements
        improvements = self.identify_improvements(python_files)
        
        # Create and return analysis result
        result = AnalysisResult(
            language=Language.PYTHON,
            files_analyzed=python_files,
            improvements=improvements,
            dependencies=dependencies,
            nodes=nodes
        )
        
        self.analysis_result = result
        return result


def create_analyzer(root_path: Path, include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> Analyzer:
    """Factory function to create the appropriate analyzer based on the codebase."""
    # For now, we only support Python, but this could be extended
    return PythonAnalyzer(root_path, include_patterns, exclude_patterns)


def analyze_codebase(root_path: Path, include_patterns: List[str] = None, exclude_patterns: List[str] = None) -> AnalysisResult:
    """Analyze a codebase and return the results."""
    analyzer = create_analyzer(root_path, include_patterns, exclude_patterns)
    return analyzer.analyze()
