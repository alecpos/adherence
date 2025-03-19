"""Dependency analyzer for Python code."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from rich.console import Console

console = Console()

@dataclass
class ImportInfo:
    """Information about an import statement."""
    name: str
    module: Optional[str] = None
    alias: Optional[str] = None
    line: int = 0
    column: int = 0
    is_from_import: bool = False
    import_path: List[str] = None
    usage_count: int = 0
    usage_examples: List[str] = None

class DependencyNode:
    """Represents a node in the dependency graph."""
    def __init__(self, name: str):
        self.name = name
        self.imports: List[str] = []
        self.imported_by: List[str] = []
        self.complexity: float = 0.0
        self.maintainability: float = 0.0
        self.best_practices: Dict[str, List[str]] = defaultdict(list)

class DependencyAnalyzer:
    """Analyzes Python code dependencies and builds a dependency graph."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.import_usage_patterns = defaultdict(lambda: defaultdict(list))
        self.import_examples = defaultdict(list)
        self.best_practices = defaultdict(list)
        self.import_stack = []
        self.visited_modules = set()
        self.import_info = {}
        self.dependency_graph = defaultdict(set)
        self.circular_dependencies = []
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges = []
        self.source_code = ""
        self._load_source()
    
    def _load_source(self) -> None:
        """Load the source code from the file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
        except Exception as e:
            console.print(f"[red]Error loading source code: {str(e)}[/red]")
            console.print(traceback.format_exc())
    
    def analyze_imports(self) -> None:
        """Analyze imports in the file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # First pass: collect all imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._process_import_node(node)
            
            # Second pass: analyze import usage
            self._analyze_import_usage(tree)
            
            # Third pass: detect circular dependencies
            self._detect_circular_dependencies()
            
            # Fourth pass: collect best practices
            self._collect_best_practices(tree)
            
        except Exception as e:
            console.print(f"[red]Error analyzing imports: {str(e)}[/red]")
            console.print(traceback.format_exc())
    
    def _process_import_node(self, node: ast.AST) -> None:
        """Process an import node and collect information."""
        if isinstance(node, ast.Import):
            for name in node.names:
                info = ImportInfo(
                    name=name.name,
                    alias=name.asname,
                    line=node.lineno,
                    column=node.col_offset,
                    import_path=[name.name],
                    usage_examples=[]
                )
                self.import_info[name.name] = info
                self.import_stack.append(name.name)
                
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for name in node.names:
                full_name = f"{module}.{name.name}" if module else name.name
                info = ImportInfo(
                    name=name.name,
                    module=module,
                    alias=name.asname,
                    line=node.lineno,
                    column=node.col_offset,
                    is_from_import=True,
                    import_path=module.split('.') + [name.name],
                    usage_examples=[]
                )
                self.import_info[name.name] = info
                self.import_stack.append(full_name)
    
    def _analyze_import_usage(self, tree: ast.AST) -> None:
        """Analyze how imports are used in the code."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self._record_function_usage(node.func.id, node)
                elif isinstance(node.func, ast.Attribute):
                    self._record_attribute_usage(node.func, node)
    
    def _record_function_usage(self, func_name: str, node: ast.Call):
        """Record how a function is used."""
        module_name = Path(self.file_path).stem
        context = ast.get_source_segment(self.source_code, node)
        if context:
            self.import_examples[func_name].append(context.strip())
            pattern = f"Function call: {func_name}()"
            if pattern not in self.import_usage_patterns[module_name][func_name]:
                self.import_usage_patterns[module_name][func_name].append(pattern)
    
    def _record_attribute_usage(self, attr: ast.Attribute, node: ast.Call):
        """Record how an attribute is used."""
        module_name = Path(self.file_path).stem
        if isinstance(attr.value, ast.Name):
            module = attr.value.id
            context = ast.get_source_segment(self.source_code, node)
            if context:
                self.import_examples[module].append(context.strip())
                pattern = f"Method call: {module}.{attr.attr}()"
                if pattern not in self.import_usage_patterns[module_name][module]:
                    self.import_usage_patterns[module_name][module].append(pattern)
    
    def _detect_circular_dependencies(self) -> None:
        """Detect circular dependencies in the import stack."""
        stack = self.import_stack.copy()
        visited = set()
        path = []
        
        def dfs(node: str, current_path: List[str]) -> None:
            if node in current_path:
                cycle = current_path[current_path.index(node):] + [node]
                if cycle not in self.circular_dependencies:
                    self.circular_dependencies.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            current_path.append(node)
            
            # Add dependencies to graph
            if node in self.import_info:
                info = self.import_info[node]
                if info.module:
                    self.dependency_graph[node].add(info.module)
            
            # Recursively check dependencies
            for dep in self.dependency_graph[node]:
                dfs(dep, current_path)
            
            current_path.pop()
        
        for node in stack:
            if node not in visited:
                dfs(node, [])
    
    def _collect_best_practices(self, tree: ast.AST) -> None:
        """Collect best practices from the code."""
        # Analyze import organization
        imports_at_top = True
        first_non_import = float('inf')
        
        for i, node in enumerate(tree.body):
            if not (isinstance(node, (ast.Import, ast.ImportFrom))):
                first_non_import = i
                break
        
        for i, node in enumerate(tree.body):
            if i > first_non_import and isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_at_top = False
                break
        
        if imports_at_top:
            self.best_practices["general"].append("All imports are organized at the top of the file")
        
        # Analyze import grouping
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if module in ['os', 'sys', 'pathlib', 'collections', 'typing']:
                    stdlib_imports.append(module)
                elif '.' in module:
                    local_imports.append(module)
                else:
                    third_party_imports.append(module)
        
        if stdlib_imports:
            self.best_practices["stdlib"].append(
                f"Standard library imports are used: {', '.join(stdlib_imports)}"
            )
        if third_party_imports:
            self.best_practices["third_party"].append(
                f"Third-party imports are used: {', '.join(third_party_imports)}"
            )
        if local_imports:
            self.best_practices["local"].append(
                f"Local imports are used: {', '.join(local_imports)}"
            )
    
    def suggest_improvements(self) -> List[str]:
        """Generate suggestions for improving the code."""
        suggestions = []
        
        # Check for circular dependencies
        if self.circular_dependencies:
            suggestions.append(
                "Circular dependencies detected. Consider restructuring imports."
            )
        
        # Check for unused imports
        for name, info in self.import_info.items():
            if info.usage_count == 0:
                suggestions.append(f"Remove unused import: {name}")
        
        # Check for import organization
        if not all(info.line == 1 for info in self.import_info.values()):
            suggestions.append(
                "Move all imports to the top of the file for better organization"
            )
        
        return suggestions
    
    def get_dependency_graph(self) -> Dict[str, DependencyNode]:
        """Get the complete dependency graph."""
        return self.nodes
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependencies in the codebase."""
        cycles = []
        visited = set()
        path = []
        
        def dfs(node: str, visited: Set[str], path: List[str]) -> List[List[str]]:
            if node in path:
                cycle = path[path.index(node):] + [node]
                if cycle not in cycles:
                    cycles.append(cycle)
                return cycles
            
            if node in visited:
                return cycles
            
            visited.add(node)
            path.append(node)
            
            for dep in self.dependency_graph[node]:
                dfs(dep, visited, path)
            
            path.pop()
            return cycles
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, visited, path)
        
        return cycles
    
    def analyze_directory(self, directory: str) -> Dict[str, DependencyNode]:
        """Analyze all Python files in a directory."""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # First pass: collect best practices and examples
        for file_path in directory_path.rglob("*.py"):
            self._collect_best_practices(ast.parse(file_path.read_text()))
        
        # Second pass: analyze files and compare against best practices
        for file_path in directory_path.rglob("*.py"):
                self._analyze_file(file_path)
        
        return self.nodes
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            # Create or get node for this file
            module_name = self._get_module_name(file_path)
            if module_name not in self.nodes:
                self.nodes[module_name] = DependencyNode(module_name)
            
            node = self.nodes[module_name]
            
            # Calculate metrics
            node.complexity = self._calculate_complexity(tree)
            node.maintainability = self._calculate_maintainability(tree)
            
            # Extract imports and patterns
            imports, patterns = self._extract_imports_and_patterns(tree)
            node.imports = imports
            
            # Add edges to graph
            for imp in imports:
                if imp not in self.nodes:
                    self.nodes[imp] = DependencyNode(imp)
                self.nodes[imp].imported_by.append(module_name)
                self.edges.append((module_name, imp))
            
        except Exception as e:
            console.print(f"[red]Error analyzing {file_path}: {str(e)}[/red]")
            console.print(traceback.format_exc())
    
    def _extract_imports_and_patterns(self, tree: ast.AST) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract imports and usage patterns from an AST."""
        imports = []
        patterns = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_name = self._get_import_name(node)
                if import_name:
                    imports.append(import_name)
                    patterns[import_name] = self._analyze_import_usage(tree, import_name)
        
        return imports, patterns
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert a file path to a module name."""
        relative_path = file_path.relative_to(file_path.parent.parent)
        return str(relative_path.with_suffix('')).replace(os.sep, '.')
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_maintainability(self, tree: ast.AST) -> float:
        """Calculate maintainability index (simplified version)."""
        lines = len(ast.unparse(tree).splitlines())
        complexity = self._calculate_complexity(tree)
        return (complexity * lines) / 1000
    
    def _get_import_name(self, node: ast.AST) -> Optional[str]:
        """Extract the import name from a node."""
        if isinstance(node, ast.Import):
            for name in node.names:
                return name.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                return f"{module}.{name.name}"
        return None