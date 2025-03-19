import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..models.schemas import CodeSuggestion

class StaticAnalyzer:
    """Performs static analysis on Python code to identify potential issues and improvements."""
    
    def __init__(self):
        self.suggestions: List[CodeSuggestion] = []
    
    def analyze_file(self, file_path: Path) -> List[CodeSuggestion]:
        """Analyze a single Python file for potential issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Run various analysis checks
            self._check_function_length(tree, file_path)
            self._check_complexity(tree, file_path)
            self._check_docstrings(tree, file_path)
            self._check_naming_conventions(tree, file_path)
            self._check_error_handling(tree, file_path)
            self._check_code_duplication(tree, file_path)
            
        except Exception as e:
            self.suggestions.append(
                CodeSuggestion(
                    title="File Analysis Error",
                    description=f"Error analyzing file {file_path}: {str(e)}",
                    priority=5,
                    affected_files=[str(file_path)],
                    suggested_changes=["Fix syntax errors or file access issues"],
                    rationale="The file could not be analyzed due to errors",
                    impact="High",
                    category="error",
                    confidence=1.0
                )
            )
        
        return self.suggestions
    
    def _check_function_length(self, tree: ast.AST, file_path: Path) -> None:
        """Check for functions that are too long."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                lines = len(node.body)
                if lines > 50:  # Arbitrary threshold
                    self.suggestions.append(
                        CodeSuggestion(
                            title="Long Function",
                            description=f"Function '{node.name}' is {lines} lines long",
                            priority=3,
                            affected_files=[str(file_path)],
                            suggested_changes=[
                                f"Break down function '{node.name}' into smaller functions",
                                "Consider using composition or inheritance"
                            ],
                            rationale="Long functions are harder to understand and maintain",
                            impact="Medium",
                            category="maintainability",
                            confidence=0.8
                        )
                    )
    
    def _check_complexity(self, tree: ast.AST, file_path: Path) -> None:
        """Check for complex code structures."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for deeply nested if statements
                depth = self._get_nesting_depth(node)
                if depth > 4:  # Arbitrary threshold
                    self.suggestions.append(
                        CodeSuggestion(
                            title="Deep Nesting",
                            description=f"Found deeply nested if statements (depth: {depth})",
                            priority=3,
                            affected_files=[str(file_path)],
                            suggested_changes=[
                                "Refactor to reduce nesting depth",
                                "Consider using early returns or guard clauses"
                            ],
                            rationale="Deep nesting makes code harder to read and maintain",
                            impact="Medium",
                            category="maintainability",
                            confidence=0.8
                        )
                    )
    
    def _check_docstrings(self, tree: ast.AST, file_path: Path) -> None:
        """Check for missing or inadequate docstrings."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if not ast.get_docstring(node):
                    self.suggestions.append(
                        CodeSuggestion(
                            title="Missing Docstring",
                            description=f"Missing docstring for {node.__class__.__name__} '{node.name}'",
                            priority=2,
                            affected_files=[str(file_path)],
                            suggested_changes=[
                                f"Add a docstring to {node.__class__.__name__} '{node.name}'",
                                "Include purpose, parameters, and return values"
                            ],
                            rationale="Docstrings help document code behavior and usage",
                            impact="Low",
                            category="documentation",
                            confidence=1.0
                        )
                    )
    
    def _check_naming_conventions(self, tree: ast.AST, file_path: Path) -> None:
        """Check for naming convention violations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    self.suggestions.append(
                        CodeSuggestion(
                            title="Naming Convention Violation",
                            description=f"Function name '{node.name}' doesn't follow Python naming conventions",
                            priority=2,
                            affected_files=[str(file_path)],
                            suggested_changes=[
                                f"Rename function '{node.name}' to follow snake_case convention"
                            ],
                            rationale="Consistent naming conventions improve code readability",
                            impact="Low",
                            category="style",
                            confidence=1.0
                        )
                    )
    
    def _check_error_handling(self, tree: ast.AST, file_path: Path) -> None:
        """Check for proper error handling."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        self.suggestions.append(
                            CodeSuggestion(
                                title="Bare Except Clause",
                                description="Found bare except clause",
                                priority=4,
                                affected_files=[str(file_path)],
                                suggested_changes=[
                                    "Specify the exception type(s) to catch",
                                    "Avoid catching all exceptions unless absolutely necessary"
                                ],
                                rationale="Bare except clauses can mask serious errors",
                                impact="High",
                                category="error_handling",
                                confidence=0.9
                            )
                        )
    
    def _check_code_duplication(self, tree: ast.AST, file_path: Path) -> None:
        """Check for potential code duplication."""
        # This is a simplified version. In a real implementation,
        # you'd want to use more sophisticated algorithms for detecting code duplication
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for i, func1 in enumerate(functions):
            for func2 in functions[i + 1:]:
                if self._are_functions_similar(func1, func2):
                    self.suggestions.append(
                        CodeSuggestion(
                            title="Potential Code Duplication",
                            description=f"Functions '{func1.name}' and '{func2.name}' appear similar",
                            priority=3,
                            affected_files=[str(file_path)],
                            suggested_changes=[
                                "Extract common code into a shared function",
                                "Consider using inheritance or composition"
                            ],
                            rationale="Code duplication increases maintenance burden",
                            impact="Medium",
                            category="maintainability",
                            confidence=0.7
                        )
                    )
    
    def _get_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate the nesting depth of a node."""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                max_depth = max(max_depth, self._get_nesting_depth(child, depth + 1))
        
        return max_depth
    
    def _are_functions_similar(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> bool:
        """Check if two functions are similar enough to be considered duplicated."""
        # This is a very simplified version. In a real implementation,
        # you'd want to use more sophisticated algorithms like:
        # - Abstract Syntax Tree comparison
        # - Token-based comparison
        # - Semantic analysis
        
        # For now, we'll just compare the number of nodes
        nodes1 = len(list(ast.walk(func1)))
        nodes2 = len(list(ast.walk(func2)))
        
        return abs(nodes1 - nodes2) < 5  # Arbitrary threshold 