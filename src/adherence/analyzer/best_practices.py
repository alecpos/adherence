"""Best practices analyzer for Python code."""

from typing import Dict, List, Any
from pathlib import Path
import ast
from collections import defaultdict

class BestPracticesAnalyzer:
    """Analyzes code for best practices and provides recommendations."""
    
    def __init__(self):
        self.import_patterns = defaultdict(list)
        self.function_patterns = defaultdict(list)
        self.class_patterns = defaultdict(list)
        self.dependency_patterns = defaultdict(list)
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file for best practices."""
        self.current_file = file_path  # Store the file path for source code access
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analysis = {
            'imports': self._analyze_imports(tree),
            'functions': self._analyze_functions(tree),
            'classes': self._analyze_classes(tree),
            'dependencies': self._analyze_dependencies(tree),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze import statements for best practices."""
        imports = {
            'stdlib': [],
            'third_party': [],
            'local': [],
            'issues': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = self._get_import_info(node)
                if import_info['type'] == 'stdlib':
                    imports['stdlib'].append(import_info)
                elif import_info['type'] == 'third_party':
                    imports['third_party'].append(import_info)
                else:
                    imports['local'].append(import_info)
                    
                # Check for import issues
                issues = self._check_import_issues(node)
                imports['issues'].extend(issues)
        
        return imports
    
    def _analyze_functions(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze function definitions."""
        functions = {
            'with_docstrings': [],
            'without_docstrings': [],
            'complex': [],
            'issues': []
        }
        
        # Get the source code for the entire file
        with open(self.current_file, 'r', encoding='utf-8') as f:
            source = f.read()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function code using line numbers
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                source_lines = source.split('\n')[start_line-1:end_line]
                code = '\n'.join(source_lines)
                
                # Get function info with error handling
                try:
                    func_info = {
                        'name': node.name,
                        'args': self._get_function_args(node),
                        'complexity': self._calculate_complexity(node),
                        'docstring': ast.get_docstring(node),
                        'code': code,
                        'returns': bool(node.returns) if hasattr(node, 'returns') else False,
                        'line': node.lineno
                    }
                except Exception as e:
                    print(f"Error analyzing function {node.name}: {str(e)}")
                    func_info = {
                        'name': node.name,
                        'args': '',
                        'complexity': 0,
                        'docstring': None,
                        'code': code,
                        'returns': False,
                        'line': node.lineno
                    }
                
                if ast.get_docstring(node):
                    functions['with_docstrings'].append(func_info)
                else:
                    functions['without_docstrings'].append(func_info)
                
                if self._calculate_complexity(node) > 7:
                    functions['complex'].append(func_info)
                
                # Check for function issues
                issues = self._check_function_issues(node)
                functions['issues'].extend(issues)
        
        return functions
    
    def _analyze_classes(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze class definitions for best practices."""
        classes = {
            'with_docstrings': [],
            'without_docstrings': [],
            'complex': [],
            'issues': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._get_class_info(node)
                if class_info['has_docstring']:
                    classes['with_docstrings'].append(class_info)
                else:
                    classes['without_docstrings'].append(class_info)
                    
                if class_info['complexity'] > 10:
                    classes['complex'].append(class_info)
                    
                # Check for class issues
                issues = self._check_class_issues(node)
                classes['issues'].extend(issues)
        
        return classes
    
    def _analyze_dependencies(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze dependencies and their usage."""
        dependencies = {
            'imported': [],
            'used': [],
            'unused': [],
            'issues': []
        }
        
        # Track imported names
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    imported_names.add(name.name)
        
        # Track used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Analyze dependencies
        for name in imported_names:
            if name in used_names:
                dependencies['used'].append(name)
            else:
                dependencies['unused'].append(name)
        
        return dependencies
    
    def _get_import_info(self, node: ast.AST) -> Dict[str, Any]:
        """Get detailed information about an import statement."""
        if isinstance(node, ast.Import):
            return {
                'type': self._get_import_type(node.names[0].name),
                'name': node.names[0].name,
                'alias': node.names[0].asname,
                'line': node.lineno
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': self._get_import_type(node.module),
                'module': node.module,
                'names': [n.name for n in node.names],
                'aliases': [n.asname for n in node.names],
                'line': node.lineno
            }
        return {}
    
    def _get_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Get detailed information about a function definition.
        
        Args:
            node: The function definition AST node.
            
        Returns:
            Dictionary containing function information.
        """
        try:
            return {
                'name': node.name,
                'args': len(node.args.args),
                'returns': bool(node.returns) if hasattr(node, 'returns') else False,
                'has_docstring': ast.get_docstring(node) is not None,
                'complexity': self._calculate_complexity(node),
                'line': node.lineno,
                'args_info': self._get_function_args(node)
            }
        except Exception as e:
            # Log the error and return basic info
            print(f"Error getting function info for {node.name}: {str(e)}")
            return {
                'name': node.name,
                'args': 0,
                'returns': False,
                'has_docstring': False,
                'complexity': 0,
                'line': node.lineno,
                'args_info': ''
            }
    
    def _get_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Get detailed information about a class definition."""
        return {
            'name': node.name,
            'bases': len(node.bases),
            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
            'has_docstring': ast.get_docstring(node) is not None,
            'complexity': self._calculate_complexity(node),
            'line': node.lineno
        }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _get_import_type(self, name: str) -> str:
        """Determine if an import is from stdlib, third-party, or local."""
        # This is a simplified check - you might want to expand this
        stdlib_modules = {'os', 'sys', 'pathlib', 'typing', 'collections', 'ast'}
        if name.split('.')[0] in stdlib_modules:
            return 'stdlib'
        elif '.' in name:
            return 'third_party'
        return 'local'
    
    def _check_import_issues(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Check for issues in import statements."""
        issues = []
        
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.asname and name.asname == name.name:
                    issues.append({
                        'type': 'redundant_alias',
                        'message': f'Redundant alias for {name.name}',
                        'line': node.lineno
                    })
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                issues.append({
                    'type': 'relative_import',
                    'message': 'Consider using absolute imports instead of relative imports',
                    'line': node.lineno
                })
        
        return issues
    
    def _check_function_issues(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Check for issues in function definitions."""
        issues = []
        
        if not ast.get_docstring(node):
            issues.append({
                'type': 'missing_docstring',
                'message': f'Function {node.name} is missing a docstring',
                'line': node.lineno
            })
        
        if self._calculate_complexity(node) > 7:
            issues.append({
                'type': 'high_complexity',
                'message': f'Function {node.name} has high complexity',
                'line': node.lineno
            })
        
        return issues
    
    def _check_class_issues(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Check for issues in class definitions."""
        issues = []
        
        if not ast.get_docstring(node):
            issues.append({
                'type': 'missing_docstring',
                'message': f'Class {node.name} is missing a docstring',
                'line': node.lineno
            })
        
        if self._calculate_complexity(node) > 10:
            issues.append({
                'type': 'high_complexity',
                'message': f'Class {node.name} has high complexity',
                'line': node.lineno
            })
        
        return issues
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Import recommendations
        if analysis['imports']['issues']:
            recommendations.append({
                'type': 'imports',
                'message': 'Consider reorganizing imports following PEP 8 guidelines',
                'details': analysis['imports']['issues']
            })
        
        # Function recommendations
        if analysis['functions']['without_docstrings']:
            recommendations.append({
                'type': 'documentation',
                'message': f'Add docstrings to {len(analysis["functions"]["without_docstrings"])} functions',
                'details': [f['name'] for f in analysis['functions']['without_docstrings']]
            })
        
        if analysis['functions']['complex']:
            recommendations.append({
                'type': 'complexity',
                'message': f'Consider refactoring {len(analysis["functions"]["complex"])} complex functions',
                'details': [f['name'] for f in analysis['functions']['complex']]
            })
        
        # Class recommendations
        if analysis['classes']['without_docstrings']:
            recommendations.append({
                'type': 'documentation',
                'message': f'Add docstrings to {len(analysis["classes"]["without_docstrings"])} classes',
                'details': [c['name'] for c in analysis['classes']['without_docstrings']]
            })
        
        if analysis['classes']['complex']:
            recommendations.append({
                'type': 'complexity',
                'message': f'Consider refactoring {len(analysis["classes"]["complex"])} complex classes',
                'details': [c['name'] for c in analysis['classes']['complex']]
            })
        
        # Dependency recommendations
        if analysis['dependencies']['unused']:
            recommendations.append({
                'type': 'dependencies',
                'message': f'Remove {len(analysis["dependencies"]["unused"])} unused imports',
                'details': analysis['dependencies']['unused']
            })
        
        return recommendations
    
    def _get_function_args(self, node: ast.FunctionDef) -> str:
        """Get the function arguments as a string.
        
        Args:
            node: The function definition AST node.
            
        Returns:
            A string representation of the function arguments.
        """
        args = []
        
        # Add positional arguments
        for arg in node.args.args:
            arg_type = self._get_annotation_type(arg.annotation) if arg.annotation else 'Any'
            args.append(f"{arg.arg}: {arg_type}")
        
        # Add keyword-only arguments
        if node.args.kwonlyargs:
            args.append("*")
            for arg in node.args.kwonlyargs:
                arg_type = self._get_annotation_type(arg.annotation) if arg.annotation else 'Any'
                default = self._get_default_value(arg)
                args.append(f"{arg.arg}: {arg_type} = {default}")
        
        # Add **kwargs if present
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return ", ".join(args)
    
    def _get_annotation_type(self, annotation: ast.AST) -> str:
        """Convert an AST annotation to a string type.
        
        Args:
            annotation: The AST annotation node.
            
        Returns:
            A string representation of the type.
        """
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                if annotation.value.id == 'List':
                    return f"List[{self._get_annotation_type(annotation.slice)}]"
                elif annotation.value.id == 'Dict':
                    return f"Dict[{self._get_annotation_type(annotation.slice)}]"
                elif annotation.value.id == 'Optional':
                    return f"Optional[{self._get_annotation_type(annotation.slice)}]"
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        return "Any"
    
    def _get_default_value(self, arg: ast.arg) -> str:
        """Get the default value for a function argument.
        
        Args:
            arg: The argument AST node.
            
        Returns:
            A string representation of the default value.
        """
        if arg.default:
            if isinstance(arg.default, ast.Constant):
                return str(arg.default.value)
            elif isinstance(arg.default, ast.Name):
                return arg.default.id
            elif isinstance(arg.default, ast.Call):
                if isinstance(arg.default.func, ast.Name):
                    return f"{arg.default.func.id}()"
        return "None"
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate the cyclomatic complexity of an AST node.
        
        Args:
            node: The AST node to analyze.
            
        Returns:
            The cyclomatic complexity score.
        """
        complexity = 1  # Base complexity
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            complexity += self._calculate_complexity(child)
        
        return complexity
    
    def _check_function_issues(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Check for issues in a function definition.
        
        Args:
            node: The function definition AST node.
            
        Returns:
            List of issues found in the function.
        """
        issues = []
        
        # Check for long functions
        if len(node.body) > 50:
            issues.append({
                'type': 'function_length',
                'message': f"Function '{node.name}' is too long ({len(node.body)} lines)",
                'line': node.lineno
            })
        
        # Check for complex functions
        complexity = self._calculate_complexity(node)
        if complexity > 7:
            issues.append({
                'type': 'complexity',
                'message': f"Function '{node.name}' is too complex (complexity: {complexity})",
                'line': node.lineno
            })
        
        # Check for missing type hints
        if not node.returns and not any(arg.annotation for arg in node.args.args):
            issues.append({
                'type': 'type_hints',
                'message': f"Function '{node.name}' is missing type hints",
                'line': node.lineno
            })
        
        return issues 