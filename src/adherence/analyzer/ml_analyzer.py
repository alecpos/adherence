import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class MLCodeAnalyzer:
    """Machine learning based code analyzer that learns patterns from the entire codebase."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            token_pattern=r'(?u)\b[A-Za-z_][A-Za-z0-9_]*\b'
        )
        self.code_embeddings = {}
        self.pattern_clusters = {}
        self.import_patterns = defaultdict(list)
        self.usage_patterns = defaultdict(list)
        self.code_graph = nx.DiGraph()
        
    def analyze_codebase(self, root_dir: str) -> Dict[str, Any]:
        """Analyze the codebase using ML techniques."""
        # Get the current file and its directory
        current_file = Path(root_dir)
        current_dir = current_file.parent
        
        # Collect Python files in the same directory
        files = list(current_dir.glob("*.py"))
        
        # Extract features and build knowledge base
        self._extract_code_features(files)
        self._cluster_patterns()
        self._build_knowledge_graph()
        
        return self._generate_insights()
    
    def _extract_code_features(self, files: List[Path]) -> None:
        """Extract features from code files using AST and text analysis."""
        code_texts = []
        file_contents = {}
        
        for file in files:
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(file, 'r', encoding=encoding) as f:
                            content = f.read()
                            break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    print(f"Warning: Could not read {file} with any supported encoding")
                    continue
                
                file_contents[file] = content
                code_texts.append(content)
                
                # Parse AST and extract patterns
                tree = ast.parse(content)
                self._extract_ast_patterns(tree, file)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        # Generate embeddings for code texts
        if code_texts:
            try:
                embeddings = self.vectorizer.fit_transform(code_texts)
                for file, embedding in zip(files, embeddings.toarray()):
                    self.code_embeddings[file] = embedding
            except Exception as e:
                print(f"Warning: Could not generate embeddings: {str(e)}")
                # Create dummy embeddings if vectorization fails
                for file in files:
                    self.code_embeddings[file] = np.zeros(100)  # Dummy embedding
    
    def _extract_ast_patterns(self, tree: ast.AST, file: Path) -> None:
        """Extract patterns from AST nodes."""
        patterns = defaultdict(list)
        
        for node in ast.walk(tree):
            # Analyze imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                pattern = self._analyze_import_pattern(node)
                if pattern:
                    self.import_patterns[file].append(pattern)
            
            # Analyze function definitions
            elif isinstance(node, ast.FunctionDef):
                pattern = self._analyze_function_pattern(node)
                if pattern:
                    patterns['functions'].append(pattern)
            
            # Analyze class definitions
            elif isinstance(node, ast.ClassDef):
                pattern = self._analyze_class_pattern(node)
                if pattern:
                    patterns['classes'].append(pattern)
            
            # Analyze variable assignments
            elif isinstance(node, ast.Assign):
                pattern = self._analyze_assignment_pattern(node)
                if pattern:
                    patterns['assignments'].append(pattern)
        
        self.usage_patterns[file] = patterns
    
    def _analyze_import_pattern(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze import statement patterns."""
        if isinstance(node, ast.Import):
            return {
                'type': 'direct_import',
                'names': [n.name for n in node.names],
                'aliases': [n.asname for n in node.names]
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [n.name for n in node.names],
                'aliases': [n.asname for n in node.names]
            }
        return None
    
    def _analyze_function_pattern(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze function definition patterns."""
        return {
            'name': node.name,
            'args': len(node.args.args),
            'returns': bool(node.returns),
            'decorators': len(node.decorator_list),
            'complexity': self._calculate_complexity(node),
            'docstring': ast.get_docstring(node) is not None
        }
    
    def _analyze_class_pattern(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze class definition patterns."""
        methods = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                methods.append(self._analyze_function_pattern(child))
        
        return {
            'name': node.name,
            'bases': len(node.bases),
            'methods': methods,
            'decorators': len(node.decorator_list),
            'docstring': ast.get_docstring(node) is not None
        }
    
    def _analyze_assignment_pattern(self, node: ast.Assign) -> Dict[str, Any]:
        """Analyze assignment patterns."""
        return {
            'targets': len(node.targets),
            'value_type': type(node.value).__name__,
            'is_constant': isinstance(node.value, (ast.Num, ast.Str, ast.NameConstant))
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
    
    def _cluster_patterns(self) -> None:
        """Cluster similar code patterns using DBSCAN."""
        if not self.code_embeddings:
            return
        
        # Convert embeddings to array
        X = np.array(list(self.code_embeddings.values()))
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(X)
        
        # Group files by cluster
        for file, label in zip(self.code_embeddings.keys(), labels):
            if label >= 0:  # Ignore noise points
                if label not in self.pattern_clusters:
                    self.pattern_clusters[label] = []
                self.pattern_clusters[label].append(file)
    
    def _build_knowledge_graph(self) -> None:
        """Build a knowledge graph of code relationships."""
        # Add nodes for files
        for file in self.code_embeddings:
            self.code_graph.add_node(str(file), type='file')
            
            # Add import relationships
            for pattern in self.import_patterns[file]:
                if pattern['type'] == 'direct_import':
                    for name in pattern['names']:
                        self.code_graph.add_edge(str(file), name, type='imports')
                elif pattern['type'] == 'from_import':
                    module = pattern['module']
                    for name in pattern['names']:
                        self.code_graph.add_edge(str(file), f"{module}.{name}", type='imports')
            
            # Add pattern relationships
            for pattern_type, patterns in self.usage_patterns[file].items():
                for pattern in patterns:
                    pattern_id = f"{pattern_type}_{pattern.get('name', 'anonymous')}"
                    self.code_graph.add_edge(str(file), pattern_id, type=pattern_type)
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate insights from the analyzed patterns."""
        insights = {
            'clusters': {},
            'common_patterns': defaultdict(list),
            'metrics': {},
            'recommendations': []
        }
        
        # Analyze clusters
        for label, files in self.pattern_clusters.items():
            insights['clusters'][f'cluster_{label}'] = {
                'files': [str(f) for f in files],
                'common_imports': self._find_common_imports(files),
                'similarity_score': self._calculate_cluster_similarity(files)
            }
        
        # Find common patterns
        for pattern_type in ['functions', 'classes', 'assignments']:
            patterns = []
            for file_patterns in self.usage_patterns.values():
                patterns.extend(file_patterns.get(pattern_type, []))
            
            if patterns:
                insights['common_patterns'][pattern_type] = self._find_common_patterns(patterns)
        
        # Calculate metrics
        insights['metrics'] = {
            'avg_complexity': self._calculate_average_complexity(),
            'import_diversity': len(set(
                name for patterns in self.import_patterns.values()
                for pattern in patterns
                for name in pattern['names']
            )),
            'code_reuse': self._calculate_code_reuse(),
            'modularity': self._calculate_modularity()
        }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def _find_common_imports(self, files: List[Path]) -> List[str]:
        """Find common imports among files."""
        if not files:
            return []
            
        common_imports = set()
        first = True
        
        for file in files:
            file_imports = set()
            for pattern in self.import_patterns[file]:
                file_imports.update(pattern['names'])
            
            if first:
                common_imports = file_imports
                first = False
            else:
                common_imports &= file_imports
        
        return list(common_imports)
    
    def _calculate_cluster_similarity(self, files: List[Path]) -> float:
        """Calculate average similarity between files in a cluster."""
        if len(files) < 2:
            return 1.0
            
        embeddings = np.array([self.code_embeddings[f] for f in files])
        similarities = cosine_similarity(embeddings)
        return float(similarities.mean())
    
    def _find_common_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common patterns in a list of patterns."""
        if not patterns:
            return []
            
        # Convert patterns to comparable format
        pattern_strs = [str(sorted(p.items())) for p in patterns]
        
        # Count pattern frequencies
        pattern_counts = defaultdict(int)
        for p in pattern_strs:
            pattern_counts[p] += 1
        
        # Return most common patterns
        common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [eval(p[0]) for p in common]
    
    def _calculate_average_complexity(self) -> float:
        """Calculate average complexity across all functions."""
        complexities = []
        for patterns in self.usage_patterns.values():
            for func in patterns.get('functions', []):
                complexities.append(func['complexity'])
        
        return float(np.mean(complexities)) if complexities else 0.0
    
    def _calculate_code_reuse(self) -> float:
        """Calculate code reuse metric based on pattern similarity."""
        total_patterns = sum(len(patterns) for patterns in self.usage_patterns.values())
        unique_patterns = len(set(
            str(pattern)
            for patterns in self.usage_patterns.values()
            for pattern_list in patterns.values()
            for pattern in pattern_list
        ))
        
        return 1 - (unique_patterns / total_patterns) if total_patterns > 0 else 0.0
    
    def _calculate_modularity(self) -> float:
        """Calculate modularity of the code graph."""
        if len(self.code_graph) < 2:
            return 0.0
            
        try:
            communities = list(nx.community.greedy_modularity_communities(self.code_graph))
            if not communities:
                return 0.0
                
            # Calculate modularity manually
            total_edges = self.code_graph.number_of_edges()
            if total_edges == 0:
                return 0.0
                
            modularity = 0.0
            for community in communities:
                subgraph = self.code_graph.subgraph(community)
                internal_edges = subgraph.number_of_edges()
                total_degree = sum(dict(self.code_graph.degree(community)).values())
                
                if total_edges > 0:
                    modularity += (internal_edges / total_edges) - (total_degree / (2 * total_edges)) ** 2
            
            return float(modularity)
            
        except Exception as e:
            print(f"Warning: Error calculating modularity: {str(e)}")
            return 0.0
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        # Check complexity
        avg_complexity = insights['metrics']['avg_complexity']
        if avg_complexity > 7:
            complex_functions = self._find_complex_functions()
            recommendations.append({
                'type': 'complexity',
                'severity': 'high',
                'message': f'Average function complexity ({avg_complexity:.2f}) is high. Consider refactoring complex functions.',
                'impact': 'maintainability',
                'examples': [
                    {
                        'function': func['name'],
                        'complexity': func['complexity'],
                        'code': func['code'],
                        'suggestion': self._generate_complexity_reduction(func)
                    }
                    for func in complex_functions[:3]  # Show top 3 most complex functions
                ]
            })
        
        # Check code reuse
        code_reuse = insights['metrics']['code_reuse']
        if code_reuse < 0.3:
            duplicate_patterns = self._find_duplicate_patterns()
            recommendations.append({
                'type': 'code_reuse',
                'severity': 'medium',
                'message': 'Low code reuse detected. Consider extracting common patterns into shared utilities.',
                'impact': 'maintainability',
                'examples': [
                    {
                        'pattern': pattern['description'],
                        'occurrences': pattern['locations'],
                        'suggestion': self._generate_code_reuse_suggestion(pattern)
                    }
                    for pattern in duplicate_patterns[:3]  # Show top 3 duplicate patterns
                ]
            })
        
        # Check modularity
        modularity = insights['metrics']['modularity']
        if modularity < 0.5:
            modularity_issues = self._find_modularity_issues()
            recommendations.append({
                'type': 'modularity',
                'severity': 'medium',
                'message': 'Code modularity could be improved. Consider reorganizing code into more cohesive modules.',
                'impact': 'architecture',
                'examples': [
                    {
                        'module': issue['module'],
                        'cohesion': issue['cohesion'],
                        'suggestion': self._generate_modularity_suggestion(issue)
                    }
                    for issue in modularity_issues[:3]  # Show top 3 modularity issues
                ]
            })
        
        # Analyze clusters
        for cluster_id, cluster in insights['clusters'].items():
            if cluster['similarity_score'] > 0.8:
                similar_files = self._find_similar_files(cluster['files'])
                recommendations.append({
                    'type': 'duplication',
                    'severity': 'medium',
                    'message': f'High code similarity in {cluster_id}. Consider refactoring to remove duplication.',
                    'impact': 'maintainability',
                    'affected_files': cluster['files'],
                    'examples': [
                        {
                            'file1': pair['file1'],
                            'file2': pair['file2'],
                            'similarity': pair['similarity'],
                            'suggestion': self._generate_duplication_reduction(pair)
                        }
                        for pair in similar_files[:3]  # Show top 3 similar file pairs
                    ]
                })
        
        return recommendations
    
    def _find_complex_functions(self) -> List[Dict[str, Any]]:
        """Find functions with high complexity."""
        complex_functions = []
        for file_path, patterns in self.usage_patterns.items():
            for func in patterns.get('functions', []):
                if func.get('complexity', 0) > 7:
                    complex_functions.append({
                        'name': func['name'],
                        'complexity': func['complexity'],
                        'code': func['code'],
                        'file': str(file_path)
                    })
        return sorted(complex_functions, key=lambda x: x['complexity'], reverse=True)
    
    def _find_duplicate_patterns(self) -> List[Dict[str, Any]]:
        """Find duplicate code patterns."""
        patterns = defaultdict(list)
        for file_path, usage in self.usage_patterns.items():
            for pattern_type, items in usage.items():
                for item in items:
                    pattern_key = self._get_pattern_key(item)
                    patterns[pattern_key].append({
                        'file': str(file_path),
                        'line': item.get('line', 0),
                        'code': item.get('code', '')
                    })
        
        return [
            {
                'description': f"Duplicate {pattern_type} pattern",
                'locations': locations,
                'count': len(locations)
            }
            for pattern_type, locations in patterns.items()
            if len(locations) > 1
        ]
    
    def _find_modularity_issues(self) -> List[Dict[str, Any]]:
        """Find modules with low cohesion."""
        issues = []
        for file_path, patterns in self.usage_patterns.items():
            cohesion = self._calculate_module_cohesion(patterns)
            if cohesion < 0.5:
                issues.append({
                    'module': str(file_path),
                    'cohesion': cohesion,
                    'patterns': patterns
                })
        return sorted(issues, key=lambda x: x['cohesion'])
    
    def _find_similar_files(self, files: List[str]) -> List[Dict[str, Any]]:
        """Find pairs of similar files."""
        similar_pairs = []
        for i, file1 in enumerate(files):
            for file2 in files[i + 1:]:
                similarity = self._calculate_file_similarity(file1, file2)
                if similarity > 0.8:
                    similar_pairs.append({
                        'file1': file1,
                        'file2': file2,
                        'similarity': similarity
                    })
        return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    def _generate_complexity_reduction(self, func: Dict[str, Any]) -> str:
        """Generate suggestion for reducing function complexity."""
        return (
            f"Consider breaking down function '{func['name']}' into smaller functions:\n"
            f"1. Extract the main logic into separate functions\n"
            f"2. Use early returns to reduce nesting\n"
            f"3. Consider using the Strategy pattern for complex conditional logic"
        )
    
    def _generate_code_reuse_suggestion(self, pattern: Dict[str, Any]) -> str:
        """Generate suggestion for improving code reuse."""
        return (
            f"Extract the common pattern into a utility function:\n"
            f"1. Create a new function in a shared utilities module\n"
            f"2. Move the common code into this function\n"
            f"3. Update all occurrences to use the new function"
        )
    
    def _generate_modularity_suggestion(self, issue: Dict[str, Any]) -> str:
        """Generate suggestion for improving module cohesion."""
        return (
            f"Reorganize module '{issue['module']}' to improve cohesion:\n"
            f"1. Group related functionality together\n"
            f"2. Extract unrelated code into separate modules\n"
            f"3. Consider using a more focused module structure"
        )
    
    def _generate_duplication_reduction(self, pair: Dict[str, Any]) -> str:
        """Generate suggestion for reducing code duplication."""
        return (
            f"Refactor to remove duplication between '{pair['file1']}' and '{pair['file2']}':\n"
            f"1. Extract common code into a shared module\n"
            f"2. Use inheritance or composition to share functionality\n"
            f"3. Consider using a template pattern for similar structures"
        )
    
    def _get_pattern_key(self, item: Dict[str, Any]) -> str:
        """Generate a unique key for a code pattern.
        
        Args:
            item: The pattern item to generate a key for.
            
        Returns:
            A string key that uniquely identifies the pattern.
        """
        if 'name' in item:
            return f"{item.get('type', 'unknown')}_{item['name']}"
        elif 'targets' in item:
            return f"assignment_{item.get('value_type', 'unknown')}"
        else:
            return str(sorted(item.items()))
    
    def _calculate_file_similarity(self, file1: str, file2: str) -> float:
        """Calculate similarity between two files using their embeddings."""
        if file1 not in self.code_embeddings or file2 not in self.code_embeddings:
            return 0.0
        return float(cosine_similarity(
            self.code_embeddings[file1].reshape(1, -1),
            self.code_embeddings[file2].reshape(1, -1)
        )[0][0])
    
    def _calculate_module_cohesion(self, patterns: Dict[str, Any]) -> float:
        """Calculate cohesion score for a module based on its patterns."""
        if not patterns:
            return 0.0
            
        # Count pattern types
        pattern_counts = defaultdict(int)
        for pattern_type, items in patterns.items():
            pattern_counts[pattern_type] = len(items)
        
        # Calculate diversity score (lower is better)
        total_patterns = sum(pattern_counts.values())
        if total_patterns == 0:
            return 0.0
            
        # Shannon's diversity index
        proportions = [count / total_patterns for count in pattern_counts.values()]
        diversity = -sum(p * np.log2(p) for p in proportions)
        
        # Convert to cohesion score (higher is better)
        max_diversity = np.log2(len(pattern_counts)) if pattern_counts else 1
        return 1 - (diversity / max_diversity) 