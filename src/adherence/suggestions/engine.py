from typing import List, Dict, Any
from ..models.schemas import CodeSuggestion, AnalysisResult
from ..analyzer.stack_trace import StackTraceAnalyzer
from ..analyzer.dependency import DependencyAnalyzer
from ..analyzer.static import StaticAnalyzer

class SuggestionEngine:
    """Generates intelligent code improvement suggestions based on multiple analysis sources."""
    
    def __init__(self):
        self.stack_trace_analyzer = StackTraceAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.static_analyzer = StaticAnalyzer()
    
    def analyze_codebase(self, directory: str) -> AnalysisResult:
        """Analyze the entire codebase and generate suggestions."""
        # Analyze dependencies
        dependency_graph = self.dependency_analyzer.analyze_directory(directory)
        
        # Get static analysis suggestions
        static_suggestions = []
        for file_path in dependency_graph.values():
            if file_path.file_path:
                static_suggestions.extend(
                    self.static_analyzer.analyze_file(file_path.file_path)
                )
        
        # Get dependency-based suggestions
        dependency_suggestions = self.dependency_analyzer.suggest_improvements()
        
        # Convert dependency suggestions to CodeSuggestion objects
        for suggestion in dependency_suggestions:
            static_suggestions.append(
                CodeSuggestion(
                    title="Dependency Issue",
                    description=suggestion,
                    priority=3,
                    affected_files=list(dependency_graph.keys()),
                    suggested_changes=["Review and refactor dependencies"],
                    rationale="Dependency issues can lead to maintenance problems",
                    impact="Medium",
                    category="dependencies",
                    confidence=0.8
                )
            )
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(dependency_graph, static_suggestions)
        
        # Generate summary
        summary = self._generate_summary(metrics, static_suggestions)
        
        return AnalysisResult(
            stack_traces=[],  # Will be populated when stack traces are provided
            dependency_graph=dependency_graph,
            suggestions=static_suggestions,
            metrics=metrics,
            summary=summary
        )
    
    def analyze_stack_trace(self, traceback_text: str) -> List[CodeSuggestion]:
        """Analyze a stack trace and generate relevant suggestions."""
        trace = self.stack_trace_analyzer.parse_stack_trace(traceback_text)
        
        # Get suggestions from stack trace analyzer
        improvement_suggestions = self.stack_trace_analyzer.suggest_improvements(trace)
        
        suggestions = []
        # Convert string suggestions to CodeSuggestion objects
        for suggestion in improvement_suggestions:
            suggestions.append(
                CodeSuggestion(
                    title="Stack Trace Analysis",
                    description=suggestion,
                    priority=1,
                    affected_files=[frame.file_path for frame in trace.frames],
                    suggested_changes=[suggestion],
                    rationale="Based on stack trace analysis",
                    impact="High",
                    category="error_handling",
                    confidence=0.9
                )
            )
        
        # Add frame-specific suggestions
        for frame in trace.frames:
            # Get frame context
            context = self.stack_trace_analyzer.get_frame_context(frame)
            
            # Analyze frame dependencies
            dependencies = self.stack_trace_analyzer.analyze_frame_dependencies(frame)
            
            # Generate suggestions based on frame analysis
            if dependencies:
                if not dependencies.get('docstring'):
                    suggestions.append(
                        CodeSuggestion(
                            title="Missing Documentation",
                            description=f"Function '{frame.function_name}' in {frame.module_name} lacks documentation",
                            priority=2,
                            affected_files=[frame.file_path],
                            suggested_changes=[
                                f"Add docstring to function '{frame.function_name}'",
                                "Include purpose, parameters, and return values"
                            ],
                            rationale="Documentation helps understand code behavior",
                            impact="Low",
                            category="documentation",
                            confidence=1.0
                        )
                    )
        
        return suggestions
    
    def _calculate_metrics(self, dependency_graph: Dict[str, Any], suggestions: List[CodeSuggestion]) -> Dict[str, float]:
        """Calculate overall codebase metrics."""
        metrics = {
            'complexity': 0.0,
            'maintainability': 0.0,
            'documentation_coverage': 0.0,
            'error_handling': 0.0,
            'code_quality': 0.0
        }
        
        # Calculate average complexity
        complexities = [node.complexity for node in dependency_graph.values() if node.complexity]
        if complexities:
            metrics['complexity'] = sum(complexities) / len(complexities)
        
        # Calculate average maintainability
        maintainabilities = [node.maintainability for node in dependency_graph.values() if node.maintainability]
        if maintainabilities:
            metrics['maintainability'] = sum(maintainabilities) / len(maintainabilities)
        
        # Calculate documentation coverage based on suggestions
        doc_suggestions = [s for s in suggestions if s.category == "documentation"]
        if suggestions:
            metrics['documentation_coverage'] = 1.0 - (len(doc_suggestions) / len(suggestions))
        
        # Calculate error handling quality
        error_suggestions = [s for s in suggestions if s.category == "error_handling"]
        if suggestions:
            metrics['error_handling'] = 1.0 - (len(error_suggestions) / len(suggestions))
        
        # Calculate overall code quality
        metrics['code_quality'] = (
            (1.0 - metrics['complexity'] / 20.0) +  # Normalize complexity
            (1.0 - metrics['maintainability']) +
            metrics['documentation_coverage'] +
            metrics['error_handling']
        ) / 4.0
        
        return metrics
    
    def _generate_summary(self, metrics: Dict[str, float], suggestions: List[CodeSuggestion]) -> str:
        """Generate a human-readable summary of the analysis."""
        summary_parts = []
        
        # Add overall quality assessment
        quality_score = metrics['code_quality']
        if quality_score >= 0.8:
            summary_parts.append("The codebase shows good overall quality.")
        elif quality_score >= 0.6:
            summary_parts.append("The codebase has moderate quality with room for improvement.")
        else:
            summary_parts.append("The codebase needs significant improvements.")
        
        # Add specific areas of concern
        concerns = []
        if metrics['complexity'] > 10:
            concerns.append("high complexity")
        if metrics['maintainability'] > 0.5:
            concerns.append("low maintainability")
        if metrics['documentation_coverage'] < 0.7:
            concerns.append("incomplete documentation")
        if metrics['error_handling'] < 0.7:
            concerns.append("improper error handling")
        
        if concerns:
            summary_parts.append(f"Key areas of concern: {', '.join(concerns)}.")
        
        # Add suggestion count
        if suggestions:
            summary_parts.append(f"Generated {len(suggestions)} improvement suggestions.")
        
        return " ".join(summary_parts) 