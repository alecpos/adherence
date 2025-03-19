"""
Adherence - Code Analysis and Improvement Tool

A tool that analyzes stack traces and provides intelligent suggestions for code improvements.
"""

from .suggestions.engine import SuggestionEngine
from .models.schemas import (
    StackFrame,
    StackTrace,
    DependencyNode,
    CodeSuggestion,
    AnalysisResult
)

__version__ = "0.1.0"
__all__ = [
    "SuggestionEngine",
    "StackFrame",
    "StackTrace",
    "DependencyNode",
    "CodeSuggestion",
    "AnalysisResult"
] 