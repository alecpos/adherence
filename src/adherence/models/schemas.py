from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class StackFrame(BaseModel):
    """Represents a single frame in a stack trace."""
    file_path: str
    line_number: int
    function_name: str
    module_name: str
    code_context: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None

class StackTrace(BaseModel):
    """Represents a complete stack trace with metadata."""
    frames: List[StackFrame]
    timestamp: datetime = Field(default_factory=datetime.now)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class DependencyNode(BaseModel):
    """Represents a node in the dependency graph."""
    module_name: str
    file_path: str
    imports: List[str] = []
    imported_by: List[str] = []
    complexity: Optional[float] = None
    maintainability: Optional[float] = None

class CodeSuggestion(BaseModel):
    """Represents a code improvement suggestion."""
    title: str
    description: str
    priority: int = Field(ge=1, le=5)  # 1-5 priority scale
    affected_files: List[str]
    suggested_changes: List[str]
    rationale: str
    impact: str
    category: str  # e.g., "performance", "security", "maintainability"
    confidence: float = Field(ge=0.0, le=1.0)  # 0-1 confidence score

class AnalysisResult(BaseModel):
    """Represents the complete analysis result."""
    stack_traces: List[StackTrace]
    dependency_graph: Dict[str, DependencyNode]
    suggestions: List[CodeSuggestion]
    metrics: Dict[str, float]  # Overall codebase metrics
    summary: str 