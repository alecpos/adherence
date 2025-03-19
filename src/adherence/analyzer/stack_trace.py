import re
import inspect
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..models.schemas import StackFrame, StackTrace

class StackTraceAnalyzer:
    """Analyzes stack traces and extracts meaningful information."""
    
    def __init__(self):
        self.frame_pattern = re.compile(
            r'File "(?P<file_path>.*?)", line (?P<line_number>\d+), in (?P<function_name>.*)'
        )
    
    def parse_stack_trace(self, traceback_text: str) -> StackTrace:
        """Parse a stack trace string into a structured format."""
        frames = []
        error_type = None
        error_message = None
        
        # Split the traceback into lines
        lines = traceback_text.strip().split('\n')
        
        # Extract error information from the last line
        if lines and ':' in lines[-1]:
            error_parts = lines[-1].split(':', 1)
            error_type = error_parts[0].strip()
            error_message = error_parts[1].strip()
        
        # Process each frame
        for i in range(0, len(lines) - 1, 2):
            frame_line = lines[i]
            context_line = lines[i + 1] if i + 1 < len(lines) else None
            
            match = self.frame_pattern.match(frame_line)
            if match:
                frame_data = match.groupdict()
                frame = StackFrame(
                    file_path=frame_data['file_path'],
                    line_number=int(frame_data['line_number']),
                    function_name=frame_data['function_name'],
                    module_name=self._extract_module_name(frame_data['file_path']),
                    code_context=context_line.strip() if context_line else None
                )
                frames.append(frame)
        
        return StackTrace(
            frames=frames,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now()
        )
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract the module name from a file path."""
        # Remove file extension and path separators
        module_name = file_path.rsplit('.', 1)[0].replace('/', '.').replace('\\', '.')
        # Remove leading/trailing dots and any empty segments
        return '.'.join(filter(None, module_name.split('.')))
    
    def get_frame_context(self, frame: StackFrame, context_lines: int = 3) -> str:
        """Get the code context around a specific frame."""
        try:
            with open(frame.file_path, 'r') as f:
                lines = f.readlines()
            
            start = max(0, frame.line_number - context_lines - 1)
            end = min(len(lines), frame.line_number + context_lines)
            
            context = []
            for i in range(start, end):
                prefix = '>' if i == frame.line_number - 1 else ' '
                context.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
            
            return '\n'.join(context)
        except Exception:
            return "Unable to load file context"
    
    def analyze_frame_dependencies(self, frame: StackFrame) -> Dict[str, Any]:
        """Analyze dependencies and context for a specific frame."""
        try:
            # Get the module object
            module = __import__(frame.module_name, fromlist=['*'])
            
            # Get the function object
            func = getattr(module, frame.function_name)
            
            # Get function signature and docstring
            sig = inspect.signature(func)
            doc = inspect.getdoc(func)
            
            # Get local variables if available
            try:
                frame_obj = inspect.currentframe()
                while frame_obj:
                    if frame_obj.f_code.co_filename == frame.file_path and frame_obj.f_lineno == frame.line_number:
                        locals_dict = frame_obj.f_locals
                        break
                    frame_obj = frame_obj.f_back
            except Exception:
                locals_dict = None
            
            return {
                'signature': str(sig),
                'docstring': doc,
                'locals': locals_dict,
                'module_doc': inspect.getdoc(module)
            }
        except Exception:
            return {}
    
    def suggest_improvements(self, trace: StackTrace) -> List[str]:
        """Generate improvement suggestions based on stack trace analysis."""
        suggestions = []
        
        # Check for common error patterns
        if trace.error_type == "ZeroDivisionError":
            suggestions.append(
                f"Division by zero detected in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Add input validation to prevent division by zero."
            )
        elif trace.error_type == "NameError":
            suggestions.append(
                f"Undefined variable or function detected in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Check for typos or missing imports."
            )
        elif trace.error_type == "TypeError":
            suggestions.append(
                f"Type mismatch detected in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Ensure correct data types are being used."
            )
        elif trace.error_type == "IndexError":
            suggestions.append(
                f"List index out of range in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Add bounds checking before accessing list elements."
            )
        elif trace.error_type == "KeyError":
            suggestions.append(
                f"Dictionary key not found in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Add key existence check before accessing dictionary."
            )
        elif trace.error_type == "AttributeError":
            suggestions.append(
                f"Object attribute not found in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Verify object type and attribute existence."
            )
        elif trace.error_type == "ImportError":
            suggestions.append(
                f"Module import failed in {trace.frames[-1].function_name}:\n"
                f"Code context:\n{trace.frames[-1].code_context}\n"
                "Check if the module is installed and accessible."
            )
        
        for frame in trace.frames:
            # Check for common patterns that might indicate issues
            if frame.function_name.startswith('_'):
                suggestions.append(
                    f"Consider making {frame.function_name} public if it's meant to be part of the API:\n"
                    f"Code context:\n{frame.code_context}"
                )
            
            # Check for error handling patterns
            if frame.code_context and 'try:' in frame.code_context:
                suggestions.append(
                    f"Add specific exception handling in {frame.function_name} to catch {trace.error_type}:\n"
                    f"Code context:\n{frame.code_context}"
                )
        
        return suggestions 