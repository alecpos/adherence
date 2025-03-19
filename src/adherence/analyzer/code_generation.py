"""Code generation using Hugging Face transformers."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    repetition_penalty: float = 1.0

class CodeGenerator:
    """Code generation using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-34b-hf"):
        """Initialize the code generator.
        
        Args:
            model_name: Name or path of the Hugging Face model to use.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = GenerationConfig()
        
        # Initialize model and tokenizer immediately
        self.load_model()
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            if self.model is None:
                print(f"Loading model {self.model_name}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device)
                print("Model loaded successfully")
            
            if self.tokenizer is None:
                print("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Set up special tokens
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    print("Set padding token to EOS token")
                
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    print("Set padding token ID to EOS token ID")
                
                # Set configuration tokens
                self.config.pad_token_id = self.tokenizer.pad_token_id
                self.config.eos_token_id = self.tokenizer.eos_token_id
                
                print("Tokenizer loaded successfully with special tokens configured")
        except Exception as e:
            print(f"Error loading model or tokenizer: {str(e)}")
            raise

    def generate_code(self, prompt: str, config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate code from a prompt.
        
        Args:
            prompt: The code generation prompt.
            config: Optional generation configuration.
            
        Returns:
            List of generated code snippets.
        """
        try:
            print("Starting code generation...")
            # Ensure model and tokenizer are loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()
            
            # Use provided config or default
            if config is None:
                config = self.config
            
            print("Tokenizing input...")
            # Tokenize input without padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            print("Generating code...")
            # Generate with timeout
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        do_sample=config.do_sample,
                        num_return_sequences=config.num_return_sequences,
                        pad_token_id=config.pad_token_id,
                        eos_token_id=config.eos_token_id,
                        repetition_penalty=config.repetition_penalty,
                        max_time=30  # 30 second timeout
                    )
                except Exception as e:
                    print(f"Generation timeout or error: {str(e)}")
                    return []
            
            print("Decoding outputs...")
            # Decode outputs
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the prompt from the generated text
                generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
            
            print("Code generation completed successfully")
            return generated_texts
            
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return []

    def generate_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code improvements from analysis results."""
        improvements = []
        
        # Generate docstrings for functions without them
        for func in analysis['functions']['without_docstrings']:
            # Format code context for better readability
            code_context = func['code']
            if code_context:
                code_context = "\n".join(line.rstrip() for line in code_context.split('\n'))
            
            # Generate docstring
            docstring_prompt = f"""Generate a comprehensive docstring for this Python function following Google style:

Current code:
{code_context}

Requirements:
1. Include a brief description of what the function does
2. Document all parameters with their types
3. Document return value with type
4. Include any exceptions that may be raised
5. Add usage examples if appropriate
6. Keep the docstring concise and clear
7. Use proper Google style formatting

Example format:
def function_name(param1: type1, param2: type2) -> return_type:
    \"\"\"Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of what is returned

    Raises:
        ExceptionType: Description of when this exception is raised

    Example:
        >>> function_name(param1, param2)
        result
    \"\"\"

Generate only the docstring, not the function code."""

            docstrings = self.generate_code(docstring_prompt)
            if docstrings:
                improvements.append({
                    'type': 'docstring',
                    'target': f"function_{func['name']}",
                    'current_code': code_context,
                    'suggested_improvement': docstrings[0],
                    'explanation': f"Generated comprehensive docstring for function {func['name']}"
                })
        
        # Generate refactoring suggestions for complex functions
        for func in analysis['functions']['complex']:
            # Format code context
            code_context = func['code']
            if code_context:
                code_context = "\n".join(line.rstrip() for line in code_context.split('\n'))
            
            # Generate refactoring suggestion
            refactoring_prompt = f"""Suggest improvements for this Python function while maintaining its functionality:

Current code:
{code_context}

Requirements:
1. Break down into smaller, focused functions if the function is too long
2. Improve readability with clear variable names and comments
3. Add proper error handling with try/except blocks
4. Include type hints for parameters and return values
5. Add comprehensive docstrings
6. Keep the same functionality and behavior
7. Do not change the function signature unless necessary
8. Do not introduce new dependencies unless absolutely necessary
9. Follow Python best practices and PEP 8 guidelines

Example format:
def helper_function(param1: type1) -> type2:
    \"\"\"Helper function description.\"\"\"
    try:
        # Implementation
        return result
    except Exception as e:
        # Error handling
        raise

def main_function(param1: type1) -> type2:
    \"\"\"Main function description.\"\"\"
    # Implementation using helper functions
    return result

Generate only the improved code, not explanations."""

            refactorings = self.generate_code(refactoring_prompt)
            if refactorings:
                improvements.append({
                    'type': 'refactoring',
                    'target': f"function_{func['name']}",
                    'current_code': code_context,
                    'suggested_improvement': refactorings[0],
                    'explanation': f"Refactoring suggestion for complex function {func['name']}"
                })
        
        # Generate import organization suggestions
        if analysis['imports']['issues']:
            # Format imports for better readability
            imports_context = "\n".join(
                f"{imp['type']}: {imp.get('name', imp.get('module', ''))}"
                for imp in analysis['imports']['stdlib'] + 
                         analysis['imports']['third_party'] + 
                         analysis['imports']['local']
            )
            
            import_prompt = f"""Suggest improvements for organizing these Python imports following PEP 8:

Current imports:
{imports_context}

Requirements:
1. Group imports in order: standard library, third-party, local
2. Sort imports alphabetically within groups
3. Remove unused imports
4. Use absolute imports instead of relative
5. Remove redundant aliases
6. Keep only necessary imports
7. Follow PEP 8 guidelines for import organization

Example format:
# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from . import local_module
from .local_module import specific_function

Generate only the improved imports, not explanations."""

            import_suggestions = self.generate_code(import_prompt)
            if import_suggestions:
                improvements.append({
                    'type': 'imports',
                    'target': 'imports',
                    'current_code': imports_context,
                    'suggested_improvement': import_suggestions[0],
                    'explanation': "Generated import organization suggestions following PEP 8"
                })
        
        return improvements

    def save_generated_code(self, improvements: List[Dict[str, Any]], output_dir: str) -> None:
        """Save generated code improvements to files.
        
        Args:
            improvements: List of generated improvements.
            output_dir: Directory to save the generated code.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save improvements metadata
        with open(output_path / "improvements.json", "w") as f:
            json.dump(improvements, f, indent=2)
        
        # Save individual generated code files
        for imp in improvements:
            if imp['type'] in ['docstring', 'refactoring']:
                file_path = output_path / f"{imp['target']}.py"
                with open(file_path, "w") as f:
                    f.write(imp['suggested_improvement'])
            elif imp['type'] == 'imports':
                file_path = output_path / "imports.py"
                with open(file_path, "w") as f:
                    f.write(imp['suggested_improvement']) 