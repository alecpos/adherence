import ast
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def example_ast_usage():
    # Example of proper AST usage
    with open("example.py", "r") as f:
        content = f.read()
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(node.name)

def example_path_usage():
    # Example of proper Path usage
    path = Path("src")
    for file in path.rglob("*.py"):
        print(file.relative_to(path))

def example_dict_usage():
    # Example of proper defaultdict usage
    patterns = defaultdict(list)
    patterns["key"].append("value")
    return patterns

def example_typing_usage() -> Dict[str, List[str]]:
    # Example of proper typing usage
    return {"key": ["value"]} 