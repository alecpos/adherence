"""Command-line interface for the adherence package."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from adherence.analyzer.dependency import DependencyAnalyzer
from adherence.analyzer.best_practices import BestPracticesAnalyzer
from adherence.analyzer.code_generation import CodeGenerator
from adherence.analyzer.ml_analyzer import MLCodeAnalyzer

console = Console()

def display_ml_insights(insights: dict, verbose: bool = False) -> None:
    """Display ML-based analysis insights.
    
    Args:
        insights: Dictionary containing ML analysis results
        verbose: Whether to show detailed output
    """
    # Display metrics
    metrics_table = Table(title="Code Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for metric, value in insights['metrics'].items():
        metrics_table.add_row(metric.replace('_', ' ').title(), f"{value:.2f}")
    
    console.print(metrics_table)
    
    # Display common patterns
    console.print("\n[bold]Common Code Patterns[/bold]")
    for pattern_type, patterns in insights['common_patterns'].items():
        console.print(f"\n[cyan]{pattern_type.title()}[/cyan]")
        for pattern in patterns:
            if verbose:
                console.print(Panel(str(pattern), title=f"Pattern Details"))
            else:
                console.print(f"- {pattern}")
    
    # Display recommendations
    console.print("\n[bold]Recommendations[/bold]")
    for rec in insights['recommendations']:
        if verbose:
            console.print(Panel(
                f"[yellow]{rec['message']}[/yellow]\n"
                f"Type: {rec['type']}\n"
                f"Severity: {rec['severity']}\n"
                f"Impact: {rec['impact']}",
                title="Detailed Recommendation"
            ))
        else:
            console.print(f"- {rec['message']}")

def display_best_practices(analysis: dict, verbose: bool = False) -> None:
    """Display best practices analysis results.
    
    Args:
        analysis: Dictionary containing best practices analysis results
        verbose: Whether to show detailed output
    """
    # Display import analysis
    console.print("\n[bold]Import Analysis[/bold]")
    for category, imports in analysis['imports'].items():
        if category != 'issues':
            console.print(f"\n[cyan]{category.title()}[/cyan]")
            for imp in imports:
                if verbose:
                    if 'module' in imp:  # ImportFrom case
                        console.print(Panel(
                            f"Module: {imp['module']}\n"
                            f"Type: {imp['type']}\n"
                            f"Names: {', '.join(imp['names'])}\n"
                            f"Line: {imp['line']}",
                            title="Import Details"
                        ))
                    else:  # Import case
                        console.print(Panel(
                            f"Name: {imp['name']}\n"
                            f"Type: {imp['type']}\n"
                            f"Line: {imp['line']}",
                            title="Import Details"
                        ))
                else:
                    if 'module' in imp:  # ImportFrom case
                        console.print(f"- {imp['module']} ({', '.join(imp['names'])})")
                    else:  # Import case
                        console.print(f"- {imp['name']}")
    
    # Display function analysis
    console.print("\n[bold]Function Analysis[/bold]")
    for category, functions in analysis['functions'].items():
        if category != 'issues':
            console.print(f"\n[cyan]{category.title()}[/cyan]")
            for func in functions:
                if verbose:
                    console.print(Panel(
                        f"Name: {func['name']}\n"
                        f"Args: {func['args']}\n"
                        f"Returns: {func.get('returns', 'None')}\n"
                        f"Complexity: {func['complexity']}\n"
                        f"Line: {func['line']}",
                        title="Function Details"
                    ))
                else:
                    console.print(f"- {func['name']}")
    
    # Display class analysis
    console.print("\n[bold]Class Analysis[/bold]")
    for category, classes in analysis['classes'].items():
        if category != 'issues':
            console.print(f"\n[cyan]{category.title()}[/cyan]")
            for cls in classes:
                if verbose:
                    console.print(Panel(
                        f"Name: {cls['name']}\n"
                        f"Bases: {cls['bases']}\n"
                        f"Methods: {cls['methods']}\n"
                        f"Complexity: {cls['complexity']}\n"
                        f"Line: {cls['line']}",
                        title="Class Details"
                    ))
                else:
                    console.print(f"- {cls['name']}")
    
    # Display recommendations
    console.print("\n[bold]Recommendations[/bold]")
    for rec in analysis['recommendations']:
        if verbose:
            console.print(Panel(
                f"[yellow]{rec['message']}[/yellow]\n"
                f"Type: {rec['type']}\n"
                f"Details: {', '.join(rec['details'])}",
                title="Detailed Recommendation"
            ))
        else:
            console.print(f"- {rec['message']}")

def display_generated_code(improvements: List[Dict[str, Any]]) -> None:
    """Display generated code improvements.
    
    Args:
        improvements: List of dictionaries containing code improvements
    """
    for imp in improvements:
        console.print(Panel(
            f"[bold cyan]Type:[/bold cyan] {imp['type']}\n"
            f"[bold cyan]Target:[/bold cyan] {imp['target']}\n\n"
            f"[bold cyan]Current Code:[/bold cyan]\n"
            f"{Syntax(imp['current_code'], 'python', theme='monokai')}\n\n"
            f"[bold cyan]Suggested Improvement:[/bold cyan]\n"
            f"{Syntax(imp['suggested_improvement'], 'python', theme='monokai')}\n\n"
            f"[bold cyan]Explanation:[/bold cyan]\n{imp['explanation']}",
            title="Code Improvement",
            expand=True
        ))

def analyze_file(file_path: str, output_dir: Optional[str] = None, verbose: bool = False) -> int:
    """Analyze a Python file and generate improvements.
    
    Args:
        file_path: Path to the Python file to analyze
        output_dir: Optional directory to save generated improvements
        verbose: Whether to show detailed output
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Initialize analyzers
        dep_analyzer = DependencyAnalyzer(file_path)
        best_practices_analyzer = BestPracticesAnalyzer()
        ml_analyzer = MLCodeAnalyzer()
        code_generator = CodeGenerator()
        
        # Perform dependency analysis
        console.print("\n[bold]Dependency Analysis Results[/bold]")
        dep_results = dep_analyzer.get_dependency_graph()
        console.print(dep_results)
        
        # Print import usage patterns
        console.print("\n[bold]Import Usage Patterns[/bold]")
        for module_name, imports in dep_analyzer.import_usage_patterns.items():
            console.print(f"\n[cyan]{module_name}:[/cyan]")
            for import_name, patterns in imports.items():
                console.print(f"\n  {import_name}:")
                for pattern in patterns:
                    console.print(f"    - {pattern}")
                if import_name in dep_analyzer.import_examples:
                    console.print("    [yellow]Examples:[/yellow]")
                    for example in dep_analyzer.import_examples[import_name][:3]:
                        console.print(f"      {example}")
        
        # Print best practices
        console.print("\n[bold]Best Practices Found[/bold]")
        for category, practices in dep_analyzer.best_practices.items():
            if practices:
                console.print(f"\n[cyan]{category.title()}:[/cyan]")
                for practice in practices:
                    console.print(f"  - {practice}")
        
        # Print suggestions
        suggestions = dep_analyzer.suggest_improvements()
        if suggestions:
            console.print("\n[bold]Suggestions for Improvement[/bold]")
            for suggestion in suggestions:
                console.print(f"  - {suggestion}")
        
        # Perform best practices analysis
        console.print("\n[bold]Best Practices Analysis[/bold]")
        best_practices_results = best_practices_analyzer.analyze_file(file_path)
        display_best_practices(best_practices_results, verbose)
        
        # Perform ML-based analysis
        console.print("\n[bold]Starting ML-Based Analysis...[/bold]")
        ml_insights = ml_analyzer.analyze_codebase(file_path)
        display_ml_insights(ml_insights, verbose)
        
        # Generate code improvements
        console.print("\n[bold]Generating Code Improvements...[/bold]")
        improvements = code_generator.generate_from_analysis(best_practices_results)
        display_generated_code(improvements)
        
        # Save generated code if output directory is specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            code_generator.save_generated_code(improvements, str(output_path))
            console.print(f"\n[green]Generated code saved to: {output_dir}[/green]")
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(Panel(
                Syntax(traceback.format_exc(), "python", theme="monokai"),
                title="Detailed Error Information",
                border_style="red"
            ))
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the adherence CLI.
    
    Args:
        argv: List of command line arguments. Defaults to sys.argv[1:].
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Analyze and improve Python code quality."
    )
    parser.add_argument(
        "command",
        choices=["analyze"],
        help="Command to execute",
    )
    parser.add_argument(
        "file",
        help="Python file to analyze",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output",
        help="Output directory for generated improvements",
    )

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return analyze_file(args.file, args.output, args.verbose)
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 