# Adherence

A Python tool for analyzing and improving code quality, with a focus on type hints, docstrings, and best practices.

## Features

- Code analysis for Python files
- Dependency analysis
- Import usage patterns
- Best practices checking
- ML-based code analysis
- Automated code improvement suggestions
- Docstring generation
- Code refactoring suggestions
- Import organization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adherence.git
cd adherence

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Analyze a Python file
python -m adherence analyze path/to/your/file.py --verbose

# Generate code improvements
python -m adherence analyze path/to/your/file.py --output improvements/
```

## Configuration

The tool can be configured through command-line arguments:

- `--verbose`: Enable verbose output
- `--output`: Specify output directory for generated improvements

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Uses [CodeLlama](https://github.com/facebookresearch/codellama) for code generation
- Inspired by various code quality tools and best practices 