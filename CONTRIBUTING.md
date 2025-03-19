# Contributing to Adherence

Thank you for your interest in contributing to Adherence! This document provides guidelines and steps for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/adherence.git
   cd adherence
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Write meaningful variable and function names

## Testing

- Write tests for new features
- Ensure all tests pass before submitting a PR
- Run the test suite:
  ```bash
  pytest
  ```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Create a Pull Request on GitHub

## Code Review

- Respond to review comments promptly
- Make requested changes
- Keep commits focused and atomic
- Update documentation as needed

## Documentation

- Update README.md if needed
- Add docstrings to new functions and classes
- Update type hints when modifying functions
- Keep documentation up to date with code changes

## Questions?

Feel free to open an issue if you have any questions about contributing to the project. 