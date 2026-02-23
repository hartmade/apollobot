# Contributing to ApolloBot

Thank you for your interest in contributing to ApolloBot! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/apollobot.git
   cd apollobot
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Run tests to verify setup:
   ```bash
   pytest
   ```

## Development Workflow

### Branching

- Create a feature branch from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```

- Use descriptive branch names:
  - `feature/` - New features
  - `fix/` - Bug fixes
  - `docs/` - Documentation changes
  - `refactor/` - Code refactoring

### Making Changes

1. Write your code following our style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Run the test suite: `pytest`
5. Run the linter: `ruff check .`

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom MCP server authentication

- Add bearer token support
- Add API key header support
- Update documentation
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### Pull Requests

1. Push your branch to your fork
2. Open a pull request against `main`
3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
4. Wait for CI checks to pass
5. Address any review feedback

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use `ruff` for linting

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md for user-facing changes

### Testing

- Write tests for all new functionality
- Maintain test coverage
- Use pytest fixtures for common setup
- Test edge cases

## Areas for Contribution

### Domain Packs

Add MCP server connectors for new data sources:

```python
# apollobot/mcp/servers/new_domain.py
from apollobot.mcp.servers.builtin import BuiltinServer

NEW_SERVERS = [
    BuiltinServer(
        name="new-data-source",
        url="https://mcp.frontierscience.ai/new-source",
        description="Description of the data source",
        domain="your_domain",
        category="data",
        api_base="https://api.example.com",
    ),
]
```

### Research Modes

Implement new research methodologies in `apollobot/agents/planner.py`.

### Analysis Methods

Add statistical and computational tools to the executor.

### Review Checks

Improve the self-review engine in `apollobot/review/`.

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
