# Contributing to LangGraph Checkpoint DynamoDB

Thank you for your interest in contributing to LangGraph Checkpoint DynamoDB! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Set up your development environment**:
   ```bash
   # Install the package in editable mode with dev dependencies
   pip install -e ".[dev]"
   ```

3. **Create a branch** for your contribution:
   ```bash
   git checkout -b your-feature-name
   ```

## Development Workflow

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Keep functions focused and maintainable
- Add docstrings for public APIs

### Testing

**All contributions must include tests.** See [tests/README.md](langgraph_checkpoint_dynamodb/tests/README.md) for detailed information on:

- Running unit tests
- Running integration tests
- Test requirements and setup
- What each test suite verifies

Quick start:
```bash
# Run all tests
pytest

# Run only unit tests (faster, no Docker required)
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```

### Pull Request Process

1. **Write clear, focused commits** - Each commit should represent a logical change
2. **Update documentation** if you're changing functionality
3. **Add tests** for new features or bug fixes
4. **Ensure all tests pass** before submitting
5. **Write a clear PR description** explaining:
   - What changes you made
   - Why you made them
   - How to test the changes

### Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes** - If you find a bug, please open an issue first
- **New features** - Discuss major features in an issue before implementing
- **Documentation** - Improvements to docs, examples, or comments
- **Tests** - Additional test coverage
- **Performance** - Optimizations and improvements

## Questions?

If you have questions about contributing, please open an issue with the `question` label.

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

