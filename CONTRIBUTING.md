# Contributing to ICD-9 Code Prediction

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/ziyuqiu/icd_prediction/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Environment details (Python version, OS, GPU)
   - Relevant error messages or logs

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - Potential impact on existing functionality

### Submitting Changes

#### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/icd_prediction.git
   cd icd_prediction
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our [coding standards](#coding-standards)

3. Write tests for new functionality:
   ```bash
   # Add tests to tests/unit/
   pytest tests/unit/test_your_module.py
   ```

4. Ensure all tests pass:
   ```bash
   pytest
   ```

5. Check code coverage (must be 80%+):
   ```bash
   pytest --cov --cov-report=term-missing
   ```

6. Commit your changes:
   ```bash
   git commit -m "feat: add your feature description"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints for function signatures

```python
def process_clinical_note(
    note: str,
    max_length: int = 512
) -> list[int]:
    """
    Process a clinical note into token IDs.

    Args:
        note: Raw clinical note text
        max_length: Maximum sequence length

    Returns:
        List of token IDs
    """
    # Implementation
    pass
```

### Code Organization

- Keep functions small and focused (< 50 lines)
- Keep files focused (< 800 lines)
- One class per file when possible
- Group related functionality into modules

### Testing

- Write tests before implementing features (TDD)
- Each test should test one thing
- Use descriptive test names
- Include edge cases

```python
class TestProcessClinicalNote:
    def test_handles_empty_input(self):
        result = process_clinical_note("")
        assert result == []

    def test_truncates_long_input(self):
        long_note = "word " * 1000
        result = process_clinical_note(long_note, max_length=100)
        assert len(result) <= 100
```

### Documentation

- Write docstrings for all public functions
- Update README.md for user-facing changes
- Include examples in docstrings

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat: add precision@k metric calculation

fix: handle empty clinical notes in preprocessing

docs: add MIMIC-III setup instructions

test: add unit tests for tokenization functions
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure CI passes (tests, linting)
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

### PR Title Format

```
<type>: <short description>
```

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How were changes tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests pass
```

## Development Tips

### Running Specific Tests

```bash
# Run tests for a specific module
pytest tests/unit/test_bert_model.py

# Run tests matching a pattern
pytest -k "test_tokenize"

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Debugging

```bash
# Run pytest with pdb
pytest --pdb

# Show print statements
pytest -s
```

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Check style with flake8
flake8 .
```

## Getting Help

- Open an issue for questions
- Tag maintainers for urgent matters
- Check existing issues and PRs

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- CONTRIBUTORS.md (coming soon)

Thank you for contributing!
