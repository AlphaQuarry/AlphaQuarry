# Contributing to Alpha Quarry

Thank you for your interest in contributing to Alpha Quarry! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for frontend development)
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/pengfeijiang320-eng/AlphaQuarry.git
   cd AlphaQuarry
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/alpha-quarry.git
   ```

## Development Setup

### Python Environment

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev,viz]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Frontend Development

```bash
cd dashboard/frontend
npm install
npm run dev  # Start dev server with hot reload
```

### Configuration

```bash
# Copy configuration template
cp configs/datasource.example.yaml configs/datasource.local.yaml

# Set your Tushare token
export TUSHARE_TOKEN="your_token_here"
```

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Maximum line length: 120 characters
- Use type hints where appropriate

#### Formatting

```bash
# Check code style
ruff check alpha_mining/ factor_research/ dashboard/api/ scripts/

# Auto-fix issues
ruff check --fix alpha_mining/ factor_research/ dashboard/api/ scripts/

# Format code
ruff format alpha_mining/ factor_research/ dashboard/api/ scripts/
```

#### Docstrings

- Use Google-style docstrings for public APIs
- Include parameter descriptions, return values, and examples where appropriate
- Keep docstrings concise but informative

Example:
```python
def calculate_ic(factor_values: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Calculate Information Coefficient (IC) between factor values and returns.

    Args:
        factor_values: Panel of factor values (date x code).
        returns: Panel of forward returns (date x code).

    Returns:
        Series of daily IC values indexed by date.

    Example:
        >>> ic = calculate_ic(factor_df, returns_df)
        >>> print(ic.mean())
        0.05
    """
```

### TypeScript/React

- Follow the existing ESLint configuration
- Use TypeScript for all new code
- Use functional components with hooks

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-robust-scaling` for new features
- `fix/issue-123-ic-calculation` for bug fixes
- `docs/update-readme` for documentation changes
- `refactor/improve-engine` for refactoring

### Commit Messages

Write clear, concise commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when applicable

Example:
```
Add MAD robust normalization to simulation module

- Implement mad_normalize() with 50% breakdown point
- Add winsorize() for extreme value handling
- Update scaling.py to support robust methods

Fixes #42
```

### Testing

Run tests before submitting:

```bash
# Run all tests
python -m pytest alpha_mining/tests tests -q

# Run specific test file
python -m pytest alpha_mining/tests/test_engine.py -q

# Run with coverage
python -m pytest alpha_mining/tests tests --cov=alpha_mining --cov=factor_research
```

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run checks**:
   ```bash
   # Lint
   ruff check alpha_mining/ factor_research/ dashboard/api/ scripts/

   # Format check
   ruff format --check alpha_mining/ factor_research/ dashboard/api/ scripts/

   # Tests
   python -m pytest alpha_mining/tests tests -q

   # Preflight guard
   python scripts/preflight_guard.py --strict
   ```

3. **Update documentation** if needed

### Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature
   ```

2. Open a Pull Request on GitHub

3. Fill in the PR template:
   - Description of changes
   - Related issues
   - Testing performed
   - Checklist completion

4. Wait for review and address feedback

### PR Requirements

- All CI checks must pass
- Code must pass linting (`ruff check`)
- Code must be formatted (`ruff format --check`)
- Tests must pass
- Documentation must be updated if applicable

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment**:
   - Python version
   - Operating system
   - Package version (`pip show alpha-quarry`)

2. **Steps to reproduce**:
   ```bash
   # Exact commands that reproduce the issue
   python scripts/run_closed_loop.py --source-backend duckdb ...
   ```

3. **Expected behavior**: What you expected to happen

4. **Actual behavior**: What actually happened, including error messages

5. **Additional context**: Screenshots, logs, or other relevant information

### Feature Requests

When requesting features:

1. **Describe the problem**: What problem does this feature solve?

2. **Propose a solution**: How should this feature work?

3. **Alternatives considered**: What other approaches did you consider?

4. **Additional context**: Examples, references, or mockups

## Project Structure

Understanding the project structure helps with contributions:

```
alpha_mining/          # Core mining engine
├── engine.py          # Expression evaluation engine
├── parser.py          # Expression parser
├── registry.py        # Operator registry
├── panel_store.py     # Panel data store
├── operators/         # 84+ expression operators
├── mining/            # Candidate generation and mutation
├── simulation/        # Neutralization, scaling, delay
└── workflow/          # Closed-loop orchestration

factor_research/       # Factor analysis library
├── single_factor.py   # IC, layer analysis, portfolios
├── screening.py       # Factor effectiveness scoring
└── diagnostics.py     # IC stability, coverage

dashboard/             # FastAPI + React Dashboard
├── api/               # Backend API
└── frontend/          # React frontend

scripts/               # Entry-point scripts
configs/               # Configuration files
docs/                  # Documentation
```

## Questions?

If you have questions about contributing, feel free to:

1. Open a GitHub issue with the `question` label
2. Check existing documentation in `docs/`
3. Review the [README](README.md) for project overview

Thank you for contributing to Alpha Quarry!
