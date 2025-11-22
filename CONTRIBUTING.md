# verif - Development Guide

## Environment Setup

We use **conda** for environment management to ensure consistent development across different machines.

### Conda Environment Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate verif

# Install pre-commit hooks
pre-commit install
```

### Automated Setup Script

```bash
# Make script executable
chmod +x setup-dev.sh

# Run automated setup
./setup-dev.sh
```

## Quick Start for Development

```bash
# Ensure environment is activated
conda activate verif

# Run code formatting
black app/ tests/
isort app/ tests/

# Run linting
flake8 app/ tests/

# Run type checking
mypy app/

# Run tests
pytest tests/ -v
```

## Pre-Commit Hooks

Pre-commit hooks automatically run before each commit to ensure code quality.

### Setup
```bash
pip install pre-commit
pre-commit install
```

### What Gets Checked
✅ **Black** - Code formatting (100 char line length)
✅ **isort** - Import sorting
✅ **Flake8** - PEP8 linting
✅ **MyPy** - Type checking
✅ **Hadolint** - Dockerfile linting
✅ **Bandit** - Security scanning
✅ **YAML/JSON** - File validation
✅ **Trailing whitespace** - File cleanup

### Manual Run
```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### Bypass (Use Sparingly)
```bash
git commit --no-verify -m "Message"
```

## CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline runs on every push and pull request.

### Pipeline Stages

#### 1. Code Quality Checks
- Black formatting
- isort import sorting
- Flake8 linting
- MyPy type checking
- Bandit security scanning

#### 2. Unit Tests
- Fast tests with mocking
- Code coverage reporting
- Uploads to Codecov

#### 3. Docker Build & Validation
- Builds Docker image
- Validates image structure
- Scans for vulnerabilities (Trivy)

#### 4. E2E Tests
- Tests full Docker container
- Validates all API endpoints
- Tests web UI functionality

#### 5. Build & Push (main branch only)
- Builds production image
- Pushes to GitHub Container Registry
- Tags with version and latest

### Viewing Results
- Go to Actions tab in GitHub
- Click on workflow run
- View job details and logs

## Code Style Guide

### Python (PEP8 + Black)
- Line length: 100 characters
- Use double quotes for strings
- Use trailing commas in multi-line structures

### Type Hints
```python
def function_name(param: str) -> Dict[str, Any]:
    """Docstring here."""
    pass
```

### Imports
```python
# Standard library
import os
import sys

# Third party
from fastapi import FastAPI
import torch

# Local
from app.model import AIDetector
```

## Testing

### Unit Tests
```bash
pytest tests/test_unit_model.py -v
```

### E2E Tests
```bash
cd tests
python test_e2e_docker.py
```

### With Coverage
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

## Docker Development

### Build Locally
```bash
docker build -t verif:dev .
```

### Run Locally
```bash
docker run -p 8000:8000 verif:dev
```

### Debug Container
```bash
docker run -it --entrypoint /bin/bash verif:dev
```

## Managing Conda Environment

### Update Environment
```bash
# After modifying environment.yml
conda env update -f environment.yml --prune
```

### Export Environment
```bash
# Export current environment (useful for reproducing issues)
conda env export > environment-locked.yml
```

### Remove Environment
```bash
conda deactivate
conda env remove -n verif
```

### Switch Between Environments
```bash
# Deactivate current environment
conda deactivate

# Activate desired environment
conda activate verif
```

## Troubleshooting

### Pre-commit fails
```bash
# Update hooks
pre-commit autoupdate

# Clean cache
pre-commit clean
```

### Conda environment conflicts
```bash
# Remove and recreate environment
conda env remove -n verif
conda env create -f environment.yml
```

### Type checking errors
```bash
# Install type stubs
pip install types-requests
```

### Docker cache issues
```bash
docker builder prune
docker build --no-cache -t verif:dev .
```

### Import errors in IDE
Make sure your IDE is using the correct Python interpreter:
- **VSCode**: Select conda environment (Ctrl+Shift+P → "Python: Select Interpreter")
- **PyCharm**: Settings → Project → Python Interpreter → Select conda env
