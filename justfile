# LOB Modeling - Justfile
# Modern command runner (replacement for Makefile)
# Usage: just <command>

# Default target
default:
    @just --list

# =============================================================================
# Setup & Dependencies
# =============================================================================

# Install all dependencies including dev tools
install:
    @echo "Installing dependencies with uv..."
    uv pip install -e ".[dev]"

# Install only production dependencies
install-prod:
    @echo "Installing production dependencies..."
    uv pip install -e "."

# Install webapp dependencies only
install-webapp:
    @echo "Installing webapp dependencies..."
    uv pip install -e ".[webapp]"

# Update all dependencies to latest compatible versions
update:
    @echo "Updating dependencies..."
    uv pip install -U -e ".[dev]"

# =============================================================================
# Testing
# =============================================================================

# Run all tests
test:
    @echo "Running tests..."
    uv run pytest

# Run tests with verbose output
test-verbose:
    @echo "Running tests (verbose)..."
    uv run pytest -v

# Run specific test file
test-file file:
    @echo "Running tests in {{file}}..."
    uv run pytest {{file}} -v

# Run tests with coverage
coverage:
    @echo "Running tests with coverage..."
    uv run pytest --cov=src/lob_modeling --cov-report=html --cov-report=term-missing

# Run tests and open coverage report
coverage-html:
    @echo "Running tests with coverage..."
    uv run pytest --cov=src/lob_modeling --cov-report=html
    @echo "Opening coverage report..."
    @xdg-open htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || echo "Coverage report generated at htmlcov/index.html"

# =============================================================================
# Linting & Formatting
# =============================================================================

# Run all lint checks
lint:
    @echo "Running flake8..."
    uv run flake8 src/lob_modeling tests
    @echo "Running black check..."
    uv run black --check src/lob_modeling tests
    @echo "Running isort check..."
    uv run isort --check-only src/lob_modeling tests
    @echo "Running pydocstyle check..."
    uv run pydocstyle src/lob_modeling
    @echo "✓ All lint checks passed!"

# Format code
format:
    @echo "Formatting code with black..."
    uv run black src/lob_modeling tests
    @echo "Sorting imports with isort..."
    uv run isort src/lob_modeling tests
    @echo "✓ Formatting complete!"

# Check formatting only (no changes)
check-format:
    @echo "Checking code formatting..."
    uv run black --check src/lob_modeling tests
    uv run isort --check-only src/lob_modeling tests
    @echo "✓ Formatting check complete!"

# Check docstring conventions
check-docstrings:
    @echo "Checking docstring conventions..."
    uv run pydocstyle src/lob_modeling

# Run type checking
typecheck:
    @echo "Running mypy type checker..."
    uv run mypy src/lob_modeling

# Run all quality checks
check: lint typecheck
    @echo "✓ All quality checks passed!"

# =============================================================================
# Running Models
# =============================================================================

# Run Kyle Model example
run-kyle:
    @echo "Running Kyle Model..."
    uv run python -c "from lob_modeling.models.kyle import KyleModel; KyleModel()"

# Run Almgren-Chriss Model example
run-almgren:
    @echo "Running Almgren-Chriss Model..."
    uv run python -c "from lob_modeling.models.almgren_chriss import AlmgrenChriss2000; AlmgrenChriss2000()"

# Run Glosten-Milgrom Model example
run-glosten:
    @echo "Running Glosten-Milgrom Model..."
    uv run python -c "from lob_modeling.models.glosten_milgrom import GlostenAndMilgromSimplest; GlostenAndMilgromSimplest()"

# Run Criscuolo-Waehlbroeck Model example
run-criscuolo:
    @echo "Running Criscuolo-Waehlbroeck Model..."
    uv run python -c "from lob_modeling.models.criscuolo_waehlbroeck import Criscuolo2014; Criscuolo2014()"

# Run De Prado Model example
run-deprado:
    @echo "Running De Prado Model..."
    uv run python -c "from lob_modeling.models.de_prado import DePrado2014; DePrado2014()"

# Run all model examples
run-all: run-kyle run-almgren run-glosten run-criscuolo
    @echo "✓ All models executed!"

# =============================================================================
# Webapp
# =============================================================================

# Start the webapp development server
dev:
    @echo "Starting webapp development server..."
    uv run uvicorn src.lob_modeling.webapp.main:app --reload --host 0.0.0.0 --port 8000

# Start webapp in production mode
serve:
    @echo "Starting webapp server..."
    uv run uvicorn src.lob_modeling.webapp.main:app --host 0.0.0.0 --port 8000

# =============================================================================
# Cleanup
# =============================================================================

# Clean build artifacts and cache
clean:
    @echo "Cleaning build artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name "htmlcov" -exec rm -rf {} +
    find . -type f -name ".coverage" -delete
    @echo "✓ Cleanup complete!"

# =============================================================================
# Help & Info
# =============================================================================

# Show project info
info:
    @echo "LOB Modeling Project"
    @echo "===================="
    @echo "Python: $(uv run python --version)"
    @echo ""
    @echo "Quick Start:"
    @echo "  just install      - Install all dependencies"
    @echo "  just test         - Run tests"
    @echo "  just lint         - Run lint checks"
    @echo "  just format       - Format code"
    @echo "  just dev          - Start webapp dev server"
    @echo ""

# Show this help message
help:
    @just --list
