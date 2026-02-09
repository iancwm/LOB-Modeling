# Agent Instructions for LOB-Modeling

This document provides guidance for AI agents working in the `LOB-Modeling` repository.

## Essential Commands

- **Install dependencies**: `make install` or `pip3 install -r requirements.txt`
- **Run linter**: `make lint` or `flake8 src/lob_modeling tests`
- **Run tests**: `make test` or `python3 -m unittest discover tests`
- **Clean build artifacts**: `make clean`

There are also `make` commands to run specific models:
- `make run-kyle`
- `make run-almgren`
- `make run-glosten`
- `make run-criscuolo`

## Code Organization

- The main source code is located in the `src/lob_modeling` directory.
- Financial models are implemented as classes in `src/lob_modeling/models`.
- Utility functions are expected to be in `src/lob_modeling/utils`.
- Tests are located in the `tests` directory.

## Naming Conventions and Style Patterns

- The project follows PEP 8 style guidelines.
- Class names are in PascalCase (e.g., `KyleModel`).
- Method and variable names are in snake_case (e.g., `one_period_price`).
- Docstrings are used to explain the purpose of classes and methods.

## Testing Approach

- The project uses the built-in `unittest` framework for testing.
- Tests are located in the `tests` directory.
- Test files are named with a `test_` prefix (e.g., `test_models.py`).
- Test methods are named with a `test_` prefix (e.g., `test_kyle_init`).
