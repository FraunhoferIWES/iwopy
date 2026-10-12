# Testing

For testing, please clone the repository and synchronize the required dependencies:

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
cd iwopy
uv sync --extra test --extra dev
```

Run the quality hooks before the final test suite:

```console
uv run pre-commit run --all-files
uv run pytest tests
```

The hooks include Ruff, mypy, and file hygiene. Run the complete optimizer
catalog independently with:

```console
uv run pytest tests/interfaces/test_optimizer_examples.py -q
```

Installed backends run their cases; missing optional packages or native features
skip only their own cases. Developer test ownership and focused-check guidance
are in [docs/development.md](https://github.com/FraunhoferIWES/iwopy/blob/dev/docs/development.md).

GitHub Actions installs all solver extras for Python 3.10 through 3.14. A named
optimizer/catalog/benchmark regression stage runs before the full suite on
regular and release workflows. Backend imports are checked first, so missing
packages fail CI rather than silently reducing coverage. The regular workflow
also supports manual runs from the Actions tab without publishing packages.
