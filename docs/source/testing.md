# Testing

For testing, please clone the repository and synchronize the required dependencies:

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
cd iwopy
uv sync --extra test --extra dev
```

Run the test suite and type checker with

```console
uv run pytest tests
uv run mypy iwopy
```

Run all repository checks with

```console
uv run pre-commit run --all-files
```
