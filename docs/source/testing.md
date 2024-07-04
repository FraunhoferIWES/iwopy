# Testing

For testing, please clone the repository and install the required dependencies
(`flake8`, `pytest`, `pygmo`, `pymoo`):

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
cd iwopy
pip install .[test]
```

If you are a developer you might want to replace the last line by
```console
pip install -e .[test]
```
for dynamic installation from the local code base.

The tests are then run by
```console
pytest tests
```
