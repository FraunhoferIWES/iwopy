# Installation

## Requirements

Supported Python versions are 3.10 through 3.14. NumPy, SciPy, and matplotlib
are installed as base dependencies.

## Installation via pip

### Virtual Python environment

Create and activate a virtual environment before installing:

```console
python -m venv .venv
source .venv/bin/activate
```

The activation command above is for POSIX shells. On Windows, use
`.venv\Scripts\activate` for Command Prompt or `.venv\Scripts\Activate.ps1`
for PowerShell. Run `deactivate` to leave the environment.

### Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/iwopy/) by

```console
pip install iwopy
```

To install the unreleased development branch instead:

```console
pip install 'iwopy @ git+https://github.com/FraunhoferIWES/iwopy@dev'
```

SciPy and the iwopy optimizers are included. Install an optional backend using
its extra: `cma`, `nevergrad`, `optuna`, `deap`, `pymoo`, or `pygmo`. For example:

```console
pip install 'iwopy[optuna]'
```

Use `opt` for all supported backends:

```console
pip install 'iwopy[opt]'
```

The [optimizer catalog](optimizer_catalog.rst) contains a runnable setup for
each optimizer. Missing optional packages produce an installation hint.
The Nevergrad extra currently requires NumPy below 2.4; the catalog explains
the upstream limitation.

### Developers

The first step as a developer is to clone the `iwopy` repository by

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
```

Enter the root directory by

```console
cd iwopy
```

Synchronize the development and test dependencies with

```console
uv sync --extra test --extra dev
```

Install the pre-commit hooks with

```console
uv run pre-commit install
```

## Installation via conda

### Conda installation

The `iwopy` package is available on the channel [conda-forge](https://anaconda.org/conda-forge/iwopy). You can install the latest version by

```console
conda install -c conda-forge iwopy
```

### Development workflow

Use the repository's `uv` workflow above for development, even when conda is
used for other environments. See [Testing](testing.md) for the quality gates.
