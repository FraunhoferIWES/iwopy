# Installation

## Requirements

The supported Python versions are:

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`
- `Python 3.11`
- `Python 3.12`

## Installation via pip

### Virtual Python environment

We recommend working in a Python virtual environment and install `iwopy` there. Such an environment can be created by

```console
python -m venv /path/to/my_venv
```

and afterwards be activated by

```console
source /path/to/my_venv/bin/activate
```

Note that in the above commands `/path/to/my_venv` is a placeholder that should be replaced by a path to a (non-existing) folder of your choice, for example `~/venv/iwopy`.

All subsequent installation commands via `pip` can then be executed directly within the active environment without changes. After your work with `iwopy` is done you can leave the environment by the command `deactivate`.

### Standard users

As a standard user, you can install the latest release via [pip](https://pypi.org/project/iwopy/) by

```console
pip install iwopy
```

This in general corresponds to the `main` branch at [github](https://github.com/FraunhoferIWES/iwopy). Alternatively, you can decide to install the latest pre-release developments (non-stable) by

```console
pip install git+https://github.com/FraunhoferIWES/iwopy@dev#egg=iwopy
```

Notice that the above default installation does not install the third-party optimization
packages. `iwopy` will tell you in an error message that it is missing a package, with
a hint of installation advice. You can avoid this step by installing all supported
optimzer packages by installing those optoinal packages by addig `[opt]`:

```console
pip install iwopy[opt]
```

or

```console
pip install git+https://github.com/FraunhoferIWES/iwopy@dev#egg=iwopy[opt]
```

### Developers

The first step as a developer is to clone the `iwopy` repository by

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
```

Enter the root directory by

```console
cd iwopy
```

Then you can either install from this directory via

```console
pip install -e .
```

Notice that the above default installation does not install the third-party optimization
packages. `iwopy` will tell you in an error message that it is missing a package, with
a hint of installation advice. You can avoid this step by installing all supported
optimzer packages by installing those optoinal packages by addig `[opt]`:

```console
pip install -e .[opt]
```

## Installation via conda

### Preparation (optional)

It is strongly recommend to use the `libmamba` dependency solver instead of the default solver. Install it once by

```console
conda install conda-libmamba-solver -n base -c conda-forge
```

We recommend that you set this to be your default solver, by

```console
conda config --set solver libmamba
```

### Standard users

The `iwopy` package is available on the channel [conda-forge](https://anaconda.org/conda-forge/iwopy). You can install the latest version by

```console
conda install -c conda-forge iwopy
```

### Developers

For developers using `conda`, we recommend first installing `iwopy` as described above, then removing only the `iwopy` package while keeping the dependencies, and then adding `iwopy` again from a git using `conda develop`:

```console
conda install iwopy conda-build -c conda-forge
conda remove iwopy --force
git clone https://github.com/FraunhoferIWES/iwopy.git
cd iwopy
conda develop .
```

Concerning the `git clone` line, we actually recommend that you fork `iwopy` on GitHub and then replace that command by cloning your fork instead.
