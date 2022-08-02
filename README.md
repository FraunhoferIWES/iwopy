# iwopy
Fraunhofer IWES optimization tools in Python

![](Logo_IWOPY.svg)

## Overview
The `iwopy` package is in fact a meta package that provides interfaces to other open-source Python optimization packages out there. Currently this includes

- [pymoo](https://pymoo.org/index.html)
- [pygmo](https://esa.github.io/pygmo2/index.html)

The basic idea of `iwopy` is to provide abstract base classes, that can be concretized for any kind of problem by the users, and the corresponding solver interfaces.

`iwopy` can thus be understood as an attempt to provide _the best of all worlds_ when it comes to solving optimization problems with Python. Obviously all the credit for implementing the invoked optimizers goes to the original package providers.

Documentation: [https://fraunhoferiwes.github.io/iwopy.docs/index.html](https://fraunhoferiwes.github.io/iwopy.docs/index.html)

Source code: [https://github.com/FraunhoferIWES/iwopy](https://github.com/FraunhoferIWES/iwopy)

PyPi reference: [https://pypi.org/project/iwopy/](https://pypi.org/project/iwopy/)

## Requirements
The supported Python versions are: 
- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`

## Installation

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

### Developers

The first step as a developer is to clone the `iwopy` repository by
```console
git clone https://github.com/FraunhoferIWES/iwopy.git
```
Enter the root directory by `cd iwopy`. Then you can either install from this directory via
```console
pip install -e .
```
Alternatively, add the `iwopy` directory to your `PYTHONPATH`, e.g. by running
```console
export PYTHONPATH=`pwd`:$PYTHONPATH
```
from the root `iwopy` directory, and then
```console
pip install -r requirements.txt
```
