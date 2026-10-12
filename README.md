# iwopy

Fraunhofer IWES optimization tools in Python

<picture>
	<source media="(prefers-color-scheme: dark)" srcset="Logo_IWOPY_white.svg">
	<img alt="IWOPY logo" src="Logo_IWOPY.svg" width="100%">
</picture>

## Overview

Define your optimization problem once and use native or external solvers through
a shared Python interface. Available external integrations include:

- [pymoo](https://pymoo.org/index.html)
- [pygmo](https://esa.github.io/pygmo2/index.html)
- [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [pycma](https://github.com/CMA-ES/pycma)
- [Nevergrad](https://facebookresearch.github.io/nevergrad/)
- [Optuna](https://optuna.org/)
- [DEAP](https://deap.readthedocs.io/)

Native `GG` and `SLSQP` optimizers are included. `LocalFD` supplies finite-difference
derivatives; `DiscretizeRegGrid` is the regular-grid problem wrapper. Population
evaluation batches candidates or derivative points when the chosen backend supports it.

The [optimizer catalog](https://fraunhoferiwes.github.io/iwopy/optimizer_catalog.html)
lists every selectable optimizer with a short description and a runnable,
explicit Python setup. 

Documentation: [https://fraunhoferiwes.github.io/iwopy](https://fraunhoferiwes.github.io/iwopy)

Source code: [https://github.com/FraunhoferIWES/iwopy](https://github.com/FraunhoferIWES/iwopy)

PyPi reference: [https://pypi.org/project/iwopy/](https://pypi.org/project/iwopy/)

Anaconda reference: [https://anaconda.org/conda-forge/iwopy](https://anaconda.org/conda-forge/iwopy)

## Citation

Please cite the JOSS paper [IWOPY: Fraunhofer IWES optimization tools in
Python](https://doi.org/10.21105/joss.06014)

 [![DOI](https://joss.theoj.org/papers/10.21105/joss.06014/status.svg)](https://doi.org/10.21105/joss.06014)

BibTeX is available on the [citation page](https://fraunhoferiwes.github.io/iwopy/citation.html).

## Requirements

Supported Python versions: 3.10 through 3.14. NumPy, SciPy, and matplotlib are
base dependencies.

## Installation

Install in an active virtual environment:

```console
pip install iwopy
```

Optional extras are `cma`, `nevergrad`, `optuna`, `deap`, `pymoo`, and `pygmo`;
use `pip install 'iwopy[opt]'` for all backends. SciPy differential evolution
needs no extra. Nevergrad currently requires NumPy below 2.4.

Conda users can install the conda-forge package:

```console
conda install -c conda-forge iwopy
```

See the [installation guide](https://fraunhoferiwes.github.io/iwopy/installation.html)
for environment setup and development-branch installation.

## Development

```console
git clone https://github.com/FraunhoferIWES/iwopy.git
cd iwopy
uv sync --extra dev --extra test
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest tests
```

Follow [AGENTS.md](AGENTS.md) and [docs/development.md](docs/development.md) for
test ownership and completion gates. Contributions are welcome through
[pull requests](https://github.com/FraunhoferIWES/iwopy/pulls).

## Support

For trouble shooting and support, please
- raise an issue [here](https://github.com/FraunhoferIWES/iwopy/issues),
- or start a discussion [here](https://github.com/FraunhoferIWES/iwopy/discussions),
- or contact the contributers.

Thanks for your help with improving *iwopy*!
