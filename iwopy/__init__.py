"""
Fraunhofer IWES optimization tools in Python

"""

from .core import Problem as Problem
from .core import Objective as Objective
from .core import Constraint as Constraint
from .core import Memory as Memory

from .wrappers import ProblemWrapper as ProblemWrapper
from .wrappers import DiscretizeRegGrid as DiscretizeRegGrid
from .wrappers import LocalFD as LocalFD
from .wrappers import SimpleProblem as SimpleProblem
from .wrappers import SimpleObjective as SimpleObjective
from .wrappers import SimpleConstraint as SimpleConstraint

from . import utils as utils
from . import interfaces as interfaces
from . import benchmarks as benchmarks
from . import optimizers as optimizers

import importlib
from pathlib import Path

try:
    tomllib = importlib.import_module("tomllib")
    source_location = Path(__file__).parent
    if (source_location.parent / "pyproject.toml").exists():
        with open(source_location.parent / "pyproject.toml", "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    else:
        __version__ = importlib.metadata.version(__package__ or __name__)
except ModuleNotFoundError:
    __version__ = importlib.metadata.version(__package__ or __name__)
