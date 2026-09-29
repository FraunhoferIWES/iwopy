"""
Fraunhofer IWES optimization tools in Python

"""

# ruff: noqa: I001
import importlib
from importlib.metadata import version as package_version
from pathlib import Path

from .core import Problem as Problem
from .core import Objective as Objective
from .core import Constraint as Constraint
from .core import Memory as Memory
from .core import Pipeline as Pipeline
from .core import PipelineStage as PipelineStage
from .core import OptimizationHistory as OptimizationHistory
from .core import OptimizerCallback as OptimizerCallback
from .core import OptimizerCallbackData as OptimizerCallbackData
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

__version__: str
try:
    tomllib = importlib.import_module("tomllib")
    source_location = Path(__file__).parent
    if (source_location.parent / "pyproject.toml").exists():
        with open(source_location.parent / "pyproject.toml", "rb") as f:
            version_value = tomllib.load(f)["project"]["version"]
            if not isinstance(version_value, str):
                raise TypeError("Project version must be a string")
            __version__ = version_value
    else:
        __version__ = package_version(__package__ or __name__)
except ModuleNotFoundError:
    __version__ = package_version(__package__ or __name__)
