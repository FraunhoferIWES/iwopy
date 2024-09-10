"""
Fraunhofer IWES optimization tools in Python
    
"""

from .core import Problem, Objective, Constraint, Memory
from .wrappers import (
    ProblemWrapper,
    DiscretizeRegGrid,
    LocalFD,
    SimpleProblem,
    SimpleObjective,
    SimpleConstraint,
)

from . import utils
from . import interfaces
from . import benchmarks
from . import optimizers

from importlib.metadata import version
__version__ = version(__package__ or __name__)
