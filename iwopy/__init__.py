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

from importlib.resources import read_text

__version__ = read_text(__package__, "VERSION")
