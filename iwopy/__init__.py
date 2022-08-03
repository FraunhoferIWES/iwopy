from .core import Problem, Objective, Constraint, Memory
from .problems import SimpleProblem
from .wrappers import ProblemWrapper, DiscretizeRegGrid

from . import utils
from . import interfaces
from . import benchmarks

from importlib.resources import read_text

__version__ = read_text(__package__, "VERSION")
