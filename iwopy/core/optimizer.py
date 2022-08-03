import numpy as np
from abc import ABCMeta, abstractmethod

from .base import Base


class Optimizer(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization solvers.

    Parameters
    ----------
    problem: iwopy.Problem
        The problem to optimize
    name: str, optional
        The name

    Attributes
    ----------
    problem: iwopy.Problem
        The problem to optimize

    """

    def __init__(self, problem, name=None):
        super().__init__(name)
        self.problem = problem

    def print_info(self):
        """
        Print solver info, called before solving
        """
        pass

    @abstractmethod
    def solve(self, verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.core.OptResults
            The optimization results object

        """

        # check problem initialization:
        if not self.problem.initialized:
            raise ValueError(
                f"Optimizer called for problem '{self.problem.name}'"
                + " before problem initialization"
            )

        # check solver initialization:
        if not self.initialized:
            raise ValueError(
                f"Optimizer called for problem '{self.problem.name}'"
                + " before solver initialization"
            )

        return None

    def finalize(self, opt_results, verbosity=1):
        """
        This function may be called after finishing
        the optimization.

        Parameters
        ----------
        opt_results: iwopy.OptResults
            The optimization results object
        verbosity : int
            The verbosity level, 0 = silent

        """
        if verbosity:

            print(f"{type(self).__name__}: Optimization run finished")
            print(f"  Success: {opt_results.success}")

            if opt_results is not None and opt_results.objs is not None:

                if self.problem.n_objectives == 1:

                    i0 = 0
                    for o in self.problem.objs.functions:
                        n = o.n_components()
                        i1 = i0 + n
                        names = o.component_names
                        if n == 1:
                            val = opt_results.objs[i0]
                            print(f"  Best {o.name} = {val}")
                        else:
                            for i in range(n):
                                val = opt_results.objs[i0 + i]
                                print(f"  Best {names[i]} = {val}")
                        i0 = i1

                else:

                    i0 = 0
                    for o in self.problem.objs.functions:
                        n = o.n_components()
                        i1 = i0 + n
                        names = o.component_names
                        if n == 1:
                            val = np.min(opt_results.objs[:, i0])
                            print(f"  Best {o.name} = {val}")
                        else:
                            for i in range(n):
                                val = np.min(opt_results.objs[:, i0 + i])
                                print(f"  Best {names[i]} = {val}")
                        i0 = i1
