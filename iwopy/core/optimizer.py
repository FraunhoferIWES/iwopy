import numpy as np
from abc import ABCMeta, abstractmethod

from iwopy.utils import new_instance
from .base import Base


class Optimizer(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization solvers.

    Attributes
    ----------
    problem: iwopy.Problem
        The problem to optimize
    name: str
        The name

    :group: core

    """

    def __init__(self, problem, name="optimizer"):
        """
        Constructor

        Parameters
        ----------
        problem: iwopy.Problem
            The problem to optimize
        name: str
            The name

        """
        super().__init__(name)
        self.problem = problem
        self.name = name

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
        verbosity: int
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
        verbosity: int
            The verbosity level, 0 = silent

        """
        if verbosity:

            print(f"{type(self).__name__}: Optimization run finished")
            if (
                isinstance(opt_results.success, bool)
                or len(opt_results.success.flat) == 1
            ):
                print(f"  Success: {opt_results.success}")
            else:
                v = np.sum(opt_results.success) / len(opt_results.success.flat)
                print(f"  Success: {100*v:.2f} %")

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
                            if self.problem.maximize_objs[i0]:
                                val = np.max(opt_results.objs[:, i0])
                            else:
                                val = np.min(opt_results.objs[:, i0])
                            print(f"  Best {o.name} = {val}")
                        else:
                            for i in range(n):
                                if self.problem.maximize_objs[i0 + 1]:
                                    val = np.max(opt_results.objs[:, i0 + i])
                                else:
                                    val = np.min(opt_results.objs[:, i0 + i])
                                print(f"  Best {names[i]} = {val}")
                        i0 = i1


    @classmethod
    def new(cls, optimizer_type, *args, **kwargs):
        """
        Run-time optimizer factory.

        Parameters
        ----------
        optimizer_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, optimizer_type, *args, **kwargs)
        