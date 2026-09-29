from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from iwopy.core.problem import Problem


class ProblemWrapper(Problem):
    """
    Generic abstract problem wrapper class.

    Attributes
    ----------
    base_problem: iwopy.Problem
        The underlying concrete problem

    :group: wrappers

    """

    def __init__(self, base_problem: Problem, name: str, **kwargs: Any) -> None:
        """
        Constructor

        Parameters
        ----------
        base_problem
            The underlying concrete problem
        name
            The problem name
        kwargs
            Additional parameters for the Problem class

        """
        super().__init__(name, **kwargs)
        self.base_problem = base_problem

    def __getattr__(self, name: str) -> Any:
        return super().__getattribute__("base_problem").__getattribute__(name)

    def var_names_int(self) -> list[str]:
        """
        The names of integer variables.

        Returns
        -------
        names: list of str
            The names of the integer variables

        """
        return self.base_problem.var_names_int()

    def initial_values_int(self) -> ArrayLike:
        """
        The initial values of the integer variables.

        Returns
        -------
        values: numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return self.base_problem.initial_values_int()

    def min_values_int(self) -> ArrayLike:
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return self.base_problem.min_values_int()

    def max_values_int(self) -> ArrayLike:
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return self.base_problem.max_values_int()

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return self.base_problem.var_names_float()

    def initial_values_float(self) -> ArrayLike | None:
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return self.base_problem.initial_values_float()

    def min_values_float(self) -> ArrayLike:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return self.base_problem.min_values_float()

    def max_values_float(self) -> ArrayLike:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return self.base_problem.max_values_float()

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        if not self.base_problem.initialized:
            self.base_problem.initialize(verbosity)

        self.objs = self.base_problem.objs
        self.cons = self.base_problem.cons

        for objective in self.objs.functions:
            objective.problem = self
        self.objs.problem = self

        for constraint in self.cons.functions:
            constraint.problem = self
        self.cons.problem = self

        super().initialize(verbosity)

    def apply_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> object | None:
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        return self.base_problem.apply_individual(vars_int, vars_float)

    def apply_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> object | None:
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        return self.base_problem.apply_population(vars_int, vars_float)

    def finalize_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray, verbosity: int = 1
    ) -> tuple[object | None, np.ndarray, np.ndarray]:
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float
            The optimal float variable values, shape: (n_vars_float,)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem
        objs: np.array
            The objective function values, shape: (n_objectives,)
        cons: np.array
            The constraints values, shape: (n_constraints,)

        """
        return self.base_problem.finalize_individual(vars_int, vars_float, verbosity)

    def finalize_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray, verbosity: int = 0
    ) -> tuple[object | None, np.ndarray, np.ndarray]:
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem
        objs: np.array
            The final objective function values, shape: (n_pop, n_components)
        cons: np.array
            The final constraint values, shape: (n_pop, n_constraints)

        """
        return self.base_problem.finalize_population(vars_int, vars_float, verbosity)
